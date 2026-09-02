"""
Adversarial Defender - Multi-stage Input Sanitization.

Implements multi-stage purification pipeline:
1. JPEG Compression (breaks pixel-level attacks)
2. Gaussian Noise (breaks gradient-based attacks)
3. Total Variation Minimization (breaks diffusion-based attacks)
4. TRIM Analysis (prediction entropy + KL divergence)
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.config import get_config
from src.core.logging import logger
from src.defense import TRIMAnalyzer, TRIMResult


@dataclass
class AdversarialCheckResult:
    """Result from adversarial check."""
    is_adversarial: bool
    purification_applied: bool
    purification_stage: Optional[str]
    trim_result: Optional[TRIMResult]
    confidence: float
    details: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_adversarial": self.is_adversarial,
            "purification_applied": self.purification_applied,
            "purification_stage": self.purification_stage,
            "trim_result": self.trim_result.to_dict() if self.trim_result else None,
            "confidence": self.confidence,
            "details": self.details or {},
        }


class AdversarialDefender:
    """
    Multi-stage Adversarial Defense.

    Purifies potentially adversarial inputs through multiple stages:
    1. JPEG Compression
    2. Gaussian Noise Injection
    3. Total Variation Minimization
    4. TRIM Analysis

    Attributes:
        jpeg_quality: JPEG compression quality (0-100)
        noise_std: Standard deviation for Gaussian noise
        tvm_weight: Weight for TVM denoising
        trim_threshold: Threshold for TRIM detection
    """

    def __init__(
        self,
        jpeg_quality: int = 85,
        noise_std: float = 0.01,
        tvm_weight: float = 0.1,
        trim_threshold: float = 0.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize adversarial defender.

        Args:
            jpeg_quality: JPEG compression quality
            noise_std: Gaussian noise standard deviation
            tvm_weight: Total variation minimization weight
            trim_threshold: TRIM detection threshold
            device: Computation device
        """
        self.jpeg_quality = jpeg_quality
        self.noise_std = noise_std
        self.tvm_weight = tvm_weight
        self.trim_threshold = trim_threshold
        self.device = device or torch.device("cpu")

        self.trim_analyzer = TRIMAnalyzer(threshold=trim_threshold)

        config = get_config()
        self.defense_config = config.defense

        logger.info(
            f"AdversarialDefender initialized: jpeg_quality={jpeg_quality}, "
            f"noise_std={noise_std}"
        )

    def _jpeg_compress(
        self,
        image: np.ndarray,
        quality: int = None,
    ) -> np.ndarray:
        """
        Apply JPEG compression to break pixel-level attacks.

        Args:
            image: Input image (BGR format)
            quality: JPEG quality (0-100)

        Returns:
            Compressed image
        """
        quality = quality or self.jpeg_quality

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode(".jpg", image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

        return decoded

    def _apply_gaussian_noise(
        self,
        image: np.ndarray,
        std: float = None,
    ) -> np.ndarray:
        """
        Apply Gaussian noise to break gradient-based attacks.

        Args:
            image: Input image
            std: Noise standard deviation

        Returns:
            Noisy image
        """
        std = std or self.noise_std

        noise = np.random.normal(0, std, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return noisy

    def _total_variation_denoise(
        self,
        image: np.ndarray,
        weight: float = None,
    ) -> np.ndarray:
        """
        Apply total variation minimization.

        Args:
            image: Input image
            weight: TVM weight parameter

        Returns:
            Denoised image
        """
        weight = weight or self.tvm_weight

        try:
            from skimage.restoration import denoise_tv_chambolle
            denoised = denoise_tv_chambolle(
                image.astype(np.float64) / 255.0,
                weight=weight,
                channel_axis=2 if len(image.shape) == 3 else None,
            )
            denoised = (denoised * 255).astype(np.uint8)
            return denoised
        except ImportError:
            logger.warning("scikit-image not available, skipping TVM")
            return image

    def _purify_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply full purification pipeline.

        Args:
            image: Input image

        Returns:
            Purified image
        """
        purified = image.copy()

        purified = self._jpeg_compress(purified)
        purified = self._apply_gaussian_noise(purified)
        purified = self._total_variation_denoise(purified)

        return purified

    def check_and_purify(
        self,
        image: np.ndarray,
        model: Optional[torch.nn.Module] = None,
        model_predict_fn: Optional[callable] = None,
    ) -> Tuple[np.ndarray, AdversarialCheckResult]:
        """
        Check if input is adversarial and purify if needed.

        Args:
            image: Input image (BGR format)
            model: Optional model for TRIM analysis
            model_predict_fn: Optional function for model prediction

        Returns:
            Tuple of (purified_image, adversarial_check_result)
        """
        details = {
            "original_shape": image.shape,
            "jpeg_quality": self.jpeg_quality,
            "noise_std": self.noise_std,
            "tvm_weight": self.tvm_weight,
        }

        trim_result = None

        if model is not None and model_predict_fn is not None:
            original_tensor = self._image_to_tensor(image)
            with torch.no_grad():
                original_pred = model_predict_fn(original_tensor)

            purified = self._purify_image(image)
            purified_tensor = self._image_to_tensor(purified)
            with torch.no_grad():
                purified_pred = model_predict_fn(purified_tensor)

            trim_result = self.trim_analyzer.analyze_predictions(
                original_pred,
                purified_pred,
            )

            details["trim_analysis"] = trim_result.to_dict()
        else:
            purified = self._purify_image(image)

        is_adversarial = trim_result.is_adversarial if trim_result else False
        purification_applied = True
        purification_stage = "multi_stage" if is_adversarial else None

        confidence = trim_result.confidence if trim_result else 0.5

        result = AdversarialCheckResult(
            is_adversarial=is_adversarial,
            purification_applied=purification_applied,
            purification_stage=purification_stage,
            trim_result=trim_result,
            confidence=confidence,
            details=details,
        )

        return purified, result

    def check_only(
        self,
        image: np.ndarray,
        model: torch.nn.Module,
        model_predict_fn: callable,
    ) -> AdversarialCheckResult:
        """
        Check if input is adversarial without purification.

        Args:
            image: Input image
            model: Model for predictions
            model_predict_fn: Function for model prediction

        Returns:
            AdversarialCheckResult
        """
        original_tensor = self._image_to_tensor(image)

        with torch.no_grad():
            original_pred = model_predict_fn(original_tensor)

        trim_result = self.trim_analyzer.analyze_single(original_pred)

        return AdversarialCheckResult(
            is_adversarial=trim_result.is_adversarial,
            purification_applied=False,
            purification_stage=None,
            trim_result=trim_result,
            confidence=trim_result.confidence,
            details={
                "original_shape": image.shape,
                "trim_analysis": trim_result.to_dict(),
            },
        )

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert image to model input tensor."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def purify_batch(
        self,
        images: list,
    ) -> list:
        """
        Purify a batch of images.

        Args:
            images: List of images

        Returns:
            List of purified images
        """
        return [self._purify_image(img) for img in images]

    def get_defense_info(self) -> Dict[str, Any]:
        """Get defense configuration information."""
        return {
            "jpeg_quality": self.jpeg_quality,
            "noise_std": self.noise_std,
            "tvm_weight": self.tvm_weight,
            "trim_threshold": self.trim_threshold,
            "device": str(self.device),
        }


def create_adversarial_defender() -> AdversarialDefender:
    """Factory function for AdversarialDefender."""
    config = get_config()
    return AdversarialDefender(
        jpeg_quality=config.defense.jpeg_quality,
        noise_std=config.defense.noise_std,
        trim_threshold=config.defense.trim_threshold,
    )
