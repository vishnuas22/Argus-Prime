"""
Heatmap Generator - Visual Deepfake Evidence.

Generates interpretable heatmaps using Grad-CAM and attention mechanisms
for visual deepfake detection evidence.

Algorithms:
- Grad-CAM: Gradient-weighted class activation mapping
- Attention-based: Multi-head attention visualization
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_config
from src.core.logging import logger


@dataclass
class HeatmapResult:
    """Result from heatmap generation."""
    heatmap_overlay: np.ndarray
    heatmap_mask: np.ndarray
    regions_of_interest: List[Dict[str, Any]]
    manipulation_score: float
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "heatmap_overlay_path": None,
            "heatmap_mask_path": None,
            "regions_of_interest": self.regions_of_interest,
            "manipulation_score": self.manipulation_score,
            "confidence": self.confidence,
        }


class HeatmapGenerator:
    """
    Heatmap Generator for Deepfake Detection Explainability.

    Generates visual evidence heatmaps using:
    - Grad-CAM for CNN-based models
    - Attention visualization for transformer models
    - Multi-scale feature fusion

    Attributes:
        target_layer: Layer to extract gradients from
        alpha: Overlay transparency
        threshold: Threshold for ROI detection
    """

    def __init__(
        self,
        target_layer: str = "layer4",
        alpha: float = 0.5,
        threshold: float = 0.6,
        use_gpu: bool = True,
    ):
        """
        Initialize heatmap generator.

        Args:
            target_layer: Layer to extract gradients from
            alpha: Heatmap overlay transparency
            threshold: Threshold for ROI detection
            use_gpu: Whether to use GPU acceleration
        """
        self.target_layer = target_layer
        self.alpha = alpha
        self.threshold = threshold
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

        logger.info(
            f"HeatmapGenerator initialized: target_layer={target_layer}, "
            f"device={self.device}"
        )

    def _preprocess_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> torch.Tensor:
        """Preprocess image for model input."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std

        image = cv2.resize(image, target_size)
        image = image.transpose(2, 0, 1)
        tensor = torch.from_numpy(image).unsqueeze(0)

        return tensor.to(self.device)

    def _postprocess_heatmap(
        self,
        heatmap: np.ndarray,
        original_size: Tuple[int, int],
    ) -> np.ndarray:
        """Postprocess heatmap for visualization."""
        heatmap = cv2.resize(heatmap, original_size)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = (heatmap * 255).astype(np.uint8)

        return heatmap

    def generate_gradcam(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            model: PyTorch model
            input_tensor: Input image tensor
            target_class: Target class for gradients

        Returns:
            Heatmap array
        """
        model.eval()
        input_tensor.requires_grad_(True)

        output = model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        one_hot = one_hot.to(self.device)

        output.backward(gradient=one_hot, retain_graph=True)

        gradients = input_tensor.grad
        if gradients is None:
            logger.warning("No gradients computed")
            return np.zeros(input_tensor.shape[2:], dtype=np.float32)

        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        features = None
        for name, module in model.named_modules():
            if name == self.target_layer:
                module.register_backward_hook(
                    lambda module, grad_input, grad_output, name=name: (
                        setattr(self, f'_grad_{name}', grad_output[0])
                    )
                )
                features = self._get_features(model, input_tensor)
                break

        if features is None:
            features = output

        if isinstance(features, tuple):
            features = features[0]

        if features.dim() == 4:
            heatmap = torch.mean(features, dim=1)[0].detach().cpu().numpy()
        else:
            heatmap = features.detach().cpu().numpy()

        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (input_tensor.shape[3], input_tensor.shape[2]))

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def _get_features(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Extract features from target layer."""
        features = None

        def hook_fn(module, input, output):
            nonlocal features
            features = output

        for name, module in model.named_modules():
            if name == self.target_layer:
                hook = module.register_forward_hook(hook_fn)
                break

        with torch.no_grad():
            _ = model(input_tensor)

        return features

    def generate_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        input_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Generate heatmap from attention weights.

        Args:
            attention_weights: Attention weights tensor
            input_size: Original input size

        Returns:
            Heatmap array
        """
        if attention_weights.dim() == 3:
            attention_weights = attention_weights.mean(dim=1)

        attention_weights = attention_weights.squeeze()

        if attention_weights.dim() == 1:
            seq_len = attention_weights.shape[0]
            n = int(np.sqrt(seq_len))
            if n * n == seq_len:
                attention_map = attention_weights.reshape(n, n)
            else:
                attention_map = attention_weights.unsqueeze(0)
        else:
            attention_map = attention_weights

        attention_map = attention_map.detach().cpu().numpy()
        attention_map = cv2.resize(attention_map, input_size)

        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

        return attention_map

    def create_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Create heatmap overlay on original image.

        Args:
            original_image: Original image
            heatmap: Heatmap array
            colormap: OpenCV colormap

        Returns:
            Overlay image
        """
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            colormap,
        )

        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(
            original_rgb,
            1 - self.alpha,
            heatmap_colored,
            self.alpha,
            0,
        )

        return overlay

    def detect_regions_of_interest(
        self,
        heatmap: np.ndarray,
        min_area: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Detect regions of interest from heatmap.

        Args:
            heatmap: Heatmap array
            min_area: Minimum contour area

        Returns:
            List of ROIs with bounding boxes and scores
        """
        _, thresholded = cv2.threshold(
            (heatmap * 255).astype(np.uint8),
            int(255 * self.threshold),
            255,
            cv2.THRESH_BINARY,
        )

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            mask = np.zeros(heatmap.shape, dtype=np.float32)
            cv2.drawContours(mask, [contour], -1, 1.0, -1)

            score = float(np.mean(heatmap * mask))

            regions.append({
                "bbox": [x, y, x + w, y + h],
                "area": int(area),
                "score": score,
                "centroid": [int(x + w / 2), int(y + h / 2)],
            })

        regions.sort(key=lambda r: r["score"], reverse=True)

        return regions

    def generate_heatmap(
        self,
        model: nn.Module,
        image: np.ndarray,
        target_class: Optional[int] = None,
        return_overlay: bool = True,
    ) -> HeatmapResult:
        """
        Generate complete heatmap result.

        Args:
            model: PyTorch model
            image: Input image
            target_class: Target class for visualization
            return_overlay: Whether to return overlay image

        Returns:
            HeatmapResult with all visualization data
        """
        original_size = (image.shape[1], image.shape[0])

        input_tensor = self._preprocess_image(image)

        heatmap = self.generate_gradcam(model, input_tensor, target_class)

        heatmap_mask = self._postprocess_heatmap(heatmap, original_size)

        regions = self.detect_regions_of_interest(heatmap)

        manipulation_score = float(np.mean(heatmap))

        confidence = float(np.std(heatmap))

        if return_overlay:
            heatmap_overlay = self.create_overlay(image, heatmap_mask)
        else:
            heatmap_overlay = heatmap_mask

        return HeatmapResult(
            heatmap_overlay=heatmap_overlay,
            heatmap_mask=heatmap_mask,
            regions_of_interest=regions,
            manipulation_score=manipulation_score,
            confidence=confidence,
        )

    def save_heatmap(
        self,
        heatmap_result: HeatmapResult,
        output_path: str,
        overlay_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Save heatmap visualization to files.

        Args:
            heatmap_result: Heatmap result to save
            output_path: Path for heatmap mask
            overlay_path: Optional path for overlay image

        Returns:
            Dictionary with saved file paths
        """
        cv2.imwrite(
            output_path,
            (heatmap_result.heatmap_mask * 255).astype(np.uint8),
        )

        saved = {"mask_path": output_path}

        if overlay_path is not None and heatmap_result.heatmap_overlay is not None:
            overlay_bgr = cv2.cvtColor(
                heatmap_result.heatmap_overlay,
                cv2.COLOR_RGB2BGR,
            )
            cv2.imwrite(overlay_path, overlay_bgr)
            saved["overlay_path"] = overlay_path

        return saved


def create_heatmap_generator() -> HeatmapGenerator:
    """Factory function for HeatmapGenerator."""
    config = get_config()
    return HeatmapGenerator(
        target_layer="layer4",
        alpha=0.5,
        threshold=0.6,
        use_gpu=True,
    )
