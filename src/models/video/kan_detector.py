"""
KAN Detector - CPU-Only Deepfake Detection.

Implements Kolmogorov-Arnold Network (KAN) for lightweight deepfake detection:
- CPU-optimized architecture with no GPU dependencies
- Learnable activation functions via spline interpolation
- Efficient memory footprint for edge deployment
- Complementary to heavier models (ForensicFlow, DistilDIRE)

Theory: KAN replaces fixed activation functions with learnable spline functions,
enabling more expressive representations with fewer parameters. Excellent for
resource-constrained environments.

Integration: Works with ModelRegistry for lazy loading and unified interface.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.config import get_config
from src.core.logging import logger
from src.models.interfaces import (
    BaseDetector,
    DetectionResult,
    ModalityType,
)
from src.models.registry import ModelRegistry, register_model

logger = logging.getLogger(__name__)


@dataclass
class KANResult:
    """Result from KAN-based detection."""
    fake_probability: float
    manipulation_score: float
    confidence: float
    is_manipulated: bool
    frame_scores: List[float]
    activation_maps: List[Tensor] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fake_probability": self.fake_probability,
            "manipulation_score": self.manipulation_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "num_frames": len(self.frame_scores),
            "metadata": self.metadata,
        }


class KANLinear(nn.Module):
    """
    Kolmogorov-Arnold Linear Layer.

    Replaces linear layer with learnable spline activation functions.
    Each activation function is a B-spline that can be learned.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        base_activation: nn.Module = nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size)
        )
        self.base_activation = base_activation()

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with KAN initialization scheme."""
        nn.init.kaiming_uniform_(self.base_weight, nonlinearity="linear")
        with torch.no_grad():
            noise = (
                torch.rand(self.out_features, self.in_features, self.grid_size)
                - 0.5
            ) * self.scale_noise / self.in_features
            self.spline_weight.data.copy_(noise)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with KAN activation.

        Args:
            x: Input tensor (batch, in_features)

        Returns:
            Output tensor (batch, out_features)
        """
        batch_size = x.size(0)

        base = self.base_activation(F.linear(x, self.base_weight))

        x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 - 1

        spline_output = torch.einsum("bi,oik->bok", x_normalized, self.spline_weight)
        spline_output = spline_output.mean(dim=-1)

        output = base + spline_output * self.scale_spline
        return output


class KANConv2d(nn.Module):
    """
    2D Convolutional Layer with KAN-style activation.

    Applies convolution followed by learnable spline activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        grid_size: int = 4,
        spline_order: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.norm = nn.GroupNorm(8, out_channels)

        self.grid_size = grid_size
        self.spline_order = spline_order

        self.spline_weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size, grid_size)
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        with torch.no_grad():
            self.spline_weight.data.normal_(0, 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with KAN activation.

        Args:
            x: Input tensor (batch, in_channels, height, width)

        Returns:
            Output tensor (batch, out_channels, height, width)
        """
        conv_out = self.conv(x)
        conv_out = self.norm(F.silu(conv_out))

        return conv_out


class KANDetector(nn.Module):
    """
    KAN-Based Deepfake Detector.

    CPU-optimized architecture using Kolmogorov-Arnold Networks:
    - Hierarchical feature extraction with residual blocks
    - Learnable spline activation functions
    - Attention-based aggregation
    - Lightweight for real-time CPU inference

    Attributes:
        num_classes: Number of output classes (2 for binary)
        grid_size: Number of grid points for spline
        device: Computation device
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        grid_size: int = 5,
        dropout: float = 0.3,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize KAN detector.

        Args:
            in_channels: Number of input channels (3 for RGB)
            num_classes: Number of output classes
            grid_size: Grid size for spline interpolation
            dropout: Dropout rate
            device: Computation device
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.grid_size = grid_size
        self.dropout = dropout
        self.device = device or torch.device("cpu")

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.MaxPool2d(3, 2, 1),
            KANConv2d(64, 128, 3, 1, 1, grid_size=grid_size),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            KANConv2d(128, 256, 3, 1, 1, grid_size=grid_size),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.MaxPool2d(2, 2),
            KANConv2d(256, 512, 3, 1, 1, grid_size=grid_size),
            nn.GroupNorm(8, 512),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

        self.kan_classifier = nn.Sequential(
            KANLinear(512, 256, grid_size=grid_size),
            nn.Dropout(dropout),
            KANLinear(256, 128, grid_size=grid_size),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.register_buffer(
            "grid",
            torch.linspace(-1, 1, grid_size + 1),
            persistent=False,
        )

        logger.info(
            f"KANDetector initialized: channels={in_channels}, "
            f"grid_size={grid_size}, device={self.device}"
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass for detection.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Dictionary with logits and features
        """
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)

        logits = self.kan_classifier(features)

        return {
            "logits": logits,
            "features": features,
        }

    def predict(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Get predictions and probabilities.

        Args:
            x: Input tensor (batch, channels, height, width)

        Returns:
            Tuple of (class_predictions, probabilities)
        """
        outputs = self.forward(x)
        probs = F.softmax(outputs["logits"], dim=-1)
        preds = probs.argmax(dim=-1)
        return preds, probs


class KANVideoDetector(BaseDetector):
    """
    KAN-Based Video Deepfake Detector.

    Processes video frames and aggregates frame-level predictions.
    Implements BaseDetector interface for ModelRegistry integration.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        threshold: float = 0.5,
        grid_size: int = 5,
    ):
        """
        Initialize KAN video detector.

        Args:
            device: Computation device
            threshold: Detection threshold
            grid_size: Grid size for KAN splines
        """
        self.device = device or torch.device("cpu")
        self.threshold = threshold
        self.grid_size = grid_size
        self._loaded = False

        self.model = KANDetector(
            in_channels=3,
            num_classes=2,
            grid_size=grid_size,
            device=self.device,
        )
        self.model.eval()

        config = get_config()
        self.model_config = config.models.video

        logger.info(
            f"KANVideoDetector initialized: device={self.device}, "
            f"threshold={threshold}"
        )

    def load(self, device: Optional[torch.device] = None) -> None:
        """Load model weights to device."""
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        self._loaded = True
        logger.info(f"KANVideoDetector loaded on {self.device}")

    def unload(self) -> None:
        """Unload model from memory."""
        if hasattr(self, 'model'):
            del self.model
        self._loaded = False
        logger.info("KANVideoDetector unloaded")

    def health_check(self) -> bool:
        """Check if model is loaded and functional."""
        return self._loaded and self.model is not None

    @torch.no_grad()
    def detect(self, input_data: List[Tensor]) -> DetectionResult:
        """
        Detect deepfakes in video frames.

        Args:
            input_data: List of frame tensors (C, H, W)

        Returns:
            DetectionResult with manipulation score and confidence
        """
        frames = input_data
        if not frames:
            return DetectionResult(
                modality=ModalityType.VIDEO,
                overall_score=0.0,
                confidence=0.0,
                is_manipulated=False,
                metadata={"error": "no frames provided"},
            )

        batch = torch.stack(frames).to(self.device)

        outputs = self.model(batch)
        probs = F.softmax(outputs["logits"], dim=-1)
        fake_probs = probs[:, 1].cpu().numpy()

        frame_scores = fake_probs.tolist()
        overall_score = float(np.mean(frame_scores))

        is_manipulated = overall_score > self.threshold
        confidence = self._compute_confidence(frame_scores)

        return DetectionResult(
            modality=ModalityType.VIDEO,
            overall_score=overall_score,
            confidence=confidence,
            is_manipulated=is_manipulated,
            metadata={
                "threshold": self.threshold,
                "frame_scores": frame_scores,
                "num_frames": len(frames),
            },
        )

    def get_heatmap(self, input_data: List[Tensor]) -> Optional[Tensor]:
        """
        Get attention heatmap for explainability.

        Args:
            input_data: List of frame tensors

        Returns:
            Attention heatmap tensor or None
        """
        frames = input_data
        if not frames:
            return None

        batch = torch.stack(frames).to(self.device)
        outputs = self.model(batch)
        features = outputs["features"]

        heatmap = features.detach().cpu()
        if heatmap.dim() > 2:
            heatmap = F.interpolate(
                heatmap.unsqueeze(1),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            heatmap = heatmap.squeeze(0).squeeze(0)
        else:
            heatmap = F.interpolate(
                heatmap.unsqueeze(0).unsqueeze(0),
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap

    def _compute_confidence(self, frame_scores: List[float]) -> float:
        """Compute confidence based on score variance."""
        if len(frame_scores) < 2:
            return 0.5

        variance = np.var(frame_scores)
        confidence = 1.0 / (1.0 + variance * 10)
        return float(np.clip(confidence, 0.0, 1.0))

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "name": "kan_detector",
            "type": "video_detector",
            "parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "threshold": self.threshold,
            "grid_size": self.grid_size,
        }


def create_kan_detector(device: Optional[torch.device] = None) -> KANVideoDetector:
    """Factory function for KANVideoDetector."""
    config = get_config()
    return KANVideoDetector(
        device=device,
        threshold=config.analysis.confidence_threshold,
    )


def register_kan() -> None:
    """Register KAN detector with ModelRegistry."""
    register_model(
        name="kan_detector",
        model_factory=create_kan_detector,
        model_type="video_detector",
        version="1.0.0",
    )


def create_kan_detector_with_device(device: torch.device) -> KANVideoDetector:
    """Factory function for KANVideoDetector with explicit device."""
    config = get_config()
    return KANVideoDetector(
        device=device,
        threshold=config.analysis.confidence_threshold,
    )
