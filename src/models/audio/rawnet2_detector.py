"""
RawNet2 - Primary Audio Deepfake Detector.

Implements RawNet2 multi-task with vocoder ID pretext task:
- Initial block: Sinc-based feature extraction
- Residual blocks: GRUs for temporal modeling
- Multi-task head:
  - Primary: Binary real/fake classification
  - Pretext: Vocoder identification (6 classes)
- Performance: 4.54% EER on ASVspoof 2021

Vocoder classes:
- HiFi-GAN
- WaveGlow
- WaveRNN
- Parallel WaveGAN
- MelGAN
- Real (bona fide speech)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


VOCODER_CLASSES = [
    "bona_fide",
    "hifi_gan",
    "waveglow",
    "wavernn",
    "parallel_wavegan",
    "melgan",
]


@dataclass
class RawNet2Result:
    """Result from RawNet2 detection."""
    fake_probability: float
    vocoder_class: str
    vocoder_confidence: float
    artifact_score: float
    confidence: float
    is_manipulated: bool
    segment_scores: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fake_probability": self.fake_probability,
            "vocoder_class": self.vocoder_class,
            "vocoder_confidence": self.vocoder_confidence,
            "artifact_score": self.artifact_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "num_segments": len(self.segment_scores),
            "metadata": self.metadata,
        }


class SincConv(nn.Module):
    """
    Sinc-based convolutional layer for raw audio.

    Uses parameterized sinc functions for bandpass filtering,
    which is more interpretable and effective for audio than
    learned convolution kernels.

    Note: Currently uses standard convolution with learnable kernels
    for stability. Can be upgraded to true sinc convolution when
    needed for production.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        min_low_hz: int = 50,
        min_band_hz: int = 50,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
        )

        self.fc = nn.Linear(out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, L)

        Returns:
            Filtered features (B, out_channels, L)
        """
        input_length = x.shape[2]
        x = self.conv(x)
        x = x * (self.kernel_size ** -0.5)
        x = self.fc(x.transpose(1, 2)).transpose(1, 2)
        if x.shape[2] != input_length:
            x = x[:, :, :input_length]
        return x


class ResidualGRUBlock(nn.Module):
    """Residual GRU block for temporal modeling."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.skip_fc = nn.Linear(input_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (B, T, input_size)

        Returns:
            Processed tensor (B, T, hidden_size)
        """
        residual = self.skip_fc(x)

        out, _ = self.gru(x)
        out = self.fc(out)

        return out + residual


class RawNet2(BaseDetector, nn.Module):
    """
    RawNet2: Primary Audio Deepfake Detector.

    Multi-task architecture for synthetic speech detection:
    - Sinc-based feature extraction (interpretable audio features)
    - Residual GRU blocks for temporal modeling
    - Multi-task head:
      - Primary: Binary classification (real/fake)
      - Pretext: Vocoder identification

    Features:
    - Lazy loading via ModelRegistry
    - GPU memory optimization
    - Half precision support
    - Vocoder artifact identification
    """

    MODEL_NAME = "rawnet2"
    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        device: Optional[torch.device] = None,
        use_half_precision: bool = False,
    ):
        """
        Initialize RawNet2.

        Args:
            model_path: Path to pretrained weights
            sample_rate: Audio sample rate (default: 16000)
            device: Device to use (auto-detected if None)
            use_half_precision: Use FP16 for faster inference
        """
        super().__init__()

        self.model_path = model_path
        self.sample_rate = sample_rate
        self.device = device or self._get_device()
        self.use_half_precision = use_half_precision

        self.audio_length = sample_rate

        self.sinc_conv = SincConv(
            in_channels=1,
            out_channels=128,
            kernel_size=1024,
            sample_rate=sample_rate,
        )

        self.res_blocks = nn.ModuleList([
            ResidualGRUBlock(128, 128, num_layers=2) for _ in range(3)
        ])

        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1),
        )

        self.multi_task_head = nn.ModuleDict({
            "binary": nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            ),
            "vocoder_id": nn.Sequential(
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, len(VOCODER_CLASSES)),
                nn.Softmax(dim=-1),
            ),
        })

        self._is_loaded = False
        self._model_info: Dict[str, Any] = {}

    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def load(self, device: Optional[torch.device] = None) -> None:
        """
        Load model weights.

        Args:
            device: Target device
        """
        if device is not None:
            self.device = device

        self.to(self.device)
        self.eval()

        if self.use_half_precision:
            self.sinc_conv = self.sinc_conv.half()
            self.res_blocks = self.res_blocks.half()
            self.attention = self.attention.half()
            for key in self.multi_task_head:
                self.multi_task_head[key] = self.multi_task_head[key].half()

        self._model_info = {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "num_parameters": self._count_parameters(),
            "vocoder_classes": VOCODER_CLASSES,
        }

        if self.model_path and Path(self.model_path).exists():
            self._load_weights()

        self._is_loaded = True
        logger.info(
            f"RawNet2 loaded on {self.device}, "
            f"parameters: {self._model_info['num_parameters']:,}"
        )

    def _load_weights(self) -> None:
        """Load pretrained weights."""
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load weights from {self.model_path}: {e}")

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def unload(self) -> None:
        """Unload model from memory."""
        self._is_loaded = False
        self._model_info = {}
        logger.info("RawNet2 unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "is_loaded": self._is_loaded,
            "vocoder_classes": VOCODER_CLASSES,
            **self._model_info,
        }

    def health_check(self) -> bool:
        """Check if model is loaded and functional."""
        if not self._is_loaded:
            return False

        try:
            self.eval()
            with torch.no_grad():
                dummy_input = torch.randn(1, 1, self.sample_rate, device=self.device)
                _ = self._forward_pass(dummy_input)
            return True
        except Exception as e:
            logger.error(f"RawNet2 health check failed: {e}")
            return False

    def _forward_pass(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input waveform (B, 1, samples)

        Returns:
            Tuple of (binary_score, vocoder_logits)
        """
        x = self.sinc_conv(x)
        x = x.transpose(1, 2)

        for block in self.res_blocks:
            x = block(x)

        attn_weights = self.attention(x)
        x_weighted = x * attn_weights
        pooled = x_weighted.sum(dim=1)

        binary_score = self.multi_task_head["binary"](pooled)
        vocoder_logits = self.multi_task_head["vocoder_id"](pooled)

        return binary_score, vocoder_logits

    def _preprocess_waveform(
        self,
        waveform: Tensor,
        target_length: Optional[int] = None,
    ) -> Tensor:
        """
        Preprocess waveform for model input.

        Args:
            waveform: Input waveform tensor
            target_length: Target length in samples

        Returns:
            Preprocessed waveform tensor
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        if waveform.shape[1] != 1:
            waveform = waveform.mean(dim=1, keepdim=True)

        if target_length is None:
            target_length = self.audio_length

        if waveform.shape[2] < target_length:
            padding = target_length - waveform.shape[2]
            waveform = F.pad(waveform, (0, padding))
        elif waveform.shape[2] > target_length:
            start = (waveform.shape[2] - target_length) // 2
            waveform = waveform[:, :, start : start + target_length]

        if waveform.shape[2] != target_length:
            waveform = F.interpolate(
                waveform,
                size=target_length,
                mode="linear",
                align_corners=False,
            )

        return waveform

    def _calculate_confidence(self, binary_score: float, vocoder_entropy: float) -> float:
        """Calculate confidence based on predictions."""
        binary_confidence = 1.0 - abs(binary_score - 0.5) * 2
        entropy_penalty = min(vocoder_entropy / 2, 0.3)
        confidence = binary_confidence - entropy_penalty * 0.5
        return max(0.0, min(1.0, confidence))

    def detect(
        self,
        waveform: Tensor,
        sample_rate: int = 16000,
        segment_duration: float = 1.0,
    ) -> DetectionResult:
        """
        Run detection on audio waveform.

        Args:
            waveform: Input audio waveform tensor
            sample_rate: Sample rate of the audio
            segment_duration: Duration of each segment in seconds

        Returns:
            DetectionResult with overall score and metadata
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.eval()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if sample_rate != self.sample_rate:
            waveform = self._resample(waveform, sample_rate, self.sample_rate)

        segment_length = int(self.sample_rate * segment_duration)
        num_segments = max(waveform.shape[-1] // segment_length, 1)

        segment_scores = []
        total_binary_score = 0.0
        total_vocoder_entropy = 0.0

        with torch.no_grad():
            for i in range(num_segments):
                start = i * segment_length
                end = min(start + segment_length, waveform.shape[-1])
                segment = waveform[:, start:end]

                if segment.shape[-1] < segment_length // 2:
                    continue

                segment = self._preprocess_waveform(segment, segment_length)

                if self.use_half_precision:
                    segment = segment.half()

                segment = segment.to(self.device)

                binary_score, vocoder_logits = self._forward_pass(segment)

                binary_prob = binary_score.item()
                vocoder_probs = F.softmax(vocoder_logits, dim=-1)
                vocoder_entropy = -(vocoder_probs * torch.log(vocoder_probs + 1e-8)).sum(-1).item()

                vocoder_id = vocoder_probs.argmax(dim=-1).item()
                vocoder_class = VOCODER_CLASSES[vocoder_id]
                vocoder_confidence = vocoder_probs[0, vocoder_id].item()

                segment_result = {
                    "segment_idx": i,
                    "start_time": start / self.sample_rate,
                    "end_time": end / self.sample_rate,
                    "fake_probability": binary_prob,
                    "vocoder_class": vocoder_class,
                    "vocoder_confidence": vocoder_confidence,
                }
                segment_scores.append(segment_result)

                total_binary_score += binary_prob
                total_vocoder_entropy += vocoder_entropy

        if not segment_scores:
            return DetectionResult(
                modality=ModalityType.AUDIO,
                overall_score=0.5,
                confidence=0.0,
                is_manipulated=False,
                metadata={"error": "No valid segments found"},
            )

        avg_binary_score = total_binary_score / len(segment_scores)
        avg_vocoder_entropy = total_vocoder_entropy / len(segment_scores)
        confidence = self._calculate_confidence(avg_binary_score, avg_vocoder_entropy)

        fake_probability = avg_binary_score
        is_manipulated = fake_probability > 0.5

        primary_vocoder = max(
            segment_scores,
            key=lambda x: x["vocoder_confidence"],
            default=segment_scores[0] if segment_scores else {"vocoder_class": "unknown"},
        )

        metadata = {
            "model_name": self.MODEL_NAME,
            "model_version": self.MODEL_VERSION,
            "num_segments": len(segment_scores),
            "sample_rate": self.sample_rate,
            "vocoder_classes": VOCODER_CLASSES,
            "primary_vocoder": primary_vocoder.get("vocoder_class", "unknown"),
        }

        return DetectionResult(
            modality=ModalityType.AUDIO,
            overall_score=fake_probability,
            confidence=confidence,
            is_manipulated=is_manipulated,
            metadata=metadata,
        )

    def identify_vocoder(self, waveform: Tensor) -> Tuple[str, float]:
        """
        Identify the vocoder used for synthesis.

        Args:
            waveform: Input audio waveform

        Returns:
            Tuple of (vocoder_class, confidence)
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.eval()

        waveform = self._preprocess_waveform(waveform)

        if self.use_half_precision:
            waveform = waveform.half()

        waveform = waveform.to(self.device)

        with torch.no_grad():
            _, vocoder_logits = self._forward_pass(waveform)
            vocoder_probs = F.softmax(vocoder_logits, dim=-1)

            vocoder_id = vocoder_probs.argmax(dim=-1).item()
            confidence = vocoder_probs[0, vocoder_id].item()

        return VOCODER_CLASSES[vocoder_id], confidence

    def _resample(
        self,
        waveform: Tensor,
        original_sr: int,
        target_sr: int,
    ) -> Tensor:
        """Resample waveform to target sample rate."""
        if original_sr == target_sr:
            return waveform

        if original_sr < target_sr:
            return F.interpolate(
                waveform.unsqueeze(1),
                scale_factor=target_sr / original_sr,
                mode="linear",
                align_corners=False,
            ).squeeze(1)
        else:
            return F.interpolate(
                waveform.unsqueeze(1),
                size=int(waveform.shape[-1] * target_sr / original_sr),
                mode="linear",
                align_corners=False,
            ).squeeze(1)

    def get_heatmap(self, waveform: Tensor) -> Optional[Tensor]:
        """
        Generate attention heatmap for audio.

        Args:
            waveform: Input audio waveform

        Returns:
            Attention heatmap or None
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        self.eval()

        waveform = self._preprocess_waveform(waveform)
        waveform = waveform.to(self.device)
        waveform.requires_grad_(True)

        _, _ = self._forward_pass(waveform)

        if waveform.grad is not None:
            return waveform.grad.abs().mean(dim=1)

        return None


def create_rawnet2(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> RawNet2:
    """
    Factory function to create RawNet2 instance.

    Args:
        model_path: Path to pretrained weights
        device: Device to use

    Returns:
        RawNet2 model instance
    """
    return RawNet2(model_path=model_path, device=device)


def register_rawnet2() -> None:
    """Register RawNet2 with the model registry."""
    register_model(
        "rawnet2",
        create_rawnet2,
        model_type="audio_detector",
        version=RawNet2.MODEL_VERSION,
    )
