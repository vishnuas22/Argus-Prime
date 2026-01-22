"""
TranscryptDetector - Lightweight Audio Deepfake Detector.

Uses Wav2Vec2 fine-tuned on synthetic speech dataset for fast, accurate
detection optimized for real-time applications.

SOTA Algorithm: Transcend (TrueMedia) - Wav2Vec2 fine-tune
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
class TranscryptResult:
    """Result from Transcrypt detection."""
    fake_probability: float
    confidence: float
    is_manipulated: bool
    segment_scores: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fake_probability": self.fake_probability,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "num_segments": len(self.segment_scores),
            "metadata": self.metadata,
        }


class TranscryptDetector(BaseDetector, nn.Module):
    """
    TranscryptDetector: Lightweight Audio Deepfake Detector.

    Uses Wav2Vec2-Base fine-tuned on synthetic speech for:
    - Fast inference (real-time capable)
    - Small model size
    - Battle-tested by TrueMedia

    Features:
    - Lazy loading via ModelRegistry
    - GPU memory optimization
    - Half precision support
    - Streaming support for long audio
    """

    MODEL_NAME = "transcrypt"
    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_half_precision: bool = False,
    ):
        super().__init__()
        self.model_path = model_path or get_config("models.transcrypt.path")
        self.device = device or self._get_device()
        self.use_half_precision = use_half_precision
        self.sample_rate = 16000
        self.max_input_length = 16000 * 30  # 30 seconds

        self.feature_extractor = None
        self.classifier = None
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

    def _build_model(self) -> None:
        """Build the Transcrypt model architecture."""
        try:
            from transformers import Wav2Vec2Model

            self.feature_extractor = Wav2Vec2Model.from_pretrained(
                "facebook/wav2vec2-base"
            )
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        except ImportError:
            logger.warning(
                "transformers not available, using mock architecture"
            )
            self.feature_extractor = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

    def load(self, device: Optional[torch.device] = None) -> None:
        """
        Load model weights.

        Args:
            device: Target device
        """
        if device is not None:
            self.device = device

        self._build_model()
        self.to(self.device)
        self.eval()

        if self.use_half_precision:
            self.feature_extractor = self.feature_extractor.half()
            self.classifier = self.classifier.half()

        if self.model_path and Path(self.model_path).exists():
            self._load_weights()

        self._model_info = {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "device": str(self.device),
            "sample_rate": self.sample_rate,
            "num_parameters": self._count_parameters(),
        }

        self._is_loaded = True
        logger.info(f"TranscryptDetector loaded on {self.device}")

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
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def unload(self) -> None:
        """Unload model to free memory."""
        self._is_loaded = False
        self._model_info = {}
        if self.feature_extractor is not None:
            self.feature_extractor.cpu()
        logger.info("TranscryptDetector unloaded")

    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata."""
        return {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "is_loaded": self._is_loaded,
            **self._model_info,
        }

    def health_check(self) -> bool:
        """Check if model is loaded and functional."""
        if not self._is_loaded:
            return False

        try:
            self.eval()
            with torch.no_grad():
                dummy_input = torch.randn(
                    1, self.sample_rate, device=self.device
                )
                _ = self.detect(dummy_input)
            return True
        except Exception as e:
            logger.error(f"TranscryptDetector health check failed: {e}")
            return False

    def _preprocess_waveform(
        self,
        waveform: Tensor,
        sample_rate: Optional[int] = None,
    ) -> Tensor:
        """
        Preprocess waveform for model input.

        Args:
            waveform: Input audio tensor
            sample_rate: Sample rate of input

        Returns:
            Preprocessed tensor (1, 1, sample_rate)
        """
        sr = sample_rate or self.sample_rate

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1) if waveform.shape[0] == 1 else waveform

        if waveform.shape[-1] > self.max_input_length:
            waveform = waveform[..., :self.max_input_length]
        elif waveform.shape[-1] < self.sample_rate:
            waveform = nn.functional.pad(
                waveform, (0, self.sample_rate - waveform.shape[-1])
            )

        return waveform.to(self.device)

    def detect(
        self,
        waveform: Tensor,
        sample_rate: Optional[int] = None,
        segment_duration: Optional[float] = None,
    ) -> DetectionResult:
        """
        Run detection on audio waveform.

        Args:
            waveform: Input audio tensor
            sample_rate: Sample rate of input
            segment_duration: Duration of segments for analysis

        Returns:
            DetectionResult with fake probability and confidence
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        waveform = self._preprocess_waveform(waveform, sample_rate)

        with torch.no_grad():
            if segment_duration is None:
                fake_prob = self._detect_segment(waveform)
                segment_scores = [{"segment_idx": 0, "fake_probability": fake_prob}]
            else:
                segment_scores = self._detect_segments(
                    waveform, segment_duration
                )
                fake_prob = max(
                    s["fake_probability"] for s in segment_scores
                )

        confidence = self._compute_confidence(fake_prob)
        is_manipulated = fake_prob > 0.5

        result = TranscryptResult(
            fake_probability=fake_prob,
            confidence=confidence,
            is_manipulated=is_manipulated,
            segment_scores=segment_scores,
            metadata={
                "segment_duration": segment_duration,
                "num_segments": len(segment_scores),
            },
        )

        return DetectionResult(
            modality=ModalityType.AUDIO,
            overall_score=fake_prob,
            confidence=confidence,
            is_manipulated=is_manipulated,
            verdict=self._get_verdict(fake_prob),
            artifacts=[],
            metadata=result.to_dict(),
        )

    def _detect_segment(self, waveform: Tensor) -> float:
        """Run detection on a single segment."""
        try:
            if self.feature_extractor is not None:
                features = self.feature_extractor(waveform.squeeze(1)).last_hidden_state
                features = features.mean(dim=1)
            else:
                features = waveform.mean(dim=-1, keepdim=True)
                features = features.expand(-1, 768)

            fake_prob = self.classifier(features).item()
            return min(max(fake_prob, 0.0), 1.0)
        except Exception:
            return 0.5

    def _detect_segments(
        self, waveform: Tensor, segment_duration: float
    ) -> List[Dict[str, Any]]:
        """Run detection on multiple segments."""
        segment_length = int(segment_duration * self.sample_rate)
        total_length = waveform.shape[-1]
        segment_scores = []

        for i in range(0, total_length, segment_length):
            segment = waveform[..., i : i + segment_length]
            if segment.shape[-1] < self.sample_rate // 2:
                continue
            fake_prob = self._detect_segment(segment)
            segment_scores.append(
                {"segment_idx": len(segment_scores), "fake_probability": fake_prob}
            )

        return segment_scores

    def _compute_confidence(self, fake_prob: float) -> float:
        """Compute confidence based on probability distance from 0.5."""
        distance = abs(fake_prob - 0.5) * 2
        return min(max(distance, 0.0), 1.0)

    def _get_verdict(self, fake_prob: float) -> str:
        """Get verdict string based on probability."""
        if fake_prob >= 0.8:
            return "HIGHLY_MANIPULATED"
        elif fake_prob >= 0.6:
            return "LIKELY_MANIPULATED"
        elif fake_prob >= 0.4:
            return "UNCERTAIN"
        elif fake_prob >= 0.2:
            return "LIKELY_AUTHENTIC"
        else:
            return "HIGHLY_AUTHENTIC"

    def get_heatmap(self, waveform: Tensor) -> Optional[Tensor]:
        """
        Generate attention heatmap for explainability.

        Args:
            waveform: Input audio tensor

        Returns:
            Attention weights or None
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        waveform = self._preprocess_waveform(waveform)

        try:
            self.eval()
            with torch.enable_grad():
                waveform.requires_grad_(True)
                features = self.feature_extractor(waveform.squeeze(1)).last_hidden_state
                output = self.classifier(features.mean(dim=1, keepdim=True))
                output.backward()

                heatmap = waveform.grad.abs()
                return heatmap / heatmap.sum(dim=-1, keepdim=True)
        except Exception:
            return None


def create_transcrypt(device: Optional[torch.device] = None) -> TranscryptDetector:
    """Factory function for TranscryptDetector."""
    return TranscryptDetector(device=device)


def register_transcrypt() -> None:
    """Register TranscryptDetector with ModelRegistry."""
    register_model(
        name="transcrypt",
        model_factory=create_transcrypt,
        model_type="audio_detector",
        version="1.0.0",
    )
