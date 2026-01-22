"""
Abstract base classes and protocols for model interfaces.

This module defines the contracts that all detection models must implement,
enabling swappable implementations and testability through mocking.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Optional, Protocol, TypeVar, runtime_checkable

import torch


class ModalityType(str, Enum):
    """Supported analysis modalities."""
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"


@dataclass
class DetectionResult:
    """Base detection result containing common fields."""
    modality: ModalityType
    overall_score: float
    confidence: float
    is_manipulated: bool
    metadata: dict
    heatmap: Optional[torch.Tensor] = None


T = TypeVar("T")
R = TypeVar("R")


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load model weights to device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if model is loaded and functional."""
        pass


class BaseDetector(BaseModel, ABC):
    """Abstract base class for detection models."""

    @abstractmethod
    def detect(self, input_data: T) -> DetectionResult:
        """Run detection on input data."""
        pass

    @abstractmethod
    def get_heatmap(self, input_data: T) -> torch.Tensor:
        """Generate attention heatmap for explainability."""
        pass


@runtime_checkable
class VideoDetector(Protocol):
    """Protocol for video detection models."""

    def detect(self, frames: List[torch.Tensor]) -> DetectionResult:
        """Detect deepfake in video frames."""
        ...

    def get_heatmap(self, frame: torch.Tensor) -> torch.Tensor:
        """Generate frame-level heatmap."""
        ...


@runtime_checkable
class AudioDetector(Protocol):
    """Protocol for audio detection models."""

    def detect(self, waveform: torch.Tensor, sample_rate: int) -> DetectionResult:
        """Detect synthetic audio."""
        ...

    def identify_vocoder(self, waveform: torch.Tensor) -> str:
        """Identify vocoder used for synthesis."""
        ...


@runtime_checkable
class TextDetector(Protocol):
    """Protocol for text detection models."""

    def detect(self, text: str, tokens: dict) -> DetectionResult:
        """Detect AI-generated text."""
        ...

    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity score."""
        ...

    def analyze_burstiness(self, text: str) -> dict:
        """Analyze burstiness metrics."""
        ...


@runtime_checkable
class Preprocessor(Protocol):
    """Protocol for preprocessing components."""

    def process(self, input_data: T) -> R:
        """Preprocess input for model."""
        ...

    def validate(self, input_data: T) -> tuple[bool, str]:
        """Validate input is processable."""
        ...


@runtime_checkable
class FusionEngine(Protocol):
    """Protocol for multi-modal fusion engines."""

    def fuse(self, results: List[DetectionResult]) -> dict:
        """Fuse multi-modal results."""
        ...

    def calculate_trust_score(self, fused: dict) -> dict:
        """Calculate final trust score."""
        ...


@dataclass
class VideoFrame:
    """Internal representation of a video frame."""
    tensor: torch.Tensor
    frame_number: int
    timestamp: float
    width: int
    height: int


@dataclass
class AudioSegment:
    """Internal representation of an audio segment."""
    waveform: torch.Tensor
    sample_rate: int
    start_time: float
    end_time: float
    spectrogram: Optional[torch.Tensor] = None


@dataclass
class FaceBox:
    """Internal representation of a detected face."""
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    confidence: float
    aligned_image: Optional[torch.Tensor] = None


@dataclass
class TextTokens:
    """Internal representation of tokenized text."""
    tokens: List[str]
    token_ids: List[int]
    attention_mask: List[int]
    language: str
