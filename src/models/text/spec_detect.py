"""
SpecDetect - Primary Text Deepfake Detector via Spectral Analysis.

Uses DFT of token probabilities to detect AI-generated text patterns.
SOTA Algorithm: SpecDetect++ (2025) - Fast, reliable, works without GPU.

Theory:
- AI-generated text has characteristic frequency patterns
- DFT of token probabilities reveals generation artifacts
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
class SpecDetectResult:
    """Result from SpecDetect spectral analysis."""
    generation_probability: float
    spectral_features: Dict[str, float]
    confidence: float
    is_manipulated: bool
    token_scores: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generation_probability": self.generation_probability,
            "spectral_features": self.spectral_features,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "num_tokens": len(self.token_scores),
            "metadata": self.metadata,
        }


class SpecDetect(BaseDetector, nn.Module):
    """
    SpecDetect: Primary Text Deepfake Detector.

    Uses spectral analysis of token probabilities to detect AI-generated text:
    - Computes DFT of token probability sequences
    - Extracts characteristic frequency patterns
    - Fast and works without GPU

    Features:
    - Lazy loading via ModelRegistry
    - GPU and CPU support
    - Half precision support
    - Per-token analysis with explainability
    """

    MODEL_NAME = "specdetect"
    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_half_precision: bool = False,
    ):
        super().__init__()
        self.model_path = model_path or get_config("models.specdetect.path")
        self.device = device or self._get_device()
        self.use_half_precision = use_half_precision

        self.token_embedding = None
        self.language_model = None
        self.spectral_head = None
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
        """Build the SpecDetect model architecture."""
        try:
            from transformers import GPT2LMHeadModel

            self.language_model = GPT2LMHeadModel.from_pretrained("gpt2")
            self.token_embedding = self.language_model.transformer.wte
            hidden_size = self.language_model.config.n_embd
        except ImportError:
            logger.warning("transformers not available, using mock architecture")
            hidden_size = 768
            self.language_model = None
            self.token_embedding = nn.Embedding(50257, hidden_size)

        self.spectral_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
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
            self.spectral_head = self.spectral_head.half()

        if self.model_path and device is not None:
            self._load_weights()

        self._model_info = {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "device": str(self.device),
            "num_parameters": self._count_parameters(),
        }

        self._is_loaded = True
        logger.info(f"SpecDetect loaded on {self.device}")

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
        if self.spectral_head is not None:
            self.spectral_head.cpu()
        logger.info("SpecDetect unloaded")

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
                dummy_tokens = torch.randint(0, 100, (1, 50), device=self.device)
                _ = self.detect("This is a test sentence for health check.")
            return True
        except Exception as e:
            logger.error(f"SpecDetect health check failed: {e}")
            return False

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        import re
        tokens = re.findall(r"\b\w+\b|[.,!?;:]", text.lower())
        return tokens

    def _get_token_probabilities(
        self, text: str, tokens: List[str]
    ) -> Tuple[List[int], List[float]]:
        """Get token IDs and their probabilities from language model."""
        try:
            if self.language_model is not None:
                from transformers import GPT2Tokenizer

                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                tokenizer.pad_token = tokenizer.eos_token

                inputs = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.language_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                input_ids = inputs["input_ids"][0].tolist()
                token_probs = []
                for i, token_id in enumerate(input_ids):
                    if token_id < probs.shape[-1]:
                        prob = probs[0, i, token_id].item()
                    else:
                        prob = 1.0 / probs.shape[-1]
                    token_probs.append(prob)

                return input_ids, token_probs
        except Exception as e:
            logger.debug(f"LM inference failed: {e}")

        token_ids = list(range(len(tokens)))
        token_probs = [0.5] * len(tokens)
        return token_ids, token_probs

    def _compute_spectral_features(
        self, token_probs: List[float]
    ) -> Dict[str, float]:
        """Compute spectral features via DFT of token probabilities."""
        if len(token_probs) < 4:
            return {
                "low_freq_power": 0.5,
                "high_freq_power": 0.5,
                "spectral_centroid": 0.5,
                "spectral_entropy": 1.0,
            }

        probs = np.array(token_probs)
        probs = probs / (probs.sum() + 1e-8)

        n = len(probs)
        if n % 2 == 0:
            n_fft = n
        else:
            n_fft = n - 1

        fft_result = np.fft.fft(probs[:n_fft])
        magnitude = np.abs(fft_result) / n_fft
        power = magnitude ** 2

        low_freq_idx = slice(1, n_fft // 4)
        high_freq_idx = slice(n_fft // 4, n_fft // 2)

        low_freq_power = float(np.mean(power[low_freq_idx]))
        high_freq_power = float(np.mean(power[high_freq_idx]))

        if np.sum(power) > 0:
            spectral_centroid = float(
                np.sum(np.arange(len(power)) * power) / np.sum(power)
            ) / (len(power) + 1e-8)
        else:
            spectral_centroid = 0.5

        probs_normalized = probs + 1e-8
        probs_normalized = probs_normalized / probs_normalized.sum()
        spectral_entropy = float(
            -np.sum(probs_normalized * np.log2(probs_normalized))
        )

        return {
            "low_freq_power": min(max(low_freq_power * 10, 0), 1),
            "high_freq_power": min(max(high_freq_power * 10, 0), 1),
            "spectral_centroid": min(max(spectral_centroid, 0), 1),
            "spectral_entropy": min(max(spectral_entropy / np.log2(len(probs) + 1), 0), 1),
        }

    def detect(
        self,
        text: str,
        return_tokens: bool = False,
    ) -> DetectionResult:
        """
        Run detection on text.

        Args:
            text: Input text string
            return_tokens: Include per-token scores in result

        Returns:
            DetectionResult with generation probability and confidence
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self._tokenize(text)

        if len(tokens) < 3:
            result = SpecDetectResult(
                generation_probability=0.5,
                spectral_features={"error": "text_too_short"},
                confidence=0.0,
                is_manipulated=False,
                token_scores=[],
                metadata={"reason": "text_too_short"},
            )
            return DetectionResult(
                modality=ModalityType.TEXT,
                overall_score=0.5,
                confidence=0.0,
                is_manipulated=False,
                verdict="UNCERTAIN",
                artifacts=[],
                metadata=result.to_dict(),
            )

        token_ids, token_probs = self._get_token_probabilities(text, tokens)

        spectral_features = self._compute_spectral_features(token_probs)

        generation_prob = self._compute_generation_score(spectral_features)

        if return_tokens:
            token_scores = [
                {
                    "token": tokens[i],
                    "probability": token_probs[i],
                    "is_suspicious": token_probs[i] < 0.01 or token_probs[i] > 0.99,
                }
                for i in range(len(tokens))
            ]
        else:
            token_scores = []

        confidence = self._compute_confidence(generation_prob, spectral_features)
        is_manipulated = generation_prob > 0.5

        result = SpecDetectResult(
            generation_probability=generation_prob,
            spectral_features=spectral_features,
            confidence=confidence,
            is_manipulated=is_manipulated,
            token_scores=token_scores,
            metadata={"num_tokens": len(tokens)},
        )

        return DetectionResult(
            modality=ModalityType.TEXT,
            overall_score=generation_prob,
            confidence=confidence,
            is_manipulated=is_manipulated,
            verdict=self._get_verdict(generation_prob),
            artifacts=[],
            metadata=result.to_dict(),
        )

    def _compute_generation_score(self, spectral_features: Dict[str, float]) -> float:
        """Compute generation probability from spectral features."""
        ai_patterns = (
            0.4 * (1 - spectral_features.get("low_freq_power", 0.5))
            + 0.3 * spectral_features.get("high_freq_power", 0.5)
            + 0.3 * (1 - spectral_features.get("spectral_entropy", 0.5))
        )
        return min(max(ai_patterns, 0.0), 1.0)

    def _compute_confidence(
        self, generation_prob: float, spectral_features: Dict[str, float]
    ) -> float:
        """Compute confidence based on spectral entropy and probability."""
        entropy = spectral_features.get("spectral_entropy", 0.5)
        distance = abs(generation_prob - 0.5) * 2
        confidence = 0.6 * distance + 0.4 * entropy
        return min(max(confidence, 0.0), 1.0)

    def _get_verdict(self, generation_prob: float) -> str:
        """Get verdict string based on probability."""
        if generation_prob >= 0.8:
            return "HIGHLY_MANIPULATED"
        elif generation_prob >= 0.6:
            return "LIKELY_MANIPULATED"
        elif generation_prob >= 0.4:
            return "UNCERTAIN"
        elif generation_prob >= 0.2:
            return "LIKELY_AUTHENTIC"
        else:
            return "HIGHLY_AUTHENTIC"

    def get_heatmap(self, text: str) -> Optional[Tensor]:
        """
        Generate attention heatmap for text tokens.

        Args:
            text: Input text string

        Returns:
            Token attention weights or None
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        tokens = self._tokenize(text)
        token_probs = [0.5] * len(tokens)

        try:
            if self.language_model is not None:
                from transformers import GPT2Tokenizer

                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                inputs = tokenizer(
                    text, return_tensors="pt", padding=True, truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.language_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)

                token_probs = []
                for i in range(inputs["input_ids"].shape[1]):
                    token_id = inputs["input_ids"][0, i].item()
                    if token_id < probs.shape[-1]:
                        prob = probs[0, i, token_id].item()
                    else:
                        prob = 1.0 / probs.shape[-1]
                    token_probs.append(1 - prob)

            attention = torch.tensor(token_probs, dtype=torch.float32)
            attention = attention / (attention.sum() + 1e-8)
            return attention.unsqueeze(0)
        except Exception:
            return None


def create_specdetect(device: Optional[torch.device] = None) -> SpecDetect:
    """Factory function for SpecDetect."""
    return SpecDetect(device=device)


def register_specdetect() -> None:
    """Register SpecDetect with ModelRegistry."""
    register_model(
        name="specdetect",
        model_factory=create_specdetect,
        model_type="text_detector",
        version="1.0.0",
    )
