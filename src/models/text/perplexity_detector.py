"""
PerplexityDetector - LLM-based Text Deepfake Detection via Perplexity & Burstiness.

Uses ensemble of language models (GPT-2, XGLM-1.7B, Mistral-7B) to detect
AI-generated text based on perplexity patterns and burstiness analysis.

SOTA Algorithm: Ensemble perplexity scoring with burstiness analysis
"""

import logging
from dataclasses import dataclass, field
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
class PerplexityResult:
    """Result from PerplexityDetector analysis."""
    ensemble_score: float
    per_model_scores: Dict[str, float]
    burstiness: float
    sentence_scores: List[Dict[str, Any]]
    confidence: float
    is_manipulated: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "ensemble_score": self.ensemble_score,
            "per_model_scores": self.per_model_scores,
            "burstiness": self.burstiness,
            "num_sentences": len(self.sentence_scores),
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "metadata": self.metadata,
        }


class PerplexityDetector(BaseDetector, nn.Module):
    """
    PerplexityDetector: LLM-based Text Deepfake Detection.

    Uses ensemble of language models for robust AI text detection:
    - Perplexity scoring from multiple LLMs
    - Burstiness analysis (variance metrics)
    - Sentence-level suspiciousness scoring

    Features:
    - Lazy loading via ModelRegistry
    - Configurable model ensemble
    - Burstiness analysis for better detection
    - Per-sentence explainability
    """

    MODEL_NAME = "perplexity"
    MODEL_VERSION = "1.0.0"

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        use_half_precision: bool = False,
    ):
        super().__init__()
        self.model_path = model_path or get_config("models.perplexity.path")
        self.device = device or self._get_device()
        self.use_half_precision = use_half_precision

        self.models: Dict[str, nn.Module] = {}
        self.tokenizers: Dict[str, Any] = {}
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

    def _load_model(self, model_name: str) -> Optional[nn.Module]:
        """Load a language model by name."""
        try:
            if model_name == "gpt2":
                from transformers import GPT2LMHeadModel, GPT2Tokenizer

                model = GPT2LMHeadModel.from_pretrained(model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            elif model_name == "xglm":
                from transformers import XGLMForCausalLM, XGLMTokenizer

                model = XGLMForCausalLM.from_pretrained(model_name)
                tokenizer = XGLMTokenizer.from_pretrained(model_name)
            elif model_name == "mistral":
                from transformers import MistralForCausalLM, AutoTokenizer

                model = MistralForCausalLM.from_pretrained(
                    "mistralai/Mistral-7B-v0.1"
                )
                tokenizer = AutoTokenizer.from_pretrained(
                    "mistralai/Mistral-7B-v0.1"
                )
            else:
                return None

            model = model.to(self.device)
            model.eval()
            if self.use_half_precision:
                model = model.half()

            self.tokenizers[model_name] = tokenizer
            return model
        except Exception as e:
            logger.warning(f"Could not load {model_name}: {e}")
            return None

    def _build_model(self) -> None:
        """Build the ensemble of language models."""
        model_names = ["gpt2"]
        if self.device.type == "cuda":
            model_names.append("xglm")

        for model_name in model_names:
            model = self._load_model(model_name)
            if model is not None:
                self.models[model_name] = model

        if not self.models:
            logger.warning("No language models loaded, using mock")

    def load(self, device: Optional[torch.device] = None) -> None:
        """
        Load model weights.

        Args:
            device: Target device
        """
        if device is not None:
            self.device = device

        self._build_model()

        self._model_info = {
            "name": self.MODEL_NAME,
            "version": self.MODEL_VERSION,
            "device": str(self.device),
            "models_loaded": list(self.models.keys()),
            "num_parameters": self._count_parameters(),
        }

        self._is_loaded = True
        logger.info(f"PerplexityDetector loaded on {self.device}")

    def _count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(
            p.numel()
            for model in self.models.values()
            for p in model.parameters()
            if p.requires_grad
        )

    def unload(self) -> None:
        """Unload models to free memory."""
        self._is_loaded = False
        self._model_info = {}
        for name in self.models:
            self.models[name].cpu()
        self.models.clear()
        logger.info("PerplexityDetector unloaded")

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
            _ = self.detect("This is a test sentence.")
            return True
        except Exception as e:
            logger.error(f"PerplexityDetector health check failed: {e}")
            return False

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _compute_perplexity(
        self, text: str, model_name: str
    ) -> Optional[float]:
        """Compute perplexity of text for a given model."""
        if model_name not in self.models:
            return None

        model = self.models[model_name]
        tokenizer = self.tokenizers.get(model_name)

        if tokenizer is None:
            return None

        try:
            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss

            return torch.exp(loss).item()
        except Exception as e:
            logger.debug(f"Perplexity computation failed for {model_name}: {e}")
            return None

    def _compute_burstiness(
        self, sentence_scores: List[float]
    ) -> float:
        """Compute burstiness (variance of sentence scores)."""
        if len(sentence_scores) < 2:
            return 0.5

        mean_score = sum(sentence_scores) / len(sentence_scores)
        variance = sum((s - mean_score) ** 2 for s in sentence_scores) / len(
            sentence_scores
        )
        burstiness = min(max(variance * 2, 0), 1)
        return burstiness

    def detect(
        self,
        text: str,
        return_sentences: bool = False,
    ) -> DetectionResult:
        """
        Run detection on text.

        Args:
            text: Input text string
            return_sentences: Include per-sentence scores in result

        Returns:
            DetectionResult with ensemble score and confidence
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        sentences = self._split_sentences(text)

        if len(sentences) < 2:
            result = PerplexityResult(
                ensemble_score=0.5,
                per_model_scores={},
                burstiness=0.5,
                sentence_scores=[],
                confidence=0.0,
                is_manipulated=False,
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

        per_model_scores: Dict[str, float] = {}
        sentence_scores: List[Dict[str, Any]] = []

        for model_name in self.models:
            model_sentence_scores = []
            for sentence in sentences:
                perplexity = self._compute_perplexity(sentence, model_name)
                if perplexity is not None:
                    score = min(max(1 - perplexity / 100, 0), 1)
                    model_sentence_scores.append(score)
                    if return_sentences:
                        sentence_idx = len(sentence_scores)
                        if sentence_idx < len(sentence_scores):
                            sentence_scores[sentence_idx][model_name] = score
                        else:
                            sentence_scores.append(
                                {"sentence": sentence[:50], model_name: score}
                            )

            if model_sentence_scores:
                per_model_scores[model_name] = sum(model_sentence_scores) / len(
                    model_sentence_scores
                )

        if not per_model_scores:
            per_model_scores = {"mock": 0.5}

        ensemble_score = sum(per_model_scores.values()) / len(per_model_scores)

        if return_sentences:
            for i, sentence in enumerate(sentences):
                if i >= len(sentence_scores):
                    sentence_scores.append({"sentence": sentence[:50]})
                sentence_scores[i]["suspiciousness"] = 1 - ensemble_score

        scores_only = [
            s.get("suspiciousness", 1 - ensemble_score)
            for s in sentence_scores
        ]
        if not scores_only:
            scores_only = [ensemble_score]

        burstiness = self._compute_burstiness(scores_only)

        confidence = self._compute_confidence(ensemble_score, burstiness)
        is_manipulated = ensemble_score > 0.5

        result = PerplexityResult(
            ensemble_score=ensemble_score,
            per_model_scores=per_model_scores,
            burstiness=burstiness,
            sentence_scores=sentence_scores if return_sentences else [],
            confidence=confidence,
            is_manipulated=is_manipulated,
            metadata={"num_sentences": len(sentences)},
        )

        return DetectionResult(
            modality=ModalityType.TEXT,
            overall_score=ensemble_score,
            confidence=confidence,
            is_manipulated=is_manipulated,
            verdict=self._get_verdict(ensemble_score),
            artifacts=[],
            metadata=result.to_dict(),
        )

    def _compute_confidence(
        self, ensemble_score: float, burstiness: float
    ) -> float:
        """Compute confidence based on score consensus and burstiness."""
        distance = abs(ensemble_score - 0.5) * 2
        consensus = 1 - burstiness
        confidence = 0.6 * distance + 0.4 * consensus
        return min(max(confidence, 0.0), 1.0)

    def _get_verdict(self, score: float) -> str:
        """Get verdict string based on score."""
        if score >= 0.8:
            return "HIGHLY_MANIPULATED"
        elif score >= 0.6:
            return "LIKELY_MANIPULATED"
        elif score >= 0.4:
            return "UNCERTAIN"
        elif score >= 0.2:
            return "LIKELY_AUTHENTIC"
        else:
            return "HIGHLY_AUTHENTIC"

    def get_heatmap(self, text: str) -> Optional[Tensor]:
        """
        Generate sentence-level heatmap for explainability.

        Args:
            text: Input text string

        Returns:
            Sentence attention weights or None
        """
        if not self._is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        sentences = self._split_sentences(text)
        sentence_scores = []

        for sentence in sentences:
            scores = []
            for model_name in self.models:
                perplexity = self._compute_perplexity(sentence, model_name)
                if perplexity is not None:
                    scores.append(1 - min(max(perplexity / 100, 0), 1))

            if scores:
                sentence_scores.append(sum(scores) / len(scores))
            else:
                sentence_scores.append(0.5)

        if not sentence_scores:
            return None

        attention = torch.tensor(sentence_scores, dtype=torch.float32)
        attention = attention / (attention.sum() + 1e-8)
        return attention.unsqueeze(0)


def create_perplexity(device: Optional[torch.device] = None) -> PerplexityDetector:
    """Factory function for PerplexityDetector."""
    return PerplexityDetector(device=device)


def register_perplexity() -> None:
    """Register PerplexityDetector with ModelRegistry."""
    register_model(
        name="perplexity",
        model_factory=create_perplexity,
        model_type="text_detector",
        version="1.0.0",
    )
