"""
Text Detection Orchestrator.

Orchestrates text deepfake detection pipeline:
1. Detect language
2. Tokenize text
3. Run SpecDetect++ (spectral analysis)
4. Run PerplexityDetector (LLM ensemble)
5. Analyze burstiness
6. Score per sentence
7. Aggregate to document level
8. Return comprehensive result
"""

from dataclasses import dataclass, field
import logging

from typing import Any, Dict, List, Optional

import numpy as np

from src.config import get_config
from src.core.logging import logger
from src.models import (
    DetectionResult,
    SpecDetect,
    PerplexityDetector,
    ModalityType,
)
from src.preprocessing import TextProcessor

logger = logging.getLogger(__name__)


@dataclass
class SentenceAnalysisResult:
    """Per-sentence analysis result."""
    sentence: str
    start_char: int
    end_char: int
    manipulation_score: float
    perplexity: Optional[float] = None
    burstiness: Optional[float] = None
    is_suspicious: bool = False


@dataclass
class TextAnalysisResult:
    """Complete text analysis result."""
    overall_score: float
    confidence: float
    is_manipulated: bool
    word_count: int
    language_detected: str
    per_sentence_results: List[SentenceAnalysisResult]
    perplexity_scores: Dict[str, float] = field(default_factory=dict)
    burstiness_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "word_count": self.word_count,
            "language_detected": self.language_detected,
            "perplexity_scores": self.perplexity_scores,
            "burstiness_metrics": self.burstiness_metrics,
            "per_sentence_scores": [
                {
                    "sentence": s.sentence,
                    "start_char": s.start_char,
                    "end_char": s.end_char,
                    "manipulation_score": s.manipulation_score,
                    "perplexity": s.perplexity,
                    "burstiness": s.burstiness,
                    "is_suspicious": s.is_suspicious,
                }
                for s in self.per_sentence_results
            ],
            "metadata": self.metadata,
        }


class TextDetector:
    """
    Text Deepfake Detection Orchestrator.

    Coordinates the text analysis pipeline with:
    - Language detection and tokenization
    - Multi-model inference (SpecDetect + Perplexity)
    - Burstiness analysis
    - Sentence-level scoring

    Attributes:
        processor: Text preprocessing processor
        primary_model: Main detection model (SpecDetect)
        secondary_model: Secondary model (PerplexityDetector)
    """

    def __init__(
        self,
        language: Optional[str] = None,
        include_perplexity: bool = True,
        include_burstiness: bool = True,
        use_gpu: bool = False,
    ):
        """
        Initialize text detector orchestrator.

        Args:
            language: Force language detection (ISO code)
            include_perplexity: Include perplexity analysis
            include_burstiness: Include burstiness analysis
            use_gpu: Whether to use GPU acceleration
        """
        config = get_config()
        self.language = language
        self.include_perplexity = include_perplexity
        self.include_burstiness = include_burstiness
        self.use_gpu = use_gpu

        self.processor = TextProcessor()
        self.primary_model = SpecDetect(device=torch.device("cpu"))
        self.secondary_model = PerplexityDetector(device=torch.device("cpu"))

        self.primary_model.load()
        self.secondary_model.load()

        logger.info(
            f"TextDetector initialized: language={language}, "
            f"perplexity={include_perplexity}, burstiness={include_burstiness}"
        )

    def analyze(
        self,
        text: str,
        progress_callback: Optional[callable] = None,
    ) -> TextAnalysisResult:
        """
        Run full text analysis pipeline.

        Args:
            text: Input text string
            progress_callback: Optional callback for progress updates

        Returns:
            TextAnalysisResult with scores, sentences, and metadata
        """
        if not text or not text.strip():
            return TextAnalysisResult(
                overall_score=0.5,
                confidence=0.0,
                is_manipulated=False,
                word_count=0,
                language_detected="unknown",
                per_sentence_results=[],
            )

        if progress_callback:
            progress_callback(10, "Detecting language")

        language = self._detect_language(text)

        if progress_callback:
            progress_callback(20, "Tokenizing text")

        sentences = self._split_sentences(text)

        if progress_callback:
            progress_callback(40, "Running spectral analysis")

        spectral_results = self._run_spectral_analysis(sentences)

        if progress_callback:
            progress_callback(70, "Running perplexity analysis")

        perplexity_results = self._run_perplexity_analysis(sentences)

        if progress_callback:
            progress_callback(85, "Analyzing burstiness")

        burstiness_metrics = self._analyze_burstiness(text, sentences)

        if progress_callback:
            progress_callback(95, "Aggregating results")

        result = self._aggregate_results(
            text, sentences, spectral_results, perplexity_results, burstiness_metrics, language
        )

        if progress_callback:
            progress_callback(100, "Analysis complete")

        logger.info(
            f"Text analysis complete: {result.word_count} words, "
            f"score={result.overall_score:.3f}"
        )

        return result

    def _detect_language(self, text: str) -> str:
        """Detect language of input text."""
        try:
            metadata = self.processor._detect_language_metadata(text)
            return metadata.language if hasattr(metadata, 'language') else "unknown"
        except Exception:
            return "unknown"

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _run_spectral_analysis(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Run SpecDetect on sentences."""
        results = []
        for sentence in sentences:
            try:
                result = self.primary_model.detect(sentence)
                results.append({
                    "sentence": sentence,
                    "score": result.overall_score,
                    "confidence": result.confidence,
                })
            except Exception:
                results.append({
                    "sentence": sentence,
                    "score": 0.5,
                    "confidence": 0.0,
                })
        return results

    def _run_perplexity_analysis(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Run PerplexityDetector on sentences."""
        if not self.include_perplexity:
            return []

        results = []
        for sentence in sentences:
            try:
                result = self.secondary_model.detect(sentence)
                if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
                    scores = result.metadata.get('per_model_scores', {})
                else:
                    scores = {"ensemble": result.overall_score}

                results.append({
                    "sentence": sentence,
                    "scores": scores,
                    "ensemble_score": result.overall_score,
                })
            except Exception:
                results.append({
                    "sentence": sentence,
                    "scores": {},
                    "ensemble_score": 0.5,
                })
        return results

    def _analyze_burstiness(
        self,
        text: str,
        sentences: List[str],
    ) -> Dict[str, float]:
        """Analyze burstiness metrics."""
        if not self.include_burstiness:
            return {}

        try:
            metrics = self.processor._calculate_burstiness(text)
            return {
                "burstiness_score": metrics.burstiness_score if hasattr(metrics, 'burstiness_score') else 0.5,
                "sentence_variance": metrics.sentence_length_std if hasattr(metrics, 'sentence_length_std') else 0.0,
            }
        except Exception:
            return {"burstiness_score": 0.5}

    def _aggregate_results(
        self,
        text: str,
        sentences: List[str],
        spectral_results: List[Dict[str, Any]],
        perplexity_results: List[Dict[str, Any]],
        burstiness_metrics: Dict[str, float],
        language: str,
    ) -> TextAnalysisResult:
        """Aggregate results into final result."""
        if not sentences:
            return TextAnalysisResult(
                overall_score=0.5,
                confidence=0.0,
                is_manipulated=False,
                word_count=0,
                language_detected=language,
                per_sentence_results=[],
            )

        word_count = len(text.split())

        sentence_results = []
        all_scores = []

        for i, sentence in enumerate(sentences):
            spectral = spectral_results[i] if i < len(spectral_results) else {"score": 0.5, "confidence": 0.0}
            perplexity = perplexity_results[i] if i < len(perplexity_results) else {"ensemble_score": 0.5}

            spectral_score = spectral.get("score", 0.5)
            perplexity_score = perplexity.get("ensemble_score", 0.5)

            fused_score = 0.6 * spectral_score + 0.4 * perplexity_score
            all_scores.append(fused_score)

            start_char = text.find(sentence) if sentence in text else 0
            end_char = start_char + len(sentence)

            sentence_results.append(
                SentenceAnalysisResult(
                    sentence=sentence,
                    start_char=start_char,
                    end_char=end_char,
                    manipulation_score=fused_score,
                    perplexity=perplexity_score,
                    is_suspicious=fused_score > 0.6,
                )
            )

        overall_score = float(np.mean(all_scores))
        confidence = self._compute_confidence(all_scores, overall_score)

        perplexity_agg = {}
        for result in perplexity_results:
            for model, score in result.get("scores", {}).items():
                if model not in perplexity_agg:
                    perplexity_agg[model] = []
                perplexity_agg[model].append(score)

        perplexity_scores = {k: float(np.mean(v)) for k, v in perplexity_agg.items()}

        metadata = {
            "num_sentences": len(sentences),
            "include_perplexity": self.include_perplexity,
            "include_burstiness": self.include_burstiness,
        }

        return TextAnalysisResult(
            overall_score=overall_score,
            confidence=confidence,
            is_manipulated=overall_score > 0.5,
            word_count=word_count,
            language_detected=language,
            per_sentence_results=sentence_results,
            perplexity_scores=perplexity_scores,
            burstiness_metrics=burstiness_metrics,
            metadata=metadata,
        )

    def _compute_confidence(self, scores: List[float], overall_score: float) -> float:
        """Compute confidence score."""
        if len(scores) < 2:
            return 0.5

        score_variance = np.var(scores)
        distance_from_middle = abs(overall_score - 0.5) * 2

        confidence = min(
            max(distance_from_middle * 0.7 + (1 - score_variance) * 0.3, 0.0),
            1.0,
        )

        return confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "primary_model": "SpecDetect",
            "secondary_model": "PerplexityDetector",
            "language": self.language,
            "include_perplexity": self.include_perplexity,
            "include_burstiness": self.include_burstiness,
        }

    def health_check(self) -> bool:
        """Check if all components are healthy."""
        try:
            if not self.primary_model.health_check():
                return False
            if not self.secondary_model.health_check():
                return False
            return True
        except Exception:
            return False


def create_text_detector(
    language: Optional[str] = None,
    include_perplexity: bool = True,
    include_burstiness: bool = True,
    use_gpu: bool = False,
) -> TextDetector:
    """Factory function for TextDetector."""
    return TextDetector(
        language=language,
        include_perplexity=include_perplexity,
        include_burstiness=include_burstiness,
        use_gpu=use_gpu,
    )
