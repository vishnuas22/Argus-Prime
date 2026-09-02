"""
Text preprocessing for deepfake detection.

Handles tokenization, perplexity calculation, and burstiness analysis.
"""

import asyncio
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.config import get_config
from src.core.logging import logger


@dataclass
class TextSegment:
    """Represents a text segment for analysis."""
    text: str
    start_idx: int
    end_idx: int
    tokens: Optional[List[int]] = None
    perplexity: Optional[float] = None
    is_ai_indicator: bool = False


@dataclass
class TextMetadata:
    """Metadata about the processed text."""
    total_characters: int
    total_words: int
    total_sentences: int
    language: str
    language_confidence: float
    avg_sentence_length: float
    word_vocabulary_size: int


@dataclass
class BurstinessMetrics:
    """Burstiness analysis metrics for AI-generated text detection."""
    sentence_length_variance: float
    sentence_length_std: float
    word_frequency_entropy: float
    repetition_ratio: float
    burstiness_score: float
    top_repeated_words: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class PerplexityMetrics:
    """Perplexity scores from multiple LLM models."""
    gpt2_perplexity: Optional[float] = None
    xglm_perplexity: Optional[float] = None
    ensemble_perplexity: Optional[float] = None
    perplexity_std: Optional[float] = None


@dataclass
class TextAnalysisBundle:
    """Complete text analysis result bundle."""
    segments: List[TextSegment]
    metadata: TextMetadata
    burstiness: BurstinessMetrics
    perplexity: PerplexityMetrics
    tokens: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "metadata": {
                "total_characters": self.metadata.total_characters,
                "total_words": self.metadata.total_words,
                "total_sentences": self.metadata.total_sentences,
                "language": self.metadata.language,
                "language_confidence": self.metadata.language_confidence,
                "avg_sentence_length": self.metadata.avg_sentence_length,
                "word_vocabulary_size": self.metadata.word_vocabulary_size,
            },
            "burstiness": {
                "sentence_length_variance": self.burstiness.sentence_length_variance,
                "sentence_length_std": self.burstiness.sentence_length_std,
                "word_frequency_entropy": self.burstiness.word_frequency_entropy,
                "repetition_ratio": self.burstiness.repetition_ratio,
                "burstiness_score": self.burstiness.burstiness_score,
                "top_repeated_words": self.burstiness.top_repeated_words,
            },
            "perplexity": {
                "gpt2_perplexity": self.perplexity.gpt2_perplexity,
                "xglm_perplexity": self.perplexity.xglm_perplexity,
                "ensemble_perplexity": self.perplexity.ensemble_perplexity,
                "perplexity_std": self.perplexity.perplexity_std,
            },
            "num_segments": len(self.segments),
        }


class TextProcessor:
    """
    Text preprocessing processor for deepfake detection.

    Handles:
    - Language detection
    - Tokenization with multilingual models
    - Perplexity calculation using multiple LLMs
    - Burstiness analysis

    Attributes:
        model_name: Name of the tokenizer model
        max_length: Maximum sequence length
        device: Device to run models on (cuda/cpu)
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        """
        Initialize the text processor.

        Args:
            model_name: HuggingFace model name for tokenization
            max_length: Maximum sequence length for tokenization
            device: Device to run models on (auto-detected if None)
        """
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"TextProcessor initialized: model={model_name}, device={self.device}")

        self._tokenizer: Optional[PreTrainedTokenizer] = None
        self._gpt2_model: Optional[PreTrainedModel] = None
        self._xglm_model: Optional[PreTrainedModel] = None
        self._langdetect_available = False

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            logger.info(f"Loading tokenizer: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            logger.info("Tokenizer loaded successfully")
        return self._tokenizer

    def _load_gpt2(self) -> PreTrainedModel:
        """Lazy load GPT-2 for perplexity."""
        if self._gpt2_model is None:
            logger.info("Loading GPT-2 for perplexity calculation")
            self._gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
            self._gpt2_model.to(self.device)
            self._gpt2_model.eval()
            logger.info("GPT-2 loaded successfully")
        return self._gpt2_model

    def _load_xglm(self) -> PreTrainedModel:
        """Lazy load XGLM for multilingual perplexity."""
        if self._xglm_model is None:
            logger.info("Loading XGLM-1.7B for multilingual perplexity")
            self._xglm_model = AutoModelForSeq2SeqLM.from_pretrained(
                "facebook/xglm-1.7B"
            )
            self._xglm_model.to(self.device)
            self._xglm_model.eval()
            logger.info("XGLM loaded successfully")
        return self._xglm_model

    def _detect_language(self, text: str) -> Tuple[str, float]:
        """
        Detect the language of the input text.

        Args:
            text: Input text to analyze

        Returns:
            Tuple of (language_code, confidence)
        """
        try:
            import fasttext

            model_path = get_config().models.fasttext_model_path
            model = fasttext.load_model(model_path)
            predictions = model.predict(text.replace("\n", " "), k=1)
            language = predictions[0][0].replace("__label__", "")
            confidence = predictions[1][0]
            self._langdetect_available = True
        except ImportError:
            logger.warning("fasttext not installed, using simple language detection")
            self._langdetect_available = False
            language = self._simple_language_detect(text)
            confidence = 0.8

        return language, confidence

    def _simple_language_detect(self, text: str) -> str:
        """Simple rule-based language detection."""
        english_markers = [" the ", " and ", " is ", " are ", " was ", " were "]
        spanish_markers = [" el ", " la ", " los ", " las ", " es ", " son "]
        french_markers = [" le ", " la ", " les ", " est ", " sont "]
        german_markers = [" der ", " die ", " das ", " ist ", " sind "]

        text_lower = text.lower()

        scores = {
            "en": sum(1 for m in english_markers if m in text_lower),
            "es": sum(1 for m in spanish_markers if m in text_lower),
            "fr": sum(1 for m in french_markers if m in text_lower),
            "de": sum(1 for m in german_markers if m in text_lower),
        }

        return max(scores, key=scores.get) if max(scores.values()) > 0 else "en"

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        sentence_endings = r"[.!?]+"
        sentences = re.split(sentence_endings, text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def tokenize(self, text: str) -> Tuple[List[int], torch.Tensor]:
        """
        Tokenize input text.

        Args:
            text: Input text to tokenize

        Returns:
            Tuple of (token_ids list, token_tensor)
        """
        tokenizer = self._load_tokenizer()

        if len(text) > self.max_length * 4:
            text = text[: self.max_length * 4]
            logger.warning(f"Text truncated to {self.max_length * 4} characters")

        encoding = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        token_ids = encoding["input_ids"].tolist()[0]
        return token_ids, encoding["input_ids"]

    def calculate_perplexity(
        self, text: str, model_type: str = "gpt2"
    ) -> float:
        """
        Calculate perplexity for a text using specified model.

        Args:
            text: Input text
            model_type: Model to use ("gpt2" or "xglm")

        Returns:
            Perplexity score
        """
        if model_type == "gpt2":
            model = self._load_gpt2()
            tokenizer = self._tokenizer
            model_name = "GPT-2"
        else:
            model = self._load_xglm()
            tokenizer = self._tokenizer
            model_name = "XGLM"

        tokenizer = self._load_tokenizer()

        encodings = tokenizer(
            text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        logger.info(f"{model_name} perplexity: {perplexity:.2f}")
        return perplexity

    def calculate_perplexity_ensemble(self, text: str) -> PerplexityMetrics:
        """
        Calculate perplexity using multiple models and ensemble.

        Args:
            text: Input text

        Returns:
            PerplexityMetrics with all scores
        """
        metrics = PerplexityMetrics()

        try:
            metrics.gpt2_perplexity = self.calculate_perplexity(text, "gpt2")
        except Exception as e:
            logger.warning(f"GPT-2 perplexity calculation failed: {e}")

        try:
            metrics.xglm_perplexity = self.calculate_perplexity(text, "xglm")
        except Exception as e:
            logger.warning(f"XGLM perplexity calculation failed: {e}")

        perplexities = [
            p for p in [metrics.gpt2_perplexity, metrics.xglm_perplexity] if p is not None
        ]

        if perplexities:
            metrics.ensemble_perplexity = float(np.mean(perplexities))
            metrics.perplexity_std = float(np.std(perplexities))
            logger.info(f"Ensemble perplexity: {metrics.ensemble_perplexity:.2f}")

        return metrics

    def _calculate_sentence_lengths(self, sentences: List[str]) -> List[int]:
        """Calculate word count for each sentence."""
        return [len(s.split()) for s in sentences]

    def _calculate_word_frequencies(
        self, text: str
    ) -> Tuple[Counter, float]:
        """
        Calculate word frequencies and entropy.

        Args:
            text: Input text

        Returns:
            Tuple of (word_counter, entropy)
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        word_counts = Counter(words)

        if not word_counts:
            return word_counts, 0.0

        total = sum(word_counts.values())
        probs = np.array(list(word_counts.values())) / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log2(probs))

        return word_counts, entropy

    def analyze_burstiness(self, text: str) -> BurstinessMetrics:
        """
        Analyze burstiness patterns in text.

        AI-generated text often shows:
        - Lower sentence length variance (more uniform)
        - Higher word frequency entropy (less natural distribution)
        - Lower repetition ratio (less natural repetition)

        Args:
            text: Input text

        Returns:
            BurstinessMetrics with analysis results
        """
        sentences = self._split_into_sentences(text)

        if not sentences:
            return BurstinessMetrics(
                sentence_length_variance=0.0,
                sentence_length_std=0.0,
                word_frequency_entropy=0.0,
                repetition_ratio=0.0,
                burstiness_score=0.0,
            )

        sentence_lengths = self._calculate_sentence_lengths(sentences)
        word_counts, word_entropy = self._calculate_word_frequencies(text)

        sentence_length_variance = float(np.var(sentence_lengths))
        sentence_length_std = float(np.std(sentence_lengths))

        total_words = sum(word_counts.values())
        repeated_words = sum(1 for c in word_counts.values() if c > 1)
        repetition_ratio = (
            repeated_words / len(word_counts) if word_counts else 0.0
        )

        max_possible_entropy = np.log2(len(word_counts)) if word_counts else 0.0
        normalized_entropy = (
            word_entropy / max_possible_entropy if max_possible_entropy > 0 else 0.0
        )

        burstiness_score = (
            0.4 * (1 - normalized_entropy)
            + 0.3 * repetition_ratio
            + 0.3 * (1 / (1 + sentence_length_std))
        )

        top_repeated = word_counts.most_common(10)

        logger.info(
            f"Burstiness analysis: variance={sentence_length_variance:.2f}, "
            f"entropy={word_entropy:.2f}, score={burstiness_score:.3f}"
        )

        return BurstinessMetrics(
            sentence_length_variance=sentence_length_variance,
            sentence_length_std=sentence_length_std,
            word_frequency_entropy=word_entropy,
            repetition_ratio=repetition_ratio,
            burstiness_score=burstiness_score,
            top_repeated_words=top_repeated,
        )

    def process(
        self,
        text: str,
        calculate_perplexity: bool = True,
        min_segment_words: int = 5,
    ) -> TextAnalysisBundle:
        """
        Complete text preprocessing pipeline.

        Args:
            text: Input text to analyze
            calculate_perplexity: Whether to calculate perplexity scores
            min_segment_words: Minimum words per segment for analysis

        Returns:
            TextAnalysisBundle with all analysis results
        """
        logger.info(f"Processing text: {len(text)} characters")

        language, language_confidence = self._detect_language(text)

        sentences = self._split_into_sentences(text)
        total_words = sum(len(s.split()) for s in sentences)

        token_ids, token_tensor = self.tokenize(text)

        segments = []
        current_segment = []
        current_idx = 0

        for sentence in sentences:
            words = sentence.split()
            if len(current_segment) + len(words) >= min_segment_words:
                if current_segment:
                    segment_text = " ".join(current_segment)
                    segments.append(
                        TextSegment(
                            text=segment_text,
                            start_idx=current_idx,
                            end_idx=current_idx + len(segment_text),
                        )
                    )
                current_segment = words
                current_idx += len(sentence) + 1
            else:
                current_segment.extend(words)

        if current_segment:
            segment_text = " ".join(current_segment)
            segments.append(
                TextSegment(
                    text=segment_text,
                    start_idx=current_idx,
                    end_idx=current_idx + len(segment_text),
                )
            )

        metadata = TextMetadata(
            total_characters=len(text),
            total_words=total_words,
            total_sentences=len(sentences),
            language=language,
            language_confidence=language_confidence,
            avg_sentence_length=total_words / len(sentences) if sentences else 0,
            word_vocabulary_size=len(set(re.findall(r"\b[a-zA-Z]+\b", text.lower()))),
        )

        burstiness = self.analyze_burstiness(text)

        perplexity = PerplexityMetrics()
        if calculate_perplexity:
            perplexity = self.calculate_perplexity_ensemble(text)

        logger.info(
            f"Text processing complete: {len(segments)} segments, "
            f"language={language}, burstiness={burstiness.burstiness_score:.3f}"
        )

        return TextAnalysisBundle(
            segments=segments,
            metadata=metadata,
            burstiness=burstiness,
            perplexity=perplexity,
            tokens=token_tensor,
        )

    async def process_async(
        self,
        text: str,
        calculate_perplexity: bool = True,
    ) -> TextAnalysisBundle:
        """
        Async version of process.

        Args:
            text: Input text to analyze
            calculate_perplexity: Whether to calculate perplexity scores

        Returns:
            TextAnalysisBundle with all analysis results
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.process(text, calculate_perplexity),
        )


def process_text_async(
    text: str,
    model_name: str = "bert-base-multilingual-cased",
    max_length: int = 512,
) -> TextAnalysisBundle:
    """
    Convenience function for async text processing.

    Args:
        text: Input text to analyze
        model_name: Tokenizer model name
        max_length: Maximum sequence length

    Returns:
        TextAnalysisBundle with analysis results
    """
    processor = TextProcessor(model_name=model_name, max_length=max_length)
    return processor.process(text)


def process_text_sync(
    text: str,
    model_name: str = "bert-base-multilingual-cased",
    max_length: int = 512,
) -> TextAnalysisBundle:
    """
    Convenience function for sync text processing.

    Args:
        text: Input text to analyze
        model_name: Tokenizer model name
        max_length: Maximum sequence length

    Returns:
        TextAnalysisBundle with analysis results
    """
    return process_text_async(text, model_name, max_length)
