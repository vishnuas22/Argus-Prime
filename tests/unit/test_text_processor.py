"""
Unit tests for text preprocessing module.

Tests tokenization, perplexity, and burstiness analysis.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

from src.preprocessing.text_processor import (
    TextProcessor,
    TextSegment,
    TextMetadata,
    TextAnalysisBundle,
    BurstinessMetrics,
    PerplexityMetrics,
    process_text_async,
    process_text_sync,
)


class TestTextSegment:
    """Test TextSegment dataclass."""

    def test_segment_creation(self):
        """Test TextSegment object creation."""
        segment = TextSegment(
            text="This is a test sentence.",
            start_idx=0,
            end_idx=25,
        )

        assert segment.text == "This is a test sentence."
        assert segment.start_idx == 0
        assert segment.end_idx == 25
        assert segment.tokens is None
        assert segment.perplexity is None
        assert segment.is_ai_indicator is False

    def test_segment_with_data(self):
        """Test TextSegment with full data."""
        segment = TextSegment(
            text="AI generated text",
            start_idx=0,
            end_idx=18,
            tokens=[1, 2, 3, 4],
            perplexity=42.5,
            is_ai_indicator=True,
        )

        assert segment.tokens == [1, 2, 3, 4]
        assert segment.perplexity == 42.5
        assert segment.is_ai_indicator is True


class TestTextMetadata:
    """Test TextMetadata dataclass."""

    def test_metadata_creation(self):
        """Test TextMetadata object creation."""
        metadata = TextMetadata(
            total_characters=1000,
            total_words=150,
            total_sentences=10,
            language="en",
            language_confidence=0.95,
            avg_sentence_length=15.0,
            word_vocabulary_size=80,
        )

        assert metadata.total_characters == 1000
        assert metadata.total_words == 150
        assert metadata.language == "en"


class TestBurstinessMetrics:
    """Test BurstinessMetrics dataclass."""

    def test_metrics_creation(self):
        """Test BurstinessMetrics object creation."""
        metrics = BurstinessMetrics(
            sentence_length_variance=25.0,
            sentence_length_std=5.0,
            word_frequency_entropy=4.5,
            repetition_ratio=0.1,
            burstiness_score=0.75,
            top_repeated_words=[("the", 50), ("and", 30)],
        )

        assert metrics.sentence_length_variance == 25.0
        assert metrics.burstiness_score == 0.75
        assert len(metrics.top_repeated_words) == 2


class TestPerplexityMetrics:
    """Test PerplexityMetrics dataclass."""

    def test_metrics_creation(self):
        """Test PerplexityMetrics object creation."""
        metrics = PerplexityMetrics(
            gpt2_perplexity=35.5,
            xglm_perplexity=42.0,
            ensemble_perplexity=38.75,
            perplexity_std=3.25,
        )

        assert metrics.gpt2_perplexity == 35.5
        assert metrics.ensemble_perplexity == 38.75


class TestTextProcessor:
    """Test TextProcessor class."""

    def test_processor_initialization(self):
        """Test TextProcessor initialization."""
        processor = TextProcessor()

        assert processor.model_name == "bert-base-multilingual-cased"
        assert processor.max_length == 512
        assert processor.device in ["cuda", "cpu"]

    def test_processor_with_custom_params(self):
        """Test TextProcessor with custom parameters."""
        processor = TextProcessor(
            model_name="bert-base-uncased",
            max_length=256,
            device="cpu",
        )

        assert processor.model_name == "bert-base-uncased"
        assert processor.max_length == 256

    def test_split_into_sentences(self):
        """Test sentence splitting."""
        processor = TextProcessor()
        text = "This is first sentence. And this is second! Is this third?"
        sentences = processor._split_into_sentences(text)

        assert len(sentences) == 3
        assert "This is first sentence" in sentences[0]

    def test_split_into_sentences_empty(self):
        """Test sentence splitting with empty text."""
        processor = TextProcessor()
        sentences = processor._split_into_sentences("")
        assert sentences == []

    def test_split_into_sentences_special_chars(self):
        """Test sentence splitting with special characters."""
        processor = TextProcessor()
        text = "Hello... world! How are you? I'm fine."
        sentences = processor._split_into_sentences(text)

        assert len(sentences) >= 2
        assert any("world" in s for s in sentences)
        assert any("How are you" in s for s in sentences)

    def test_calculate_sentence_lengths(self):
        """Test sentence length calculation."""
        processor = TextProcessor()
        sentences = ["One two three", "A B C D E", "Short"]
        lengths = processor._calculate_sentence_lengths(sentences)

        assert lengths == [3, 5, 1]

    def test_calculate_word_frequencies(self):
        """Test word frequency calculation."""
        processor = TextProcessor()
        text = "the cat sat on the mat the dog"
        counter, entropy = processor._calculate_word_frequencies(text)

        assert counter["the"] == 3
        assert counter["cat"] == 1
        assert entropy > 0

    def test_calculate_word_frequencies_empty(self):
        """Test word frequency with empty text."""
        processor = TextProcessor()
        counter, entropy = processor._calculate_word_frequencies("")

        assert counter == {}
        assert entropy == 0.0

    def test_simple_language_detect_english(self):
        """Test simple language detection for English."""
        processor = TextProcessor()
        lang = processor._simple_language_detect("The quick brown fox jumps over the lazy dog")

        assert lang == "en"

    def test_simple_language_detect_spanish(self):
        """Test simple language detection for Spanish."""
        processor = TextProcessor()
        lang = processor._simple_language_detect("El perro marrón rápido salta sobre el perro perezoso")

        assert lang == "es"

    def test_simple_language_detect_french(self):
        """Test simple language detection for French."""
        processor = TextProcessor()
        lang = processor._simple_language_detect("Le chien brun rapide saute sur le chien paresseux")

        assert lang == "fr"

    def test_simple_language_detect_german(self):
        """Test simple language detection for German."""
        processor = TextProcessor()
        lang = processor._simple_language_detect("Der Hund läuft schnell über die Straße")

        assert lang == "de"


class TestTextAnalysisBundle:
    """Test TextAnalysisBundle class."""

    def test_bundle_creation(self):
        """Test bundle creation."""
        segments = [
            TextSegment(text="First sentence.", start_idx=0, end_idx=15),
            TextSegment(text="Second sentence.", start_idx=16, end_idx=33),
        ]
        metadata = TextMetadata(
            total_characters=33,
            total_words=4,
            total_sentences=2,
            language="en",
            language_confidence=0.9,
            avg_sentence_length=2.0,
            word_vocabulary_size=4,
        )
        burstiness = BurstinessMetrics(
            sentence_length_variance=0.0,
            sentence_length_std=0.0,
            word_frequency_entropy=2.0,
            repetition_ratio=0.0,
            burstiness_score=0.5,
        )
        perplexity = PerplexityMetrics(
            gpt2_perplexity=30.0,
            xglm_perplexity=35.0,
            ensemble_perplexity=32.5,
            perplexity_std=2.5,
        )

        bundle = TextAnalysisBundle(
            segments=segments,
            metadata=metadata,
            burstiness=burstiness,
            perplexity=perplexity,
        )

        assert len(bundle.segments) == 2
        assert bundle.metadata.language == "en"

    def test_bundle_to_dict(self):
        """Test bundle serialization."""
        segments = [TextSegment(text="Test.", start_idx=0, end_idx=5)]
        metadata = TextMetadata(
            total_characters=5,
            total_words=1,
            total_sentences=1,
            language="en",
            language_confidence=0.9,
            avg_sentence_length=1.0,
            word_vocabulary_size=1,
        )
        burstiness = BurstinessMetrics(
            sentence_length_variance=0.0,
            sentence_length_std=0.0,
            word_frequency_entropy=0.0,
            repetition_ratio=0.0,
            burstiness_score=0.5,
        )
        perplexity = PerplexityMetrics()

        bundle = TextAnalysisBundle(
            segments=segments,
            metadata=metadata,
            burstiness=burstiness,
            perplexity=perplexity,
        )

        bundle_dict = bundle.to_dict()

        assert "metadata" in bundle_dict
        assert "burstiness" in bundle_dict
        assert "perplexity" in bundle_dict
        assert bundle_dict["metadata"]["language"] == "en"


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_process_text_sync_function(self):
        """Test the sync convenience function."""
        with patch('src.preprocessing.text_processor.TextProcessor') as MockProcessor:
            mock_instance = MagicMock()
            mock_instance.process.return_value = TextAnalysisBundle(
                segments=[],
                metadata=TextMetadata(
                    total_characters=100,
                    total_words=20,
                    total_sentences=5,
                    language="en",
                    language_confidence=0.9,
                    avg_sentence_length=4.0,
                    word_vocabulary_size=15,
                ),
                burstiness=BurstinessMetrics(
                    sentence_length_variance=0.0,
                    sentence_length_std=0.0,
                    word_frequency_entropy=0.0,
                    repetition_ratio=0.0,
                    burstiness_score=0.5,
                ),
                perplexity=PerplexityMetrics(),
            )
            MockProcessor.return_value = mock_instance

            result = process_text_sync("This is a test text for analysis.")

            assert result.metadata.total_characters == 100
            mock_instance.process.assert_called_once()

    def test_process_text_async_function(self):
        """Test the async convenience function."""
        with patch('src.preprocessing.text_processor.TextProcessor') as MockProcessor:
            mock_instance = MagicMock()
            mock_bundle = TextAnalysisBundle(
                segments=[],
                metadata=TextMetadata(
                    total_characters=50,
                    total_words=10,
                    total_sentences=2,
                    language="en",
                    language_confidence=0.85,
                    avg_sentence_length=5.0,
                    word_vocabulary_size=10,
                ),
                burstiness=BurstinessMetrics(
                    sentence_length_variance=0.0,
                    sentence_length_std=0.0,
                    word_frequency_entropy=0.0,
                    repetition_ratio=0.0,
                    burstiness_score=0.5,
                ),
                perplexity=PerplexityMetrics(),
            )
            mock_instance.process.return_value = mock_bundle
            MockProcessor.return_value = mock_instance

            result = process_text_async("Async test text")

            assert result.metadata.total_characters == 50
            mock_instance.process.assert_called_once()


class TestTextProcessorIntegration:
    """Integration tests for TextProcessor."""

    def test_tokenize(self):
        """Test text tokenization."""
        processor = TextProcessor()
        text = "This is a test sentence for tokenization."
        token_ids, token_tensor = processor.tokenize(text)

        assert len(token_ids) > 0
        assert isinstance(token_ids, list)
        assert isinstance(token_tensor, torch.Tensor)

    def test_analyze_burstiness(self):
        """Test burstiness analysis."""
        processor = TextProcessor()
        text = "First sentence here. Second sentence there. Third sentence everywhere."
        burstiness = processor.analyze_burstiness(text)

        assert burstiness.sentence_length_variance >= 0
        assert 0 <= burstiness.burstiness_score <= 1

    def test_analyze_burstiness_single_sentence(self):
        """Test burstiness with single sentence."""
        processor = TextProcessor()
        text = "Just one sentence."
        burstiness = processor.analyze_burstiness(text)

        assert burstiness.sentence_length_variance == 0.0
        assert burstiness.sentence_length_std == 0.0

    def test_process_full_pipeline(self):
        """Test complete processing pipeline."""
        processor = TextProcessor()
        text = "This is a test. This is another test. And a third test sentence."
        bundle = processor.process(text, calculate_perplexity=False)

        assert isinstance(bundle, TextAnalysisBundle)
        assert bundle.metadata.total_characters > 0
        assert bundle.metadata.total_sentences == 3
        assert len(bundle.segments) > 0
        assert bundle.burstiness is not None

    def test_process_short_text(self):
        """Test processing short text."""
        processor = TextProcessor()
        text = "Short text."
        bundle = processor.process(text, calculate_perplexity=False)

        assert bundle.metadata.total_characters == len("Short text.")
        assert bundle.metadata.total_sentences == 1

    def test_process_with_english_text(self):
        """Test processing English text."""
        processor = TextProcessor()
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump!"
        )
        bundle = processor.process(text, calculate_perplexity=False)

        assert bundle.metadata.language == "en"
        assert bundle.metadata.total_sentences == 3

    def test_process_with_multilingual_text(self):
        """Test processing multilingual text."""
        processor = TextProcessor()
        text = "Hello world. Hola mundo. Bonjour le monde."

        with patch.object(processor, '_detect_language', return_value=('en', 0.85)):
            bundle = processor.process(text, calculate_perplexity=False)

        assert bundle.metadata.language == "en"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
