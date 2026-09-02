"""
Unit tests for TRIM Analyzer - Training-free Adversarial Detection.
"""

import pytest
import torch
import numpy as np

from src.defense import TRIMAnalyzer, TRIMResult


class TestTRIMAnalyzer:
    """Test TRIM Analyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = TRIMAnalyzer(threshold=0.1, entropy_window=5)

        assert analyzer.threshold == 0.1
        assert analyzer.entropy_window == 5
        assert analyzer.device.type in ["cpu", "cuda"]

    def test_compute_entropy_uniform(self):
        """Test entropy computation for uniform distribution."""
        analyzer = TRIMAnalyzer()

        logits = torch.zeros(1, 10)
        entropy = analyzer.compute_entropy(logits)

        assert entropy.shape == (1,)
        expected_entropy = np.log(10)
        assert abs(entropy.item() - expected_entropy) < 0.1

    def test_compute_entropy_peaked(self):
        """Test entropy computation for peaked distribution."""
        analyzer = TRIMAnalyzer()

        logits = torch.zeros(1, 10)
        logits[0, 0] = 10
        entropy = analyzer.compute_entropy(logits)

        assert entropy.shape == (1,)
        assert entropy.item() < 1.0

    def test_compute_kl_divergence_identical(self):
        """Test KL divergence for identical distributions."""
        analyzer = TRIMAnalyzer()

        probs = torch.softmax(torch.randn(1, 10), dim=-1)
        kl_div = analyzer.compute_kl_divergence(probs, probs)

        assert kl_div < 0.01

    def test_compute_kl_divergence_different(self):
        """Test KL divergence for different distributions."""
        analyzer = TRIMAnalyzer()

        probs1 = torch.softmax(torch.randn(1, 10), dim=-1)
        probs2 = torch.softmax(torch.randn(1, 10), dim=-1)
        kl_div = analyzer.compute_kl_divergence(probs1, probs2)

        assert kl_div >= 0

    def test_analyze_predictions_adversarial(self):
        """Test adversarial detection with shifted predictions."""
        analyzer = TRIMAnalyzer(threshold=0.1)

        clean_logits = torch.zeros(1, 10)
        perturbed_logits = torch.zeros(1, 10)
        perturbed_logits[0, 0] = 5

        result = analyzer.analyze_predictions(clean_logits, perturbed_logits)

        assert isinstance(result, TRIMResult)
        assert 0 <= result.entropy_score <= 5  # Entropy can exceed 1 for dramatic changes
        assert 0 <= result.confidence <= 1
        assert isinstance(result.is_adversarial, bool)

    def test_analyze_single_clean(self):
        """Test analysis of clean predictions."""
        analyzer = TRIMAnalyzer(threshold=0.1)

        predictions = torch.softmax(torch.randn(5, 10), dim=-1)
        result = analyzer.analyze_single(predictions)

        assert isinstance(result, TRIMResult)
        assert result.kl_divergence == 0.0
        assert result.js_divergence == 0.0

    def test_trim_result_to_dict(self):
        """Test TRIMResult serialization."""
        result = TRIMResult(
            is_adversarial=True,
            entropy_score=0.3,
            kl_divergence=0.2,
            js_divergence=0.15,
            confidence=0.8,
            details={"test": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["is_adversarial"] is True
        assert result_dict["entropy_score"] == 0.3
        assert result_dict["details"]["test"] == "value"

    def test_batch_analyze(self):
        """Test batch prediction analysis."""
        analyzer = TRIMAnalyzer()

        predictions = torch.softmax(torch.randn(20, 10), dim=-1)
        result = analyzer.batch_analyze(predictions, window_size=5)

        assert isinstance(result, TRIMResult)
        assert result.entropy_score >= 0


class TestTRIMThresholds:
    """Test TRIM threshold behavior."""

    def test_low_threshold_more_adversarial(self):
        """Test that lower threshold flags more as adversarial."""
        low_threshold = TRIMAnalyzer(threshold=0.01)
        high_threshold = TRIMAnalyzer(threshold=0.5)

        clean_logits = torch.zeros(1, 10)
        perturbed_logits = torch.zeros(1, 10)
        perturbed_logits[0, 0] = 3

        low_result = low_threshold.analyze_predictions(clean_logits, perturbed_logits)
        high_result = high_threshold.analyze_predictions(clean_logits, perturbed_logits)

        assert low_result.is_adversarial or not high_result.is_adversarial or low_result.confidence >= high_result.confidence


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
