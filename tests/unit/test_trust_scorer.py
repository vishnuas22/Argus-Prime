"""
Unit tests for Trust Scorer Module.
"""

import pytest

from src.fusion.trust_scorer import TrustScore, TrustScorer, create_trust_scorer


class TestTrustScore:
    """Test TrustScore dataclass."""

    def test_initialization(self):
        """Test trust score initialization."""
        score = TrustScore(
            overall_score=87.5,
            confidence=0.89,
            confidence_level="high",
            breakdown={"video": {"weight": 0.4}},
            verdict="HIGHLY_AUTHENTIC",
            fused_score=0.875,
        )

        assert score.overall_score == 87.5
        assert score.confidence == 0.89
        assert score.confidence_level == "high"
        assert score.verdict == "HIGHLY_AUTHENTIC"
        assert score.fused_score == 0.875

    def test_to_dict(self):
        """Test trust score serialization."""
        score = TrustScore(
            overall_score=75.0,
            confidence=0.75,
            confidence_level="medium",
            breakdown={"audio": {"weight": 0.35}},
            verdict="LIKELY_AUTHENTIC",
            fused_score=0.75,
        )

        result = score.to_dict()

        assert result["overall_score"] == 75.0
        assert result["confidence"] == 0.75
        assert result["confidence_level"] == "medium"
        assert result["verdict"] == "LIKELY_AUTHENTIC"

    def test_repr(self):
        """Test trust score string representation."""
        score = TrustScore(
            overall_score=50.0,
            confidence=0.5,
            confidence_level="medium",
            breakdown={},
            verdict="UNCERTAIN",
            fused_score=0.5,
        )

        repr_str = repr(score)

        assert "TrustScore" in repr_str
        assert "score=50.0" in repr_str


class TestTrustScorer:
    """Test TrustScorer calculator."""

    def test_initialization(self):
        """Test scorer initialization."""
        scorer = TrustScorer()

        assert scorer is not None

    def test_compute_trust_score_basic(self):
        """Test basic trust score computation."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.85,
            "confidence": 0.9,
            "weights_used": {
                "video": 0.40,
                "audio": 0.35,
                "text": 0.25,
            },
        }

        score = scorer.compute_trust_score(fusion_result)

        assert isinstance(score, TrustScore)
        assert score.overall_score == 85.0
        assert score.confidence == 0.9
        assert score.confidence_level == "high"
        assert score.verdict == "HIGHLY_AUTHENTIC"

    def test_compute_trust_score_low(self):
        """Test trust score computation for low scores."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.15,
            "confidence": 0.7,
            "weights_used": {
                "video": 0.40,
                "audio": 0.35,
                "text": 0.25,
            },
        }

        score = scorer.compute_trust_score(fusion_result)

        assert score.overall_score == 15.0
        assert score.verdict == "HIGHLY_MANIPULATED"

    def test_compute_trust_score_uncertain(self):
        """Test trust score computation for uncertain scores."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.45,
            "confidence": 0.4,
            "weights_used": {
                "video": 0.40,
                "audio": 0.35,
                "text": 0.25,
            },
        }

        score = scorer.compute_trust_score(fusion_result)

        assert score.overall_score == 45.0
        assert score.confidence_level == "low"
        assert score.verdict == "UNCERTAIN"

    def test_compute_trust_score_with_metadata(self):
        """Test trust score computation with metadata."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.70,
            "confidence": 0.8,
            "weights_used": {
                "video": 0.40,
                "audio": 0.35,
                "text": 0.25,
            },
        }

        metadata = {
            "rppg_quality": 0.3,
            "adversarial_detected": True,
        }

        score = scorer.compute_trust_score(fusion_result, metadata=metadata)

        assert "adjustment_notes" in score.breakdown
        assert any("rPPG quality" in note for note in score.breakdown["adjustment_notes"])
        assert any("Adversarial" in note for note in score.breakdown["adjustment_notes"])

    def test_compute_trust_score_missing_modality(self):
        """Test trust score computation with missing modality."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.60,
            "confidence": 0.6,
            "weights_used": {
                "video": 0.50,
                "audio": 0.50,
                "text": 0.0,
            },
        }

        metadata = {
            "missing_modality": "text",
        }

        score = scorer.compute_trust_score(fusion_result, metadata=metadata)

        assert "adjustment_notes" in score.breakdown
        assert any("text" in note for note in score.breakdown["adjustment_notes"])

    def test_breakdown_computation(self):
        """Test breakdown computation."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.80,
            "confidence": 0.85,
            "weights_used": {
                "video": 0.40,
                "audio": 0.35,
                "text": 0.25,
            },
        }

        score = scorer.compute_trust_score(fusion_result)

        assert "video" in score.breakdown
        assert "audio" in score.breakdown
        assert "text" in score.breakdown

        assert score.breakdown["video"]["weight"] == 0.40
        assert score.breakdown["video"]["contribution"] == 32.0  # 0.40 * 0.80 * 100
        assert "percentage" in score.breakdown["video"]

    def test_verdict_thresholds(self):
        """Test verdict thresholds."""
        scorer = TrustScorer()

        test_cases = [
            (85.0, "HIGHLY_AUTHENTIC"),
            (65.0, "LIKELY_AUTHENTIC"),
            (50.0, "UNCERTAIN"),
            (30.0, "LIKELY_MANIPULATED"),
            (10.0, "HIGHLY_MANIPULATED"),
        ]

        for score_value, expected_verdict in test_cases:
            fusion_result = {
                "fused_score": score_value / 100,
                "confidence": 0.5,
                "weights_used": {
                    "video": 0.40,
                    "audio": 0.35,
                    "text": 0.25,
                },
            }
            score = scorer.compute_trust_score(fusion_result)
            assert score.verdict == expected_verdict, f"Score {score_value}: expected {expected_verdict}, got {score.verdict}"

    def test_confidence_levels(self):
        """Test confidence level mapping."""
        scorer = TrustScorer()

        test_cases = [
            (0.9, "high"),
            (0.8, "high"),
            (0.6, "medium"),
            (0.5, "medium"),
            (0.4, "low"),
            (0.2, "low"),
        ]

        for conf_value, expected_level in test_cases:
            fusion_result = {
                "fused_score": 0.5,
                "confidence": conf_value,
                "weights_used": {
                    "video": 0.40,
                    "audio": 0.35,
                    "text": 0.25,
                },
            }
            score = scorer.compute_trust_score(fusion_result)
            assert score.confidence_level == expected_level, f"Confidence {conf_value}: expected {expected_level}, got {score.confidence_level}"

    def test_empty_fusion_result(self):
        """Test handling of empty fusion result."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.5,
            "confidence": 0.5,
            "weights_used": {},
        }

        score = scorer.compute_trust_score(fusion_result)

        assert score.overall_score == 50.0
        assert score.verdict == "UNCERTAIN"


class TestTrustScorerFactory:
    """Test trust scorer factory function."""

    def test_create_trust_scorer(self):
        """Test factory function."""
        scorer = create_trust_scorer()

        assert isinstance(scorer, TrustScorer)


class TestTrustScorerIntegration:
    """Integration tests for trust scorer."""

    def test_complete_workflow(self):
        """Test complete trust scoring workflow."""
        scorer = TrustScorer()

        fusion_result = {
            "fused_score": 0.85,
            "confidence": 0.85,
            "weights_used": {
                "video": 0.40,
                "audio": 0.35,
                "text": 0.25,
            },
            "consistency_status": "consistent",
            "adjustments_made": ["rppg_quality_adjustment"],
        }

        score = scorer.compute_trust_score(fusion_result)

        assert isinstance(score, TrustScore)
        assert score.overall_score == 85.0
        assert score.confidence == 0.85
        assert score.confidence_level == "high"
        assert score.verdict == "HIGHLY_AUTHENTIC"
        assert score.fused_score == 0.85

        result_dict = score.to_dict()
        assert "breakdown" in result_dict
        assert "overall_score" in result_dict


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
