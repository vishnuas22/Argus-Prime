"""
Unit Tests for Cross-Modal Consistency Checker.

Tests consistency verification across video, audio, text, and biological signals.
"""

import pytest
from typing import Dict, Any, Optional

from src.analysis.crossmodal import (
    CrossModalChecker,
    CrossModalResult,
    Contradiction,
)


class TestContradiction:
    """Test cases for Contradiction dataclass."""

    def test_contradiction_creation(self):
        """Test creating a Contradiction."""
        contradiction = Contradiction(
            type="lip_sync_mismatch",
            description="Video shows manipulation but audio appears authentic",
            confidence=0.75,
            severity="high",
            evidence={"video_score": 0.8, "audio_score": 0.2},
            recommendation="Manual review recommended",
        )
        assert contradiction.type == "lip_sync_mismatch"
        assert contradiction.confidence == 0.75
        assert contradiction.severity == "high"
        assert len(contradiction.evidence) == 2

    def test_contradiction_defaults(self):
        """Test Contradiction with default values."""
        contradiction = Contradiction(
            type="test_type",
            description="Test description",
            confidence=0.5,
            severity="medium",
        )
        assert contradiction.evidence == {}
        assert contradiction.recommendation == ""


class TestCrossModalResult:
    """Test cases for CrossModalResult."""

    def test_result_creation(self):
        """Test creating a CrossModalResult."""
        contradictions = [
            Contradiction(
                type="test1",
                description="Test 1",
                confidence=0.6,
                severity="medium",
            ),
            Contradiction(
                type="test2",
                description="Test 2",
                confidence=0.7,
                severity="high",
            ),
        ]
        result = CrossModalResult(
            consistency_score=0.85,
            is_consistent=True,
            contradictions=contradictions,
            lip_sync_score=0.9,
            temporal_score=0.8,
            content_score=0.95,
            biological_score=0.75,
            recommendations=["Recommendation 1"],
            raw_scores={"lip_sync": 0.9, "temporal": 0.8},
        )
        assert result.consistency_score == 0.85
        assert result.is_consistent is True
        assert len(result.contradictions) == 2

    def test_result_defaults(self):
        """Test CrossModalResult with defaults."""
        result = CrossModalResult(
            consistency_score=1.0,
            is_consistent=True,
        )
        assert result.contradictions == []
        assert result.lip_sync_score is None
        assert result.recommendations == []

    def test_result_to_dict(self):
        """Test CrossModalResult serialization."""
        result = CrossModalResult(
            consistency_score=0.9,
            is_consistent=True,
            contradictions=[
                Contradiction(
                    type="test",
                    description="Test",
                    confidence=0.8,
                    severity="high",
                )
            ],
            raw_scores={"score": 0.9},
        )
        as_dict = result.to_dict()
        assert as_dict["consistency_score"] == 0.9
        assert as_dict["is_consistent"] is True
        assert len(as_dict["contradictions"]) == 1
        assert as_dict["contradictions"][0]["type"] == "test"
        assert as_dict["raw_scores"]["score"] == 0.9


class TestCrossModalChecker:
    """Test cases for CrossModalChecker."""

    @pytest.fixture
    def checker(self):
        """Create a CrossModalChecker instance."""
        return CrossModalChecker()

    def test_checker_initialization(self, checker):
        """Test that checker initializes correctly."""
        assert checker is not None
        assert checker.consistency_threshold == 0.3
        assert len(checker.severity_weights) > 0

    def test_check_lip_sync_consistent(self, checker):
        """Test lip-sync check with consistent scores."""
        video_result = {"overall_score": 0.5, "confidence": 0.8}
        audio_result = {"overall_score": 0.5, "confidence": 0.8}
        
        score, contradictions = checker._check_lip_sync(video_result, audio_result)
        assert score == 1.0
        assert len(contradictions) == 0

    def test_check_lip_sync_mismatch_high(self, checker):
        """Test lip-sync check with high manipulation mismatch."""
        video_result = {"overall_score": 0.9, "confidence": 0.8}
        audio_result = {"overall_score": 0.2, "confidence": 0.8}
        
        score, contradictions = checker._check_lip_sync(video_result, audio_result)
        assert score < 1.0
        assert len(contradictions) > 0
        assert contradictions[0].type == "lip_sync_mismatch"
        assert contradictions[0].severity == "high"

    def test_check_lip_sync_vocoder_detected(self, checker):
        """Test lip-sync check with vocoder detected."""
        video_result = {"overall_score": 0.5, "confidence": 0.8}
        audio_result = {
            "overall_score": 0.7,
            "confidence": 0.8,
            "vocoder_detected": "HiFi-GAN",
        }
        
        score, contradictions = checker._check_lip_sync(video_result, audio_result)
        assert len(contradictions) > 0
        vocoder_contradictions = [c for c in contradictions if c.type == "vocoder_detected"]
        assert len(vocoder_contradictions) > 0

    def test_check_lip_sync_no_video(self, checker):
        """Test lip-sync check with no video result."""
        score, contradictions = checker._check_lip_sync(None, {"overall_score": 0.5})
        assert score is None
        assert len(contradictions) == 0

    def test_check_lip_sync_no_audio(self, checker):
        """Test lip-sync check with no audio result."""
        score, contradictions = checker._check_lip_sync({"overall_score": 0.5}, None)
        assert score is None
        assert len(contradictions) == 0

    def test_check_temporal_consistent(self, checker):
        """Test temporal check with consistent data."""
        video_result = {
            "frames_analyzed": 150,
            "per_frame_scores": [{"manipulation_score": 0.5}] * 10,
        }
        audio_result = {
            "duration_seconds": 5.0,
            "segments_analyzed": 10,
            "per_segment_scores": [{"manipulation_score": 0.5}] * 10,
        }
        
        score, contradictions = checker._check_temporal_consistency(video_result, audio_result)
        assert score == 1.0
        assert len(contradictions) == 0

    def test_check_temporal_duration_mismatch(self, checker):
        """Test temporal check with duration mismatch."""
        video_result = {
            "frames_analyzed": 300,  # ~10 seconds
            "per_frame_scores": [{"manipulation_score": 0.5}] * 10,
        }
        audio_result = {
            "duration_seconds": 3.0,  # Much shorter
            "segments_analyzed": 10,
            "per_segment_scores": [{"manipulation_score": 0.5}] * 10,
        }
        
        score, contradictions = checker._check_temporal_consistency(video_result, audio_result)
        assert score < 1.0
        duration_contradictions = [c for c in contradictions if c.type == "duration_mismatch"]
        assert len(duration_contradictions) > 0

    def test_check_content_consistent(self, checker):
        """Test content check with consistent data."""
        video_result = {"overall_score": 0.5, "confidence": 0.8}
        text_result = {"overall_score": 0.5, "confidence": 0.8}
        
        score, contradictions = checker._check_content_consistency(video_result, text_result)
        assert score == 1.0
        assert len(contradictions) == 0

    def test_check_content_mismatch(self, checker):
        """Test content check with mismatch."""
        video_result = {"overall_score": 0.9, "confidence": 0.8}
        text_result = {"overall_score": 0.2, "confidence": 0.8}
        
        score, contradictions = checker._check_content_consistency(video_result, text_result)
        assert score < 0.7
        assert len(contradictions) > 0
        assert contradictions[0].type == "content_mismatch"

    def test_check_biological_consistent(self, checker):
        """Test biological check with consistent data."""
        video_result = {"overall_score": 0.5}
        rppg_result = {
            "quality_score": 0.8,
            "temporal_consistency": 0.9,
        }
        
        score, contradictions = checker._check_biological_consistency(video_result, rppg_result)
        assert score >= 0.9
        assert len(contradictions) == 0

    def test_check_biological_poor_quality(self, checker):
        """Test biological check with poor signal quality."""
        video_result = {"overall_score": 0.5}
        rppg_result = {
            "quality_score": 0.2,  # Poor quality
            "temporal_consistency": 0.9,
        }
        
        score, contradictions = checker._check_biological_consistency(video_result, rppg_result)
        assert score < 0.7
        quality_contradictions = [c for c in contradictions if c.type == "poor_biological_signal"]
        assert len(quality_contradictions) > 0

    def test_check_biological_manipulation_mismatch(self, checker):
        """Test biological check with manipulation vs quality mismatch."""
        video_result = {"overall_score": 0.8}  # High manipulation
        rppg_result = {
            "quality_score": 0.2,  # Poor quality
            "temporal_consistency": 0.9,
        }
        
        score, contradictions = checker._check_biological_consistency(video_result, rppg_result)
        assert score < 0.5
        mismatch_contradictions = [
            c for c in contradictions if c.type == "manipulation_biological_mismatch"
        ]
        assert len(mismatch_contradictions) > 0

    def test_check_consistency_all_consistent(self, checker):
        """Test full consistency check with all consistent data."""
        video_result = {"overall_score": 0.5, "confidence": 0.8}
        audio_result = {"overall_score": 0.5, "confidence": 0.8}
        text_result = {"overall_score": 0.5, "confidence": 0.8}
        
        result = checker.check_consistency(
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
        )
        assert result.consistency_score >= 0.9
        assert result.is_consistent is True
        assert len(result.contradictions) == 0

    def test_check_consistency_with_contradictions(self, checker):
        """Test full consistency check with contradictions."""
        video_result = {"overall_score": 0.9, "confidence": 0.8}
        audio_result = {"overall_score": 0.2, "confidence": 0.8}
        
        result = checker.check_consistency(
            video_result=video_result,
            audio_result=audio_result,
        )
        assert result.consistency_score < 0.8
        assert len(result.contradictions) > 0
        assert result.lip_sync_score < 1.0

    def test_check_consistency_no_data(self, checker):
        """Test full consistency check with no data."""
        result = checker.check_consistency()
        assert result.consistency_score == 1.0
        assert result.is_consistent is True

    def test_check_consistency_recommendations(self, checker):
        """Test that recommendations are generated."""
        video_result = {"overall_score": 0.9, "confidence": 0.8}
        audio_result = {"overall_score": 0.9, "confidence": 0.8}
        
        result = checker.check_consistency(
            video_result=video_result,
            audio_result=audio_result,
        )
        # Should have recommendations when there are contradictions
        assert len(result.recommendations) >= 0

    def test_check_from_results(self, checker):
        """Test convenience method check_from_results."""
        video_result = {"overall_score": 0.5, "confidence": 0.8}
        audio_result = {"overall_score": 0.5, "confidence": 0.8}
        
        result = checker.check_from_results(
            video_result=video_result,
            audio_result=audio_result,
        )
        assert result.consistency_score >= 0.9
        assert result.is_consistent is True

    def test_get_crossmodal_checker_singleton(self):
        """Test that get_crossmodal_checker returns singleton."""
        from src.analysis.crossmodal import get_crossmodal_checker
        checker1 = get_crossmodal_checker()
        checker2 = get_crossmodal_checker()
        assert checker1 is checker2


class TestCrossModalCheckerEdgeCases:
    """Edge case tests for CrossModalChecker."""

    @pytest.fixture
    def checker(self):
        return CrossModalChecker()

    def test_check_with_missing_confidence(self, checker):
        """Test checks when confidence is missing."""
        video_result = {"overall_score": 0.5}  # No confidence
        audio_result = {"overall_score": 0.5}
        
        score, _ = checker._check_lip_sync(video_result, audio_result)
        assert score is not None

    def test_check_with_empty_per_frame_scores(self, checker):
        """Test temporal check with empty per-frame scores."""
        video_result = {"frames_analyzed": 150, "per_frame_scores": []}
        audio_result = {"duration_seconds": 5.0, "segments_analyzed": 5, "per_segment_scores": []}
        
        score, contradictions = checker._check_temporal_consistency(video_result, audio_result)
        assert score == 1.0  # Should still pass

    def test_severity_weights(self, checker):
        """Test severity weight mapping."""
        assert checker.severity_weights["critical"] == 1.0
        assert checker.severity_weights["high"] == 0.75
        assert checker.severity_weights["medium"] == 0.5
        assert checker.severity_weights["low"] == 0.25
        assert checker.severity_weights["info"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
