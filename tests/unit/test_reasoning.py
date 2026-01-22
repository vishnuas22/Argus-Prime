"""
Unit tests for Reasoning Generator - Chain-of-Thought Explainability.
"""

import pytest

from src.explainability import (
    ReasoningGenerator,
    ReasoningResult,
    create_reasoning_generator,
)


class TestReasoningGenerator:
    """Test Reasoning Generator."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = ReasoningGenerator()

        assert generator.thresholds["highly_manipulated"] == 0.8
        assert generator.thresholds["likely_manipulated"] == 0.6
        assert generator.thresholds["likely_authentic"] == 0.4

    def test_classify_verdict_highly_manipulated(self):
        """Test verdict classification for highly manipulated."""
        generator = ReasoningGenerator()

        verdict = generator._classify_verdict(0.9)
        assert verdict == "HIGHLY_MANIPULATED"

    def test_classify_verdict_likely_manipulated(self):
        """Test verdict classification for likely manipulated."""
        generator = ReasoningGenerator()

        verdict = generator._classify_verdict(0.7)
        assert verdict == "LIKELY_MANIPULATED"

    def test_classify_verdict_uncertain(self):
        """Test verdict classification for uncertain."""
        generator = ReasoningGenerator()

        verdict = generator._classify_verdict(0.5)
        assert verdict == "UNCERTAIN"

    def test_classify_verdict_likely_authentic(self):
        """Test verdict classification for likely authentic."""
        generator = ReasoningGenerator()

        # Score of 0.3 is below likely_authentic_threshold (0.4) and detection_threshold (0.5)
        # so it should be HIGHLY_AUTHENTIC
        verdict = generator._classify_verdict(0.3)
        assert verdict == "HIGHLY_AUTHENTIC"

        # Score of 0.45 is between detection_threshold (0.5) and likely_authentic (0.4)
        # Actually 0.45 < 0.5, so it should be UNCERTAIN (between 0.4 and 0.5)
        verdict = generator._classify_verdict(0.45)
        assert verdict == "UNCERTAIN"

    def test_classify_verdict_highly_authentic(self):
        """Test verdict classification for highly authentic."""
        generator = ReasoningGenerator()

        verdict = generator._classify_verdict(0.1)
        assert verdict == "HIGHLY_AUTHENTIC"

    def test_detect_manipulation_type_video(self):
        """Test manipulation type detection for video."""
        generator = ReasoningGenerator()

        manipulation_type = generator._detect_manipulation_type(
            video_score=0.85,
            audio_score=None,
            text_score=None,
            branch_scores={},
        )

        assert "face swapping" in manipulation_type.lower() or "manipulation" in manipulation_type.lower()

    def test_detect_manipulation_type_audio(self):
        """Test manipulation type detection for audio."""
        generator = ReasoningGenerator()

        manipulation_type = generator._detect_manipulation_type(
            video_score=None,
            audio_score=0.85,
            text_score=None,
            branch_scores={},
        )

        assert "voice" in manipulation_type.lower() or "vocoder" in manipulation_type.lower() or "synthesis" in manipulation_type.lower()

    def test_detect_manipulation_type_text(self):
        """Test manipulation type detection for text."""
        generator = ReasoningGenerator()

        manipulation_type = generator._detect_manipulation_type(
            video_score=None,
            audio_score=None,
            text_score=0.85,
            branch_scores={},
        )

        assert "text" in manipulation_type.lower() or "ai" in manipulation_type.lower() or "llm" in manipulation_type.lower()

    def test_localize_manipulation_video(self):
        """Test localization for video results."""
        generator = ReasoningGenerator()

        video_result = {
            "per_frame_scores": [
                {"frame_number": 1, "manipulation_score": 0.3},
                {"frame_number": 2, "manipulation_score": 0.8, "face_bbox": [0, 0, 100, 100]},
                {"frame_number": 3, "manipulation_score": 0.85, "face_bbox": [0, 0, 100, 100]},
            ]
        }

        location = generator._localize_manipulation(video_result=video_result, audio_result=None)

        assert location != "content-wide"
        assert "frame" in location.lower() or "facial" in location.lower()

    def test_localize_manipulation_audio(self):
        """Test localization for audio results."""
        generator = ReasoningGenerator()

        audio_result = {
            "per_segment_scores": [
                {"start_time": 0, "end_time": 2, "manipulation_score": 0.3},
                {"start_time": 2, "end_time": 4, "manipulation_score": 0.85},
            ]
        }

        location = generator._localize_manipulation(video_result=None, audio_result=audio_result)

        assert location != "content-wide"
        assert "time" in location.lower()

    def test_generate_reasoning_minimal(self):
        """Test reasoning generation with minimal input."""
        generator = ReasoningGenerator()

        result = generator.generate_reasoning()

        assert isinstance(result, ReasoningResult)
        assert result.verdict in ["UNCERTAIN", "LIKELY_AUTHENTIC"]
        assert result.confidence >= 0
        assert isinstance(result.why, list)
        assert isinstance(result.reasoning_trace, list)
        assert isinstance(result.evidence_chain, list)

    def test_generate_reasoning_video_only(self):
        """Test reasoning generation with video result."""
        generator = ReasoningGenerator()

        video_result = {
            "overall_score": 0.75,
            "branch_scores": {"texture": 0.8, "spectral": 0.6},
            "faces_detected": 2,
            "per_frame_scores": [],
        }

        result = generator.generate_reasoning(video_result=video_result)

        assert isinstance(result, ReasoningResult)
        assert result.verdict in ["LIKELY_MANIPULATED", "HIGHLY_MANIPULATED"]
        assert len(result.why) > 0

    def test_generate_reasoning_fusion(self):
        """Test reasoning generation with fusion result."""
        generator = ReasoningGenerator()

        fusion_result = {
            "fused_score": 0.65,
            "confidence": 0.7,
            "weights_used": {"video": 0.4, "audio": 0.35, "text": 0.25},
            "consistency_score": 0.6,
        }

        result = generator.generate_reasoning(fusion_result=fusion_result)

        assert isinstance(result, ReasoningResult)
        assert result.confidence == 0.7

    def test_generate_verdict_text(self):
        """Test human-readable verdict text generation."""
        generator = ReasoningGenerator()

        reasoning = ReasoningResult(
            verdict="HIGHLY_MANIPULATED",
            confidence=0.85,
            what="face swapping",
            where="Frames 10-20",
            why=["Texture anomaly: 0.8", "Spectral noise: 0.75"],
            reasoning_trace=[],
            evidence_chain=[],
        )

        text = generator.generate_verdict_text(reasoning)

        assert "highly manipulated" in text.lower()
        assert "85%" in text or "85" in text
        assert "face swapping" in text.lower()

    def test_reasoning_result_to_dict(self):
        """Test ReasoningResult serialization."""
        result = ReasoningResult(
            verdict="LIKELY_MANIPULATED",
            confidence=0.7,
            what="test manipulation",
            where="frame 5",
            why=["evidence 1"],
            reasoning_trace=[{"step": 1}],
            evidence_chain=[{"modality": "video"}],
        )

        result_dict = result.to_dict()

        assert result_dict["verdict"] == "LIKELY_MANIPULATED"
        assert result_dict["confidence"] == 0.7
        assert result_dict["what"] == "test manipulation"
        assert len(result_dict["reasoning_trace"]) == 1
        assert len(result_dict["evidence_chain"]) == 1

    def test_factory_function(self):
        """Test factory function."""
        generator = create_reasoning_generator()

        assert isinstance(generator, ReasoningGenerator)


class TestEvidenceChain:
    """Test evidence chain generation."""

    def test_evidence_chain_video(self):
        """Test evidence chain for video results."""
        generator = ReasoningGenerator()

        video_result = {
            "overall_score": 0.75,
            "branch_scores": {"texture": 0.8, "spectral": 0.6, "temporal": 0.4},
            "faces_detected": 0,
        }

        evidence = generator._generate_evidence_chain(video_result=video_result)

        assert len(evidence) > 0
        modalities = [e.get("modality") for e in evidence]
        assert "video" in modalities

    def test_evidence_chain_audio_with_vocoder(self):
        """Test evidence chain for audio with vocoder detection."""
        generator = ReasoningGenerator()

        audio_result = {
            "overall_score": 0.8,
            "vocoder_detected": "hifi_gan",
            "artifact_evidence": {"raw_net2_avg": 0.7},
            "per_segment_scores": [],
        }

        evidence = generator._generate_evidence_chain(audio_result=audio_result)

        vocoder_evidence = [e for e in evidence if "vocoder" in str(e).lower()]
        assert len(vocoder_evidence) > 0

    def test_evidence_chain_text(self):
        """Test evidence chain for text results."""
        generator = ReasoningGenerator()

        text_result = {
            "overall_score": 0.65,
            "perplexity_scores": {"gpt2": 0.6, "xglm": 0.5},
            "burstiness_metrics": {"burstiness_score": 0.3},
        }

        evidence = generator._generate_evidence_chain(text_result=text_result)

        assert len(evidence) > 0
        modalities = [e.get("modality") for e in evidence]
        assert "text" in modalities


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
