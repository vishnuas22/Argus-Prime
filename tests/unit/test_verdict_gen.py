"""
Unit Tests for Verdict Generator.

Tests human-readable verdict generation and document formatting.
"""

import pytest
from pathlib import Path

from src.reporting.verdict_gen import (
    VerdictGenerator,
    VerdictDocument,
)


class TestVerdictGenerator:
    """Test cases for VerdictGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a VerdictGenerator instance."""
        return VerdictGenerator(
            highly_authentic_threshold=80.0,
            likely_authentic_threshold=60.0,
            likely_manipulated_threshold=40.0,
        )

    def test_generator_initialization(self, generator):
        """Test that generator initializes correctly."""
        assert generator is not None
        assert generator.thresholds["highly_authentic"] == 80.0
        assert generator.thresholds["likely_authentic"] == 60.0

    def test_classify_verdict_highly_authentic(self, generator):
        """Test verdict classification for highly authentic content."""
        assert generator._classify_verdict(85.0) == "HIGHLY_AUTHENTIC"
        assert generator._classify_verdict(95.0) == "HIGHLY_AUTHENTIC"

    def test_classify_verdict_likely_authentic(self, generator):
        """Test verdict classification for likely authentic content."""
        assert generator._classify_verdict(70.0) == "LIKELY_AUTHENTIC"
        assert generator._classify_verdict(65.0) == "LIKELY_AUTHENTIC"

    def test_classify_verdict_uncertain(self, generator):
        """Test verdict classification for uncertain content."""
        assert generator._classify_verdict(50.0) == "UNCERTAIN"
        assert generator._classify_verdict(45.0) == "UNCERTAIN"

    def test_classify_verdict_likely_manipulated(self, generator):
        """Test verdict classification for likely manipulated content."""
        assert generator._classify_verdict(35.0) == "HIGHLY_MANIPULATED"
        assert generator._classify_verdict(45.0) == "UNCERTAIN"
        assert generator._classify_verdict(55.0) == "UNCERTAIN"

    def test_generate_summary_highly_authentic(self, generator):
        """Test summary generation for highly authentic content."""
        summary = generator._generate_summary("HIGHLY_AUTHENTIC", 85.0, 0.9)
        assert "authenticity" in summary.lower()
        assert "85" in summary

    def test_generate_summary_highly_manipulated(self, generator):
        """Test summary generation for highly manipulated content."""
        summary = generator._generate_summary("HIGHLY_MANIPULATED", 15.0, 0.85)
        assert "manipulation" in summary.lower()

    def test_generate_summary_with_manipulation_type(self, generator):
        """Test summary generation with manipulation type."""
        summary = generator._generate_summary("LIKELY_MANIPULATED", 30.0, 0.7, "face swapping")
        assert "face swapping" in summary.lower()

    def test_extract_key_findings_empty(self, generator):
        """Test key findings extraction with no results."""
        findings = generator._extract_key_findings()
        assert len(findings) == 1
        assert "No significant anomalies" in findings[0]

    def test_extract_key_findings_video_manipulated(self, generator):
        """Test key findings extraction from video result."""
        video_result = {
            "overall_score": 0.8,
            "branch_scores": {"texture": 0.75, "spectral": 0.6},
            "faces_detected": 2,
        }
        findings = generator._extract_key_findings(video_result=video_result)
        assert len(findings) >= 1
        assert any("manipulation" in f.lower() for f in findings)

    def test_extract_key_findings_video_no_faces(self, generator):
        """Test key findings when no faces detected."""
        video_result = {
            "overall_score": 0.5,
            "faces_detected": 0,
        }
        findings = generator._extract_key_findings(video_result=video_result)
        assert any("No faces detected" in f for f in findings)

    def test_extract_key_findings_audio(self, generator):
        """Test key findings extraction from audio result."""
        audio_result = {
            "overall_score": 0.85,
            "vocoder_detected": "HiFi-GAN",
            "confidence": 0.8,
        }
        findings = generator._extract_key_findings(audio_result=audio_result)
        assert len(findings) >= 1
        assert any("synthetic" in f.lower() or "vocoder" in f.lower() for f in findings)

    def test_extract_key_findings_text(self, generator):
        """Test key findings extraction from text result."""
        text_result = {
            "overall_score": 0.9,
            "perplexity_scores": {"gpt2": 0.85, "llama": 0.88},
            "confidence": 0.75,
        }
        findings = generator._extract_key_findings(text_result=text_result)
        assert len(findings) >= 1
        assert any("AI-generated" in f for f in findings)

    def test_extract_key_findings_crossmodal(self, generator):
        """Test key findings extraction from cross-modal result."""
        crossmodal_result = {
            "consistency_score": 0.3,
            "contradictions": [
                {"type": "lip_sync", "description": "Mouth movements don't match audio"},
                {"type": "temporal", "description": "Frame timestamps inconsistent"},
            ],
        }
        findings = generator._extract_key_findings(crossmodal_result=crossmodal_result)
        assert len(findings) >= 1
        assert any("consistency" in f.lower() for f in findings)

    def test_generate_evidence_summary(self, generator):
        """Test evidence summary generation."""
        video_result = {"confidence": 0.8}
        audio_result = {"confidence": 0.7}
        summary = generator._generate_evidence_summary(video_result, audio_result)
        assert "modalities" in summary.lower()
        assert "confidence" in summary.lower()

    def test_generate_evidence_summary_with_localization(self, generator):
        """Test evidence summary with localization data."""
        video_result = {"confidence": 0.8}
        localization_result = {
            "temporal_regions": [
                {"start_frame": 10, "end_frame": 20, "confidence": 0.8},
                {"start_frame": 30, "end_frame": 35, "confidence": 0.7},
            ]
        }
        summary = generator._generate_evidence_summary(
            video_result, localization_result=localization_result
        )
        assert "regions" in summary.lower() or "manipulation" in summary.lower()

    def test_generate_recommendation_highly_authentic(self, generator):
        """Test recommendation for highly authentic content."""
        rec = generator._generate_recommendation("HIGHLY_AUTHENTIC", 0.9)
        assert "authentic" in rec.lower()
        assert "action required" in rec.lower() or "sufficient" in rec.lower()

    def test_generate_recommendation_highly_manipulated(self, generator):
        """Test recommendation for highly manipulated content."""
        rec = generator._generate_recommendation("HIGHLY_MANIPULATED", 0.85)
        assert "manipulation" in rec.lower() or "warning" in rec.lower()
        assert "expert" in rec.lower() or "forensic" in rec.lower()

    def test_generate_verdict(self, generator):
        """Test complete verdict generation."""
        video_result = {"overall_score": 0.8, "confidence": 0.85, "faces_detected": 3}
        trust_result = {
            "overall_score": 75.0,
            "confidence": 0.8,
            "verdict": "LIKELY_AUTHENTIC",
        }

        verdict = generator.generate_verdict(
            trust_score=75.0,
            confidence=0.8,
            video_result=video_result,
        )

        assert isinstance(verdict, VerdictDocument)
        assert verdict.confidence == 0.8
        assert len(verdict.key_findings) >= 1

    def test_generate_from_trust_result(self, generator):
        """Test verdict generation from trust result dict."""
        trust_result = {
            "overall_score": 45.0,
            "confidence": 0.7,
            "verdict": "UNCERTAIN",
            "what": "face swapping",
        }
        video_result = {"overall_score": 0.5, "confidence": 0.7}

        verdict = generator.generate_from_trust_result(
            trust_result=trust_result,
            video_result=video_result,
        )

        assert isinstance(verdict, VerdictDocument)
        assert verdict.confidence == 0.7
        assert "face swapping" in verdict.technical_details.get("manipulation_type", "")


class TestVerdictDocument:
    """Test cases for VerdictDocument dataclass."""

    def test_to_dict(self):
        """Test dictionary serialization."""
        doc = VerdictDocument(
            verdict="HIGHLY_AUTHENTIC",
            confidence=0.9,
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            evidence_summary="Test evidence",
            recommendation="Test recommendation",
        )
        as_dict = doc.to_dict()
        assert as_dict["verdict"] == "HIGHLY_AUTHENTIC"
        assert as_dict["confidence"] == 0.9
        assert len(as_dict["key_findings"]) == 2

    def test_to_markdown(self):
        """Test markdown conversion."""
        doc = VerdictDocument(
            verdict="HIGHLY_AUTHENTIC",
            confidence=0.9,
            summary="Test summary",
            key_findings=["Finding 1", "Finding 2"],
            evidence_summary="Test evidence",
            recommendation="Test recommendation",
        )
        markdown = doc.to_markdown()
        assert "# Argus Prime Forensic Analysis Verdict" in markdown
        assert "## Verdict: HIGHLY_AUTHENTIC" in markdown
        assert "## Executive Summary" in markdown
        assert "## Key Findings" in markdown
        assert "1. Finding 1" in markdown

    def test_to_markdown_with_technical_details(self):
        """Test markdown with technical details."""
        doc = VerdictDocument(
            verdict="UNCERTAIN",
            confidence=0.5,
            summary="Test",
            key_findings=[],
            evidence_summary="Test",
            recommendation="Test",
            technical_details={"video_score": 0.6, "audio_score": 0.4},
        )
        markdown = doc.to_markdown()
        assert "## Technical Details" in markdown
        assert "```json" in markdown


class TestVerdictGeneratorIntegration:
    """Integration tests for VerdictGenerator."""

    def test_create_verdict_generator_factory(self):
        """Test factory function."""
        from src.reporting.verdict_gen import create_verdict_generator
        generator = create_verdict_generator()
        assert isinstance(generator, VerdictGenerator)

    def test_complete_workflow(self):
        """Test complete verdict generation workflow."""
        generator = VerdictGenerator()

        video_result = {
            "overall_score": 0.75,
            "confidence": 0.85,
            "faces_detected": 5,
            "branch_scores": {"texture": 0.7, "spectral": 0.5},
        }

        audio_result = {
            "overall_score": 0.6,
            "confidence": 0.7,
            "vocoder_detected": None,
        }

        crossmodal_result = {
            "consistency_score": 0.8,
            "contradictions": [],
        }

        verdict = generator.generate_verdict(
            trust_score=68.0,
            confidence=0.78,
            video_result=video_result,
            audio_result=audio_result,
            crossmodal_result=crossmodal_result,
        )

        assert isinstance(verdict, VerdictDocument)
        assert verdict.verdict == "LIKELY_AUTHENTIC"
        assert verdict.confidence == 0.78
        assert len(verdict.key_findings) >= 1
        assert "confidence" in verdict.evidence_summary.lower()

        markdown = verdict.to_markdown()
        assert "## Verdict:" in markdown

    def test_manipulated_content_verdict(self):
        """Test verdict generation for manipulated content."""
        generator = VerdictGenerator()

        video_result = {
            "overall_score": 0.92,
            "confidence": 0.9,
            "faces_detected": 2,
            "branch_scores": {"texture": 0.9, "spectral": 0.88},
        }

        audio_result = {
            "overall_score": 0.85,
            "confidence": 0.85,
            "vocoder_detected": "HiFi-GAN",
        }

        verdict = generator.generate_verdict(
            trust_score=25.0,
            confidence=0.88,
            video_result=video_result,
            audio_result=audio_result,
        )

        assert verdict.verdict == "HIGHLY_MANIPULATED"
        assert "manipulation" in verdict.summary.lower()
        assert "expert" in verdict.recommendation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
