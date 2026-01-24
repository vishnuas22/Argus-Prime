"""
Verdict Generator - Human-Readable Forensic Verdicts.

Generates structured, human-readable verdict documents with:
1. Executive summary
2. Key findings
3. Evidence summary
4. Recommendations
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.core.logging import logger


@dataclass
class VerdictDocument:
    """Complete verdict document."""
    verdict: str
    confidence: float
    summary: str
    key_findings: List[str]
    evidence_summary: str
    recommendation: str
    technical_details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "summary": self.summary,
            "key_findings": self.key_findings,
            "evidence_summary": self.evidence_summary,
            "recommendation": self.recommendation,
            "technical_details": self.technical_details,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# Argus Prime Forensic Analysis Verdict",
            "",
            f"## Verdict: {self.verdict}",
            f"**Confidence:** {self.confidence:.1%}",
            "",
            "---",
            "",
            "## Executive Summary",
            self.summary,
            "",
            "## Key Findings",
        ]

        for i, finding in enumerate(self.key_findings, 1):
            lines.append(f"{i}. {finding}")

        lines.extend([
            "",
            "## Evidence Summary",
            self.evidence_summary,
            "",
            "## Recommendation",
            self.recommendation,
        ])

        if self.technical_details:
            lines.extend([
                "",
                "## Technical Details",
                "```json",
                str(self.technical_details),
                "```",
            ])

        return "\n".join(lines)


class VerdictGenerator:
    """
    Verdict Document Generator.

    Generates human-readable forensic verdicts with:
    - Clear verdict classification
    - Evidence-based reasoning
    - Actionable recommendations
    - Multi-language support

    Attributes:
        confidence_thresholds: Thresholds for verdict classification
    """

    def __init__(
        self,
        highly_authentic_threshold: float = 80.0,
        likely_authentic_threshold: float = 60.0,
        likely_manipulated_threshold: float = 40.0,
    ):
        """
        Initialize verdict generator.

        Args:
            highly_authentic_threshold: Threshold for HIGHLY_AUTHENTIC
            likely_authentic_threshold: Threshold for LIKELY_AUTHENTIC
            likely_manipulated_threshold: Threshold for LIKELY_MANIPULATED
        """
        self.thresholds = {
            "highly_authentic": highly_authentic_threshold,
            "likely_authentic": likely_authentic_threshold,
            "likely_manipulated": likely_manipulated_threshold,
        }

        config = get_config()
        self.detection_threshold = config.analysis.confidence_threshold

        logger.info("VerdictGenerator initialized")

    def _classify_verdict(self, trust_score: float) -> str:
        """Classify verdict based on trust score."""
        if trust_score >= self.thresholds["highly_authentic"]:
            return "HIGHLY_AUTHENTIC"
        elif trust_score >= self.thresholds["likely_authentic"]:
            return "LIKELY_AUTHENTIC"
        elif trust_score >= self.thresholds["likely_manipulated"]:
            return "UNCERTAIN"
        elif trust_score >= self.detection_threshold * 100:
            return "LIKELY_MANIPULATED"
        else:
            return "HIGHLY_MANIPULATED"

    def _generate_summary(
        self,
        verdict: str,
        trust_score: float,
        confidence: float,
        manipulation_type: Optional[str] = None,
    ) -> str:
        """Generate executive summary text."""
        confidence_pct = int(confidence * 100)

        if verdict == "HIGHLY_AUTHENTIC":
            return (
                f"This content shows strong indicators of authenticity with a trust score of "
                f"{trust_score:.0f}/100. The analysis confidence is {confidence_pct}%. "
                f"All detection models consistently indicate genuine content."
            )
        elif verdict == "LIKELY_AUTHENTIC":
            return (
                f"This content appears to be authentic with a trust score of "
                f"{trust_score:.0f}/100. The analysis confidence is {confidence_pct}%. "
                f"Most indicators support authenticity, with minor uncertainties."
            )
        elif verdict == "UNCERTAIN":
            if manipulation_type:
                return (
                    f"The analysis produced uncertain results (trust score: {trust_score:.0f}/100, "
                    f"confidence: {confidence_pct}%). Potential {manipulation_type} was detected "
                    f"but requires additional verification."
                )
            return (
                f"The analysis could not determine a clear verdict. Trust score: "
                f"{trust_score:.0f}/100, confidence: {confidence_pct}%. "
                f"Additional review may be required."
            )
        elif verdict == "LIKELY_MANIPULATED":
            return (
                f"This content shows indicators of manipulation with a trust score of "
                f"{trust_score:.0f}/100. The analysis confidence is {confidence_pct}%. "
                f"Detected patterns suggest potential {manipulation_type or 'digital manipulation'}."
            )
        else:
            return (
                f"This content shows strong indicators of manipulation. Trust score: "
                f"{trust_score:.0f}/100, confidence: {confidence_pct}%. "
                f"Clear evidence of {manipulation_type or 'digital manipulation'} has been detected."
            )

    def _extract_key_findings(
        self,
        video_result: Optional[Dict] = None,
        audio_result: Optional[Dict] = None,
        text_result: Optional[Dict] = None,
        crossmodal_result: Optional[Dict] = None,
    ) -> List[str]:
        """Extract key findings from analysis results."""
        findings = []

        if video_result:
            overall_score = video_result.get("overall_score", 0.5)
            if overall_score > 0.7:
                findings.append(
                    f"Video analysis detected manipulation with score {overall_score:.2f}"
                )
                branch_scores = video_result.get("branch_scores", {})
                for branch, score in branch_scores.items():
                    if score > 0.6:
                        findings.append(f"  - {branch} anomaly detected: {score:.2f}")
            elif overall_score < 0.3:
                findings.append(f"Video analysis indicates authenticity (score: {overall_score:.2f})")

            faces = video_result.get("faces_detected", 0)
            if faces == 0:
                findings.append("No faces detected in video content")

        if audio_result:
            overall_score = audio_result.get("overall_score", 0.5)
            if overall_score > 0.7:
                findings.append(
                    f"Audio analysis detected synthetic speech with score {overall_score:.2f}"
                )
                vocoder = audio_result.get("vocoder_detected")
                if vocoder:
                    findings.append(f"  - Vocoder signature detected: {vocoder}")
            elif overall_score < 0.3:
                findings.append(f"Audio analysis indicates authentic speech (score: {overall_score:.2f})")

        if text_result:
            overall_score = text_result.get("overall_score", 0.5)
            if overall_score > 0.7:
                findings.append(
                    f"Text analysis detected AI-generated content with score {overall_score:.2f}"
                )
                perplexity = text_result.get("perplexity_scores", {})
                if perplexity:
                    avg_perplexity = sum(perplexity.values()) / len(perplexity)
                    findings.append(f"  - Average perplexity: {avg_perplexity:.2f}")
            elif overall_score < 0.3:
                findings.append(f"Text analysis indicates human-written content (score: {overall_score:.2f})")

        if crossmodal_result:
            consistency = crossmodal_result.get("consistency_score", 0.5)
            if consistency < 0.5:
                findings.append(
                    f"Cross-modal consistency check failed (score: {consistency:.2f})"
                )
                contradictions = crossmodal_result.get("contradictions", [])
                for contra in contradictions[:3]:
                    findings.append(f"  - {contra.get('type', 'Unknown')}: {contra.get('description', '')[:80]}")

        if not findings:
            findings.append("No significant anomalies detected across all modalities")

        return findings

    def _generate_evidence_summary(
        self,
        video_result: Optional[Dict] = None,
        audio_result: Optional[Dict] = None,
        text_result: Optional[Dict] = None,
        localization_result: Optional[Dict] = None,
    ) -> str:
        """Generate evidence summary."""
        evidence_count = 0
        total_confidence = 0.0

        if video_result:
            evidence_count += 1
            total_confidence += video_result.get("confidence", 0.5)

        if audio_result:
            evidence_count += 1
            total_confidence += audio_result.get("confidence", 0.5)

        if text_result:
            evidence_count += 1
            total_confidence += text_result.get("confidence", 0.5)

        if evidence_count == 0:
            return "Insufficient evidence for conclusive analysis."

        avg_confidence = total_confidence / evidence_count

        if localization_result:
            temporal_regions = localization_result.get("temporal_regions", [])
            if temporal_regions:
                regions_count = len(temporal_regions)
                return (
                    f"Analysis based on {evidence_count} modalities with average confidence "
                    f"{avg_confidence:.0%}. {regions_count} manipulation regions localized temporally. "
                    f"Evidence supports the verdict with {'high' if avg_confidence > 0.7 else 'moderate' if avg_confidence > 0.5 else 'low'} confidence."
                )

        return (
            f"Analysis based on {evidence_count} modalities with average confidence "
            f"{avg_confidence:.0%}. Evidence supports the verdict with "
            f"{'high' if avg_confidence > 0.7 else 'moderate' if avg_confidence > 0.5 else 'low'} confidence."
        )

    def _generate_recommendation(self, verdict: str, confidence: float) -> str:
        """Generate recommendation based on verdict."""
        if verdict == "HIGHLY_AUTHENTIC":
            return (
                "This content appears to be authentic. Standard verification procedures are sufficient. "
                "No immediate action required."
            )
        elif verdict == "LIKELY_AUTHENTIC":
            return (
                "This content appears to be authentic with high confidence. "
                "Routine verification is recommended."
            )
        elif verdict == "UNCERTAIN":
            return (
                "This content requires additional verification. Consider:"
                " 1. Manual review by a forensic expert"
                " 2. Comparison with authentic samples"
                " 3. Check for metadata (C2PA, EXIF) that may provide provenance"
            )
        elif verdict == "LIKELY_MANIPULATED":
            return (
                "This content shows indicators of manipulation. Recommended actions:"
                " 1. Do not use without additional verification"
                " 2. Consult with a forensic expert"
                " 3. Check source and chain of custody"
            )
        else:
            return (
                "WARNING: This content shows strong indicators of manipulation. "
                "Do not use without thorough forensic analysis. "
                "Consult with a forensic expert immediately."
            )

    def generate_verdict(
        self,
        trust_score: float,
        confidence: float,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        crossmodal_result: Optional[Dict[str, Any]] = None,
        localization_result: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> VerdictDocument:
        """
        Generate complete verdict document.

        Args:
            trust_score: Overall trust score (0-100)
            confidence: Analysis confidence (0-1)
            video_result: Video analysis result
            audio_result: Audio analysis result
            text_result: Text analysis result
            crossmodal_result: Cross-modal consistency result
            localization_result: Localization result
            metadata: Additional metadata

        Returns:
            Complete VerdictDocument
        """
        verdict = self._classify_verdict(trust_score)

        manipulation_type = metadata.get("manipulation_type") if metadata else None

        summary = self._generate_summary(
            verdict, trust_score, confidence, manipulation_type
        )

        key_findings = self._extract_key_findings(
            video_result, audio_result, text_result, crossmodal_result
        )

        evidence_summary = self._generate_evidence_summary(
            video_result, audio_result, text_result, localization_result
        )

        recommendation = self._generate_recommendation(verdict, confidence)

        technical_details = {}
        if metadata:
            technical_details.update(metadata)
        if video_result:
            technical_details["video_score"] = video_result.get("overall_score")
        if audio_result:
            technical_details["audio_score"] = audio_result.get("overall_score")
        if text_result:
            technical_details["text_score"] = text_result.get("overall_score")

        return VerdictDocument(
            verdict=verdict,
            confidence=confidence,
            summary=summary,
            key_findings=key_findings,
            evidence_summary=evidence_summary,
            recommendation=recommendation,
            technical_details=technical_details,
        )

    def generate_from_trust_result(
        self,
        trust_result: Dict[str, Any],
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        crossmodal_result: Optional[Dict[str, Any]] = None,
        localization_result: Optional[Dict[str, Any]] = None,
    ) -> VerdictDocument:
        """
        Generate verdict from trust score result.

        Args:
            trust_result: Trust score result dictionary
            video_result: Video analysis result
            audio_result: Audio analysis result
            text_result: Text analysis result
            crossmodal_result: Cross-modal consistency result
            localization_result: Localization result

        Returns:
            Complete VerdictDocument
        """
        return self.generate_verdict(
            trust_score=trust_result.get("overall_score", 50.0),
            confidence=trust_result.get("confidence", 0.5),
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
            crossmodal_result=crossmodal_result,
            localization_result=localization_result,
            metadata={"manipulation_type": trust_result.get("what")},
        )


def create_verdict_generator() -> VerdictGenerator:
    """Factory function for VerdictGenerator."""
    return VerdictGenerator()
