"""
Forensic Report Generator.

Generates comprehensive forensic reports:
1. Report structure with executive summary
2. Detailed findings per modality
3. Evidence collection
4. Recommendations
"""

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
import uuid

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.core.logging import logger
from src.fusion import FusionResult, TrustScorer

logger = logging.getLogger(__name__)


@dataclass
class ForensicReport:
    """Complete forensic analysis report."""
    report_id: str
    request_id: str
    created_at: datetime
    file_info: Dict[str, Any]
    executive_summary: Dict[str, Any]
    trust_score: Dict[str, Any]
    findings: Dict[str, Any]
    evidence: Dict[str, Any]
    recommendations: List[str]
    raw_output: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "report_id": self.report_id,
            "request_id": self.request_id,
            "created_at": self.created_at.isoformat(),
            "file_info": self.file_info,
            "executive_summary": self.executive_summary,
            "trust_score": self.trust_score,
            "findings": self.findings,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
            "raw_output": self.raw_output,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


class ReportGenerator:
    """
    Forensic Report Generator.

    Generates comprehensive forensic reports with:
    - Executive summary with verdict
    - Per-modality findings
    - Evidence and visualizations
    - Actionable recommendations
    """

    def __init__(self):
        """Initialize report generator."""
        self.trust_scorer = TrustScorer()
        logger.info("ReportGenerator initialized")

    def generate_report(
        self,
        request_id: str,
        file_info: Dict[str, Any],
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        fusion_result: Optional[FusionResult] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ForensicReport:
        """
        Generate comprehensive forensic report.

        Args:
            request_id: Associated request identifier
            file_info: Original file information
            video_result: Video analysis result
            audio_result: Audio analysis result
            text_result: Text analysis result
            fusion_result: Fused score result
            metadata: Additional metadata

        Returns:
            Complete ForensicReport
        """
        report_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        trust_result = self._compute_trust_score(fusion_result, metadata)

        executive_summary = self._generate_executive_summary(trust_result)

        findings = self._aggregate_findings(
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
        )

        evidence = self._collect_evidence(
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
        )

        recommendations = self._generate_recommendations(trust_result, findings)

        raw_output = self._collect_raw_outputs(
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
            fusion_result=fusion_result,
        )

        return ForensicReport(
            report_id=report_id,
            request_id=request_id,
            created_at=created_at,
            file_info=file_info,
            executive_summary=executive_summary,
            trust_score=trust_result,
            findings=findings,
            evidence=evidence,
            recommendations=recommendations,
            raw_output=raw_output,
        )

    def _compute_trust_score(
        self,
        fusion_result: Optional[FusionResult],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute trust score from fusion result."""
        if fusion_result is None:
            return {
                "overall_score": 50.0,
                "confidence": 0.0,
                "confidence_level": "low",
                "breakdown": {},
                "verdict": "UNCERTAIN",
                "fused_score": 0.5,
            }

        return self.trust_scorer.compute_trust_score(fusion_result, metadata)

    def _generate_executive_summary(
        self,
        trust_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate executive summary."""
        return {
            "verdict": trust_result.get("verdict", "UNCERTAIN"),
            "trust_score": trust_result.get("overall_score", 50.0),
            "confidence": trust_result.get("confidence", 0.0),
            "confidence_level": trust_result.get("confidence_level", "medium"),
            "summary_text": self._generate_summary_text(trust_result),
        }

    def _generate_summary_text(self, trust_result: Dict[str, Any]) -> str:
        """Generate human-readable summary."""
        verdict = trust_result.get("verdict", "UNCERTAIN")
        score = trust_result.get("overall_score", 50.0)
        confidence = trust_result.get("confidence", 0.0)

        if verdict == "HIGHLY_AUTHENTIC":
            return (
                "This content shows strong indicators of authenticity. "
                f"The analysis confidence is {confidence:.0%}."
            )
        elif verdict == "LIKELY_AUTHENTIC":
            return (
                "This content appears to be authentic with moderate confidence. "
                f"Analysis confidence: {confidence:.0%}."
            )
        elif verdict == "UNCERTAIN":
            return (
                "The analysis could not determine a clear verdict. "
                f"Additional review may be required. Confidence: {confidence:.0%}."
            )
        elif verdict == "LIKELY_MANIPULATED":
            return (
                "This content shows indicators of manipulation. "
                f"Further investigation is recommended. Confidence: {confidence:.0%}."
            )
        else:
            return (
                "This content shows strong indicators of manipulation. "
                f"Immediate review is advised. Confidence: {confidence:.0%}."
            )

    def _aggregate_findings(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Aggregate findings from all modalities."""
        findings = {}

        if video_result:
            findings["video"] = {
                "analyzed": True,
                "frames_analyzed": video_result.get("frames_analyzed", 0),
                "faces_detected": video_result.get("faces_detected", 0),
                "manipulation_score": video_result.get("overall_score", 0.5),
                "confidence": video_result.get("confidence", 0.5),
                "verdict": "MANIPULATED" if video_result.get("is_manipulated", False) else "AUTHENTIC",
                "branch_scores": video_result.get("branch_scores", {}),
            }

        if audio_result:
            findings["audio"] = {
                "analyzed": True,
                "duration_seconds": audio_result.get("duration_seconds", 0),
                "segments_analyzed": audio_result.get("segments_analyzed", 0),
                "manipulation_score": audio_result.get("overall_score", 0.5),
                "confidence": audio_result.get("confidence", 0.5),
                "verdict": "MANIPULATED" if audio_result.get("is_manipulated", False) else "AUTHENTIC",
                "vocoder_detected": audio_result.get("vocoder_detected"),
                "artifact_evidence": audio_result.get("artifact_evidence", {}),
            }

        if text_result:
            findings["text"] = {
                "analyzed": True,
                "word_count": text_result.get("word_count", 0),
                "language": text_result.get("language_detected", "unknown"),
                "manipulation_score": text_result.get("overall_score", 0.5),
                "confidence": text_result.get("confidence", 0.5),
                "verdict": "MANIPULATED" if text_result.get("is_manipulated", False) else "AUTHENTIC",
                "perplexity_scores": text_result.get("perplexity_scores", {}),
                "burstiness_metrics": text_result.get("burstiness_metrics", {}),
            }

        return findings

    def _collect_evidence(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Collect evidence from analysis results."""
        evidence = {
            "heatmaps": [],
            "spectrograms": [],
            "localization": {
                "timestamps": [],
                "spatial_regions": [],
            },
        }

        if video_result:
            heatmap_path = video_result.get("heatmap_path")
            if heatmap_path:
                evidence["heatmaps"].append(heatmap_path)

            for frame in video_result.get("per_frame_scores", []):
                if frame.get("heatmap_path"):
                    evidence["heatmaps"].append(frame["heatmap_path"])

                if frame.get("face_bbox") and frame.get("manipulation_score", 0) > 0.7:
                    evidence["localization"]["spatial_regions"].append({
                        "frame": frame.get("frame_number"),
                        "bbox": frame.get("face_bbox"),
                        "score": frame.get("manipulation_score"),
                    })

        if audio_result:
            for segment in audio_result.get("per_segment_scores", []):
                if segment.get("spectrogram_path"):
                    evidence["spectrograms"].append(segment["spectrogram_path"])

                if segment.get("manipulation_score", 0) > 0.7:
                    evidence["localization"]["timestamps"].append({
                        "start_time": segment.get("start_time"),
                        "end_time": segment.get("end_time"),
                        "score": segment.get("manipulation_score"),
                    })

        return evidence

    def _generate_recommendations(
        self,
        trust_result: Dict[str, Any],
        findings: Dict[str, Any],
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        verdict = trust_result.get("verdict", "UNCERTAIN")
        score = trust_result.get("overall_score", 50.0)

        if verdict in ["HIGHLY_MANIPULATED", "LIKELY_MANIPULATED"]:
            recommendations.append("Review the content with a forensic expert before making decisions.")
            recommendations.append("Check for associated metadata (EXIF, C2PA) that may provide provenance.")

            if findings.get("video", {}).get("analyzed"):
                if findings["video"].get("faces_detected", 0) == 0:
                    recommendations.append("No faces detected - consider image-based manipulation.")

            if findings.get("audio", {}).get("vocoder_detected"):
                recommendations.append(
                    f"Vocoder detected: {findings['audio']['vocoder_detected']}. "
                    "This may indicate synthetic speech generation."
                )

        elif verdict == "UNCERTAIN":
            recommendations.append("Consider manual review of the content.")
            recommendations.append("Request additional authentic samples for comparison if available.")

        else:
            recommendations.append("The content appears authentic based on current analysis.")
            recommendations.append("Maintain standard verification procedures.")

        return recommendations

    def _collect_raw_outputs(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        fusion_result: Optional[FusionResult] = None,
    ) -> Dict[str, Any]:
        """Collect raw model outputs for debugging."""
        raw = {}

        if fusion_result:
            raw["fusion"] = fusion_result.to_dict()

        if video_result:
            raw["video"] = {
                "per_frame": video_result.get("per_frame_scores", []),
                "branch_scores": video_result.get("branch_scores", {}),
            }

        if audio_result:
            raw["audio"] = {
                "per_segment": audio_result.get("per_segment_scores", []),
                "artifact_evidence": audio_result.get("artifact_evidence", {}),
            }

        if text_result:
            raw["text"] = {
                "per_sentence": text_result.get("per_sentence_scores", []),
                "perplexity_scores": text_result.get("perplexity_scores", {}),
            }

        return raw


def create_report_generator() -> ReportGenerator:
    """Factory function for ReportGenerator."""
    return ReportGenerator()
