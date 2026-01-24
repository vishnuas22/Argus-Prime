"""
Reasoning Generator - Chain-of-Thought Explainability.

Generates structured reasoning for deepfake detection verdicts
with evidence chains and human-readable explanations.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import get_config
from src.core.logging import logger


@dataclass
class ReasoningResult:
    """Result from reasoning generation."""
    verdict: str
    confidence: float
    what: str
    where: str
    why: List[str]
    reasoning_trace: List[Dict[str, Any]]
    evidence_chain: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "what": self.what,
            "where": self.where,
            "why": self.why,
            "reasoning_trace": self.reasoning_trace,
            "evidence_chain": self.evidence_chain,
        }


@dataclass
class EvidenceItem:
    """Individual evidence item."""
    source: str
    metric: str
    value: float
    threshold: float
    interpretation: str


class ReasoningGenerator:
    """
    Chain-of-Thought Reasoning Generator.

    Generates explainable verdicts with:
    - What: Type of manipulation detected
    - Where: Location in time/space
    - Why: Evidence chain with reasoning
    - How confident: Confidence level

    Attributes:
        confidence_thresholds: Thresholds for verdict classification
    """

    def __init__(
        self,
        highly_manipulated_threshold: float = 0.8,
        likely_manipulated_threshold: float = 0.6,
        likely_authentic_threshold: float = 0.4,
    ):
        """
        Initialize reasoning generator.

        Args:
            highly_manipulated_threshold: Threshold for HIGHLY_MANIPULATED
            likely_manipulated_threshold: Threshold for LIKELY_MANIPULATED
            likely_authentic_threshold: Threshold for LIKELY_AUTHENTIC
        """
        self.thresholds = {
            "highly_manipulated": highly_manipulated_threshold,
            "likely_manipulated": likely_manipulated_threshold,
            "likely_authentic": likely_authentic_threshold,
        }

        config = get_config()
        self.detection_threshold = config.analysis.confidence_threshold

        logger.info("ReasoningGenerator initialized")

    def _classify_verdict(self, score: float) -> str:
        """Classify verdict based on score."""
        if score >= self.thresholds["highly_manipulated"]:
            return "HIGHLY_MANIPULATED"
        elif score >= self.thresholds["likely_manipulated"]:
            return "LIKELY_MANIPULATED"
        elif score >= self.thresholds["likely_authentic"]:
            return "UNCERTAIN"
        elif score >= self.detection_threshold:
            return "LIKELY_AUTHENTIC"
        else:
            return "HIGHLY_AUTHENTIC"

    def _detect_manipulation_type(
        self,
        video_score: Optional[float],
        audio_score: Optional[float],
        text_score: Optional[float],
        branch_scores: Dict[str, float],
    ) -> str:
        """Detect type of manipulation."""
        scores = {}
        if video_score is not None:
            scores["video"] = video_score
        if audio_score is not None:
            scores["audio"] = audio_score
        if text_score is not None:
            scores["text"] = text_score

        if not scores:
            return "unknown manipulation"

        modality = max(scores, key=scores.get)
        score = scores[modality]

        if branch_scores:
            texture_score = branch_scores.get("texture", 0)
            spectral_score = branch_scores.get("spectral", 0)
            temporal_score = branch_scores.get("temporal", 0)

            if texture_score > 0.7:
                return "face swapping or identity manipulation"
            elif spectral_score > 0.7:
                return "GAN/Vocoder-based synthesis"
            elif temporal_score > 0.7:
                return "temporal inconsistency or frame tampering"

        if modality == "video":
            if score > 0.8:
                return "face swapping (e.g., DeepFaceLab, FaceSwap)"
            elif score > 0.6:
                return "partial face manipulation"
            else:
                return "subtle video tampering"
        elif modality == "audio":
            if score > 0.8:
                return "voice cloning (e.g., Vocoder synthesis)"
            elif score > 0.6:
                return "partial voice synthesis"
            else:
                return "audio manipulation"
        elif modality == "text":
            if score > 0.8:
                return "AI-generated text (LLM output)"
            elif score > 0.6:
                return "partial AI text generation"
            else:
                return "text tampering"

        return "multi-modal manipulation"

    def _localize_manipulation(
        self,
        video_result: Optional[Dict] = None,
        audio_result: Optional[Dict] = None,
    ) -> str:
        """Generate localization description."""
        location_info = []

        if video_result:
            frames = video_result.get("per_frame_scores", [])
            if frames:
                high_score_frames = [
                    f for f in frames
                    if f.get("manipulation_score", 0) > 0.7
                ]

                if high_score_frames:
                    frame_nums = [f["frame_number"] for f in high_score_frames]
                    if len(frame_nums) == 1:
                        location_info.append(f"Frame {frame_nums[0]}")
                    elif len(frame_nums) > 1:
                        location_info.append(
                            f"Frames {min(frame_nums)}-{max(frame_nums)}"
                        )

                face_bboxes = [
                    f.get("face_bbox")
                    for f in frames
                    if f.get("face_bbox") and f.get("manipulation_score", 0) > 0.7
                ]
                if face_bboxes:
                    location_info.append("facial region")

        if audio_result:
            segments = audio_result.get("per_segment_scores", [])
            if segments:
                high_score_segments = [
                    s for s in segments
                    if s.get("manipulation_score", 0) > 0.7
                ]
                if high_score_segments:
                    times = [
                        (s["start_time"] + s["end_time"]) / 2
                        for s in high_score_segments
                    ]
                    if len(times) == 1:
                        location_info.append(f"Time {times[0]:.1f}s")
                    else:
                        location_info.append(
                            f"Time {min(times):.1f}s - {max(times):.1f}s"
                        )

        if not location_info:
            return "content-wide"

        return ", ".join(location_info)

    def _generate_evidence_chain(
        self,
        video_result: Optional[Dict] = None,
        audio_result: Optional[Dict] = None,
        text_result: Optional[Dict] = None,
        fusion_result: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Generate evidence chain for reasoning."""
        evidence = []

        if video_result:
            branch_scores = video_result.get("branch_scores", {})
            for branch, score in branch_scores.items():
                evidence.append({
                    "modality": "video",
                    "branch": branch,
                    "score": score,
                    "threshold": 0.5,
                    "flag": score > 0.5,
                })

            if video_result.get("faces_detected", 0) == 0:
                evidence.append({
                    "modality": "video",
                    "observation": "No faces detected",
                    "interpretation": "May indicate full-frame manipulation",
                })

        if audio_result:
            vocoder = audio_result.get("vocoder_detected")
            if vocoder:
                evidence.append({
                    "modality": "audio",
                    "observation": f"Vocoder detected: {vocoder}",
                    "interpretation": "Indicates synthetic speech generation",
                })

            artifact_evidence = audio_result.get("artifact_evidence", {})
            for artifact, score in artifact_evidence.items():
                evidence.append({
                    "modality": "audio",
                    "artifact": artifact,
                    "score": score,
                    "flag": score > 0.5,
                })

        if text_result:
            perplexity_scores = text_result.get("perplexity_scores", {})
            for model, score in perplexity_scores.items():
                evidence.append({
                    "modality": "text",
                    "model": model,
                    "perplexity_score": score,
                    "flag": score > 0.5,
                })

            burstiness = text_result.get("burstiness_metrics", {})
            if burstiness:
                evidence.append({
                    "modality": "text",
                    "burstiness_score": burstiness.get("burstiness_score", 0),
                    "interpretation": "AI text often shows lower burstiness",
                })

        if fusion_result:
            weights = fusion_result.get("weights_used", {})
            for modality, weight in weights.items():
                evidence.append({
                    "modality": "fusion",
                    "component": modality,
                    "weight": weight,
                })

        return evidence

    def generate_reasoning(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        fusion_result: Optional[Dict[str, Any]] = None,
    ) -> ReasoningResult:
        """
        Generate complete reasoning for a verdict.

        Args:
            video_result: Video analysis result
            audio_result: Audio analysis result
            text_result: Text analysis result
            fusion_result: Fusion result

        Returns:
            ReasoningResult with full explanation
        """
        fusion_score = 0.5
        confidence = 0.5

        if fusion_result:
            fusion_score = fusion_result.get("fused_score", 0.5)
            confidence = fusion_result.get("confidence", 0.5)
        else:
            scores = []
            if video_result:
                scores.append(video_result.get("overall_score", 0.5))
            if audio_result:
                scores.append(audio_result.get("overall_score", 0.5))
            if text_result:
                scores.append(text_result.get("overall_score", 0.5))

            if scores:
                fusion_score = sum(scores) / len(scores)

        verdict = self._classify_verdict(fusion_score)

        branch_scores = {}
        if video_result:
            branch_scores = video_result.get("branch_scores", {})

        what = self._detect_manipulation_type(
            video_result.get("overall_score") if video_result else None,
            audio_result.get("overall_score") if audio_result else None,
            text_result.get("overall_score") if text_result else None,
            branch_scores,
        )

        where = self._localize_manipulation(video_result, audio_result)

        why = []
        if video_result:
            branch_scores = video_result.get("branch_scores", {})
            for branch, score in branch_scores.items():
                if score > 0.5:
                    why.append(
                        f"{branch} anomaly: {score:.2f} (threshold: 0.5)"
                    )

        if audio_result:
            vocoder = audio_result.get("vocoder_detected")
            if vocoder:
                why.append(f"Vocoder artifact: {vocoder}")

        if text_result:
            perplexity = text_result.get("perplexity_scores", {})
            for model, score in perplexity.items():
                if score > 0.5:
                    why.append(f"Perplexity ({model}): {score:.2f}")

        if fusion_result:
            consistency = fusion_result.get("consistency_score", 0.5)
            if consistency < 0.5:
                why.append(f"Cross-modal inconsistency: {consistency:.2f}")

        if not why:
            why.append(f"Overall manipulation score: {fusion_score:.2f}")

        reasoning_trace = [
            {
                "step": 1,
                "observation": "Multi-modal analysis completed",
                "input": {
                    "video_score": video_result.get("overall_score") if video_result else None,
                    "audio_score": audio_result.get("overall_score") if audio_result else None,
                    "text_score": text_result.get("overall_score") if text_result else None,
                },
            },
            {
                "step": 2,
                "thought": f"Fused score: {fusion_score:.2f}",
                "action": "Classify verdict",
            },
            {
                "step": 3,
                "thought": f"Verdict: {verdict}",
                "confidence": confidence,
            },
        ]

        evidence_chain = self._generate_evidence_chain(
            video_result, audio_result, text_result, fusion_result
        )

        return ReasoningResult(
            verdict=verdict,
            confidence=confidence,
            what=what,
            where=where,
            why=why,
            reasoning_trace=reasoning_trace,
            evidence_chain=evidence_chain,
        )

    def generate_verdict_text(self, reasoning: ReasoningResult) -> str:
        """Generate human-readable verdict text."""
        confidence_pct = int(reasoning.confidence * 100)

        text = [
            f"Based on our multi-modal analysis, this content has been "
            f"{reasoning.verdict.replace('_', ' ').lower()} with a confidence of {confidence_pct}%.",
        ]

        if reasoning.what != "unknown manipulation":
            text.append(f"Detected manipulation type: {reasoning.what}.")

        text.append(f"Location: {reasoning.where}.")

        if reasoning.why:
            text.append("Key evidence:")
            for i, evidence in enumerate(reasoning.why[:3], 1):
                text.append(f"  {i}. {evidence}")

        return " ".join(text)


def create_reasoning_generator() -> ReasoningGenerator:
    """Factory function for ReasoningGenerator."""
    return ReasoningGenerator()
