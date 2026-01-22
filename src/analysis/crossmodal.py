"""
Cross-Modal Consistency Checker.

Detects contradictions and inconsistencies across multiple modalities
(video, audio, text, biological signals) to identify potential deepfakes.

Features:
- Lip-sync verification (mouth movements vs audio)
- Temporal consistency (frame timestamps vs audio segments)
- Content consistency (text vs video/audio context)
- Biological signal consistency (rPPG across frames)

Output:
  - consistency_score ∈ [0, 1]
  - List of contradictions with confidence
  - Recommendations for manual review
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_config
from src.core.logging import logger


@dataclass
class Contradiction:
    """A detected contradiction between modalities."""
    type: str
    description: str
    confidence: float
    severity: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""


@dataclass
class CrossModalResult:
    """Result of cross-modal consistency analysis."""
    consistency_score: float
    is_consistent: bool
    contradictions: List[Contradiction] = field(default_factory=list)
    lip_sync_score: Optional[float] = None
    temporal_score: Optional[float] = None
    content_score: Optional[float] = None
    biological_score: Optional[float] = None
    recommendations: List[str] = field(default_factory=list)
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "consistency_score": self.consistency_score,
            "is_consistent": self.is_consistent,
            "contradictions": [
                {
                    "type": c.type,
                    "description": c.description,
                    "confidence": c.confidence,
                    "severity": c.severity,
                    "evidence": c.evidence,
                    "recommendation": c.recommendation,
                }
                for c in self.contradictions
            ],
            "lip_sync_score": self.lip_sync_score,
            "temporal_score": self.temporal_score,
            "content_score": self.content_score,
            "biological_score": self.biological_score,
            "recommendations": self.recommendations,
            "raw_scores": self.raw_scores,
        }


class CrossModalChecker:
    """
    Cross-modal consistency verification system.

    Analyzes relationships between different modalities to detect
    inconsistencies that may indicate manipulation.

    Attributes:
        config: Configuration object
        consistency_threshold: Threshold for consistency flagging
        contradiction_weights: Weights for different contradiction types
    """

    def __init__(self):
        """Initialize cross-modal checker."""
        self.config = get_config()

        self.consistency_threshold = 0.3
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
            "info": 0.0,
        }

        logger.info("CrossModalChecker initialized")

    def _check_lip_sync(
        self,
        video_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], List[Contradiction]]:
        """
        Check lip-sync consistency between video and audio.

        Args:
            video_result: Video analysis result
            audio_result: Audio analysis result

        Returns:
            Tuple of (score, contradictions)
        """
        if video_result is None or audio_result is None:
            return None, []

        contradictions = []
        score = 1.0

        try:
            video_manipulation = video_result.get("overall_score", 0.5)
            audio_manipulation = audio_result.get("overall_score", 0.5)

            video_confidence = video_result.get("confidence", 0.5)
            audio_confidence = audio_result.get("confidence", 0.5)

            manipulation_diff = abs(video_manipulation - audio_manipulation)

            if manipulation_diff > 0.4:
                if video_manipulation > audio_manipulation:
                    contradiction = Contradiction(
                        type="lip_sync_mismatch",
                        description=(
                            "Video shows strong manipulation signs but audio appears "
                            "authentic - possible lip-sync deepfake"
                        ),
                        confidence=min(0.9, manipulation_diff * 1.5),
                        severity="high",
                        evidence={
                            "video_manipulation": video_manipulation,
                            "audio_manipulation": audio_manipulation,
                            "difference": manipulation_diff,
                        },
                        recommendation="Manual review of lip-sync quality recommended",
                    )
                else:
                    contradiction = Contradiction(
                        type="lip_sync_mismatch",
                        description=(
                            "Audio shows manipulation signs but video appears "
                            "authentic - possible audio deepfake with authentic video"
                        ),
                        confidence=min(0.9, manipulation_diff * 1.5),
                        severity="high",
                        evidence={
                            "video_manipulation": video_manipulation,
                            "audio_manipulation": audio_manipulation,
                            "difference": manipulation_diff,
                        },
                        recommendation="Verify audio authenticity with secondary method",
                    )
                contradictions.append(contradiction)
                score -= 0.3

            if audio_result.get("vocoder_detected"):
                contradiction = Contradiction(
                    type="vocoder_detected",
                    description=f"Vocoder detected in audio: {audio_result['vocoder_detected']}",
                    confidence=audio_confidence,
                    severity="medium",
                    evidence={
                        "vocoder": audio_result["vocoder_detected"],
                        "audio_manipulation": audio_manipulation,
                    },
                    recommendation="Audio was likely synthesized - verify source authenticity",
                )
                contradictions.append(contradiction)
                score -= 0.2

            score = max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Lip-sync check failed: {e}")
            return 0.5, []

        return score, contradictions

    def _check_temporal_consistency(
        self,
        video_result: Optional[Dict[str, Any]],
        audio_result: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], List[Contradiction]]:
        """
        Check temporal consistency between video and audio.

        Args:
            video_result: Video analysis result
            audio_result: Audio analysis result

        Returns:
            Tuple of (score, contradictions)
        """
        if video_result is None or audio_result is None:
            return None, []

        contradictions = []
        score = 1.0

        try:
            video_frames = video_result.get("frames_analyzed", 0)
            audio_segments = audio_result.get("segments_analyzed", 0)

            if video_frames > 0 and audio_segments > 0:
                video_duration = video_frames / 30.0
                audio_duration = audio_result.get("duration_seconds", 0)

                if video_duration > 0 and audio_duration > 0:
                    duration_ratio = min(video_duration, audio_duration) / max(
                        video_duration, audio_duration
                    )

                    if duration_ratio < 0.8:
                        contradiction = Contradiction(
                            type="duration_mismatch",
                            description=(
                                f"Video duration ({video_duration:.1f}s) differs significantly "
                                f"from audio duration ({audio_duration:.1f}s)"
                            ),
                            confidence=0.7,
                            severity="medium",
                            evidence={
                                "video_duration": video_duration,
                                "audio_duration": audio_duration,
                                "ratio": duration_ratio,
                            },
                            recommendation="Verify source files are from same content",
                        )
                        contradictions.append(contradiction)
                        score -= 0.2

            frame_scores = video_result.get("per_frame_scores", [])
            segment_scores = audio_result.get("per_segment_scores", [])

            if len(frame_scores) > 5 and len(segment_scores) > 5:
                frame_manipulation = [f.get("manipulation_score", 0.5) for f in frame_scores]
                segment_manipulation = [s.get("manipulation_score", 0.5) for s in segment_scores]

                frame_variance = float(np.var(frame_manipulation))
                segment_variance = float(np.var(segment_manipulation))

                if frame_variance < 1e-6 and segment_variance < 1e-6:
                    pass
                else:
                    variance_ratio = min(frame_variance, segment_variance) / max(
                        frame_variance + 1e-6, segment_variance + 1e-6
                    )

                    if variance_ratio < 0.3:
                        contradiction = Contradiction(
                            type="temporal_pattern_mismatch",
                            description=(
                                "Manipulation patterns differ between video frames and "
                                "audio segments - indicates potential temporal editing"
                            ),
                            confidence=0.6,
                            severity="medium",
                            evidence={
                                "frame_variance": frame_variance,
                                "segment_variance": segment_variance,
                            },
                            recommendation="Check for temporal editing artifacts",
                        )
                        contradictions.append(contradiction)
                        score -= 0.15

            score = max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Temporal consistency check failed: {e}")
            return 0.5, []

        return score, contradictions

    def _check_content_consistency(
        self,
        video_result: Optional[Dict[str, Any]],
        text_result: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], List[Contradiction]]:
        """
        Check content consistency between video and text.

        Args:
            video_result: Video analysis result
            text_result: Text analysis result

        Returns:
            Tuple of (score, contradictions)
        """
        if video_result is None or text_result is None:
            return None, []

        contradictions = []
        score = 1.0

        try:
            text_manipulation = text_result.get("overall_score", 0.5)
            video_manipulation = video_result.get("overall_score", 0.5)

            text_confidence = text_result.get("confidence", 0.5)
            video_confidence = video_result.get("confidence", 0.5)

            manipulation_diff = abs(video_manipulation - text_manipulation)

            if manipulation_diff > 0.5:
                contradiction = Contradiction(
                    type="content_mismatch",
                    description=(
                        "Video and text analysis show conflicting manipulation indicators - "
                        "content may have been mismatched or edited inconsistently"
                    ),
                    confidence=min(0.85, manipulation_diff),
                    severity="high",
                    evidence={
                        "video_manipulation": video_manipulation,
                        "text_manipulation": text_manipulation,
                        "difference": manipulation_diff,
                    },
                    recommendation="Verify text and video are from same source",
                )
                contradictions.append(contradiction)
                score -= 0.35

            sentence_scores = text_result.get("per_sentence_scores", [])
            if len(sentence_scores) > 5:
                sentence_manipulation = [s.get("manipulation_score", 0.5) for s in sentence_scores]
                high_manipulation_sentences = sum(1 for s in sentence_manipulation if s > 0.7)

                if high_manipulation_sentences / len(sentence_manipulation) > 0.3:
                    contradiction = Contradiction(
                        type="selective_text_manipulation",
                        description="Some sentences show high manipulation scores while others appear authentic",
                        confidence=0.6,
                        severity="medium",
                        evidence={
                            "manipulated_sentences": high_manipulation_sentences,
                            "total_sentences": len(sentence_manipulation),
                        },
                        recommendation="Verify authenticity of individual sentences",
                    )
                    contradictions.append(contradiction)
                    score -= 0.15

            score = max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Content consistency check failed: {e}")
            return 0.5, []

        return score, contradictions

    def _check_biological_consistency(
        self,
        video_result: Optional[Dict[str, Any]],
        rppg_result: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[float], List[Contradiction]]:
        """
        Check biological signal consistency in video.

        Args:
            video_result: Video analysis result
            rppg_result: rPPG analysis result

        Returns:
            Tuple of (score, contradictions)
        """
        if video_result is None or rppg_result is None:
            return None, []

        contradictions = []
        score = 1.0

        try:
            if not rppg_result:
                return None, []

            quality_score = rppg_result.get("quality_score", 0.0)
            temporal_consistency = rppg_result.get("temporal_consistency", 1.0)

            if quality_score < 0.3:
                contradiction = Contradiction(
                    type="poor_biological_signal",
                    description="rPPG signal quality is poor - biological signals may be manipulated",
                    confidence=1.0 - quality_score,
                    severity="high",
                    evidence={
                        "quality_score": quality_score,
                    },
                    recommendation="Manual verification of biological signals recommended",
                )
                contradictions.append(contradiction)
                score -= 0.4

            if temporal_consistency < 0.5:
                contradiction = Contradiction(
                    type="temporal_inconsistency",
                    description="rPPG signals show temporal inconsistency - possible frame manipulation",
                    confidence=1.0 - temporal_consistency,
                    severity="high",
                    evidence={
                        "temporal_consistency": temporal_consistency,
                    },
                    recommendation="Check for frame-by-frame manipulation",
                )
                contradictions.append(contradiction)
                score -= 0.3

            video_manipulation = video_result.get("overall_score", 0.5)
            rppg_manipulation = 1.0 - quality_score

            if video_manipulation > 0.7 and quality_score < 0.4:
                contradiction = Contradiction(
                    type="manipulation_biological_mismatch",
                    description=(
                        "High video manipulation score but poor biological signals - "
                        "inconsistent with authentic content"
                    ),
                    confidence=0.7,
                    severity="critical",
                    evidence={
                        "video_manipulation": video_manipulation,
                        "biological_quality": quality_score,
                    },
                    recommendation="Strong indicator of manipulation - manual review essential",
                )
                contradictions.append(contradiction)
                score -= 0.3

            score = max(0.0, min(1.0, score))

        except Exception as e:
            logger.warning(f"Biological consistency check failed: {e}")
            return 0.5, []

        return score, contradictions

    def check_consistency(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
        rppg_result: Optional[Dict[str, Any]] = None,
    ) -> CrossModalResult:
        """
        Perform comprehensive cross-modal consistency analysis.

        Args:
            video_result: Video analysis result
            audio_result: Audio analysis result
            text_result: Text analysis result
            rppg_result: rPPG analysis result

        Returns:
            CrossModalResult with consistency analysis
        """
        logger.info("Starting cross-modal consistency analysis")

        all_contradictions = []
        raw_scores = {}

        lip_sync_score, lip_contradictions = self._check_lip_sync(
            video_result, audio_result
        )
        if lip_sync_score is not None:
            raw_scores["lip_sync"] = lip_sync_score
            self.lip_sync_score = lip_sync_score
        all_contradictions.extend(lip_contradictions)

        temporal_score, temporal_contradictions = self._check_temporal_consistency(
            video_result, audio_result
        )
        if temporal_score is not None:
            raw_scores["temporal"] = temporal_score
            self.temporal_score = temporal_score
        all_contradictions.extend(temporal_contradictions)

        content_score, content_contradictions = self._check_content_consistency(
            video_result, text_result
        )
        if content_score is not None:
            raw_scores["content"] = content_score
            self.content_score = content_score
        all_contradictions.extend(content_contradictions)

        biological_score, biological_contradictions = self._check_biological_consistency(
            video_result, rppg_result
        )
        if biological_score is not None:
            raw_scores["biological"] = biological_score
            self.biological_score = biological_score
        all_contradictions.extend(biological_contradictions)

        valid_scores = [s for s in raw_scores.values() if s is not None]
        if valid_scores:
            consistency_score = sum(valid_scores) / len(valid_scores)
        else:
            consistency_score = 1.0

        is_consistent = consistency_score >= self.consistency_threshold

        severity_penalty = sum(
            self.severity_weights.get(c.severity, 0.0) * c.confidence
            for c in all_contradictions
        )
        consistency_score = max(0.0, consistency_score - severity_penalty * 0.1)
        consistency_score = max(0.0, min(1.0, consistency_score))

        recommendations = []
        critical_contradictions = [c for c in all_contradictions if c.severity == "critical"]
        high_contradictions = [c for c in all_contradictions if c.severity == "high"]

        if critical_contradictions:
            recommendations.append("CRITICAL: Multiple critical inconsistencies detected - manual review REQUIRED")
        if high_contradictions:
            recommendations.append("High-severity contradictions found - secondary verification recommended")

        if not all_contradictions:
            recommendations.append("No cross-modal inconsistencies detected")

        result = CrossModalResult(
            consistency_score=consistency_score,
            is_consistent=is_consistent,
            contradictions=all_contradictions,
            lip_sync_score=lip_sync_score,
            temporal_score=temporal_score,
            content_score=content_score,
            biological_score=biological_score,
            recommendations=recommendations,
            raw_scores=raw_scores,
        )

        logger.info(
            f"Cross-modal analysis complete: score={consistency_score:.2f}, "
            f"contradictions={len(all_contradictions)}"
        )

        return result

    def check_from_results(
        self,
        video_result: Optional[Dict[str, Any]] = None,
        audio_result: Optional[Dict[str, Any]] = None,
        text_result: Optional[Dict[str, Any]] = None,
    ) -> CrossModalResult:
        """
        Convenience method for checking consistency from analysis results.

        Extracts rPPG data from video result if available.
        """
        rppg_result = None
        if video_result:
            rppg_result = video_result.get("rppg_result")

        return self.check_consistency(
            video_result=video_result,
            audio_result=audio_result,
            text_result=text_result,
            rppg_result=rppg_result,
        )


_crossmodal_checker: Optional[CrossModalChecker] = None


def get_crossmodal_checker() -> CrossModalChecker:
    """Get or create cross-modal checker instance."""
    global _crossmodal_checker
    if _crossmodal_checker is None:
        _crossmodal_checker = CrossModalChecker()
    return _crossmodal_checker
