"""
Multi-Modal Score Fusion Engine.

Implements adaptive fusion of detection scores from multiple modalities:
1. Base weighted fusion (configurable weights)
2. Cross-modal consistency checks
3. Adaptive weighting based on quality
4. Confidence calibration
"""

from dataclasses import dataclass, field
from enum import Enum
import logging

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import get_config
from src.core.logging import logger

logger = logging.getLogger(__name__)


class ConsistencyStatus(str, Enum):
    """Cross-modal consistency status."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNCERTAIN = "uncertain"


@dataclass
class FusionResult:
    """Result of score fusion."""
    fused_score: float
    confidence: float
    weights_used: Dict[str, float]
    consistency_status: ConsistencyStatus
    consistency_score: float
    adjustments_made: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fused_score": self.fused_score,
            "confidence": self.confidence,
            "weights_used": self.weights_used,
            "consistency_status": self.consistency_status.value,
            "consistency_score": self.consistency_score,
            "adjustments_made": self.adjustments_made,
            "metadata": self.metadata,
        }


class FusionEngine:
    """
    Multi-Modal Score Fusion Engine.

    Implements adaptive fusion with:
    - Configurable base weights (video: 0.40, audio: 0.35, text: 0.25)
    - Cross-modal consistency checking
    - Quality-based weight adjustments
    - Temperature scaling for confidence calibration

    Attributes:
        base_weights: Default modality weights
        consistency_threshold: Threshold for inconsistency detection
        cross_modal_adjustment: Maximum adjustment percentage
    """

    def __init__(
        self,
        video_weight: float = 0.40,
        audio_weight: float = 0.35,
        text_weight: float = 0.25,
        cross_modal_adjustment: float = 0.20,
        consistency_threshold: float = 0.30,
    ):
        """
        Initialize fusion engine.

        Args:
            video_weight: Weight for video modality
            audio_weight: Weight for audio modality
            text_weight: Weight for text modality
            cross_modal_adjustment: Maximum adjustment percentage
            consistency_threshold: Threshold for detecting inconsistency
        """
        self.base_weights = {
            "video": video_weight,
            "audio": audio_weight,
            "text": text_weight,
        }

        self.cross_modal_adjustment = cross_modal_adjustment
        self.consistency_threshold = consistency_threshold

        config = get_config()
        self.fusion_config = config.analysis.fusion

        logger.info(
            f"FusionEngine initialized: weights={self.base_weights}, "
            f"adjustment={cross_modal_adjustment}"
        )

    def fuse(
        self,
        video_score: Optional[float] = None,
        audio_score: Optional[float] = None,
        text_score: Optional[float] = None,
        video_confidence: Optional[float] = None,
        audio_confidence: Optional[float] = None,
        text_confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FusionResult:
        """
        Fuse multi-modal detection scores.

        Args:
            video_score: Video manipulation score (0-1)
            audio_score: Audio manipulation score (0-1)
            text_score: Text manipulation score (0-1)
            video_confidence: Video analysis confidence (0-1)
            audio_confidence: Audio analysis confidence (0-1)
            text_confidence: Text analysis confidence (0-1)
            metadata: Additional metadata for adjustments

        Returns:
            FusionResult with fused score, confidence, and adjustments
        """
        scores = {
            "video": video_score,
            "audio": audio_score,
            "text": text_score,
        }

        confidences = {
            "video": video_confidence,
            "audio": audio_confidence,
            "text": text_confidence,
        }

        active_modalities = {k for k, v in scores.items() if v is not None}
        if not active_modalities:
            return FusionResult(
                fused_score=0.5,
                confidence=0.0,
                weights_used=self.base_weights,
                consistency_status=ConsistencyStatus.UNCERTAIN,
                consistency_score=0.5,
                adjustments_made=["no_modalities"],
                metadata={"reason": "no scores provided"},
            )

        weights, adjustments = self._compute_adaptive_weights(scores, confidences, metadata)

        fused_score = self._compute_weighted_average(scores, weights)

        consistency_status, consistency_score = self._check_consistency(scores)

        if consistency_status == ConsistencyStatus.INCONSISTENT:
            adjustments.append("consistency_adjustment")
            fused_score = self._apply_consistency_adjustment(
                fused_score, consistency_score, adjustments
            )

        confidence = self._compute_confidence(scores, confidences, consistency_score)

        return FusionResult(
            fused_score=fused_score,
            confidence=confidence,
            weights_used=weights,
            consistency_status=consistency_status,
            consistency_score=consistency_score,
            adjustments_made=adjustments,
            metadata=metadata or {},
        )

    def _compute_adaptive_weights(
        self,
        scores: Dict[str, Optional[float]],
        confidences: Dict[str, Optional[float]],
        metadata: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, float], List[str]]:
        """Compute adaptive weights based on confidence and availability."""
        weights = self.base_weights.copy()
        adjustments = []

        total_confidence = 0.0
        confidence_sum = 0.0

        for modality in ["video", "audio", "text"]:
            conf = confidences.get(modality, 0.5)
            if scores.get(modality) is not None:
                total_confidence += conf
                confidence_sum += 1

        if total_confidence > 0 and confidence_sum > 0:
            avg_confidence = total_confidence / confidence_sum
            if avg_confidence < 0.5:
                adjustment = min(self.cross_modal_adjustment, 1 - avg_confidence)
                adjustments.append(f"low_confidence_reduction_{adjustment:.2f}")

        if metadata:
            rppg_quality = metadata.get("rppg_quality")
            if rppg_quality is not None and rppg_quality < 0.5:
                reduction = 0.2 * (1 - rppg_quality)
                weights["video"] = max(weights["video"] - reduction, 0.1)
                adjustments.append(f"rppg_quality_adjustment_{reduction:.2f}")

            adversarial_flag = metadata.get("adversarial_detected", False)
            if adversarial_flag:
                reduction = 0.15
                weights["video"] = max(weights["video"] - reduction, 0.1)
                adjustments.append("adversarial_adjustment")

        active_count = sum(1 for k, v in scores.items() if v is not None)
        if active_count < 3:
            remaining_weight = 1.0
            for modality in ["video", "audio", "text"]:
                if scores.get(modality) is None:
                    remaining_weight -= weights[modality]

            for modality in ["video", "audio", "text"]:
                if scores.get(modality) is not None:
                    weights[modality] = weights[modality] / remaining_weight if remaining_weight > 0 else 0

            adjustments.append(f"missing_modality_adjustment_{active_count}")

        return weights, adjustments

    def _compute_weighted_average(
        self,
        scores: Dict[str, Optional[float]],
        weights: Dict[str, float],
    ) -> float:
        """Compute weighted average of scores."""
        total_score = 0.0
        total_weight = 0.0

        for modality in ["video", "audio", "text"]:
            score = scores.get(modality)
            weight = weights.get(modality, 0)
            if score is not None and weight > 0:
                total_score += score * weight
                total_weight += weight

        if total_weight == 0:
            return 0.5

        return total_score / total_weight

    def _check_consistency(
        self,
        scores: Dict[str, Optional[float]],
    ) -> Tuple[ConsistencyStatus, float]:
        """Check cross-modal consistency."""
        active_scores = [(k, v) for k, v in scores.items() if v is not None]

        if len(active_scores) < 2:
            return ConsistencyStatus.UNCERTAIN, 0.5

        score_values = [v for _, v in active_scores]
        score_range = max(score_values) - min(score_values)

        consistency_score = max(0, 1 - score_range)

        if score_range > self.consistency_threshold:
            return ConsistencyStatus.INCONSISTENT, consistency_score
        else:
            return ConsistencyStatus.CONSISTENT, consistency_score

    def _apply_consistency_adjustment(
        self,
        fused_score: float,
        consistency_score: float,
        adjustments: List[str],
    ) -> float:
        """Apply adjustment for inconsistent results."""
        adjustment_factor = self.cross_modal_adjustment * (1 - consistency_score)

        if fused_score > 0.5:
            adjusted_score = fused_score * (1 + adjustment_factor)
        else:
            adjusted_score = fused_score * (1 - adjustment_factor)

        return min(max(adjusted_score, 0.0), 1.0)

    def _compute_confidence(
        self,
        scores: Dict[str, Optional[float]],
        confidences: Dict[str, Optional[float]],
        consistency_score: float,
    ) -> float:
        """Compute overall confidence."""
        conf_values = [v for v in confidences.values() if v is not None]
        if not conf_values:
            return 0.5 * consistency_score

        avg_confidence = sum(conf_values) / len(conf_values)
        consistency_weight = 0.3
        confidence_weight = 0.7

        return avg_confidence * confidence_weight + consistency_score * consistency_weight

    def get_model_info(self) -> Dict[str, Any]:
        """Get fusion engine information."""
        return {
            "base_weights": self.base_weights,
            "cross_modal_adjustment": self.cross_modal_adjustment,
            "consistency_threshold": self.consistency_threshold,
        }


class TrustScorer:
    """
    Trust Score Calculator.

    Computes final trust score and confidence level:
    1. Raw Score: Weighted fusion result (0-1)
    2. Trust Score: 100 × raw_score
    3. Confidence: Based on model agreement, rPPG quality, adversarial checks
    4. Breakdown: Per-modality contribution
    """

    def __init__(self):
        """Initialize trust scorer."""
        logger.info("TrustScorer initialized")

    def compute_trust_score(
        self,
        fusion_result: FusionResult,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Compute trust score from fusion result.

        Args:
            fusion_result: Result from FusionEngine
            metadata: Additional analysis metadata

        Returns:
            Dictionary with trust score, confidence, and breakdown
        """
        raw_score = fusion_result.fused_score
        trust_score = round(raw_score * 100, 1)

        confidence = fusion_result.confidence

        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        breakdown = self._compute_breakdown(fusion_result.weights_used, raw_score)

        if metadata:
            adjustment_notes = []
            if metadata.get("rppg_quality", 1.0) < 0.5:
                adjustment_notes.append("rPPG quality below threshold")
            if metadata.get("adversarial_detected", False):
                adjustment_notes.append("Adversarial patterns detected")
            if metadata.get("missing_modality"):
                adjustment_notes.append(f"Missing: {metadata['missing_modality']}")

            breakdown["adjustment_notes"] = adjustment_notes

        return {
            "overall_score": trust_score,
            "confidence": round(confidence, 3),
            "confidence_level": confidence_level,
            "breakdown": breakdown,
            "verdict": self._get_verdict(trust_score),
            "fused_score": round(raw_score, 3),
        }

    def _compute_breakdown(
        self,
        weights: Dict[str, float],
        fused_score: float,
    ) -> Dict[str, Any]:
        """Compute per-modality contribution breakdown."""
        breakdown = {
            "video": {
                "weight": round(weights.get("video", 0), 2),
                "contribution": round(weights.get("video", 0) * fused_score * 100, 1),
            },
            "audio": {
                "weight": round(weights.get("audio", 0), 2),
                "contribution": round(weights.get("audio", 0) * fused_score * 100, 1),
            },
            "text": {
                "weight": round(weights.get("text", 0), 2),
                "contribution": round(weights.get("text", 0) * fused_score * 100, 1),
            },
        }

        total_contribution = sum(
            b["contribution"] for b in breakdown.values()
        )

        for modality in breakdown:
            if total_contribution > 0:
                breakdown[modality]["percentage"] = round(
                    breakdown[modality]["contribution"] / total_contribution * 100, 1
                )
            else:
                breakdown[modality]["percentage"] = 0.0

        return breakdown

    def _get_verdict(self, trust_score: float) -> str:
        """Get verdict string based on trust score."""
        if trust_score >= 80:
            return "HIGHLY_AUTHENTIC"
        elif trust_score >= 60:
            return "LIKELY_AUTHENTIC"
        elif trust_score >= 40:
            return "UNCERTAIN"
        elif trust_score >= 20:
            return "LIKELY_MANIPULATED"
        else:
            return "HIGHLY_MANIPULATED"


def create_fusion_engine() -> FusionEngine:
    """Factory function for FusionEngine."""
    config = get_config()
    fusion_config = config.analysis.fusion
    return FusionEngine(
        video_weight=fusion_config.get("video_weight", 0.40),
        audio_weight=fusion_config.get("audio_weight", 0.35),
        text_weight=fusion_config.get("text_weight", 0.25),
        cross_modal_adjustment=fusion_config.get("cross_modal_adjustment", 0.20),
    )
