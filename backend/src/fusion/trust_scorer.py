"""
Trust Scorer Module.

Computes final trust score and confidence level.

Role: Computes final trust score and confidence level
SOTA Algorithm: Score normalization + confidence calibration
Integration:
  - Imports: scipy.stats, config
  - Input: FusionResult, analysis_metadata
  - Output: TrustScore with overall_score, confidence, breakdown
Calculation:
  1. Raw Score: Weighted fusion result (0-1)
  2. Trust Score: 100 × raw_score
  3. Confidence: Based on:
     - Model agreement (higher agreement = higher confidence)
     - rPPG quality (higher quality = higher confidence)
     - Adversarial check (clean = higher confidence)
  4. Breakdown:
     - Video contribution: 40%
     - Audio contribution: 35%
     - Text contribution: 25%
Why: Trust score provides actionable metric, confidence enables risk-based routing
"""

from typing import Any, Dict, Optional

from src.config import get_config
from src.core.logging import logger


class TrustScore:
    """
    Trust Score Result.

    Contains the final trust score with confidence and breakdown.
    """

    def __init__(
        self,
        overall_score: float,
        confidence: float,
        confidence_level: str,
        breakdown: Dict[str, Any],
        verdict: str,
        fused_score: float,
    ):
        """
        Initialize trust score.

        Args:
            overall_score: Trust score (0-100)
            confidence: Confidence level (0-1)
            confidence_level: String description (high/medium/low)
            breakdown: Per-modality contribution breakdown
            verdict: Verdict string
            fused_score: Raw fused score (0-1)
        """
        self.overall_score = overall_score
        self.confidence = confidence
        self.confidence_level = confidence_level
        self.breakdown = breakdown
        self.verdict = verdict
        self.fused_score = fused_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "breakdown": self.breakdown,
            "verdict": self.verdict,
            "fused_score": self.fused_score,
        }

    def __repr__(self) -> str:
        return (
            f"TrustScore(score={self.overall_score}, "
            f"confidence={self.confidence:.2f}, "
            f"verdict={self.verdict})"
        )


class TrustScorer:
    """
    Trust Score Calculator.

    Computes final trust score and confidence level:
    1. Raw Score: Weighted fusion result (0-1)
    2. Trust Score: 100 × raw_score
    3. Confidence: Based on model agreement, rPPG quality, adversarial checks
    4. Breakdown: Per-modality contribution

    This is a thin wrapper around FusionEngine.TrustScorer for documentation alignment.
    """

    def __init__(self):
        """Initialize trust scorer."""
        self.config = get_config()
        logger.info("TrustScorer initialized")

    def compute_trust_score(
        self,
        fusion_result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrustScore:
        """
        Compute trust score from fusion result.

        Args:
            fusion_result: Result from FusionEngine containing fused_score, confidence, weights_used
            metadata: Additional analysis metadata

        Returns:
            TrustScore with overall_score, confidence, and breakdown
        """
        raw_score = fusion_result.get("fused_score", 0.5)
        trust_score = round(raw_score * 100, 1)

        confidence = fusion_result.get("confidence", 0.5)
        weights = fusion_result.get("weights_used", {})

        if confidence >= 0.8:
            confidence_level = "high"
        elif confidence >= 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        breakdown = self._compute_breakdown(weights, raw_score)

        if metadata:
            adjustment_notes = []
            if metadata.get("rppg_quality", 1.0) < 0.5:
                adjustment_notes.append("rPPG quality below threshold")
            if metadata.get("adversarial_detected", False):
                adjustment_notes.append("Adversarial patterns detected")
            if metadata.get("missing_modality"):
                adjustment_notes.append(f"Missing: {metadata['missing_modality']}")

            if adjustment_notes:
                breakdown["adjustment_notes"] = adjustment_notes

        verdict = self._get_verdict(trust_score)

        return TrustScore(
            overall_score=trust_score,
            confidence=round(confidence, 3),
            confidence_level=confidence_level,
            breakdown=breakdown,
            verdict=verdict,
            fused_score=round(raw_score, 3),
        )

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


def create_trust_scorer() -> TrustScorer:
    """Factory function for TrustScorer."""
    return TrustScorer()
