"""Fusion module - Multi-modal score fusion and trust scoring."""

from .fusion_engine import (
    FusionEngine,
    FusionResult,
    TrustScorer,
    ConsistencyStatus,
    create_fusion_engine,
)
from .trust_scorer import (
    TrustScore,
    TrustScorer as SimpleTrustScorer,
    create_trust_scorer,
)

__all__ = [
    "FusionEngine",
    "FusionResult",
    "TrustScorer",
    "ConsistencyStatus",
    "create_fusion_engine",
    "TrustScore",
    "SimpleTrustScorer",
    "create_trust_scorer",
]
