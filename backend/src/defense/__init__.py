"""Defense module - Adversarial defense and input sanitization."""

from .trim_analyzer import (
    TRIMAnalyzer,
    TRIMResult,
    create_trim_analyzer,
)
from .adversarial_defender import (
    AdversarialDefender,
    AdversarialCheckResult,
    create_adversarial_defender,
)
from .purifier import (
    DGGANPurifier,
    PurifyResult,
    create_dggan_purifier,
)

__all__ = [
    "TRIMAnalyzer",
    "TRIMResult",
    "create_trim_analyzer",
    "AdversarialDefender",
    "AdversarialCheckResult",
    "create_adversarial_defender",
    "DGGANPurifier",
    "PurifyResult",
    "create_dggan_purifier",
]
