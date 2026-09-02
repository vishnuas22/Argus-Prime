"""Text detection models."""

from .spec_detect import (
    SpecDetect,
    SpecDetectResult,
    create_specdetect,
    register_specdetect,
)
from .perplexity_detector import (
    PerplexityDetector,
    PerplexityResult,
    create_perplexity,
    register_perplexity,
)

__all__ = [
    "SpecDetect",
    "SpecDetectResult",
    "create_specdetect",
    "register_specdetect",
    "PerplexityDetector",
    "PerplexityResult",
    "create_perplexity",
    "register_perplexity",
]
