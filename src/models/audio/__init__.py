"""Audio detection models."""

from .rawnet2_detector import (
    RawNet2,
    RawNet2Result,
    VOCODER_CLASSES,
    create_rawnet2,
    register_rawnet2,
)
from .transcrypt_detector import (
    TranscryptDetector,
    TranscryptResult,
    create_transcrypt,
    register_transcrypt,
)

__all__ = [
    "RawNet2",
    "RawNet2Result",
    "VOCODER_CLASSES",
    "create_rawnet2",
    "register_rawnet2",
    "TranscryptDetector",
    "TranscryptResult",
    "create_transcrypt",
    "register_transcrypt",
]
