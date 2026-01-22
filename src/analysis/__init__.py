"""Analysis module - Detection orchestrators."""

from .video_detector import (
    VideoDetector,
    VideoAnalysisResult,
    FrameAnalysisResult,
    create_video_detector,
)
from .audio_detector import (
    AudioDetector,
    AudioAnalysisResult,
    SegmentAnalysisResult,
    create_audio_detector,
)
from .text_detector import (
    TextDetector,
    TextAnalysisResult,
    SentenceAnalysisResult,
    create_text_detector,
)
from .rppg_analyzer import (
    RPPGAnalyzer,
    RPPGResult,
    get_rppg_analyzer,
)
from .crossmodal import (
    CrossModalChecker,
    CrossModalResult,
    Contradiction,
    get_crossmodal_checker,
)

__all__ = [
    "VideoDetector",
    "VideoAnalysisResult",
    "FrameAnalysisResult",
    "create_video_detector",
    "AudioDetector",
    "AudioAnalysisResult",
    "SegmentAnalysisResult",
    "create_audio_detector",
    "TextDetector",
    "TextAnalysisResult",
    "SentenceAnalysisResult",
    "create_text_detector",
    "RPPGAnalyzer",
    "RPPGResult",
    "get_rppg_analyzer",
    "CrossModalChecker",
    "CrossModalResult",
    "Contradiction",
    "get_crossmodal_checker",
]
