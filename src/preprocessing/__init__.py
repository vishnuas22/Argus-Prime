"""
Preprocessing module - Media preprocessing pipelines.

Provides video, audio, and text preprocessing for deepfake detection.
"""

from .video_processor import (
    VideoProcessor,
    Frame,
    VideoMetadata,
    extract_frames_async,
    extract_frames_sync,
)

from .face_detector import (
    FaceDetector,
    FaceBox,
    MTCNN,
    detect_faces,
    align_face,
)

from .audio_processor import (
    AudioProcessor,
    AudioSegment,
    AudioMetadata,
    AudioAnalysisBundle,
    process_audio_async,
    process_audio_sync,
)

from .text_processor import (
    TextProcessor,
    TextSegment,
    TextMetadata,
    TextAnalysisBundle,
    BurstinessMetrics,
    PerplexityMetrics,
    process_text_async,
    process_text_sync,
)

from .face_aligner import (
    FaceAligner,
    Landmark5Points,
    AlignedFace,
    align_face as align_face_from_landmarks,
    align_face_from_array,
)

from .preprocessor import (
    UnifiedPreprocessor,
    PreprocessingResult,
    PreprocessingOptions,
    get_preprocessor,
)

__all__ = [
    "VideoProcessor",
    "Frame",
    "VideoMetadata",
    "extract_frames_async",
    "extract_frames_sync",
    "FaceDetector",
    "FaceBox",
    "MTCNN",
    "detect_faces",
    "align_face",
    "AudioProcessor",
    "AudioSegment",
    "AudioMetadata",
    "AudioAnalysisBundle",
    "process_audio_async",
    "process_audio_sync",
    "TextProcessor",
    "TextSegment",
    "TextMetadata",
    "TextAnalysisBundle",
    "BurstinessMetrics",
    "PerplexityMetrics",
    "process_text_async",
    "process_text_sync",
    "FaceAligner",
    "Landmark5Points",
    "AlignedFace",
    "align_face_from_landmarks",
    "align_face_from_array",
    "UnifiedPreprocessor",
    "PreprocessingResult",
    "PreprocessingOptions",
    "get_preprocessor",
]
