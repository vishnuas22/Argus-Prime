"""
Unified Preprocessing Facade.

Provides a single interface for preprocessing all content types
(video, audio, text, image) with automatic type detection and routing.

Features:
- Unified interface for all preprocessing tasks
- Automatic file type detection
- Consistent progress reporting
- Resource cleanup management
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import concurrent.futures

from src.config import get_config
from src.core.logging import logger

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _run_async_in_executor(coro):
    """Run async coroutine in thread pool to avoid event loop conflicts."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, lambda: loop.run_until_complete(coro))


@dataclass
class PreprocessingResult:
    """Result of preprocessing operation."""
    success: bool
    file_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "file_type": self.file_type,
            "data": self.data,
            "metadata": self.metadata,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class PreprocessingOptions:
    """Options for preprocessing pipeline."""
    video_fps: int = 5
    video_max_frames: Optional[int] = None
    video_extract_faces: bool = True
    audio_sample_rate: int = 16000
    audio_segment_duration: float = 2.0
    audio_apply_vad: bool = True
    text_language: Optional[str] = None
    image_size: Tuple[int, int] = (224, 224)
    generate_spectrograms: bool = True
    generate_heatmaps: bool = True


class UnifiedPreprocessor:
    """
    Unified preprocessing facade for all content types.

    Routes preprocessing requests to appropriate specialized processors
    based on detected file type.

    Attributes:
        options: Preprocessing configuration options
        video_processor: Video preprocessing processor
        audio_processor: Audio preprocessing processor
        text_processor: Text preprocessing processor
        face_detector: Face detection processor
        face_aligner: Face alignment processor
    """

    def __init__(self, options: Optional[PreprocessingOptions] = None):
        """
        Initialize unified preprocessor.

        Args:
            options: Preprocessing options (uses defaults if None)
        """
        self.options = options or PreprocessingOptions()
        self.config = get_config()

        self._video_processor = None
        self._audio_processor = None
        self._text_processor = None
        self._face_detector = None
        self._face_aligner = None

        logger.info("UnifiedPreprocessor initialized")

    def _get_video_processor(self):
        """Get or create video processor."""
        if self._video_processor is None:
            from src.preprocessing import VideoProcessor
            self._video_processor = VideoProcessor()
        return self._video_processor

    def _get_audio_processor(self):
        """Get or create audio processor."""
        if self._audio_processor is None:
            from src.preprocessing import AudioProcessor
            self._audio_processor = AudioProcessor(
                sample_rate=self.options.audio_sample_rate,
                segment_duration=self.options.audio_segment_duration,
            )
        return self._audio_processor

    def _get_text_processor(self):
        """Get or create text processor."""
        if self._text_processor is None:
            from src.preprocessing import TextProcessor
            self._text_processor = TextProcessor()
        return self._text_processor

    def _get_face_detector(self):
        """Get or create face detector."""
        if self._face_detector is None:
            from src.preprocessing import FaceDetector
            self._face_detector = FaceDetector()
        return self._face_detector

    def _get_face_aligner(self):
        """Get or create face aligner."""
        if self._face_aligner is None:
            from src.preprocessing import FaceAligner
            self._face_aligner = FaceAligner()
        return self._face_aligner

    def detect_file_type(
        self, file_path: str, content_type: Optional[str] = None
    ) -> str:
        """
        Detect file type from path and content type.

        Args:
            file_path: Path to file
            content_type: Optional declared content type

        Returns:
            File type string: video, audio, image, text, or unknown
        """
        path = Path(file_path)
        ext = path.suffix.lower()

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        text_exts = {".txt", ".md", ".pdf", ".docx"}

        if ext in video_exts or (content_type and "video" in content_type):
            return "video"
        elif ext in audio_exts or (content_type and "audio" in content_type):
            return "audio"
        elif ext in image_exts or (content_type and "image" in content_type):
            return "image"
        elif ext in text_exts or (content_type and "text" in content_type):
            return "text"
        else:
            return "unknown"

    async def preprocess(
        self,
        file_path: str,
        content_type: Optional[str] = None,
        options: Optional[PreprocessingOptions] = None,
        progress_callback=None,
    ) -> PreprocessingResult:
        """
        Preprocess a file based on its type.

        Args:
            file_path: Path to file
            content_type: Optional declared content type
            options: Override preprocessing options
            progress_callback: Optional callback(progress, message)

        Returns:
            PreprocessingResult with processed data
        """
        file_type = self.detect_file_type(file_path, content_type)

        logger.info(f"Starting preprocessing: {file_path} (type: {file_type})")

        if progress_callback:
            progress_callback(5, f"Detected file type: {file_type}")

        if file_type == "unknown":
            return PreprocessingResult(
                success=False,
                file_type="unknown",
                errors=["Unable to determine file type"],
            )

        try:
            if file_type == "video":
                return await self._preprocess_video(
                    file_path, options or self.options, progress_callback
                )
            elif file_type == "audio":
                return await self._preprocess_audio(
                    file_path, options or self.options, progress_callback
                )
            elif file_type == "image":
                return await self._preprocess_image(
                    file_path, options or self.options, progress_callback
                )
            elif file_type == "text":
                return await self._preprocess_text(
                    file_path, options or self.options, progress_callback
                )
            else:
                return PreprocessingResult(
                    success=False,
                    file_type=file_type,
                    errors=[f"Unsupported file type: {file_type}"],
                )

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return PreprocessingResult(
                success=False,
                file_type=file_type,
                errors=[str(e)],
            )

    async def _preprocess_video(
        self,
        file_path: str,
        options: PreprocessingOptions,
        progress_callback=None,
    ) -> PreprocessingResult:
        """Preprocess video file."""
        video_processor = self._get_video_processor()

        if progress_callback:
            progress_callback(10, "Extracting video metadata")

        metadata = video_processor.get_metadata(file_path)

        if progress_callback:
            progress_callback(20, "Extracting frames")

        frames = await video_processor.extract_frames(
            file_path,
            fps=options.video_fps,
            max_frames=options.video_max_frames,
        )

        if progress_callback:
            progress_callback(50, "Detecting faces")

        face_detector = self._get_face_detector()
        aligned_faces = []
        face_boxes = []

        for frame in frames:
            try:
                import cv2
                image = cv2.imread(frame.path)
                faces = face_detector.detect(image)

                for face in faces:
                    aligned = self._get_face_aligner().align(image, face)
                    aligned_faces.append(aligned)
                    face_boxes.append(face.bbox)

            except Exception as e:
                logger.warning(f"Face detection failed for frame {frame.frame_number}: {e}")

        if progress_callback:
            progress_callback(80, "Generating heatmaps")

        heatmaps = []
        if options.generate_heatmaps and aligned_faces:
            try:
                from src.explainability import HeatmapGenerator
                heatmap_gen = HeatmapGenerator()
                for face_tensor in aligned_faces[:10]:
                    heatmap = heatmap_gen.generate(face_tensor)
                    heatmaps.append(heatmap)
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")

        if progress_callback:
            progress_callback(100, "Video preprocessing complete")

        return PreprocessingResult(
            success=True,
            file_type="video",
            data={
                "frames": [f.to_dict() for f in frames],
                "aligned_faces_count": len(aligned_faces),
                "face_boxes": [b.tolist() for b in face_boxes],
                "heatmaps": heatmaps,
            },
            metadata={
                "duration_seconds": metadata.duration_seconds,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.fps,
                "frame_count": len(frames),
                "face_count": len(aligned_faces),
            },
        )

    async def _preprocess_audio(
        self,
        file_path: str,
        options: PreprocessingOptions,
        progress_callback=None,
    ) -> PreprocessingResult:
        """Preprocess audio file."""
        audio_processor = self._get_audio_processor()

        if progress_callback:
            progress_callback(10, "Analyzing audio metadata")

        metadata = audio_processor.get_metadata(file_path)

        if progress_callback:
            progress_callback(30, "Normalizing audio")

        normalized_path, _ = await audio_processor.normalize_audio(file_path)

        if progress_callback:
            progress_callback(50, "Segmenting audio")

        segments, _ = await audio_processor.segment_audio(
            normalized_path, normalize=False
        )

        if progress_callback:
            progress_callback(80, "Generating spectrograms")

        spectrograms = []
        if options.generate_spectrograms:
            spectrograms = audio_processor.generate_spectrograms(segments)

        if progress_callback:
            progress_callback(100, "Audio preprocessing complete")

        return PreprocessingResult(
            success=True,
            file_type="audio",
            data={
                "segments": [s.to_dict() for s in segments],
                "spectrograms_shape": [s.shape for s in spectrograms] if spectrograms else [],
            },
            metadata={
                "duration_seconds": metadata.duration_seconds,
                "sample_rate": metadata.sample_rate,
                "channels": metadata.channels,
                "segment_count": len(segments),
                "speech_segments": sum(1 for s in segments if s.is_speech),
            },
        )

    async def _preprocess_image(
        self,
        file_path: str,
        options: PreprocessingOptions,
        progress_callback=None,
    ) -> PreprocessingResult:
        """Preprocess image file."""
        import cv2
        import numpy as np

        if progress_callback:
            progress_callback(20, "Loading image")

        image = cv2.imread(file_path)
        if image is None:
            return PreprocessingResult(
                success=False,
                file_type="image",
                errors=["Failed to load image"],
            )

        height, width = image.shape[:2]

        if progress_callback:
            progress_callback(40, "Detecting faces")

        face_detector = self._get_face_detector()
        faces = face_detector.detect(image)

        aligned_faces = []
        face_data = []

        for face in faces:
            aligned = self._get_face_aligner().align(image, face)
            aligned_faces.append(aligned)
            face_data.append({
                "bbox": face.bbox.tolist(),
                "landmarks": face.landmarks.tolist() if hasattr(face, 'landmarks') else None,
                "confidence": face.confidence if hasattr(face, 'confidence') else None,
            })

        if progress_callback:
            progress_callback(70, "Generating heatmaps")

        heatmaps = []
        if options.generate_heatmaps and aligned_faces:
            try:
                from src.explainability import HeatmapGenerator
                heatmap_gen = HeatmapGenerator()
                for face_tensor in aligned_faces:
                    heatmap = heatmap_gen.generate(face_tensor)
                    heatmaps.append(heatmap)
            except Exception as e:
                logger.warning(f"Heatmap generation failed: {e}")

        if progress_callback:
            progress_callback(100, "Image preprocessing complete")

        return PreprocessingResult(
            success=True,
            file_type="image",
            data={
                "faces_count": len(aligned_faces),
                "face_data": face_data,
                "heatmaps": heatmaps,
            },
            metadata={
                "width": width,
                "height": height,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
            },
        )

    async def _preprocess_text(
        self,
        file_path: str,
        options: PreprocessingOptions,
        progress_callback=None,
    ) -> PreprocessingResult:
        """Preprocess text file."""
        text_processor = self._get_text_processor()

        if progress_callback:
            progress_callback(30, "Reading text")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if progress_callback:
            progress_callback(50, "Analyzing text")

        result = text_processor.analyze(content)

        if progress_callback:
            progress_callback(100, "Text preprocessing complete")

        return PreprocessingResult(
            success=True,
            file_type="text",
            data=result.to_dict() if hasattr(result, 'to_dict') else result,
            metadata={
                "word_count": getattr(result, 'word_count', len(content.split())),
                "language": getattr(result, 'language_detected', 'unknown'),
            },
        )

    def preprocess_sync(
        self,
        file_path: str,
        content_type: Optional[str] = None,
        options: Optional[PreprocessingOptions] = None,
        progress_callback=None,
    ) -> PreprocessingResult:
        """
        Synchronous version of preprocess.

        Args:
            file_path: Path to file
            content_type: Optional declared content type
            options: Override preprocessing options
            progress_callback: Optional callback(progress, message)

        Returns:
            PreprocessingResult with processed data
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return _run_async_in_executor(self.preprocess(file_path, content_type, options, progress_callback))
            else:
                return loop.run_until_complete(self.preprocess(file_path, content_type, options, progress_callback))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(self.preprocess(file_path, content_type, options, progress_callback))


_preprocessor: Optional[UnifiedPreprocessor] = None


def get_preprocessor(
    options: Optional[PreprocessingOptions] = None,
) -> UnifiedPreprocessor:
    """Get or create unified preprocessor instance."""
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = UnifiedPreprocessor(options)
    return _preprocessor
