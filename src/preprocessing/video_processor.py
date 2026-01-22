"""
Video preprocessing module for Argus Prime.

Provides video frame extraction with configurable FPS and quality using FFmpeg.
Supports keyframe extraction, scene detection, and GPU-accelerated decoding.
"""

import asyncio
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from src.config import get_config
from src.core.logging import logger


@dataclass
class Frame:
    """Internal representation of a video frame."""
    path: str
    frame_number: int
    timestamp: float
    width: int
    height: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame to dictionary for serialization."""
        return {
            "path": self.path,
            "frame_number": self.frame_number,
            "timestamp": self.timestamp,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class VideoMetadata:
    """Metadata extracted from video file."""
    duration_seconds: float
    width: int
    height: int
    fps: float
    codec: str
    bitrate: int
    has_audio: bool
    frame_count: int


class VideoProcessor:
    """
    Video frame extraction processor using FFmpeg.

    This class handles video preprocessing by extracting frames at configurable
    FPS rates, with support for GPU acceleration and smart frame selection.

    Attributes:
        temp_dir: Directory for temporary frame storage
        ffmpeg_path: Path to FFmpeg binary
        gpu_enabled: Whether GPU acceleration is available

    Example:
        >>> processor = VideoProcessor()
        >>> frames = processor.extract_frames(
        ...     video_path="input.mp4",
        ...     fps=5,
        ...     output_dir="/tmp/frames"
        ... )
        >>> metadata = processor.get_metadata("input.mp4")
    """

    def __init__(
        self,
        temp_dir: Optional[str] = None,
        ffmpeg_path: Optional[str] = None,
    ):
        """
        Initialize the video processor.

        Args:
            temp_dir: Directory for temporary frame storage
            ffmpeg_path: Path to FFmpeg binary (auto-detected if None)
        """
        config = get_config()
        self.temp_dir = temp_dir or config.storage.temp_dir
        self.ffmpeg_path = ffmpeg_path or self._detect_ffmpeg()
        self.gpu_enabled = self._check_gpu_support()
        self._supported_formats = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

        # Create temp directory if it doesn't exist
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

        logger.info(
            f"VideoProcessor initialized: ffmpeg={self.ffmpeg_path}, "
            f"gpu={self.gpu_enabled}, temp_dir={self.temp_dir}"
        )

    def _detect_ffmpeg(self) -> str:
        """
        Detect FFmpeg binary path.

        Returns:
            Path to FFmpeg binary or 'ffmpeg' if not found in common locations
        """
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
        ]

        for path in common_paths:
            if Path(path).exists():
                logger.info(f"Found FFmpeg at: {path}")
                return path

        # Default to ffmpeg in PATH
        logger.warning("FFmpeg not found in common locations, using system PATH")
        return "ffmpeg"

    def _check_gpu_support(self) -> bool:
        """
        Check if GPU acceleration is available.

        Returns:
            True if NVENC/NVDEC is available
        """
        try:
            # Check for NVIDIA GPU and NVENC support
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if "h264_nvenc" in result.stdout or "hevc_nvenc" in result.stdout:
                logger.info("GPU acceleration (NVENC) available")
                return True

            # Check for VAAPI (Linux)
            result = subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if "vaapi" in result.stdout:
                logger.info("GPU acceleration (VAAPI) available")
                return True

            logger.info("GPU acceleration not available, using CPU")
            return False

        except Exception as e:
            logger.warning(f"Failed to check GPU support: {e}")
            return False

    def validate_input(self, video_path: str) -> tuple[bool, str]:
        """
        Validate that the input file is a valid video.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(video_path)

        if not path.exists():
            return False, f"Video file not found: {video_path}"

        if path.suffix.lower() not in self._supported_formats:
            return False, (
                f"Unsupported video format: {path.suffix}. "
                f"Supported formats: {', '.join(self._supported_formats)}"
            )

        if path.stat().st_size == 0:
            return False, f"Video file is empty: {video_path}"

        # Verify it's actually a video file using ffprobe
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name",
                    "-of", "csv=p=0",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if not result.stdout.strip():
                return False, "No video stream found in file"

            logger.info(f"Validated video: {path.name}, codec={result.stdout.strip()}")
            return True, ""

        except subprocess.TimeoutExpired:
            return False, "ffprobe timed out"
        except FileNotFoundError:
            return False, "ffprobe not found - please install ffmpeg"

    def get_metadata(self, video_path: str) -> VideoMetadata:
        """
        Extract metadata from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            VideoMetadata object with video information

        Raises:
            ValueError: If video file is invalid
            RuntimeError: If ffprobe fails
        """
        path = Path(video_path)

        if not path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            import json
            info = json.loads(result.stdout)

            # Find video stream
            video_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "video":
                    video_stream = stream
                    break

            if not video_stream:
                raise RuntimeError("No video stream found")

            format_info = info.get("format", {})
            video_info = video_stream

            # Calculate frame count
            duration = float(format_info.get("duration", 0))
            fps_str = video_info.get("r_frame_rate", "30/1")
            fps_parts = fps_str.split("/")
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else 30.0
            frame_count = int(duration * fps) if duration > 0 else 0

            metadata = VideoMetadata(
                duration_seconds=duration,
                width=int(video_info.get("width", 0)),
                height=int(video_info.get("height", 0)),
                fps=fps,
                codec=video_info.get("codec_name", "unknown"),
                bitrate=int(format_info.get("bit_rate", 0)),
                has_audio=any(
                    s.get("codec_type") == "audio"
                    for s in info.get("streams", [])
                ),
                frame_count=frame_count,
            )

            logger.info(
                f"Extracted metadata: {path.name}, "
                f"{metadata.width}x metadata.height, "
                f"{metadata.duration_seconds:.1f}s, "
                f"{metadata.frame_count} frames"
            )

            return metadata

        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe timed out")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")

    async def extract_frames(
        self,
        video_path: str,
        fps: int = 5,
        output_dir: Optional[str] = None,
        quality: int = 2,
        max_frames: Optional[int] = None,
        use_gpu: bool = False,
    ) -> List[Frame]:
        """
        Extract frames from a video file at specified FPS.

        Args:
            video_path: Path to the video file
            fps: Frames per second to extract (1-30, default: 5)
            output_dir: Output directory for frames (auto-generated if None)
            quality: FFmpeg quality parameter (0-31, lower is better)
            max_frames: Maximum number of frames to extract (None for all)
            use_gpu: Whether to use GPU acceleration

        Returns:
            List of Frame objects with extracted frame information

        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If frame extraction fails
        """
        # Validate input
        is_valid, error_msg = self.validate_input(video_path)
        if not is_valid:
            raise ValueError(error_msg)

        # Validate parameters
        if not 1 <= fps <= 30:
            raise ValueError(f"FPS must be between 1 and 30, got {fps}")

        if not 0 <= quality <= 31:
            raise ValueError(f"Quality must be between 0 and 31, got {quality}")

        # Generate output directory
        if output_dir is None:
            request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_dir = Path(self.temp_dir) / f"frames_{request_id}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get video metadata for frame count estimation
        metadata = self.get_metadata(video_path)
        estimated_frames = min(
            int(metadata.duration_seconds * fps),
            max_frames or float("inf")
        )

        logger.info(
            f"Extracting frames from {Path(video_path).name}: "
            f"fps={fps}, estimated_frames={estimated_frames}, "
            f"output_dir={output_dir}"
        )

        # Build FFmpeg command
        scale_filter = "scale=640:360"
        if use_gpu and self.gpu_enabled:
            # Use GPU-accelerated scaling
            hwaccel = "cuda" if "cuda" in subprocess.run(
                [self.ffmpeg_path, "-hide_banner", "-hwaccels"],
                capture_output=True, text=True
            ).stdout else "vaapi"
            cmd = [
                self.ffmpeg_path,
                "-hwaccel", hwaccel,
                "-i", video_path,
                "-vf", f"{scale_filter},fps={fps}",
                "-q:v", str(quality),
                "-f", "image2",
                str(output_path / "frame_%06d.jpg"),
            ]
        else:
            # CPU-based extraction
            cmd = [
                self.ffmpeg_path,
                "-i", video_path,
                "-vf", f"fps={fps},scale=640:360",
                "-q:v", str(quality),
                "-f", "image2",
                str(output_path / "frame_%06d.jpg"),
            ]

        # Execute FFmpeg
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace")
                raise RuntimeError(f"FFmpeg failed: {error_msg}")

        except asyncio.CancelledError:
            raise RuntimeError("Frame extraction was cancelled")

        # Collect extracted frames
        frames = self._collect_frames(output_path, max_frames)

        logger.info(
            f"Extracted {len(frames)} frames from {Path(video_path).name}"
        )

        return frames

    def extract_frames_sync(
        self,
        video_path: str,
        fps: int = 5,
        output_dir: Optional[str] = None,
        quality: int = 2,
        max_frames: Optional[int] = None,
        use_gpu: bool = False,
    ) -> List[Frame]:
        """
        Synchronous version of extract_frames.

        This method is provided for compatibility with synchronous code.
        For async code, use extract_frames() instead.
        """
        return asyncio.run(
            self.extract_frames(
                video_path=video_path,
                fps=fps,
                output_dir=output_dir,
                quality=quality,
                max_frames=max_frames,
                use_gpu=use_gpu,
            )
        )

    def _collect_frames(self, output_dir: Path, max_frames: Optional[int]) -> List[Frame]:
        """
        Collect extracted frames and create Frame objects.

        Args:
            output_dir: Directory containing extracted frames
            max_frames: Maximum number of frames to collect

        Returns:
            List of Frame objects sorted by frame number
        """
        frames = []
        frame_files = sorted(output_dir.glob("frame_*.jpg"))

        if max_frames:
            frame_files = frame_files[:max_frames]

        for frame_path in frame_files:
            # Extract frame number from filename
            frame_num = int(frame_path.stem.split("_")[1])

            # Get frame dimensions
            from PIL import Image
            with Image.open(frame_path) as img:
                width, height = img.size

            # Calculate timestamp
            fps = 5  # Default FPS, actual timestamp comes from frame number
            timestamp = frame_num / fps

            frames.append(
                Frame(
                    path=str(frame_path),
                    frame_number=frame_num,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                )
            )

        return frames

    def extract_keyframes(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        quality: int = 2,
    ) -> List[Frame]:
        """
        Extract only keyframes (I-frames) from a video.

        Keyframes are more efficient for processing as they don't require
        decoding of dependent frames.

        Args:
            video_path: Path to the video file
            output_dir: Output directory for frames
            quality: FFmpeg quality parameter (0-31)

        Returns:
            List of Frame objects containing keyframes
        """
        if output_dir is None:
            request_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_dir = Path(self.temp_dir) / f"keyframes_{request_id}"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            "-i", video_path,
            "-vf", "select=eq(pict_type\\,I)",
            "-q:v", str(quality),
            "-f", "image2",
            str(output_path / "keyframe_%06d.jpg"),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for keyframe extraction
            )

            if result.returncode != 0:
                raise RuntimeError(f"Keyframe extraction failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Keyframe extraction timed out")

        frames = self._collect_frames(output_path, None)

        logger.info(
            f"Extracted {len(frames)} keyframes from {Path(video_path).name}"
        )

        return frames

    def cleanup_frames(self, frames: List[Frame]) -> None:
        """
        Clean up extracted frame files.

        Args:
            frames: List of Frame objects to clean up
        """
        unique_dirs = set(Path(f.path).parent for f in frames)

        for dir_path in unique_dirs:
            try:
                if dir_path.exists():
                    shutil.rmtree(dir_path)
                    logger.info(f"Cleaned up frame directory: {dir_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up {dir_path}: {e}")

    def cleanup_dir(self, output_dir: str) -> None:
        """
        Clean up a specific output directory.

        Args:
            output_dir: Directory to clean up
        """
        dir_path = Path(output_dir)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                logger.info(f"Cleaned up directory: {output_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {output_dir}: {e}")


async def extract_frames_async(
    video_path: str,
    fps: int = 5,
    output_dir: Optional[str] = None,
    quality: int = 2,
    max_frames: Optional[int] = None,
) -> List[Frame]:
    """
    Convenience function for async frame extraction.

    Args:
        video_path: Path to the video file
        fps: Frames per second to extract
        output_dir: Output directory for frames
        quality: FFmpeg quality parameter
        max_frames: Maximum frames to extract

    Returns:
        List of Frame objects
    """
    processor = VideoProcessor()
    return await processor.extract_frames(
        video_path=video_path,
        fps=fps,
        output_dir=output_dir,
        quality=quality,
        max_frames=max_frames,
    )


def extract_frames_sync(
    video_path: str,
    fps: int = 5,
    output_dir: Optional[str] = None,
    quality: int = 2,
    max_frames: Optional[int] = None,
) -> List[Frame]:
    """
    Convenience function for sync frame extraction.

    Args:
        video_path: Path to the video file
        fps: Frames per second to extract
        output_dir: Output directory for frames
        quality: FFmpeg quality parameter
        max_frames: Maximum frames to extract

    Returns:
        List of Frame objects
    """
    processor = VideoProcessor()
    return processor.extract_frames_sync(
        video_path=video_path,
        fps=fps,
        output_dir=output_dir,
        quality=quality,
        max_frames=max_frames,
    )
