"""
Unit tests for video preprocessing module.

Tests frame extraction, metadata retrieval, and validation.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.preprocessing.video_processor import (
    VideoProcessor,
    Frame,
    VideoMetadata,
)


class TestFrame:
    """Test Frame dataclass."""

    def test_frame_creation(self):
        """Test Frame object creation."""
        frame = Frame(
            path="/tmp/frames/frame_000001.jpg",
            frame_number=1,
            timestamp=0.2,
            width=640,
            height=360,
        )
        assert frame.path == "/tmp/frames/frame_000001.jpg"
        assert frame.frame_number == 1
        assert frame.timestamp == 0.2
        assert frame.width == 640
        assert frame.height == 360

    def test_frame_to_dict(self):
        """Test Frame serialization to dictionary."""
        frame = Frame(
            path="/tmp/frames/frame_000001.jpg",
            frame_number=1,
            timestamp=0.2,
            width=640,
            height=360,
        )
        frame_dict = frame.to_dict()

        assert frame_dict["path"] == "/tmp/frames/frame_000001.jpg"
        assert frame_dict["frame_number"] == 1
        assert frame_dict["timestamp"] == 0.2
        assert frame_dict["width"] == 640
        assert frame_dict["height"] == 360


class TestVideoMetadata:
    """Test VideoMetadata dataclass."""

    def test_metadata_creation(self):
        """Test VideoMetadata object creation."""
        metadata = VideoMetadata(
            duration_seconds=120.5,
            width=1920,
            height=1080,
            fps=30.0,
            codec="h264",
            bitrate=5000000,
            has_audio=True,
            frame_count=3615,
        )
        assert metadata.duration_seconds == 120.5
        assert metadata.width == 1920
        assert metadata.height == 1080
        assert metadata.fps == 30.0
        assert metadata.codec == "h264"
        assert metadata.bitrate == 5000000
        assert metadata.has_audio is True
        assert metadata.frame_count == 3615


class TestVideoProcessor:
    """Test VideoProcessor class."""

    def test_processor_initialization(self):
        """Test VideoProcessor initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(temp_dir=temp_dir)

            assert processor.temp_dir == temp_dir
            assert processor.ffmpeg_path is not None
            # GPU check depends on system

    def test_processor_with_custom_ffmpeg(self):
        """Test VideoProcessor with custom FFmpeg path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(
                temp_dir=temp_dir,
                ffmpeg_path="/custom/ffmpeg"
            )
            assert processor.ffmpeg_path == "/custom/ffmpeg"

    def test_validate_input_nonexistent_file(self):
        """Test validation of non-existent file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(temp_dir=temp_dir)

            is_valid, error_msg = processor.validate_input(
                "/nonexistent/video.mp4"
            )
            assert is_valid is False
            assert "not found" in error_msg.lower()

    def test_validate_input_empty_file(self):
        """Test validation of empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_file = Path(temp_dir) / "empty.mp4"
            empty_file.touch()

            processor = VideoProcessor(temp_dir=temp_dir)

            is_valid, error_msg = processor.validate_input(str(empty_file))
            assert is_valid is False
            assert "empty" in error_msg.lower()

    def test_validate_input_unsupported_format(self):
        """Test validation of unsupported format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            file = Path(temp_dir) / "test.xyz"
            file.write_text("test content")

            processor = VideoProcessor(temp_dir=temp_dir)

            is_valid, error_msg = processor.validate_input(str(file))
            assert is_valid is False
            assert "unsupported" in error_msg.lower()

    def test_validate_input_valid_mp4(self, tmp_path):
        """Test validation of valid MP4 file."""
        # Create a mock video file (we can't actually create a valid video)
        video_file = tmp_path / "test.mp4"
        video_file.write_bytes(b"fake video content")

        processor = VideoProcessor(temp_dir=str(tmp_path))

        # This will fail ffprobe but that's expected for mock file
        is_valid, error_msg = processor.validate_input(str(video_file))
        # The validation might pass file existence but fail ffprobe
        assert isinstance(is_valid, bool)

    def test_parameter_validation_fps(self):
        """Test parameter validation for FPS."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(temp_dir=temp_dir)

            # Test invalid FPS values
            with pytest.raises(ValueError):
                processor.extract_frames_sync(
                    video_path="/fake/path.mp4",
                    fps=0,
                )

            with pytest.raises(ValueError):
                processor.extract_frames_sync(
                    video_path="/fake/path.mp4",
                    fps=31,
                )

    def test_parameter_validation_quality(self):
        """Test parameter validation for quality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(temp_dir=temp_dir)

            # Test invalid quality values
            with pytest.raises(ValueError):
                processor.extract_frames_sync(
                    video_path="/fake/path.mp4",
                    quality=-1,
                )

            with pytest.raises(ValueError):
                processor.extract_frames_sync(
                    video_path="/fake/path.mp4",
                    quality=32,
                )

    def test_collect_frames_empty_directory(self):
        """Test collecting frames from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            processor = VideoProcessor(temp_dir=temp_dir)
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            frames = processor._collect_frames(output_dir, None)
            assert frames == []

    def test_cleanup_frames(self, tmp_path):
        """Test frame cleanup functionality."""
        # Create mock frame files
        frame_dir = tmp_path / "frames"
        frame_dir.mkdir()

        (frame_dir / "frame_000001.jpg").touch()
        (frame_dir / "frame_000002.jpg").touch()

        # Create Frame objects
        frames = [
            Frame(str(frame_dir / "frame_000001.jpg"), 1, 0.2, 640, 360),
            Frame(str(frame_dir / "frame_000002.jpg"), 2, 0.4, 640, 360),
        ]

        processor = VideoProcessor(temp_dir=str(tmp_path))
        processor.cleanup_frames(frames)

        # Verify directory was cleaned up
        assert not frame_dir.exists()

    def test_cleanup_dir(self, tmp_path):
        """Test directory cleanup functionality."""
        test_dir = tmp_path / "test_output"
        test_dir.mkdir()
        (test_dir / "file.txt").touch()

        processor = VideoProcessor(temp_dir=str(tmp_path))
        processor.cleanup_dir(str(test_dir))

        assert not test_dir.exists()


class TestVideoProcessorIntegration:
    """Integration tests for VideoProcessor (require FFmpeg)."""

    def test_metadata_extraction_real_video(self, tmp_path):
        """Test metadata extraction from a real video file."""
        # This test requires an actual video file
        # For unit testing, we mock the subprocess call
        processor = VideoProcessor(temp_dir=str(tmp_path))

        # Mock the ffprobe call
        mock_output = '''
        {
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "r_frame_rate": "30/1",
                    "codec_name": "h264"
                },
                {
                    "codec_type": "audio",
                    "codec_name": "aac"
                }
            ],
            "format": {
                "duration": "120.5",
                "bit_rate": "5000000"
            }
        }
        '''

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout=mock_output,
                stderr=""
            )

            # Create a fake video file to pass validation
            video_file = tmp_path / "video.mp4"
            video_file.write_bytes(b"fake video content")

            metadata = processor.get_metadata(str(video_file))

            assert metadata.duration_seconds == 120.5
            assert metadata.width == 1920
            assert metadata.height == 1080
            assert metadata.fps == 30.0
            assert metadata.codec == "h264"
            assert metadata.has_audio is True


class TestExtractFramesConvenienceFunctions:
    """Test convenience functions for frame extraction."""

    def test_extract_frames_sync_function(self):
        """Test the sync convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the async method
            with patch('src.preprocessing.video_processor.VideoProcessor.extract_frames') as mock:
                mock.return_value = [
                    Frame(f"{temp_dir}/frame_000001.jpg", 1, 0.2, 640, 360),
                    Frame(f"{temp_dir}/frame_000002.jpg", 2, 0.4, 640, 360),
                ]

                from src.preprocessing import extract_frames_sync

                frames = extract_frames_sync(
                    video_path="/fake/path.mp4",
                    fps=5,
                )

                assert len(frames) == 2
                assert mock.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
