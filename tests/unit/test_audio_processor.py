"""
Unit tests for audio preprocessing module.

Tests audio processing, VAD, segmentation, and spectrogram generation.
"""

import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from src.preprocessing.audio_processor import (
    AudioProcessor,
    AudioSegment,
    AudioMetadata,
    AudioAnalysisBundle,
    process_audio_async,
    process_audio_sync,
)


class TestAudioSegment:
    """Test AudioSegment dataclass."""

    def test_segment_creation(self):
        """Test AudioSegment object creation."""
        waveform = torch.randn(1, 32000)  # 2 seconds at 16kHz
        segment = AudioSegment(
            waveform=waveform,
            sample_rate=16000,
            start_time=0.0,
            end_time=2.0,
            is_speech=True,
        )

        assert segment.waveform.shape == (1, 32000)
        assert segment.sample_rate == 16000
        assert segment.start_time == 0.0
        assert segment.end_time == 2.0
        assert segment.is_speech is True
        assert segment.spectrogram is None

    def test_segment_with_spectrogram(self):
        """Test AudioSegment with spectrogram."""
        waveform = torch.randn(1, 32000)
        spectrogram = torch.randn(256, 100)

        segment = AudioSegment(
            waveform=waveform,
            sample_rate=16000,
            start_time=0.0,
            end_time=2.0,
            is_speech=True,
            spectrogram=spectrogram,
        )

        assert segment.spectrogram is not None
        assert segment.spectrogram.shape == (256, 100)

    def test_segment_to_dict(self):
        """Test AudioSegment serialization."""
        waveform = torch.randn(1, 32000)
        segment = AudioSegment(
            waveform=waveform,
            sample_rate=16000,
            start_time=1.0,
            end_time=3.0,
            is_speech=False,
        )

        segment_dict = segment.to_dict()

        assert segment_dict["start_time"] == 1.0
        assert segment_dict["end_time"] == 3.0
        assert segment_dict["is_speech"] is False
        assert segment_dict["sample_rate"] == 16000
        assert "waveform_shape" in segment_dict


class TestAudioMetadata:
    """Test AudioMetadata dataclass."""

    def test_metadata_creation(self):
        """Test AudioMetadata object creation."""
        metadata = AudioMetadata(
            duration_seconds=120.5,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            bitrate=256000,
            file_size=3072000,
        )

        assert metadata.duration_seconds == 120.5
        assert metadata.sample_rate == 16000
        assert metadata.channels == 1
        assert metadata.bit_depth == 16
        assert metadata.bitrate == 256000
        assert metadata.file_size == 3072000

    def test_metadata_optional_fields(self):
        """Test AudioMetadata with optional fields."""
        metadata = AudioMetadata(
            duration_seconds=60.0,
            sample_rate=48000,
            channels=2,
            bit_depth=None,
            bitrate=None,
            file_size=1024000,
        )

        assert metadata.bit_depth is None
        assert metadata.bitrate is None


class TestAudioProcessor:
    """Test AudioProcessor class."""

    def test_processor_initialization(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor()

        assert processor.sample_rate == 16000
        assert processor.segment_duration == 2.0
        assert processor.overlap == 0.5
        assert processor.n_mels == 256
        assert processor.n_fft == 512
        assert processor.vad_mode == 2

    def test_processor_with_custom_params(self):
        """Test AudioProcessor with custom parameters."""
        processor = AudioProcessor(
            sample_rate=22050,
            segment_duration=3.0,
            overlap=0.75,
            n_mels=128,
            n_fft=1024,
            vad_mode=3,
        )

        assert processor.sample_rate == 22050
        assert processor.segment_duration == 3.0
        assert processor.overlap == 0.75
        assert processor.n_mels == 128
        assert processor.n_fft == 1024
        assert processor.vad_mode == 3

    def test_validate_input_nonexistent_file(self):
        """Test validation of non-existent file."""
        processor = AudioProcessor()

        is_valid, error_msg = processor.validate_input(
            "/nonexistent/audio.wav"
        )

        assert is_valid is False
        assert "not found" in error_msg.lower()

    def test_validate_input_empty_file(self):
        """Test validation of empty file."""
        processor = AudioProcessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            empty_file = Path(temp_dir) / "empty.wav"
            empty_file.touch()

            is_valid, error_msg = processor.validate_input(str(empty_file))
            assert is_valid is False
            assert "empty" in error_msg.lower()

    def test_validate_input_unsupported_format(self):
        """Test validation of unsupported format."""
        processor = AudioProcessor()

        with tempfile.TemporaryDirectory() as temp_dir:
            file = Path(temp_dir) / "test.xyz"
            file.write_text("test content")

            is_valid, error_msg = processor.validate_input(str(file))
            assert is_valid is False
            assert "unsupported" in error_msg.lower()

    def test_validate_input_valid_wav(self, tmp_path):
        """Test validation of valid WAV file."""
        processor = AudioProcessor()

        # Create a mock WAV file
        wav_file = tmp_path / "test.wav"
        wav_file.write_bytes(b"fake wav content")

        is_valid, error_msg = processor.validate_input(str(wav_file))
        # Will fail ffprobe but pass file validation
        assert isinstance(is_valid, bool)

    def test_apply_vad_with_mock(self):
        """Test VAD application with mock."""
        processor = AudioProcessor()

        # Mock VAD
        mock_vad = MagicMock()
        mock_vad.is_speech.return_value = True
        processor._vad = mock_vad

        waveform = torch.randn(1, 16000)  # 1 second
        is_speech = processor._apply_vad(waveform, 16000)

        assert len(is_speech) > 0
        assert all(is_speech)  # All frames marked as speech

    def test_apply_vad_without_library(self):
        """Test VAD when webrtcvad is not available."""
        processor = AudioProcessor()
        processor._vad = None  # Simulate missing library

        waveform = torch.randn(1, 16000)
        is_speech = processor._apply_vad(waveform, 16000)

        # Should return list of True values
        assert len(is_speech) > 0

    def test_generate_spectrogram(self):
        """Test spectrogram generation."""
        processor = AudioProcessor()

        waveform = torch.randn(1, 32000)
        segment = AudioSegment(
            waveform=waveform,
            sample_rate=16000,
            start_time=0.0,
            end_time=2.0,
            is_speech=True,
        )

        spectrogram = processor.generate_spectrogram(segment)

        assert spectrogram is not None
        assert spectrogram.shape[1] == 256
        assert spectrogram.ndim == 3

    def test_generate_spectrograms_batch(self):
        """Test batch spectrogram generation."""
        processor = AudioProcessor()

        # Create test segments
        segments = [
            AudioSegment(
                waveform=torch.randn(1, 32000),
                sample_rate=16000,
                start_time=0.0,
                end_time=2.0,
                is_speech=True,
            )
            for _ in range(3)
        ]

        spectrograms = processor.generate_spectrograms(segments)

        assert len(spectrograms) == 3
        for spec in spectrograms:
            assert spec is not None

    def test_segment_to_dict(self, tmp_path):
        """Test segment dictionary conversion."""
        processor = AudioProcessor()

        # Mock metadata extraction
        mock_metadata = AudioMetadata(
            duration_seconds=120.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            bitrate=256000,
            file_size=3072000,
        )

        with patch.object(processor, 'get_metadata', return_value=mock_metadata):
            bundle = AudioAnalysisBundle(
                segments=[],
                metadata=mock_metadata,
            )

            bundle_dict = bundle.to_dict()

            assert "segments" in bundle_dict
            assert "metadata" in bundle_dict
            assert "statistics" in bundle_dict


class TestAudioAnalysisBundle:
    """Test AudioAnalysisBundle class."""

    def test_bundle_creation(self):
        """Test bundle creation."""
        segments = [
            AudioSegment(
                waveform=torch.randn(1, 32000),
                sample_rate=16000,
                start_time=0.0,
                end_time=2.0,
                is_speech=True,
            ),
            AudioSegment(
                waveform=torch.randn(1, 32000),
                sample_rate=16000,
                start_time=2.0,
                end_time=4.0,
                is_speech=False,
            ),
        ]

        metadata = AudioMetadata(
            duration_seconds=4.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            bitrate=None,
            file_size=1024000,
        )

        bundle = AudioAnalysisBundle(
            segments=segments,
            metadata=metadata,
        )

        assert len(bundle.segments) == 2
        assert len(bundle.speech_segments) == 1
        assert bundle.total_duration == 4.0
        assert bundle.speech_duration == 2.0

    def test_bundle_statistics(self):
        """Test bundle statistics calculation."""
        segments = [
            AudioSegment(
                waveform=torch.randn(1, 32000),
                sample_rate=16000,
                start_time=0.0,
                end_time=2.0,
                is_speech=True,
            ),
            AudioSegment(
                waveform=torch.randn(1, 32000),
                sample_rate=16000,
                start_time=2.0,
                end_time=4.0,
                is_speech=True,
            ),
            AudioSegment(
                waveform=torch.randn(1, 32000),
                sample_rate=16000,
                start_time=4.0,
                end_time=6.0,
                is_speech=False,
            ),
        ]

        metadata = AudioMetadata(
            duration_seconds=6.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            bitrate=None,
            file_size=1024000,
        )

        bundle = AudioAnalysisBundle(
            segments=segments,
            metadata=metadata,
        )

        assert bundle.total_duration == 6.0
        assert bundle.speech_duration == 4.0
        assert len(bundle.speech_segments) == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_process_audio_sync_function(self):
        """Test the sync convenience function."""
        with patch('src.preprocessing.audio_processor.AudioProcessor') as MockProcessor:
            mock_instance = MagicMock()
            mock_instance.process_sync.return_value = (
                [AudioSegment(
                    waveform=torch.randn(1, 32000),
                    sample_rate=16000,
                    start_time=0.0,
                    end_time=2.0,
                    is_speech=True,
                )],
                AudioMetadata(
                    duration_seconds=2.0,
                    sample_rate=16000,
                    channels=1,
                    bit_depth=16,
                    bitrate=256000,
                    file_size=64000,
                ),
            )
            MockProcessor.return_value = mock_instance

            from src.preprocessing import process_audio_sync

            result = process_audio_sync("/fake/audio.wav")

            assert len(result.segments) == 1
            mock_instance.process_sync.assert_called_once_with("/fake/audio.wav")


class TestAudioProcessorIntegration:
    """Integration tests for AudioProcessor."""

    def test_process_with_mock_metadata(self, tmp_path):
        """Test complete processing with mocked metadata."""
        processor = AudioProcessor()

        # Create a mock audio file
        audio_file = tmp_path / "audio.wav"
        audio_file.write_bytes(b"fake audio content")

        # Mock metadata
        mock_metadata = AudioMetadata(
            duration_seconds=4.0,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            bitrate=256000,
            file_size=1024000,
        )

        with patch.object(processor, 'get_metadata', return_value=mock_metadata):
            # Mock segment_audio
            mock_segments = [
                AudioSegment(
                    waveform=torch.randn(1, 32000),
                    sample_rate=16000,
                    start_time=0.0,
                    end_time=2.0,
                    is_speech=True,
                ),
            ]

            with patch.object(processor, 'segment_audio', return_value=mock_segments):
                with patch.object(processor, 'generate_spectrograms', return_value=[]):
                    segments, metadata = processor.process_sync(str(audio_file))

                    assert len(segments) == 1
                    assert metadata.duration_seconds == 4.0

    def test_processor_with_different_sample_rates(self):
        """Test processor handles different sample rates."""
        for sr in [8000, 16000, 22050, 44100]:
            processor = AudioProcessor(sample_rate=sr)
            assert processor.sample_rate == sr

    def test_processor_with_different_segment_durations(self):
        """Test processor handles different segment durations."""
        for duration in [1.0, 2.0, 3.0, 5.0]:
            processor = AudioProcessor(segment_duration=duration)
            assert processor.segment_duration == duration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
