"""
Audio preprocessing module for Argus Prime.

Provides audio normalization, voice activity detection (VAD),
segmentation, and spectrogram generation using Librosa and WebRTC VAD.
"""

import asyncio
import subprocess
import tempfile
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, TypeVar

import numpy as np
import torch
import torchaudio

from src.config import get_config
from src.core.logging import logger

_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

T = TypeVar('T')


def _run_async_in_executor(coro) -> T:
    """Run async coroutine in thread pool to avoid event loop conflicts."""
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(_executor, lambda: loop.run_until_complete(coro))


@dataclass
class AudioSegment:
    """Internal representation of an audio segment."""
    waveform: torch.Tensor  # (1, samples)
    sample_rate: int
    start_time: float
    end_time: float
    is_speech: bool
    spectrogram: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_speech": self.is_speech,
            "sample_rate": self.sample_rate,
            "waveform_shape": list(self.waveform.shape),
        }


@dataclass
class AudioMetadata:
    """Metadata extracted from audio file."""
    duration_seconds: float
    sample_rate: int
    channels: int
    bit_depth: Optional[int]
    bitrate: Optional[int]
    file_size: int


class AudioProcessor:
    """
    Audio preprocessing processor for deepfake detection.

    This class handles audio preprocessing by:
    1. Converting to 16kHz, mono, 16-bit PCM format
    2. Applying WebRTC VAD for speech detection
    3. Segmenting into fixed-length windows
    4. Generating Mel-spectrograms for model input

    Attributes:
        sample_rate: Target sample rate (default: 16000)
        segment_duration: Segment length in seconds (default: 2)
        overlap: Overlap ratio between segments (default: 0.5)
        n_mels: Number of Mel frequency bins (default: 256)
        n_fft: FFT window size (default: 512)
        vad_mode: WebRTC VAD mode (0-3, default: 2)

    Example:
        >>> processor = AudioProcessor()
        >>> segments = processor.process("audio.wav")
        >>> spectrograms = processor.generate_spectrograms(segments)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_duration: float = 2.0,
        overlap: float = 0.5,
        n_mels: int = 256,
        n_fft: int = 512,
        vad_mode: int = 2,
        ffmpeg_path: Optional[str] = None,
    ):
        """
        Initialize the audio processor.

        Args:
            sample_rate: Target sample rate in Hz (default: 16000)
            segment_duration: Segment length in seconds (default: 2.0)
            overlap: Overlap ratio between segments (default: 0.5)
            n_mels: Number of Mel frequency bins (default: 256)
            n_fft: FFT window size (default: 512)
            vad_mode: WebRTC VAD mode 0-3 (default: 2)
            ffmpeg_path: Path to FFmpeg binary
        """
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.overlap = overlap
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.vad_mode = vad_mode

        # FFmpeg path
        self.ffmpeg_path = ffmpeg_path or self._detect_ffmpeg()

        # VAD initialization (lazy loaded)
        self._vad = None

        # Mel spectrogram transform
        self._mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            normalized=True,
        )

        logger.info(
            f"AudioProcessor initialized: sr={sample_rate}, "
            f"segment={segment_duration}s, overlap={overlap}, "
            f"n_mels={n_mels}, vad_mode={vad_mode}"
        )

    def _detect_ffmpeg(self) -> str:
        """Detect FFmpeg binary path."""
        common_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
        ]

        for path in common_paths:
            if Path(path).exists():
                return path

        return "ffmpeg"

    def _get_vad(self):
        """Get or create WebRTC VAD instance."""
        if self._vad is None:
            try:
                import webrtcvad
                self._vad = webrtcvad.create(mode=self.vad_mode)
                logger.info("WebRTC VAD initialized successfully")
            except ImportError:
                logger.warning("webrtcvad not installed, VAD will be simulated")
                self._vad = None
        return self._vad

    def validate_input(self, audio_path: str) -> Tuple[bool, str]:
        """
        Validate that the input file is a valid audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(audio_path)

        if not path.exists():
            return False, f"Audio file not found: {audio_path}"

        supported_formats = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}
        if path.suffix.lower() not in supported_formats:
            return False, (
                f"Unsupported audio format: {path.suffix}. "
                f"Supported: {', '.join(supported_formats)}"
            )

        if path.stat().st_size == 0:
            return False, f"Audio file is empty: {audio_path}"

        return True, ""

    def get_metadata(self, audio_path: str) -> AudioMetadata:
        """
        Extract metadata from an audio file.

        Args:
            audio_path: Path to the audio file

        Returns:
            AudioMetadata object with audio information

        Raises:
            ValueError: If audio file is invalid
        """
        path = Path(audio_path)

        if not path.exists():
            raise ValueError(f"Audio file not found: {audio_path}")

        try:
            # Use ffprobe for metadata
            cmd = [
                "ffprobe",
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(path),
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")

            import json
            info = json.loads(result.stdout)

            # Find audio stream
            audio_stream = None
            for stream in info.get("streams", []):
                if stream.get("codec_type") == "audio":
                    audio_stream = stream
                    break

            format_info = info.get("format", {})

            duration = float(format_info.get("duration", 0))
            sample_rate = int(audio_stream.get("sample_rate", 16000))
            channels = int(audio_stream.get("channels", 1))

            # Determine bit depth
            codec = audio_stream.get("codec_name", "")
            if codec in ("pcm_s16le", "pcm_fltle"):
                bit_depth = 16
            elif codec == "pcm_s32le":
                bit_depth = 32
            else:
                bit_depth = None

            metadata = AudioMetadata(
                duration_seconds=duration,
                sample_rate=sample_rate,
                channels=channels,
                bit_depth=bit_depth,
                bitrate=int(format_info.get("bit_rate", 0)),
                file_size=path.stat().st_size,
            )

            logger.info(
                f"Extracted audio metadata: {path.name}, "
                f"{metadata.duration_seconds:.1f}s, "
                f"{metadata.sample_rate}Hz, {metadata.channels} ch"
            )

            return metadata

        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe timed out")

    async def normalize_audio(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[str, AudioMetadata]:
        """
        Normalize audio to 16kHz, mono, 16-bit PCM format.

        Args:
            audio_path: Path to input audio
            output_path: Path for normalized output

        Returns:
            Tuple of (output_path, AudioMetadata)

        Raises:
            ValueError: If audio file is invalid
        """
        # Validate input
        is_valid, error_msg = self.validate_input(audio_path)
        if not is_valid:
            raise ValueError(error_msg)

        # Generate output path if not provided
        if output_path is None:
            output_path = str(Path(audio_path).with_suffix(".wav"))

        # Build FFmpeg command
        cmd = [
            self.ffmpeg_path,
            "-y",  # Overwrite output
            "-i", audio_path,
            "-ar", str(self.sample_rate),
            "-ac", "1",  # Mono
            "-acodec", "pcm_s16le",
            "-vn",  # No video
            output_path,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {stderr.decode()}")

        except asyncio.CancelledError:
            raise RuntimeError("Audio normalization was cancelled")

        # Get metadata
        metadata = self.get_metadata(output_path)

        logger.info(f"Normalized audio saved to: {output_path}")

        return output_path, metadata

    def normalize_audio_sync(
        self,
        audio_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[str, AudioMetadata]:
        """
        Synchronous version of normalize_audio.

        Args:
            audio_path: Path to input audio
            output_path: Path for normalized output

        Returns:
            Tuple of (output_path, AudioMetadata)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return _run_async_in_executor(self.normalize_audio(audio_path, output_path))
            else:
                return loop.run_until_complete(self.normalize_audio(audio_path, output_path))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(self.normalize_audio(audio_path, output_path))

    def _apply_vad(self, waveform: torch.Tensor, sample_rate: int) -> List[bool]:
        """
        Apply WebRTC VAD to detect speech segments.

        Args:
            waveform: Audio waveform tensor (1, samples)
            sample_rate: Sample rate in Hz

        Returns:
            List of booleans indicating speech for each frame
        """
        vad = self._get_vad()

        if vad is None:
            # Simulate VAD - assume all non-silent audio is speech
            return [True] * (len(waveform[0]) // (sample_rate // 100))

        # Convert to 16-bit PCM
        audio_data = (waveform[0].numpy() * 32767).astype(np.int16)

        # Frame size for VAD (10ms, 20ms, or 30ms)
        frame_duration_ms = 20
        frame_size = int(sample_rate * frame_duration_ms / 1000)

        is_speech = []
        for i in range(0, len(audio_data), frame_size):
            frame = audio_data[i:i + frame_size]
            if len(frame) == frame_size:
                try:
                    speech = vad.is_speech(frame.tobytes(), sample_rate, frame_duration_ms)
                    is_speech.append(speech)
                except Exception:
                    is_speech.append(False)
            else:
                is_speech.append(False)

        return is_speech

    async def segment_audio(
        self,
        audio_path: str,
        normalize: bool = True,
    ) -> List[AudioSegment]:
        """
        Segment audio into fixed-length windows with VAD.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio first

        Returns:
            List of AudioSegment objects

        Raises:
            ValueError: If audio file is invalid
        """
        # Validate input
        is_valid, error_msg = self.validate_input(audio_path)
        if not is_valid:
            raise ValueError(error_msg)

        # Normalize audio if required
        if normalize:
            normalized_path, _ = await self.normalize_audio(audio_path)
        else:
            normalized_path = audio_path

        # Load audio
        try:
            waveform, sample_rate = torchaudio.load(normalized_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load audio: {e}")

        # Ensure mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Apply VAD
        speech_flags = self._apply_vad(waveform, sample_rate)

        # Calculate segment parameters
        segment_samples = int(self.segment_duration * sample_rate)
        hop_samples = int(segment_samples * (1 - self.overlap))

        # Create segments
        segments = []
        num_segments = (waveform.shape[1] - segment_samples) // hop_samples + 1

        for i in range(num_segments):
            start_sample = i * hop_samples
            end_sample = start_sample + segment_samples
            segment_waveform = waveform[:, start_sample:end_sample]

            # Determine if segment contains speech
            start_frame = start_sample // (sample_rate // 100)
            end_frame = end_sample // (sample_rate // 100)
            is_speech = any(speech_flags[start_frame:end_frame])

            start_time = start_sample / sample_rate
            end_time = end_sample / sample_rate

            segments.append(AudioSegment(
                waveform=segment_waveform,
                sample_rate=sample_rate,
                start_time=start_time,
                end_time=end_time,
                is_speech=is_speech,
            ))

        logger.info(
            f"Segmented audio into {len(segments)} segments "
            f"({sum(1 for s in segments if s.is_speech)} with speech)"
        )

        # Clean up normalized file if we created it
        if normalize and normalized_path != audio_path:
            Path(normalized_path).unlink(missing_ok=True)

        return segments

    def segment_audio_sync(
        self,
        audio_path: str,
        normalize: bool = True,
    ) -> List[AudioSegment]:
        """
        Synchronous version of segment_audio.

        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio first

        Returns:
            List of AudioSegment objects
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return _run_async_in_executor(self.segment_audio(audio_path, normalize))
            else:
                return loop.run_until_complete(self.segment_audio(audio_path, normalize))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(self.segment_audio(audio_path, normalize))

    def generate_spectrogram(
        self,
        segment: AudioSegment,
    ) -> torch.Tensor:
        """
        Generate Mel-spectrogram for an audio segment.

        Args:
            segment: AudioSegment to process

        Returns:
            Mel-spectrogram tensor
        """
        # Ensure correct sample rate
        if segment.sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                segment.sample_rate,
                self.sample_rate,
            )
            waveform = resampler(segment.waveform)
        else:
            waveform = segment.waveform

        # Generate Mel spectrogram
        with torch.no_grad():
            spectrogram = self._mel_transform(waveform)

        # Log-scale for better visualization
        spectrogram = torch.log(spectrogram + 1e-9)

        return spectrogram

    def generate_spectrograms(
        self,
        segments: List[AudioSegment],
    ) -> List[torch.Tensor]:
        """
        Generate spectrograms for all segments.

        Args:
            segments: List of AudioSegment objects

        Returns:
            List of spectrogram tensors
        """
        spectrograms = []
        for segment in segments:
            spec = self.generate_spectrogram(segment)
            spectrograms.append(spec)

            # Store in segment for convenience
            segment.spectrogram = spec

        logger.info(f"Generated {len(spectrograms)} spectrograms")

        return spectrograms

    async def process(
        self,
        audio_path: str,
        generate_spectrograms: bool = True,
    ) -> Tuple[List[AudioSegment], AudioMetadata]:
        """
        Complete audio preprocessing pipeline.

        Args:
            audio_path: Path to audio file
            generate_spectrograms: Whether to generate spectrograms

        Returns:
            Tuple of (list of segments, metadata)
        """
        # Get metadata
        metadata = self.get_metadata(audio_path)

        # Segment audio
        segments = await self.segment_audio(audio_path, normalize=True)

        # Generate spectrograms
        if generate_spectrograms:
            self.generate_spectrograms(segments)

        return segments, metadata

    def process_sync(
        self,
        audio_path: str,
        generate_spectrograms: bool = True,
    ) -> Tuple[List[AudioSegment], AudioMetadata]:
        """
        Synchronous version of process.

        Args:
            audio_path: Path to audio file
            generate_spectrograms: Whether to generate spectrograms

        Returns:
            Tuple of (list of segments, metadata)
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return _run_async_in_executor(self.process(audio_path, generate_spectrograms))
            else:
                return loop.run_until_complete(self.process(audio_path, generate_spectrograms))
        except RuntimeError:
            return asyncio.get_event_loop().run_until_complete(self.process(audio_path, generate_spectrograms))


class AudioAnalysisBundle:
    """Bundle containing all processed audio data."""

    def __init__(
        self,
        segments: List[AudioSegment],
        metadata: AudioMetadata,
        spectrograms: Optional[List[torch.Tensor]] = None,
    ):
        self.segments = segments
        self.metadata = metadata
        self.spectrograms = spectrograms or []

        # Filter speech segments
        self.speech_segments = [s for s in segments if s.is_speech]

        # Statistics
        self.total_duration = sum(
            s.end_time - s.start_time for s in segments
        )
        self.speech_duration = sum(
            s.end_time - s.start_time for s in self.speech_segments
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert bundle to dictionary for serialization."""
        return {
            "segments": [s.to_dict() for s in self.segments],
            "metadata": {
                "duration_seconds": self.metadata.duration_seconds,
                "sample_rate": self.metadata.sample_rate,
                "channels": self.metadata.channels,
            },
            "statistics": {
                "total_duration": self.total_duration,
                "speech_duration": self.speech_duration,
                "num_segments": len(self.segments),
                "speech_segments": len(self.speech_segments),
            },
        }


async def process_audio_async(
    audio_path: str,
    sample_rate: int = 16000,
    segment_duration: float = 2.0,
    overlap: float = 0.5,
) -> AudioAnalysisBundle:
    """
    Convenience function for async audio processing.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        segment_duration: Segment length in seconds
        overlap: Overlap ratio

    Returns:
        AudioAnalysisBundle with processed data
    """
    processor = AudioProcessor(
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        overlap=overlap,
    )

    segments, metadata = await processor.process(audio_path)

    return AudioAnalysisBundle(
        segments=segments,
        metadata=metadata,
        spectrograms=[s.spectrogram for s in segments if s.spectrogram is not None],
    )


def process_audio_sync(
    audio_path: str,
    sample_rate: int = 16000,
    segment_duration: float = 2.0,
    overlap: float = 0.5,
) -> AudioAnalysisBundle:
    """
    Convenience function for sync audio processing.

    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        segment_duration: Segment length in seconds
        overlap: Overlap ratio

    Returns:
        AudioAnalysisBundle with processed data
    """
    processor = AudioProcessor(
        sample_rate=sample_rate,
        segment_duration=segment_duration,
        overlap=overlap,
    )

    segments, metadata = processor.process_sync(audio_path)

    return AudioAnalysisBundle(
        segments=segments,
        metadata=metadata,
        spectrograms=[s.spectrogram for s in segments if s.spectrogram is not None],
    )
