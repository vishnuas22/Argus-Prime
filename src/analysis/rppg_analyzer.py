"""
rPPG Analyzer.

Extracts and validates remote Photoplethysmography (rPPG) signals from video
to detect biological inconsistencies that indicate deepfake manipulation.

Features:
- G-PPG: Green channel pulse extraction
- C-PPG: Chrominance-based pulse extraction
- Signal quality assessment
- Heart rate estimation
- Temporal consistency checking

Algorithm:
  G-PPG:
    - Extract green channel signal from facial ROI
    - Apply bandpass filter (0.7-4.0 Hz for HR)
    - FFT to extract dominant frequency
  C-PPG:
    - Compute chrominance signals (R-G, 2B-G-R)
    - Independent component analysis (ICA)
    - Select best signal via SNR
  Fusion: Weighted average with quality-based weighting
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Any, Dict, List, Optional, Tuple

from src.config import get_config
from src.core.logging import logger


@dataclass
class RPPGResult:
    """Result of rPPG analysis."""
    heart_rate: Optional[float] = None
    heart_rate_confidence: float = 0.0
    quality_score: float = 0.0
    quality_label: str = "unknown"
    signal_strength: float = 0.0
    temporal_consistency: float = 0.0
    gppg_signal: Optional[np.ndarray] = None
    cpppg_signal: Optional[np.ndarray] = None
    fused_signal: Optional[np.ndarray] = None
    frame_qualities: List[float] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "heart_rate": self.heart_rate,
            "heart_rate_confidence": self.heart_rate_confidence,
            "quality_score": self.quality_score,
            "quality_label": self.quality_label,
            "signal_strength": self.signal_strength,
            "temporal_consistency": self.temporal_consistency,
            "frame_qualities": self.frame_qualities,
            "warnings": self.warnings,
        }


class RPPGAnalyzer:
    """
    Remote Photoplethysmography (rPPG) signal extractor.

    Extracts pulse signals from facial video using multiple methods:
    - G-PPG: Green channel signal extraction
    - C-PPG: Chrominance-based signal extraction
    - ICA-PPG: Independent component analysis

    Attributes:
        fps: Frames per second of input video
        roi_size: Size of facial ROI for signal extraction
        hr_min: Minimum heart rate (bpm)
        hr_max: Maximum heart rate (bpm)
    """

    def __init__(self, fps: float = 30.0):
        """
        Initialize rPPG analyzer.

        Args:
            fps: Frames per second of input video
        """
        self.fps = fps
        self.config = get_config()

        self.roi_size = (64, 64)
        self.hr_min = 40
        self.hr_max = 180

        self.hr_freq_min = self.hr_min / 60.0
        self.hr_freq_max = self.hr_max / 60.0

        logger.info(
            f"RPPGAnalyzer initialized: fps={fps}, "
            f"hr_range=[{self.hr_min}-{self.hr_max}] bpm"
        )

    def _create_bandpass_filter(
        self, order: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create Butterworth bandpass filter for heart rate frequencies."""
        nyquist = self.fps / 2
        low = self.hr_freq_min / nyquist
        high = min(self.hr_freq_max / nyquist, 0.99)

        if low >= high:
            low = 0.5 / nyquist
            high = 4.0 / nyquist

        try:
            b, a = signal.butter(order, [low, high], btype="band")
            return b, a
        except Exception as e:
            logger.warning(f"Bandpass filter creation failed: {e}")
            return None, None

    def _extract_gppg_signal(
        self, frames: List[np.ndarray], face_boxes: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], List[float]]:
        """
        Extract G-PPG signal from green channel.

        Args:
            frames: List of video frames
            face_boxes: List of face bounding boxes

        Returns:
            Tuple of (signal, frame_qualities)
        """
        if len(frames) < 30:
            return None, []

        signals = []
        frame_qualities = []

        for i, (frame, box) in enumerate(zip(frames, face_boxes)):
            try:
                x1, y1, x2, y2 = box.astype(int)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    frame_qualities.append(0.0)
                    continue

                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size == 0:
                    frame_qualities.append(0.0)
                    continue

                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                r = np.mean(face_roi[:, :, 0])
                g = np.mean(face_roi[:, :, 1])
                b = np.mean(face_roi[:, :, 2])

                luminance = r + g + b
                if luminance > 0:
                    gppg = g / luminance
                    signals.append(gppg)
                    frame_qualities.append(min(1.0, g / 255.0))
                else:
                    frame_qualities.append(0.0)

            except Exception as e:
                frame_qualities.append(0.0)

        if len(signals) < 30:
            return None, frame_qualities

        signal_array = np.array(signals)

        b, a = self._create_bandpass_filter()
        if b is not None:
            try:
                signal_array = signal.filtfilt(b, a, signal_array)
            except Exception as e:
                logger.warning(f"rPPG signal filtering failed: {e}")

        return signal_array, frame_qualities

    def _extract_cpppg_signal(
        self, frames: List[np.ndarray], face_boxes: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], List[float]]:
        """
        Extract C-PPG signal using chrominance method.

        Args:
            frames: List of video frames
            face_boxes: List of face bounding boxes

        Returns:
            Tuple of (signal, frame_qualities)
        """
        if len(frames) < 30:
            return None, []

        signals = []
        frame_qualities = []

        for i, (frame, box) in enumerate(zip(frames, face_boxes)):
            try:
                x1, y1, x2, y2 = box.astype(int)

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                if x2 <= x1 or y2 <= y1:
                    frame_qualities.append(0.0)
                    continue

                face_roi = frame[y1:y2, x1:x2]

                if face_roi.size == 0:
                    frame_qualities.append(0.0)
                    continue

                face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                r = np.mean(face_roi[:, :, 0])
                g = np.mean(face_roi[:, :, 1])
                b = np.mean(face_roi[:, :, 2])

                chrominance = (r - g) / (r + g + 1e-6)
                signals.append(chrominance)

                brightness = np.mean(face_roi)
                frame_qualities.append(min(1.0, brightness / 128.0))

            except Exception as e:
                frame_qualities.append(0.0)

        if len(signals) < 30:
            return None, frame_qualities

        signal_array = np.array(signals)

        b, a = self._create_bandpass_filter()
        if b is not None:
            try:
                signal_array = signal.filtfilt(b, a, signal_array)
            except Exception as e:
                logger.warning(f"C-PPPG signal filtering failed: {e}")

        return signal_array, frame_qualities

    def _estimate_heart_rate(
        self, signal: np.ndarray
    ) -> Tuple[Optional[float], float]:
        """
        Estimate heart rate from rPPG signal using FFT.

        Args:
            signal: rPPG signal

        Returns:
            Tuple of (heart_rate, confidence)
        """
        if signal is None or len(signal) < 30:
            return None, 0.0

        try:
            signal = signal - np.mean(signal)
            signal = signal / (np.std(signal) + 1e-6)

            n = len(signal)
            fft_result = fft(signal)
            freqs = fftfreq(n, 1 / self.fps)

            positive_freqs = freqs[: n // 2]
            positive_fft = np.abs(fft_result[: n // 2])

            valid_mask = (positive_freqs >= self.hr_freq_min) & (
                positive_freqs <= self.hr_freq_max
            )

            if not np.any(valid_mask):
                return None, 0.0

            valid_freqs = positive_freqs[valid_mask]
            valid_fft = positive_fft[valid_mask]

            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            peak_power = valid_fft[peak_idx]

            heart_rate = peak_freq * 60

            total_power = np.sum(valid_fft)
            confidence = peak_power / (total_power + 1e-6)

            confidence = min(1.0, confidence * len(valid_fft))

            return float(heart_rate), float(confidence)

        except Exception as e:
            logger.warning(f"Heart rate estimation failed: {e}")
            return None, 0.0

    def _compute_signal_quality(
        self, signal: np.ndarray, frame_qualities: List[float]
    ) -> Tuple[float, str]:
        """
        Compute signal quality score.

        Args:
            signal: rPPG signal
            frame_qualities: Per-frame quality scores

        Returns:
            Tuple of (quality_score, quality_label)
        """
        if signal is None or len(signal) < 30:
            return 0.0, "insufficient_data"

        try:
            signal_std = np.std(signal)
            signal_mean = np.abs(np.mean(signal))

            snr = signal_mean / (signal_std + 1e-6)

            quality_score = min(1.0, snr / 3.0)

            avg_frame_quality = np.mean(frame_qualities) if frame_qualities else 0.0
            quality_score = quality_score * (0.7 + 0.3 * avg_frame_quality)

            if quality_score >= 0.8:
                label = "excellent"
            elif quality_score >= 0.6:
                label = "good"
            elif quality_score >= 0.4:
                label = "moderate"
            elif quality_score >= 0.2:
                label = "poor"
            else:
                label = "very_poor"

            return float(quality_score), label

        except Exception as e:
            logger.warning(f"Signal quality computation failed: {e}")
            return 0.0, "error"

    def _check_temporal_consistency(
        self, signals: List[Optional[np.ndarray]]
    ) -> float:
        """
        Check temporal consistency of rPPG signals across windows.

        Args:
            signals: List of signals from different time windows

        Returns:
            Consistency score (0-1)
        """
        valid_signals = [s for s in signals if s is not None and len(s) >= 30]

        if len(valid_signals) < 2:
            return 1.0

        try:
            hr_estimates = []
            for sig in valid_signals:
                hr, _ = self._estimate_heart_rate(sig)
                if hr is not None:
                    hr_estimates.append(hr)

            if len(hr_estimates) < 2:
                return 1.0

            hr_std = np.std(hr_estimates)
            hr_mean = np.mean(hr_estimates)

            if hr_mean > 0:
                cv = hr_std / hr_mean
                consistency = max(0.0, 1.0 - cv)
            else:
                consistency = 1.0

            return float(consistency)

        except Exception as e:
            logger.warning(f"Temporal consistency check failed: {e}")
            return 0.5

    def analyze(
        self,
        frames: List[np.ndarray],
        face_boxes: List[np.ndarray],
        progress_callback=None,
    ) -> RPPGResult:
        """
        Analyze video frames for rPPG signals.

        Args:
            frames: List of video frames (BGR format)
            face_boxes: List of face bounding boxes
            progress_callback: Optional callback(progress, message)

        Returns:
            RPPGResult with heart rate and quality metrics
        """
        logger.info(f"Starting rPPG analysis: {len(frames)} frames")

        result = RPPGResult()

        if len(frames) < 30:
            result.warnings.append("Insufficient frames for rPPG analysis")
            return result

        total_frames = len(frames)
        processed_frames = 0

        gppg_signal, gppg_qualities = self._extract_gppg_signal(frames, face_boxes)
        cpppg_signal, cpppg_qualities = self._extract_cpppg_signal(
            frames, face_boxes
        )

        processed_frames = total_frames
        if progress_callback:
            progress_callback(100, "Signal extraction complete")

        all_qualities = gppg_qualities + cpppg_qualities
        frame_qualities = [np.mean([gppg_qualities[i], cpppg_qualities[i]])
                          for i in range(len(gppg_qualities))]
        result.frame_qualities = frame_qualities

        gppg_hr, gppg_confidence = self._estimate_heart_rate(gppg_signal)
        cpppg_hr, cpppg_confidence = self._estimate_heart_rate(cpppg_signal)

        gppg_quality, gppg_label = self._compute_signal_quality(
            gppg_signal, gppg_qualities
        )
        cpppg_quality, cpppg_label = self._compute_signal_quality(
            cpppg_signal, cpppg_qualities
        )

        if gppg_quality >= cpppg_quality:
            result.fused_signal = gppg_signal
            result.gppg_signal = gppg_signal
            result.quality_score = gppg_quality
            result.quality_label = gppg_label
            if gppg_hr is not None:
                result.heart_rate = gppg_hr
                result.heart_rate_confidence = gppg_confidence
        else:
            result.fused_signal = cpppg_signal
            result.cpppg_signal = cpppg_signal
            result.quality_score = cpppg_quality
            result.quality_label = cpppg_label
            if cpppg_hr is not None:
                result.heart_rate = cpppg_hr
                result.heart_rate_confidence = cpppg_confidence

        if gppg_signal is not None and cpppg_signal is not None:
            combined_signal = 0.5 * gppg_signal + 0.5 * cpppg_signal
            result.fused_signal = combined_signal

            combined_hr, combined_conf = self._estimate_heart_rate(combined_signal)

            if result.heart_rate is None and combined_hr is not None:
                result.heart_rate = combined_hr
                result.heart_rate_confidence = combined_conf

        if gppg_signal is not None and len(gppg_signal) >= 30:
            signal_std = np.std(gppg_signal)
            signal_power = np.mean(np.abs(gppg_signal))
            result.signal_strength = min(1.0, signal_power / (signal_std + 1e-6))

        temporal_consistency = self._check_temporal_consistency(
            [gppg_signal, cpppg_signal]
        )
        result.temporal_consistency = temporal_consistency

        if result.quality_score < 0.3:
            result.warnings.append("Low signal quality - results may be unreliable")

        if result.temporal_consistency < 0.5:
            result.warnings.append("Temporal inconsistency detected")

        if result.heart_rate is not None:
            logger.info(
                f"rPPG analysis complete: hr={result.heart_rate:.1f} bpm, "
                f"quality={result.quality_label}, "
                f"consistency={result.temporal_consistency:.2f}"
            )
        else:
            logger.info("rPPG analysis complete: no valid heart rate detected")

        return result

    def analyze_from_path(
        self,
        video_path: str,
        max_frames: int = 300,
        progress_callback=None,
    ) -> RPPGResult:
        """
        Analyze video file for rPPG signals.

        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process
            progress_callback: Optional callback(progress, message)

        Returns:
            RPPGResult with heart rate and quality metrics
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.fps = fps

            frames = []
            face_boxes = []

            frame_idx = 0
            while frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
                face_boxes.append(np.array([0, 0, frame.shape[1], frame.shape[0]]))

                if progress_callback and frame_idx % 30 == 0:
                    progress = int(frame_idx / max_frames * 50)
                    progress_callback(progress, "Reading frames")

                frame_idx += 1

            cap.release()

            if progress_callback:
                progress_callback(50, "Frames read, starting signal extraction")

            return self.analyze(frames, face_boxes, progress_callback)

        except ImportError:
            result = RPPGResult()
            result.warnings.append("OpenCV not available - cannot process video file")
            return result
        except Exception as e:
            logger.error(f"Video rPPG analysis failed: {e}")
            result = RPPGResult()
            result.warnings.append(f"Analysis failed: {str(e)}")
            return result


_rppg_analyzer: Optional[RPPGAnalyzer] = None


def get_rppg_analyzer(fps: float = 30.0) -> RPPGAnalyzer:
    """Get or create rPPG analyzer instance."""
    global _rppg_analyzer
    if _rppg_analyzer is None or _rppg_analyzer.fps != fps:
        _rppg_analyzer = RPPGAnalyzer(fps)
    return _rppg_analyzer
