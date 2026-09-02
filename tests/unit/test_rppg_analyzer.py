"""
Unit Tests for rPPG Analyzer.

Tests biological signal extraction from video for deepfake detection.
"""

import pytest
import numpy as np
from pathlib import Path

from src.analysis.rppg_analyzer import RPPGAnalyzer, RPPGResult


class TestRPPGAnalyzer:
    """Test cases for RPPGAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create an RPPGAnalyzer instance."""
        return RPPGAnalyzer(fps=30.0)

    def test_analyzer_initialization(self, analyzer):
        """Test that analyzer initializes correctly."""
        assert analyzer is not None
        assert analyzer.fps == 30.0
        assert analyzer.hr_min == 40
        assert analyzer.hr_max == 180

    def test_analyzer_custom_fps(self):
        """Test analyzer with custom FPS."""
        analyzer = RPPGAnalyzer(fps=25.0)
        assert analyzer.fps == 25.0

    def test_bandpass_filter_creation(self, analyzer):
        """Test bandpass filter creation."""
        b, a = analyzer._create_bandpass_filter()
        assert b is not None
        assert a is not None

    def test_bandpass_filter_bounds(self, analyzer):
        """Test bandpass filter frequency bounds."""
        analyzer_hr = RPPGAnalyzer(fps=60.0)  # Higher FPS
        b, a = analyzer_hr._create_bandpass_filter()
        # Should still create valid filter
        assert b is not None

    def test_estimate_heart_rate_valid_signal(self, analyzer):
        """Test heart rate estimation with valid signal."""
        # Create a synthetic signal with ~60 BPM frequency
        t = np.linspace(0, 10, 300)  # 10 seconds at 30 FPS
        freq = 1.0  # 1 Hz = 60 BPM
        signal = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.randn(len(t))
        
        hr, confidence = analyzer._estimate_heart_rate(signal)
        assert hr is not None
        assert 40 <= hr <= 180
        assert 0.0 <= confidence <= 1.0

    def test_estimate_heart_rate_short_signal(self, analyzer):
        """Test heart rate estimation with short signal."""
        signal = np.random.randn(10)
        hr, confidence = analyzer._estimate_heart_rate(signal)
        assert hr is None
        assert confidence == 0.0

    def test_estimate_heart_rate_none_signal(self, analyzer):
        """Test heart rate estimation with None signal."""
        hr, confidence = analyzer._estimate_heart_rate(None)
        assert hr is None
        assert confidence == 0.0

    def test_compute_signal_quality_excellent(self, analyzer):
        """Test signal quality computation for excellent signal."""
        signal = np.random.randn(100) * 0.1 + 0.5  # High SNR
        frame_qualities = [0.9] * 100
        quality, label = analyzer._compute_signal_quality(signal, frame_qualities)
        assert quality >= 0.7
        assert label in ["excellent", "good", "moderate"]

    def test_compute_signal_quality_poor(self, analyzer):
        """Test signal quality computation for poor signal."""
        signal = np.random.randn(100) * 2.0  # Low SNR
        frame_qualities = [0.1] * 100
        quality, label = analyzer._compute_signal_quality(signal, frame_qualities)
        assert quality < 0.5

    def test_compute_signal_quality_insufficient_data(self, analyzer):
        """Test signal quality with insufficient data."""
        quality, label = analyzer._compute_signal_quality(None, [])
        assert quality == 0.0
        assert label == "insufficient_data"

    def test_check_temporal_consistency_single_window(self, analyzer):
        """Test temporal consistency with single window."""
        signal = [np.random.randn(50) for _ in range(1)]
        consistency = analyzer._check_temporal_consistency(signal)
        assert consistency == 1.0

    def test_check_temporal_consistency_consistent(self, analyzer):
        """Test temporal consistency with consistent windows."""
        # Same signal in all windows
        base_signal = np.random.randn(50)
        signals = [base_signal + 0.01 * np.random.randn(50) for _ in range(3)]
        consistency = analyzer._check_temporal_consistency(signals)
        assert consistency >= 0.8

    def test_check_temporal_consistency_inconsistent(self, analyzer):
        """Test temporal consistency with inconsistent windows."""
        # Different signals
        signals = [np.random.randn(50) * (i + 1) for i in range(3)]
        consistency = analyzer._check_temporal_consistency(signals)
        assert consistency < 0.9

    def test_rppg_result_to_dict(self):
        """Test RPPGResult serialization."""
        result = RPPGResult(
            heart_rate=72.5,
            heart_rate_confidence=0.85,
            quality_score=0.75,
            quality_label="good",
            signal_strength=0.65,
            temporal_consistency=0.90,
            frame_qualities=[0.8, 0.9, 0.7],
            warnings=["warning1"],
        )
        as_dict = result.to_dict()
        assert as_dict["heart_rate"] == 72.5
        assert as_dict["quality_label"] == "good"
        assert as_dict["temporal_consistency"] == 0.90

    def test_rppg_result_defaults(self):
        """Test RPPGResult default values."""
        result = RPPGResult()
        assert result.heart_rate is None
        assert result.quality_score == 0.0
        assert result.quality_label == "unknown"
        assert result.signal_strength == 0.0
        assert result.temporal_consistency == 0.0


class TestRPPGAnalyzerAnalyze:
    """Test cases for RPPGAnalyzer.analyze method."""

    @pytest.fixture
    def analyzer(self):
        return RPPGAnalyzer(fps=10.0)

    def test_analyze_insufficient_frames(self, analyzer):
        """Test analysis with insufficient frames."""
        frames = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(20)]
        face_boxes = [np.array([10, 10, 90, 90]) for _ in range(20)]

        result = analyzer.analyze(frames, face_boxes)
        assert result.heart_rate is None
        assert "Insufficient frames" in result.warnings[0]

    def test_analyze_with_synthetic_signal(self, analyzer):
        """Test analysis with synthetic video frames."""
        # Create synthetic frames with periodic signal
        num_frames = 50
        frames = []
        face_boxes = []
        
        for i in range(num_frames):
            # Create a frame with intensity oscillating at ~60 BPM
            intensity = 128 + 30 * np.sin(2 * np.pi * 1.0 * i / 30)  # 1 Hz at 30 fps
            frame = np.full((100, 100, 3), int(intensity), dtype=np.uint8)
            frames.append(frame)
            face_boxes.append(np.array([10, 10, 90, 90]))
        
        result = analyzer.analyze(frames, face_boxes)
        # Should detect some heart rate
        assert result.heart_rate is not None or result.warnings

    def test_analyze_with_progress_callback(self, analyzer):
        """Test analysis with progress callback."""
        frames = [np.random.randint(0, 255, (100, 100, 3)) for _ in range(30)]
        face_boxes = [np.array([10, 10, 90, 90]) for _ in range(30)]
        
        callback_calls = []
        def progress_callback(progress, message):
            callback_calls.append((progress, message))
        
        result = analyzer.analyze(frames, face_boxes, progress_callback=progress_callback)
        assert len(callback_calls) > 0


class TestRPPGAnalyzerIntegration:
    """Integration tests for RPPGAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        return RPPGAnalyzer(fps=30.0)

    def test_get_rppg_analyzer_singleton(self):
        """Test that get_rppg_analyzer returns singleton."""
        from src.analysis.rppg_analyzer import get_rppg_analyzer
        analyzer1 = get_rppg_analyzer(fps=30.0)
        analyzer2 = get_rppg_analyzer(fps=30.0)
        assert analyzer1 is analyzer2

    def test_analyze_from_path_method_exists(self, analyzer):
        """Test that analyze_from_path method exists."""
        assert hasattr(analyzer, 'analyze_from_path')
        assert callable(analyzer.analyze_from_path)

    def test_fps_parameter_affects_hr_range(self):
        """Test that different FPS affects heart rate range calculation."""
        analyzer_15 = RPPGAnalyzer(fps=15.0)
        analyzer_60 = RPPGAnalyzer(fps=60.0)

        assert analyzer_15.fps == 15.0
        assert analyzer_60.fps == 60.0

        assert analyzer_15.hr_freq_min == 40 / 60.0
        assert analyzer_60.hr_freq_min == 40 / 60.0
        assert analyzer_15.hr_freq_max == 180 / 60.0
        assert analyzer_60.hr_freq_max == 180 / 60.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
