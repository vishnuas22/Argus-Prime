"""
Unit tests for KAN Detector - CPU-Only Deepfake Detection.
"""

import pytest
import torch
import numpy as np

from src.models.video.kan_detector import (
    KANDetector,
    KANVideoDetector,
    KANResult,
    KANLinear,
    KANConv2d,
)


class TestKANLinear:
    """Test KAN Linear layer."""

    def test_initialization(self):
        """Test KAN linear layer initialization."""
        layer = KANLinear(
            in_features=10,
            out_features=5,
            grid_size=5,
            spline_order=3,
        )

        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.grid_size == 5
        assert layer.spline_order == 3

    def test_forward_pass(self):
        """Test forward pass of KAN linear layer."""
        layer = KANLinear(in_features=10, out_features=5, grid_size=4)

        x = torch.randn(2, 10)
        output = layer(x)

        assert output.shape == (2, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_parameter_count(self):
        """Test that layer has learnable parameters."""
        layer = KANLinear(in_features=10, out_features=5, grid_size=4)

        params = list(layer.parameters())
        assert len(params) >= 2


class TestKANConv2d:
    """Test KAN 2D Convolutional layer."""

    def test_initialization(self):
        """Test KAN conv layer initialization."""
        layer = KANConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            grid_size=4,
        )

        assert layer.in_channels == 3
        assert layer.out_channels == 64

    def test_forward_pass(self):
        """Test forward pass of KAN conv layer."""
        layer = KANConv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1,
            grid_size=4,
        )

        x = torch.randn(2, 3, 32, 32)
        output = layer(x)

        assert output.shape == (2, 64, 32, 32)
        assert not torch.isnan(output).any()


class TestKANDetector:
    """Test KAN Detector model."""

    def test_initialization(self):
        """Test KAN detector initialization."""
        model = KANDetector(
            in_channels=3,
            num_classes=2,
            grid_size=5,
        )

        assert model.in_channels == 3
        assert model.num_classes == 2
        assert model.grid_size == 5

    def test_forward_pass(self):
        """Test forward pass of KAN detector."""
        model = KANDetector(
            in_channels=3,
            num_classes=2,
            grid_size=4,
        )

        x = torch.randn(2, 3, 224, 224)
        outputs = model(x)

        assert "logits" in outputs
        assert "features" in outputs
        assert outputs["logits"].shape == (2, 2)
        assert outputs["features"].shape[0] == 2

    def test_predict(self):
        """Test prediction method."""
        model = KANDetector(
            in_channels=3,
            num_classes=2,
            grid_size=4,
        )

        x = torch.randn(2, 3, 224, 224)
        preds, probs = model.predict(x)

        assert preds.shape == (2,)
        assert probs.shape == (2, 2)
        assert (probs.sum(dim=-1) - 1.0).abs().max() < 1e-5

    def test_parameter_count(self):
        """Test model parameter count."""
        model = KANDetector(
            in_channels=3,
            num_classes=2,
            grid_size=4,
        )

        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0


class TestKANVideoDetector:
    """Test KAN Video Detector wrapper."""

    def test_initialization(self):
        """Test video detector initialization."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        assert detector.threshold == 0.5
        assert detector.device.type in ["cpu", "cuda"]

    def test_detect_single_frame(self):
        """Test detection of single frame."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        frame = torch.randn(3, 224, 224)
        result = detector.detect([frame])

        assert isinstance(result.is_manipulated, bool)
        assert 0 <= result.confidence <= 1
        assert 0 <= result.overall_score <= 1

    def test_detect_multiple_frames(self):
        """Test detection of multiple frames."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        frames = [torch.randn(3, 224, 224) for _ in range(5)]
        result = detector.detect(frames)

        assert isinstance(result.is_manipulated, bool)
        assert 0 <= result.confidence <= 1
        assert 0 <= result.overall_score <= 1

    def test_detect_empty_frames(self):
        """Test detection with empty frames."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        result = detector.detect([])

        assert result.is_manipulated is False
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_get_heatmap(self):
        """Test heatmap generation."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        frames = [torch.randn(3, 224, 224) for _ in range(3)]
        heatmap = detector.get_heatmap(frames)

        assert heatmap is not None
        assert heatmap.shape == (224, 224)
        assert 0 <= heatmap.min() <= 1
        assert 0 <= heatmap.max() <= 1

    def test_get_model_info(self):
        """Test model info retrieval."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        info = detector.get_model_info()

        assert info["name"] == "kan_detector"
        assert info["type"] == "video_detector"
        assert info["parameters"] > 0
        assert info["threshold"] == 0.5
        assert info["grid_size"] == 4


class TestKANResult:
    """Test KAN Result dataclass."""

    def test_result_creation(self):
        """Test KAN result creation."""
        result = KANResult(
            fake_probability=0.7,
            manipulation_score=0.7,
            confidence=0.8,
            is_manipulated=True,
            frame_scores=[0.6, 0.7, 0.8],
        )

        assert result.fake_probability == 0.7
        assert result.is_manipulated is True

    def test_result_to_dict(self):
        """Test KAN result serialization."""
        result = KANResult(
            fake_probability=0.7,
            manipulation_score=0.7,
            confidence=0.8,
            is_manipulated=True,
            frame_scores=[0.6, 0.7, 0.8],
            metadata={"test": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["fake_probability"] == 0.7
        assert result_dict["is_manipulated"] is True
        assert result_dict["num_frames"] == 3
        assert result_dict["metadata"]["test"] == "value"


class TestKANConfidence:
    """Test confidence computation."""

    def test_confidence_range(self):
        """Test that confidence is in valid range."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        frames = [torch.randn(3, 224, 224) for _ in range(10)]
        result = detector.detect(frames)

        assert 0 <= result.confidence <= 1

    def test_single_frame_confidence(self):
        """Test confidence with single frame."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        frame = torch.randn(3, 224, 224)
        result = detector.detect([frame])

        assert result.confidence == 0.5

    def test_multiple_frames_variance_confidence(self):
        """Test confidence computation with different variance."""
        detector = KANVideoDetector(
            device=torch.device("cpu"),
            threshold=0.5,
            grid_size=4,
        )

        low_variance_frames = [torch.ones(3, 224, 224) * 0.5 for _ in range(5)]
        result_low = detector.detect(low_variance_frames)

        high_variance_frames = [
            torch.randn(3, 224, 224) * 0.5 + 0.5 for _ in range(5)
        ]
        result_high = detector.detect(high_variance_frames)

        assert result_low.confidence >= result_high.confidence


class TestKANDeviceCompatibility:
    """Test device compatibility."""

    def test_cpu_execution(self):
        """Test that model runs on CPU."""
        model = KANDetector(
            in_channels=3,
            num_classes=2,
            grid_size=4,
        )
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        outputs = model(x)

        assert not torch.isnan(outputs["logits"]).any()
        assert not torch.isinf(outputs["logits"]).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
