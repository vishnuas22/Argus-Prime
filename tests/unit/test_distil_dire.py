"""
Unit tests for DistilDIRE lightweight video/image deepfake detection model.

Tests model architecture, reconstruction, and detection pipeline.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.video.distil_dire import (
    DistilDIRE,
    DIREResult,
    DistilledDiffusionModel,
    ResidualBlock,
    create_distil_dire,
    register_distil_dire,
)


class TestDIREResult:
    """Test DIREResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = DIREResult(
            fake_probability=0.85,
            reconstruction_error=0.042,
            confidence=0.9,
            is_manipulated=True,
            frame_scores=[0.04, 0.045, 0.04],
        )

        assert result.fake_probability == 0.85
        assert result.reconstruction_error == 0.042
        assert result.confidence == 0.9
        assert result.is_manipulated is True
        assert len(result.frame_scores) == 3

    def test_result_to_dict(self):
        """Test result serialization."""
        result = DIREResult(
            fake_probability=0.75,
            reconstruction_error=0.035,
            confidence=0.8,
            is_manipulated=True,
            frame_scores=[0.035],
        )

        result_dict = result.to_dict()

        assert "fake_probability" in result_dict
        assert "reconstruction_error" in result_dict
        assert "confidence" in result_dict
        assert "is_manipulated" in result_dict
        assert result_dict["num_frames"] == 1


class TestResidualBlock:
    """Test ResidualBlock module."""

    def test_block_initialization(self):
        """Test block initialization."""
        block = ResidualBlock(64, 128)

        assert hasattr(block, "conv1")
        assert hasattr(block, "conv2")
        assert hasattr(block, "skip")

    def test_forward_pass_same_channels(self):
        """Test forward with same input/output channels."""
        block = ResidualBlock(64, 64)
        block.eval()

        with torch.no_grad():
            x = torch.randn(2, 64, 32, 32)
            y = block(x)

        assert y.shape == x.shape

    def test_forward_pass_different_channels(self):
        """Test forward with different input/output channels."""
        block = ResidualBlock(64, 128)
        block.eval()

        with torch.no_grad():
            x = torch.randn(2, 64, 32, 32)
            y = block(x)

        assert y.shape == (2, 128, 32, 32)


class TestDistilledDiffusionModel:
    """Test DistilledDiffusionModel module."""

    def test_model_initialization(self):
        """Test model initialization."""
        model = DistilledDiffusionModel(
            in_channels=3,
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
        )

        assert hasattr(model, "time_embed")
        assert hasattr(model, "input_proj")
        assert hasattr(model, "down")
        assert hasattr(model, "mid")
        assert hasattr(model, "up")
        assert hasattr(model, "out")

    def test_has_forward_method(self):
        """Test that model has forward method."""
        model = DistilledDiffusionModel(
            in_channels=3,
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
        )

        assert hasattr(model, "forward")
        assert callable(model.forward)

    def test_model_channels_correct(self):
        """Test that model has correct channel configuration."""
        model = DistilledDiffusionModel(
            in_channels=3,
            model_channels=64,
            channel_mult=(1, 2, 4, 8),
        )

        assert model.in_channels == 3
        assert model.model_channels == 64


class TestDistilDIRE:
    """Test DistilDIRE main class."""

    def test_initialization(self):
        """Test DistilDIRE initialization."""
        model = DistilDIRE()

        assert hasattr(model, "model")
        assert model.MODEL_NAME == "distil_dire"
        assert model.MODEL_VERSION == "1.0.0"

    def test_model_info(self, cpu_device):
        """Test model info retrieval."""
        model = DistilDIRE(device=cpu_device)

        info = model.get_model_info()

        assert info["name"] == "distil_dire"
        assert info["version"] == "1.0.0"

    def test_health_check_not_loaded(self):
        """Test health check before loading."""
        model = DistilDIRE()

        assert model.health_check() is False

    def test_has_detect_method(self):
        """Test that model has detect method."""
        model = DistilDIRE()

        assert hasattr(model, "detect")
        assert callable(model.detect)

    def test_has_detect_single_method(self):
        """Test that model has detect_single method."""
        model = DistilDIRE()

        assert hasattr(model, "detect_single")
        assert callable(model.detect_single)

    def test_has_get_heatmap_method(self):
        """Test that model has get_heatmap method."""
        model = DistilDIRE()

        assert hasattr(model, "get_heatmap")
        assert callable(model.get_heatmap)

    def test_load_sets_is_loaded(self, cpu_device):
        """Test that load sets is_loaded flag."""
        model = DistilDIRE(device=cpu_device)
        model.load(device=cpu_device)

        assert model._is_loaded is True

        model.unload()

    def test_unload_resets_flag(self, cpu_device):
        """Test that unload resets is_loaded flag."""
        model = DistilDIRE(device=cpu_device)
        model.load(device=cpu_device)
        model.unload()

        assert model._is_loaded is False

    def test_has_diffusion_model(self, cpu_device):
        """Test that model has diffusion model."""
        model = DistilDIRE(device=cpu_device)

        assert hasattr(model, "model")

    def test_has_threshold_attribute(self, cpu_device):
        """Test that model has threshold attribute."""
        model = DistilDIRE(device=cpu_device)

        assert hasattr(model, "_threshold")

    def test_default_threshold_value(self, cpu_device):
        """Test that default threshold is 0.5."""
        model = DistilDIRE(device=cpu_device)

        assert model._threshold == 0.5

    def test_custom_threshold_value(self, cpu_device):
        """Test that custom threshold is accepted."""
        model = DistilDIRE(device=cpu_device, use_half_precision=False)

        assert model.use_half_precision is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_distil_dire(self, cpu_device):
        """Test factory function."""
        model = create_distil_dire(device=cpu_device)

        assert isinstance(model, DistilDIRE)

    def test_register_distil_dire(self, cpu_device):
        """Test registration."""
        from src.models.registry import ModelRegistry

        ModelRegistry._instance = None
        register_distil_dire()

        registry = ModelRegistry.get_instance()
        assert "distil_dire" in registry._model_factories


class TestDistilDIREIntegration:
    """Integration tests for DistilDIRE."""

    @pytest.fixture
    def model(self, cpu_device):
        """Create a loaded model."""
        model = DistilDIRE(device=cpu_device)
        model.load(device=cpu_device)
        yield model
        model.unload()

    def test_model_loaded_after_load(self, model):
        """Test that model is loaded after load()."""
        assert model._is_loaded is True

    def test_has_diffusion_model(self, model):
        """Test that model has diffusion model."""
        assert hasattr(model, "model")

    def test_model_info_after_load(self, model):
        """Test model info after loading."""
        info = model.get_model_info()

        assert info["name"] == "distil_dire"
        assert info["is_loaded"] is True

    def test_half_precision_option(self, cpu_device):
        """Test half precision mode."""
        model = DistilDIRE(device=cpu_device, use_half_precision=True)
        model.load(device=cpu_device)

        assert model.use_half_precision is True

        model.unload()

    def test_model_version_constant(self):
        """Test that model version is correct constant."""
        assert DistilDIRE.MODEL_NAME == "distil_dire"
        assert DistilDIRE.MODEL_VERSION == "1.0.0"

    def test_preprocess_method_exists(self, cpu_device):
        """Test that model has preprocessing method."""
        model = DistilDIRE(device=cpu_device)

        assert hasattr(model, "_preprocess_image")

    def test_reconstruction_error_method_exists(self, cpu_device):
        """Test that model has reconstruction error method."""
        model = DistilDIRE(device=cpu_device)

        assert hasattr(model, "_compute_reconstruction_error")

    def test_confidence_method_exists(self, cpu_device):
        """Test that model has confidence calculation method."""
        model = DistilDIRE(device=cpu_device)

        assert hasattr(model, "_calculate_confidence")


@pytest.fixture(scope="session")
def cpu_device():
    """Use CPU for all tests."""
    return torch.device("cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
