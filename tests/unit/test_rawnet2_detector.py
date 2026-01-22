"""
Unit tests for RawNet2 audio deepfake detection model.

Tests model architecture, sinc convolution, GRU blocks, and detection pipeline.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.audio.rawnet2_detector import (
    RawNet2,
    RawNet2Result,
    VOCODER_CLASSES,
    SincConv,
    ResidualGRUBlock,
    create_rawnet2,
    register_rawnet2,
)


class TestRawNet2Result:
    """Test RawNet2Result dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = RawNet2Result(
            fake_probability=0.85,
            vocoder_class="hifi_gan",
            vocoder_confidence=0.92,
            artifact_score=0.78,
            confidence=0.9,
            is_manipulated=True,
            segment_scores=[
                {"segment_idx": 0, "fake_probability": 0.85, "vocoder_class": "hifi_gan"},
                {"segment_idx": 1, "fake_probability": 0.88, "vocoder_class": "waveglow"},
            ],
        )

        assert result.fake_probability == 0.85
        assert result.vocoder_class == "hifi_gan"
        assert result.vocoder_confidence == 0.92
        assert result.is_manipulated is True
        assert len(result.segment_scores) == 2

    def test_result_to_dict(self):
        """Test result serialization."""
        result = RawNet2Result(
            fake_probability=0.75,
            vocoder_class="waveglow",
            vocoder_confidence=0.85,
            artifact_score=0.70,
            confidence=0.8,
            is_manipulated=True,
            segment_scores=[],
        )

        result_dict = result.to_dict()

        assert "fake_probability" in result_dict
        assert "vocoder_class" in result_dict
        assert "confidence" in result_dict
        assert "is_manipulated" in result_dict
        assert result_dict["num_segments"] == 0


class TestVocoderClasses:
    """Test vocoder classes constant."""

    def test_vocoder_classes_count(self):
        """Test that there are 6 vocoder classes."""
        assert len(VOCODER_CLASSES) == 6

    def test_vocoder_classes_values(self):
        """Test vocoder class values."""
        expected = ["bona_fide", "hifi_gan", "waveglow", "wavernn", "parallel_wavegan", "melgan"]
        assert VOCODER_CLASSES == expected


class TestSincConv:
    """Test SincConv module."""

    def test_sinc_conv_initialization(self):
        """Test SincConv initialization."""
        conv = SincConv(
            in_channels=1,
            out_channels=128,
            kernel_size=1024,
            sample_rate=16000,
        )

        assert hasattr(conv, "conv")
        assert hasattr(conv, "fc")
        assert conv.in_channels == 1
        assert conv.out_channels == 128

    def test_sinc_conv_forward_shape(self):
        """Test SincConv forward pass shape."""
        conv = SincConv(
            in_channels=1,
            out_channels=64,
            kernel_size=512,
            sample_rate=16000,
        )
        conv.eval()

        with torch.no_grad():
            x = torch.randn(2, 1, 16000)
            out = conv(x)

        assert out.shape[0] == 2
        assert out.shape[1] == 64
        assert out.shape[2] == x.shape[2]


class TestResidualGRUBlock:
    """Test ResidualGRUBlock module."""

    def test_block_initialization(self):
        """Test block initialization."""
        block = ResidualGRUBlock(
            input_size=128,
            hidden_size=128,
            num_layers=2,
        )

        assert hasattr(block, "gru")
        assert hasattr(block, "fc")
        assert hasattr(block, "skip_fc")

    def test_block_forward_shape(self):
        """Test block forward pass shape."""
        block = ResidualGRUBlock(
            input_size=128,
            hidden_size=128,
            num_layers=2,
        )
        block.eval()

        with torch.no_grad():
            x = torch.randn(2, 100, 128)
            out = block(x)

        assert out.shape == x.shape


class TestRawNet2:
    """Test RawNet2 main class."""

    def test_initialization(self):
        """Test RawNet2 initialization."""
        model = RawNet2()

        assert hasattr(model, "sinc_conv")
        assert hasattr(model, "res_blocks")
        assert hasattr(model, "attention")
        assert hasattr(model, "multi_task_head")
        assert model.MODEL_NAME == "rawnet2"
        assert model.MODEL_VERSION == "1.0.0"

    def test_model_info(self, cpu_device):
        """Test model info retrieval."""
        model = RawNet2(device=cpu_device)

        info = model.get_model_info()

        assert info["name"] == "rawnet2"
        assert info["version"] == "1.0.0"
        assert "vocoder_classes" in info

    def test_health_check_not_loaded(self):
        """Test health check before loading."""
        model = RawNet2()

        assert model.health_check() is False

    def test_has_detect_method(self):
        """Test that model has detect method."""
        model = RawNet2()

        assert hasattr(model, "detect")
        assert callable(model.detect)

    def test_has_identify_vocoder_method(self):
        """Test that model has identify_vocoder method."""
        model = RawNet2()

        assert hasattr(model, "identify_vocoder")
        assert callable(model.identify_vocoder)

    def test_has_get_heatmap_method(self):
        """Test that model has get_heatmap method."""
        model = RawNet2()

        assert hasattr(model, "get_heatmap")
        assert callable(model.get_heatmap)

    def test_load_sets_is_loaded(self, cpu_device):
        """Test that load sets is_loaded flag."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)

        assert model._is_loaded is True

        model.unload()

    def test_unload_resets_flag(self, cpu_device):
        """Test that unload resets is_loaded flag."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)
        model.unload()

        assert model._is_loaded is False

    def test_has_sinc_conv(self, cpu_device):
        """Test that model has sinc convolution."""
        model = RawNet2(device=cpu_device)

        assert hasattr(model, "sinc_conv")

    def test_has_res_blocks(self, cpu_device):
        """Test that model has residual blocks."""
        model = RawNet2(device=cpu_device)

        assert hasattr(model, "res_blocks")
        assert len(model.res_blocks) == 3

    def test_has_multi_task_head(self, cpu_device):
        """Test that model has multi-task head."""
        model = RawNet2(device=cpu_device)

        assert hasattr(model, "multi_task_head")
        assert "binary" in model.multi_task_head
        assert "vocoder_id" in model.multi_task_head

    def test_detect_not_loaded(self, cpu_device):
        """Test detection before loading."""
        model = RawNet2(device=cpu_device)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.detect(torch.randn(16000))

    def test_identify_vocoder_not_loaded(self, cpu_device):
        """Test vocoder identification before loading."""
        model = RawNet2(device=cpu_device)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.identify_vocoder(torch.randn(16000))

    def test_identify_vocoder_returns_tuple(self, cpu_device):
        """Test that identify_vocoder returns (class, confidence)."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)
        model.eval()

        with torch.no_grad():
            waveform = torch.randn(16000, device=cpu_device)
            vocoder_class, confidence = model.identify_vocoder(waveform)

        assert isinstance(vocoder_class, str)
        assert vocoder_class in VOCODER_CLASSES
        assert 0 <= confidence <= 1

    def test_preprocess_waveform_shape(self, cpu_device):
        """Test waveform preprocessing shape."""
        model = RawNet2(device=cpu_device)

        waveform = torch.randn(8000)
        processed = model._preprocess_waveform(waveform)

        assert processed.shape[-1] == model.sample_rate

    def test_detect_returns_detection_result(self, cpu_device):
        """Test that detect returns DetectionResult."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)
        model.eval()

        with torch.no_grad():
            waveform = torch.randn(model.sample_rate, device=cpu_device)
            result = model.detect(waveform)

        assert result.modality.value == "audio"
        assert 0 <= result.overall_score <= 1
        assert 0 <= result.confidence <= 1

    def test_detect_with_segmentation(self, cpu_device):
        """Test detection with multiple segments."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)
        model.eval()

        with torch.no_grad():
            waveform = torch.randn(model.sample_rate * 3, device=cpu_device)
            result = model.detect(waveform, segment_duration=1.0)

        assert "num_segments" in result.metadata

    def test_get_heatmap_not_loaded(self, cpu_device):
        """Test heatmap generation before loading."""
        model = RawNet2(device=cpu_device)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.get_heatmap(torch.randn(16000))


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_rawnet2(self, cpu_device):
        """Test factory function."""
        model = create_rawnet2(device=cpu_device)

        assert isinstance(model, RawNet2)

    def test_register_rawnet2(self, cpu_device):
        """Test registration."""
        from src.models.registry import ModelRegistry

        ModelRegistry._instance = None
        register_rawnet2()

        registry = ModelRegistry.get_instance()
        assert "rawnet2" in registry._model_factories


class TestRawNet2Integration:
    """Integration tests for RawNet2."""

    @pytest.fixture
    def model(self, cpu_device):
        """Create a loaded model."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)
        yield model
        model.unload()

    def test_model_loaded_after_load(self, model):
        """Test that model is loaded after load()."""
        assert model._is_loaded is True

    def test_has_sinc_conv_layer(self, model):
        """Test that model has sinc convolution layer."""
        assert hasattr(model, "sinc_conv")

    def test_has_residual_blocks(self, model):
        """Test that model has residual GRU blocks."""
        assert hasattr(model, "res_blocks")
        assert len(model.res_blocks) == 3

    def test_has_attention_layer(self, model):
        """Test that model has attention layer."""
        assert hasattr(model, "attention")

    def test_model_info_after_load(self, model):
        """Test model info after loading."""
        info = model.get_model_info()

        assert info["name"] == "rawnet2"
        assert info["is_loaded"] is True
        assert "vocoder_classes" in info

    def test_half_precision_option(self, cpu_device):
        """Test half precision mode."""
        model = RawNet2(device=cpu_device, use_half_precision=True)
        model.load(device=cpu_device)

        assert model.use_half_precision is True

        model.unload()

    def test_different_sample_rates(self, cpu_device):
        """Test with different input sample rates."""
        model = RawNet2(device=cpu_device)
        model.load(device=cpu_device)
        model.eval()

        with torch.no_grad():
            waveform_16k = torch.randn(model.sample_rate, device=cpu_device)
            waveform_8k = torch.randn(model.sample_rate // 2, device=cpu_device)

            result_16k = model.detect(waveform_16k, sample_rate=16000)
            result_8k = model.detect(waveform_8k, sample_rate=8000)

        assert 0 <= result_16k.overall_score <= 1
        assert 0 <= result_8k.overall_score <= 1

    def test_model_constants(self):
        """Test that model constants are correct."""
        assert RawNet2.MODEL_NAME == "rawnet2"
        assert RawNet2.MODEL_VERSION == "1.0.0"


@pytest.fixture(scope="session")
def cpu_device():
    """Use CPU for all tests."""
    return torch.device("cpu")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
