"""
Unit tests for ForensicFlow video deepfake detection model.

Tests model architecture, branches, and detection pipeline.
"""

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from src.models.video.forensic_flow import (
    ForensicFlow,
    ForensicFlowResult,
    ConvNeXtBranch,
    SwinTransformerBranch,
    SpectralBranch,
    AttentionTemporalPooling,
    GatedFusion,
    create_forensic_flow,
    register_forensic_flow,
)
from src.models.interfaces import DetectionResult


@pytest.fixture(scope="session")
def cpu_device():
    """Use CPU for all tests to avoid MPS/CPU issues."""
    return torch.device("cpu")


class TestForensicFlowResult:
    """Test ForensicFlowResult dataclass."""

    def test_result_creation(self):
        """Test result creation."""
        result = ForensicFlowResult(
            overall_score=0.85,
            confidence=0.9,
            is_manipulated=True,
            branch_scores={
                "global_consistency": 0.8,
                "texture_anomaly": 0.9,
                "spectral_noise": 0.85,
            },
            frame_scores=[0.8, 0.85, 0.9],
        )

        assert result.overall_score == 0.85
        assert result.confidence == 0.9
        assert result.is_manipulated is True
        assert len(result.frame_scores) == 3

    def test_result_to_dict(self):
        """Test result serialization."""
        result = ForensicFlowResult(
            overall_score=0.75,
            confidence=0.8,
            is_manipulated=True,
            branch_scores={"test": 0.5},
            frame_scores=[0.7, 0.8],
        )

        result_dict = result.to_dict()

        assert "overall_score" in result_dict
        assert "confidence" in result_dict
        assert "is_manipulated" in result_dict
        assert result_dict["num_frames"] == 2


class TestConvNeXtBranch:
    """Test ConvNeXt branch."""

    def test_branch_initialization(self, cpu_device):
        """Test branch initialization."""
        branch = ConvNeXtBranch(pretrained=False)
        branch.to(cpu_device)

        assert hasattr(branch, "backbone")
        assert hasattr(branch, "classifier")
        assert branch.feature_dim > 0

    def test_branch_has_required_methods(self, cpu_device):
        """Test that branch has required methods."""
        branch = ConvNeXtBranch(pretrained=False)
        branch.to(cpu_device)

        assert hasattr(branch, "forward")

    def test_feature_dim(self, cpu_device):
        """Test feature dimension."""
        branch = ConvNeXtBranch(pretrained=False)
        branch.to(cpu_device)

        assert branch.feature_dim > 0
        assert isinstance(branch.feature_dim, int)


class TestSwinTransformerBranch:
    """Test Swin Transformer branch."""

    def test_branch_initialization(self, cpu_device):
        """Test branch initialization."""
        branch = SwinTransformerBranch(pretrained=False)
        branch.to(cpu_device)

        assert hasattr(branch, "backbone")
        assert hasattr(branch, "classifier")

    def test_feature_dim(self, cpu_device):
        """Test feature dimension."""
        branch = SwinTransformerBranch(pretrained=False)
        branch.to(cpu_device)

        assert branch.feature_dim > 0


class TestSpectralBranch:
    """Test Spectral branch."""

    def test_branch_initialization(self, cpu_device):
        """Test branch initialization."""
        branch = SpectralBranch()
        branch.to(cpu_device)

        assert hasattr(branch, "fft_conv")
        assert hasattr(branch, "channel_attention")
        assert hasattr(branch, "classifier")

    def test_has_compute_spectrum(self, cpu_device):
        """Test that branch has compute_spectrum method."""
        branch = SpectralBranch()
        branch.to(cpu_device)

        assert hasattr(branch, "compute_spectrum")


class TestAttentionTemporalPooling:
    """Test attention temporal pooling."""

    def test_initialization(self, cpu_device):
        """Test initialization."""
        pool = AttentionTemporalPooling(feature_dim=384)
        pool.to(cpu_device)

        assert hasattr(pool, "attention")
        assert hasattr(pool, "query")

    def test_has_attention_module(self, cpu_device):
        """Test that pooling has attention module."""
        pool = AttentionTemporalPooling(feature_dim=384)
        pool.to(cpu_device)

        assert hasattr(pool, "attention")
        assert hasattr(pool, "query")


class TestGatedFusion:
    """Test gated fusion module."""

    def test_initialization(self, cpu_device):
        """Test initialization."""
        fusion = GatedFusion(num_branches=3)
        fusion.to(cpu_device)
        fusion.eval()

        assert hasattr(fusion, "gate_network")
        assert hasattr(fusion, "fusion_network")

    def test_returns_weights_dict(self, cpu_device):
        """Test that fusion returns weights dictionary."""
        fusion = GatedFusion(num_branches=3)
        fusion.to(cpu_device)
        fusion.eval()

        with torch.no_grad():
            scores = [
                torch.tensor([0.8], device=cpu_device),
                torch.tensor([0.6], device=cpu_device),
                torch.tensor([0.7], device=cpu_device),
            ]
            fused, weights = fusion(scores)

        assert isinstance(fused.item(), float)
        assert len(weights) == 3
        assert "global_branch" in weights
        assert "texture_branch" in weights
        assert "spectral_branch" in weights


class TestForensicFlow:
    """Test ForensicFlow main class."""

    def test_initialization(self, cpu_device):
        """Test ForensicFlow initialization."""
        model = ForensicFlow(pretrained=False, device=cpu_device)

        assert hasattr(model, "branch_a")
        assert hasattr(model, "branch_b")
        assert hasattr(model, "branch_c")
        assert hasattr(model, "temporal_pool")
        assert hasattr(model, "fusion")
        assert model.MODEL_NAME == "forensic_flow"

    def test_model_info(self, cpu_device):
        """Test model info retrieval."""
        model = ForensicFlow(pretrained=False, device=cpu_device)

        info = model.get_model_info()

        assert info["name"] == "forensic_flow"
        assert info["version"] == "1.0.0"

    def test_health_check_not_loaded(self, cpu_device):
        """Test health check before loading."""
        model = ForensicFlow(pretrained=False, device=cpu_device)

        assert model.health_check() is False

    def test_has_detect_method(self, cpu_device):
        """Test that model has detect method."""
        model = ForensicFlow(pretrained=False, device=cpu_device)

        assert hasattr(model, "detect")
        assert callable(model.detect)

    def test_has_get_heatmap_method(self, cpu_device):
        """Test that model has get_heatmap method."""
        model = ForensicFlow(pretrained=False, device=cpu_device)

        assert hasattr(model, "get_heatmap")
        assert callable(model.get_heatmap)

    def test_load_sets_is_loaded(self, cpu_device):
        """Test that load sets is_loaded flag."""
        model = ForensicFlow(pretrained=False, device=cpu_device)
        model.load(device=cpu_device)

        assert model._is_loaded is True

        model.unload()

    def test_unload_resets_flag(self, cpu_device):
        """Test that unload resets is_loaded flag."""
        model = ForensicFlow(pretrained=False, device=cpu_device)
        model.load(device=cpu_device)
        model.unload()

        assert model._is_loaded is False


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_forensic_flow(self, cpu_device):
        """Test factory function."""
        model = create_forensic_flow(device=cpu_device)

        assert isinstance(model, ForensicFlow)

    def test_register_forensic_flow(self, cpu_device):
        """Test registration."""
        from src.models.registry import ModelRegistry

        ModelRegistry._instance = None
        register_forensic_flow()

        registry = ModelRegistry.get_instance()
        assert "forensic_flow" in registry._model_factories


class TestForensicFlowIntegration:
    """Integration tests for ForensicFlow."""

    @pytest.fixture
    def model(self, cpu_device):
        """Create a loaded model."""
        model = ForensicFlow(pretrained=False, device=cpu_device)
        model.load(device=cpu_device)
        yield model
        model.unload()

    def test_model_loaded_after_load(self, model):
        """Test that model is loaded after load()."""
        assert model._is_loaded is True

    def test_has_three_branches(self, model):
        """Test that model has three branches."""
        assert hasattr(model, "branch_a")
        assert hasattr(model, "branch_b")
        assert hasattr(model, "branch_c")

    def test_has_temporal_pooling(self, model):
        """Test that model has temporal pooling."""
        assert hasattr(model, "temporal_pool")

    def test_has_fusion_module(self, model):
        """Test that model has fusion module."""
        assert hasattr(model, "fusion")

    def test_model_info_after_load(self, model):
        """Test model info after loading."""
        info = model.get_model_info()

        assert info["name"] == "forensic_flow"
        assert info["is_loaded"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
