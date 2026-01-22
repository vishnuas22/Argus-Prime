"""
Unit tests for model registry module.

Tests lazy loading, GPU memory management, and health checks.
"""

import pytest
import time
import torch
from unittest.mock import MagicMock, patch

from src.models.registry import (
    ModelRegistry,
    ModelMetadata,
    ModelHealthStatus,
    ModelLoadError,
    ModelNotFoundError,
    GPUMemoryManager,
    load_model_context,
    get_model,
    register_model,
)


class TestGPUMemoryManager:
    """Test GPU memory manager."""

    def test_device_type_cuda(self):
        """Test device type detection with CUDA."""
        manager = GPUMemoryManager()
        if torch.cuda.is_available():
            assert manager.device_type == "cuda"
        else:
            assert manager.device_type in ["mps", "cpu"]

    def test_device_type_fallback(self):
        """Test device type fallback."""
        manager = GPUMemoryManager(max_memory_mb=1000)
        device = manager.device_type
        assert device in ["cuda", "mps", "cpu"]

    def test_get_available_memory(self):
        """Test available memory detection."""
        manager = GPUMemoryManager()
        memory = manager.get_available_memory_mb()
        assert memory >= 0

    def test_can_allocate(self):
        """Test memory allocation check."""
        manager = GPUMemoryManager(max_memory_mb=1000)

        assert manager.can_allocate(100) is True
        assert manager.can_allocate(2000) is False

    def test_allocate_and_deallocate(self):
        """Test memory allocation and deallocation."""
        manager = GPUMemoryManager(max_memory_mb=1000)

        assert manager.allocate("model_a", 100) is True
        assert manager.allocate("model_b", 200) is True
        assert manager._current_memory_mb == 300

        assert manager.deallocate("model_a") is True
        assert manager._current_memory_mb == 200

    def test_allocate_with_eviction(self):
        """Test allocation with automatic eviction."""
        manager = GPUMemoryManager(max_memory_mb=500, eviction_threshold_mb=100)

        assert manager.allocate("model_a", 300) is True
        assert manager.allocate("model_b", 300) is True

    def test_allocate_within_limit(self):
        """Test allocation within memory limit."""
        manager = GPUMemoryManager(max_memory_mb=1000)

        assert manager.allocate("model_a", 300) is True
        assert manager.allocate("model_b", 200) is True
        assert manager._current_memory_mb == 500

    def test_get_memory_info(self):
        """Test memory info retrieval."""
        manager = GPUMemoryManager(max_memory_mb=1000)
        manager.allocate("model_a", 100)

        info = manager.get_memory_info()

        assert "device_type" in info
        assert "current_usage_mb" in info
        assert "max_memory_mb" in info
        assert "available_mb" in info
        assert "model_count" in info

    def test_update_access(self):
        """Test access time update."""
        manager = GPUMemoryManager()
        manager.allocate("model_a", 100)

        initial_time = manager._lru_order["model_a"]
        time.sleep(0.01)
        manager.update_access("model_a")

        assert manager._lru_order["model_a"] >= initial_time


class TestModelMetadata:
    """Test ModelMetadata dataclass."""

    def test_metadata_creation(self):
        """Test metadata creation."""
        metadata = ModelMetadata(
            name="test_model",
            model_type="detector",
            version="1.0.0",
        )

        assert metadata.name == "test_model"
        assert metadata.model_type == "detector"
        assert metadata.version == "1.0.0"
        assert metadata.is_loaded is False
        assert metadata.access_count == 0

    def test_metadata_with_times(self):
        """Test metadata with timestamps."""
        now = time.time()
        metadata = ModelMetadata(
            name="test_model",
            model_type="detector",
            version="2.0.0",
            loaded_at=now,
            last_accessed=now,
            access_count=10,
            memory_usage_mb=256.5,
            device="cuda:0",
            is_loaded=True,
        )

        assert metadata.loaded_at == now
        assert metadata.access_count == 10
        assert metadata.memory_usage_mb == 256.5


class TestModelHealthStatus:
    """Test ModelHealthStatus dataclass."""

    def test_health_status_healthy(self):
        """Test healthy status."""
        status = ModelHealthStatus(
            model_name="test",
            is_healthy=True,
            last_check=time.time(),
        )

        assert status.is_healthy is True
        assert status.error_message is None

    def test_health_status_unhealthy(self):
        """Test unhealthy status."""
        status = ModelHealthStatus(
            model_name="test",
            is_healthy=False,
            last_check=time.time(),
            error_message="CUDA out of memory",
        )

        assert status.is_healthy is False
        assert status.error_message == "CUDA out of memory"


class TestModelRegistry:
    """Test ModelRegistry class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ModelRegistry._instance = None
        yield
        ModelRegistry._instance = None

    def test_singleton_pattern(self):
        """Test singleton pattern implementation."""
        registry1 = ModelRegistry.get_instance()
        registry2 = ModelRegistry.get_instance()

        assert registry1 is registry2

    def test_register_model(self):
        """Test model registration."""
        registry = ModelRegistry.get_instance()

        def mock_factory():
            return MagicMock()

        registry.register("test_model", mock_factory, "detector", "1.0.0")

        assert "test_model" in registry._model_factories
        assert registry._model_metadata["test_model"].name == "test_model"

    def test_register_duplicate(self):
        """Test duplicate registration warning."""
        registry = ModelRegistry.get_instance()

        def mock_factory():
            return MagicMock()

        registry.register("test_model", mock_factory)
        registry.register("test_model", mock_factory)

        assert "test_model" in registry._model_factories

    def test_unregister_model(self):
        """Test model unregistration."""
        registry = ModelRegistry.get_instance()

        def mock_factory():
            return MagicMock()

        registry.register("test_model", mock_factory)
        registry.unregister("test_model")

        assert "test_model" not in registry._model_factories

    def test_get_device_cuda(self):
        """Test device detection with CUDA."""
        registry = ModelRegistry.get_instance()
        device = registry.get_device()

        assert device.type in ["cuda", "mps", "cpu"]

    def test_load_model(self):
        """Test model loading."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()
        mock_model.load = MagicMock()
        mock_model.unload = MagicMock()
        mock_model.health_check = MagicMock(return_value=True)

        def factory():
            return mock_model

        registry.register("load_test", factory, "detector", "1.0.0")
        model = registry.load("load_test")

        assert model is mock_model
        mock_model.load.assert_called_once()
        assert registry._model_metadata["load_test"].is_loaded is True

    def test_load_model_not_found(self):
        """Test loading non-existent model."""
        registry = ModelRegistry.get_instance()

        with pytest.raises(ModelNotFoundError):
            registry.load("non_existent")

    def test_load_already_loaded(self):
        """Test loading already loaded model."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()
        mock_model.load = MagicMock()
        mock_model.unload = MagicMock()

        def factory():
            return mock_model

        registry.register("reload_test", factory)
        registry.load("reload_test")
        registry.load("reload_test")

        assert mock_model.load.call_count == 1

    def test_load_with_device(self):
        """Test loading with specific device."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()
        mock_model.load = MagicMock()

        def factory():
            return mock_model

        registry.register("device_test", factory)
        device = torch.device("cpu")
        registry.load("device_test", device=device)

        mock_model.load.assert_called_with(device)

    def test_unload_model(self):
        """Test model unloading."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()
        mock_model.unload = MagicMock()

        def factory():
            return mock_model

        registry.register("unload_test", factory)
        registry.load("unload_test")
        result = registry.unload("unload_test")

        assert result is True
        mock_model.unload.assert_called_once()
        assert registry._model_metadata["unload_test"].is_loaded is False

    def test_unload_not_loaded(self):
        """Test unloading not loaded model."""
        registry = ModelRegistry.get_instance()

        def mock_factory():
            return MagicMock()

        registry.register("unload_not_loaded", mock_factory)
        result = registry.unload("unload_not_loaded")

        assert result is False

    def test_unload_all(self):
        """Test unloading all models."""
        registry = ModelRegistry.get_instance()

        mock_model1 = MagicMock()
        mock_model1.unload = MagicMock()
        mock_model2 = MagicMock()
        mock_model2.unload = MagicMock()

        def factory1():
            return mock_model1

        def factory2():
            return mock_model2

        registry.register("unload_all_1", factory1)
        registry.register("unload_all_2", factory2)
        registry.load("unload_all_1")
        registry.load("unload_all_2")

        registry.unload_all()

        assert len(registry._loaded_models) == 0

    def test_get_loaded_model(self):
        """Test getting loaded model."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()

        def factory():
            return mock_model

        registry.register("get_test", factory)
        registry.load("get_test")

        result = registry.get("get_test")
        assert result is mock_model

        not_loaded = registry.get("non_existent")
        assert not_loaded is None

    def test_get_metadata(self):
        """Test getting model metadata."""
        registry = ModelRegistry.get_instance()

        def factory():
            return MagicMock()

        registry.register("metadata_test", factory, "detector", "2.0.0")
        metadata = registry.get_metadata("metadata_test")

        assert metadata is not None
        assert metadata.name == "metadata_test"
        assert metadata.version == "2.0.0"

    def test_health_check_loaded(self):
        """Test health check on loaded model."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()
        mock_model.health_check = MagicMock(return_value=True)

        def factory():
            return mock_model

        registry.register("health_check_loaded", factory)
        registry.load("health_check_loaded")

        status = registry.health_check("health_check_loaded")

        assert status.is_healthy is True
        assert status.model_name == "health_check_loaded"

    def test_health_check_not_registered(self):
        """Test health check on not registered model."""
        registry = ModelRegistry.get_instance()

        status = registry.health_check("not_registered")

        assert status.is_healthy is False
        assert status.error_message is not None

    def test_health_check_all(self):
        """Test health check on all models."""
        registry = ModelRegistry.get_instance()

        def factory():
            return MagicMock()

        registry.register("health_all_1", factory)
        registry.register("health_all_2", factory)

        results = registry.health_check_all()

        assert len(results) == 2
        assert "health_all_1" in results
        assert "health_all_2" in results

    def test_get_status(self):
        """Test getting registry status."""
        registry = ModelRegistry.get_instance()

        def factory():
            return MagicMock()

        registry.register("status_test", factory)
        registry.load("status_test")

        status = registry.get_status()

        assert "loaded_models" in status
        assert "registered_models" in status
        assert "memory" in status
        assert "models" in status
        assert status["loaded_models"] >= 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ModelRegistry._instance = None
        yield
        ModelRegistry._instance = None

    def test_register_model_function(self):
        """Test register_model convenience function."""
        def mock_factory():
            return MagicMock()

        register_model("convenience_test", mock_factory, "detector", "1.0.0")

        registry = ModelRegistry.get_instance()
        assert "convenience_test" in registry._model_factories

    def test_get_model_function(self):
        """Test get_model convenience function."""
        mock_model = MagicMock()

        def factory():
            return mock_model

        register_model("get_convenience", factory)
        result = get_model("get_convenience")

        assert result is mock_model

    def test_load_model_context(self):
        """Test load_model_context context manager."""
        mock_model = MagicMock()

        def factory():
            return mock_model

        register_model("context_test", factory)

        with load_model_context("context_test") as model:
            assert model is mock_model


class TestModelRegistryIntegration:
    """Integration tests for ModelRegistry."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset singleton before each test."""
        ModelRegistry._instance = None
        yield
        ModelRegistry._instance = None

    def test_full_lifecycle(self):
        """Test complete model lifecycle."""
        registry = ModelRegistry.get_instance()

        call_count = {"load": 0, "unload": 0}

        class MockModel:
            def load(self, device):
                call_count["load"] += 1

            def unload(self):
                call_count["unload"] += 1

            def health_check(self):
                return True

            def get_model_info(self):
                return {"version": "1.0.0"}

        def factory():
            return MockModel()

        registry.register("lifecycle_test", factory)

        assert registry.load("lifecycle_test") is not None
        assert call_count["load"] == 1

        assert registry.get("lifecycle_test") is not None

        registry.unload("lifecycle_test")
        assert call_count["unload"] == 1

    def test_multiple_models_same_type(self):
        """Test loading multiple models of same type."""
        registry = ModelRegistry.get_instance()

        mock_models = [MagicMock() for _ in range(3)]

        for i, mock_model in enumerate(mock_models):
            def factory(m=mock_model):
                return m
            registry.register(f"multi_{i}", factory)

        for i in range(3):
            registry.load(f"multi_{i}")

        assert len(registry._loaded_models) == 3

    def test_force_reload(self):
        """Test force reload functionality."""
        registry = ModelRegistry.get_instance()

        mock_model = MagicMock()
        call_count = {"load": 0}

        def factory():
            call_count["load"] += 1
            return mock_model

        registry.register("force_reload_test", factory)
        registry.load("force_reload_test")
        registry.load("force_reload_test")
        registry.load("force_reload_test", force_reload=True)

        assert call_count["load"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
