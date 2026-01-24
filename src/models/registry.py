"""
Model registry for centralized model loading, caching, and lifecycle management.

This module implements:
- Lazy loading (load only when first accessed)
- GPU memory caching with eviction policy
- Automatic device placement (cuda/mps/cpu)
- Model versioning support
- Health checks and reload on failure
"""

import asyncio
import functools
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, TypeVar

import torch

from src.config import get_config
from src.core.logging import logger

T = TypeVar("T")


@dataclass
class ModelMetadata:
    """Metadata for a registered model."""
    name: str
    model_type: str
    version: str
    loaded_at: Optional[float] = None
    last_accessed: Optional[float] = None
    access_count: int = 0
    memory_usage_mb: float = 0.0
    device: str = "cpu"
    is_loaded: bool = False


@dataclass
class ModelHealthStatus:
    """Health status of a model."""
    model_name: str
    is_healthy: bool
    last_check: float
    error_message: Optional[str] = None
    load_time_ms: float = 0.0


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""

    pass


class ModelNotFoundError(Exception):
    """Exception raised when model is not registered."""

    pass


class GPUMemoryManager:
    """
    Manages GPU memory allocation and eviction.

    Features:
    - Tracks memory usage across models
    - Evicts least recently used models when memory is full
    - Supports CUDA, MPS (Apple Silicon), and CPU fallback
    """

    def __init__(
        self,
        max_memory_mb: Optional[int] = None,
        eviction_threshold_mb: int = 512,
    ):
        """
        Initialize GPU memory manager.

        Args:
            max_memory_mb: Maximum memory budget. None means auto-detect.
            eviction_threshold_mb: Memory threshold to trigger eviction
        """
        self.max_memory_mb = max_memory_mb
        self.eviction_threshold_mb = eviction_threshold_mb
        self._current_memory_mb: float = 0.0
        self._model_memory: Dict[str, float] = {}
        self._lru_order: Dict[str, float] = {}

    @property
    def device_type(self) -> str:
        """Get the current device type."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def get_available_memory_mb(self) -> float:
        """Get available GPU memory in MB."""
        if self.device_type == "cuda":
            torch.cuda.empty_cache()
            reserved = torch.cuda.memory_reserved(0) / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return total - reserved
        elif self.device_type == "mps":
            return 4096.0
        else:
            return float("inf")

    def get_model_memory_mb(self, model: torch.nn.Module) -> float:
        """Estimate model memory usage in MB."""
        param_count = sum(p.numel() for p in model.parameters())
        param_bytes = param_count * 4
        buffer_bytes = sum(
            b.numel() * b.element_size()
            for b in model.buffers()
        )
        total_bytes = param_bytes + buffer_bytes
        return total_bytes / (1024 * 1024)

    def can_allocate(self, required_mb: float) -> bool:
        """Check if memory can be allocated."""
        if self.max_memory_mb is None:
            return True
        return (self._current_memory_mb + required_mb) < self.max_memory_mb

    def allocate(self, model_name: str, memory_mb: float) -> bool:
        """
        Allocate memory for a model.

        Args:
            model_name: Name of the model
            memory_mb: Memory to allocate in MB

        Returns:
            True if allocation successful, False otherwise
        """
        if not self.can_allocate(memory_mb):
            if not self.evict_lru_models(memory_mb):
                logger.warning(
                    f"Cannot allocate {memory_mb}MB for {model_name}, "
                    f"current usage: {self._current_memory_mb:.2f}MB"
                )
                return False

        self._model_memory[model_name] = memory_mb
        self._current_memory_mb += memory_mb
        self._lru_order[model_name] = time.time()
        logger.debug(f"Allocated {memory_mb:.2f}MB for {model_name}")
        return True

    def deallocate(self, model_name: str) -> bool:
        """
        Deallocate memory for a model.

        Args:
            model_name: Name of the model

        Returns:
            True if deallocation successful
        """
        if model_name in self._model_memory:
            memory_mb = self._model_memory.pop(model_name)
            self._current_memory_mb -= memory_mb
            self._lru_order.pop(model_name, None)
            logger.debug(f"Deallocated memory for {model_name}")
            return True
        return False

    def evict_lru_models(self, required_mb: float) -> bool:
        """
        Evict least recently used models to free memory.

        Args:
            required_mb: Memory that needs to be freed

        Returns:
            True if enough memory was freed
        """
        freed_mb = 0.0
        models_to_evict = sorted(
            self._lru_order.items(),
            key=lambda x: x[1]
        )

        for model_name, _ in models_to_evict:
            if model_name in self._model_memory:
                memory_mb = self._model_memory[model_name]
                self.deallocate(model_name)
                registry = ModelRegistry.get_instance()
                if model_name in registry._loaded_models:
                    registry._loaded_models[model_name].unload()
                    del registry._loaded_models[model_name]
                freed_mb += memory_mb

                if freed_mb >= required_mb + self.eviction_threshold_mb:
                    break

        logger.info(f"Evicted models, freed {freed_mb:.2f}MB")
        return freed_mb >= required_mb

    def update_access(self, model_name: str) -> None:
        """Update last accessed time for LRU tracking."""
        if model_name in self._lru_order:
            self._lru_order[model_name] = time.time()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        return {
            "device_type": self.device_type,
            "current_usage_mb": self._current_memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "available_mb": self.get_available_memory_mb(),
            "model_count": len(self._model_memory),
        }


class ModelRegistry:
    """
    Centralized model registry for lazy loading and lifecycle management.

    Implements singleton pattern with LRU cache for model instances.
    Supports automatic device placement and health monitoring.
    """

    _instance: Optional["ModelRegistry"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls) -> "ModelRegistry":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the model registry."""
        if hasattr(self, "_initialized"):
            return

        self._loaded_models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, ModelMetadata] = {}
        self._model_factories: Dict[str, Callable[[], Any]] = {}
        self._memory_manager = GPUMemoryManager()
        self._health_check_interval = 300
        self._last_health_check: Dict[str, float] = {}

        self._initialized = True
        logger.info("ModelRegistry initialized")

    @classmethod
    def get_instance(cls) -> "ModelRegistry":
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        name: str,
        model_factory: Callable[[], T],
        model_type: str = "detector",
        version: str = "1.0.0",
    ) -> None:
        """
        Register a model factory for lazy loading.

        Args:
            name: Unique model identifier
            model_factory: Callable that returns a model instance
            model_type: Type of model (detector, preprocessor, etc.)
            version: Model version string
        """
        if name in self._model_factories:
            logger.warning(f"Model {name} already registered, overwriting")

        self._model_factories[name] = model_factory
        self._model_metadata[name] = ModelMetadata(
            name=name,
            model_type=model_type,
            version=version,
        )
        logger.info(f"Registered model: {name} (v{version}, type={model_type})")

    def unregister(self, name: str) -> None:
        """Unregister a model and unload if loaded."""
        if name in self._loaded_models:
            self.unload(name)

        self._model_factories.pop(name, None)
        self._model_metadata.pop(name, None)
        logger.info(f"Unregistered model: {name}")

    def get_device(self) -> torch.device:
        """
        Get the best available device for inference.

        Returns:
            torch.device (cuda, mps, or cpu)
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    @functools.lru_cache(maxsize=128)
    def load(
        self,
        name: str,
        device: Optional[torch.device] = None,
        force_reload: bool = False,
    ) -> Any:
        """
        Load a model by name (lazy loading).

        Args:
            name: Model identifier
            device: Device to load on (auto-detected if None)
            force_reload: Force reload even if already loaded

        Returns:
            Loaded model instance

        Raises:
            ModelNotFoundError: If model is not registered
            ModelLoadError: If model loading fails
        """
        if name not in self._model_factories:
            raise ModelNotFoundError(f"Model {name} is not registered")

        if name in self._loaded_models and not force_reload:
            self._memory_manager.update_access(name)
            self._model_metadata[name].access_count += 1
            self._model_metadata[name].last_accessed = time.time()
            return self._loaded_models[name]

        if device is None:
            device = self.get_device()

        logger.info(f"Loading model {name} on {device}")

        start_time = time.time()

        try:
            model = self._model_factories[name]()
            if hasattr(model, "load"):
                model.load(device)

            memory_mb = 0.0
            if hasattr(model, "__class__"):
                try:
                    if isinstance(model, torch.nn.Module):
                        memory_mb = self._memory_manager.get_model_memory_mb(model)
                except Exception:
                    pass

            if memory_mb > 0 and not self._memory_manager.allocate(name, memory_mb):
                logger.warning(
                    f"Could not allocate memory for {name}, continuing anyway"
                )

            self._loaded_models[name] = model

            self._model_metadata[name].loaded_at = start_time
            self._model_metadata[name].last_accessed = start_time
            self._model_metadata[name].access_count = 1
            self._model_metadata[name].device = str(device)
            self._model_metadata[name].is_loaded = True
            self._model_metadata[name].memory_usage_mb = memory_mb

            load_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Model {name} loaded successfully in {load_time_ms:.2f}ms "
                f"on {device}, memory: {memory_mb:.2f}MB"
            )

            return model

        except Exception as e:
            logger.error(f"Failed to load model {name}: {e}")
            raise ModelLoadError(f"Model {name} failed to load: {e}") from e

    async def load_async(
        self,
        name: str,
        device: Optional[torch.device] = None,
        force_reload: bool = False,
    ) -> Any:
        """
        Async version of load.

        Args:
            name: Model identifier
            device: Device to load on
            force_reload: Force reload even if loaded

        Returns:
            Loaded model instance
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.load(name, device, force_reload),
        )

    def unload(self, name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            name: Model identifier

        Returns:
            True if unloaded successfully
        """
        if name not in self._loaded_models:
            logger.debug(f"Model {name} not loaded, nothing to unload")
            return False

        model = self._loaded_models[name]
        try:
            if hasattr(model, "unload"):
                model.unload()

            self._memory_manager.deallocate(name)

            del self._loaded_models[name]

            if name in self._model_metadata:
                self._model_metadata[name].is_loaded = False
                self._model_metadata[name].memory_usage_mb = 0.0

            logger.info(f"Model {name} unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model {name}: {e}")
            return False

    def unload_all(self) -> None:
        """Unload all loaded models."""
        for name in list(self._loaded_models.keys()):
            self.unload(name)
        logger.info("All models unloaded")

    def get(self, name: str) -> Optional[Any]:
        """
        Get a loaded model if available.

        Args:
            name: Model identifier

        Returns:
            Model instance or None if not loaded
        """
        if name in self._loaded_models:
            self._memory_manager.update_access(name)
            return self._loaded_models[name]
        return None

    def get_metadata(self, name: str) -> Optional[ModelMetadata]:
        """
        Get metadata for a model.

        Args:
            name: Model identifier

        Returns:
            ModelMetadata or None if not found
        """
        return self._model_metadata.get(name)

    def get_all_metadata(self) -> Dict[str, ModelMetadata]:
        """Get metadata for all registered models."""
        return dict(self._model_metadata)

    def health_check(self, name: str) -> ModelHealthStatus:
        """
        Perform health check on a model.

        Args:
            name: Model identifier

        Returns:
            ModelHealthStatus with health information
        """
        if name not in self._model_factories:
            return ModelHealthStatus(
                model_name=name,



def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        ModelRegistry: Singleton registry instance
    """
    return ModelRegistry.get_instance()

                is_healthy=False,
                last_check=time.time(),
                error_message="Model not registered",
            )

        start_time = time.time()

        try:
            if name in self._loaded_models:
                model = self._loaded_models[name]
                if hasattr(model, "health_check"):
                    is_healthy = model.health_check()
                else:
                    is_healthy = True

                load_time_ms = (time.time() - start_time) * 1000

                return ModelHealthStatus(
                    model_name=name,
                    is_healthy=is_healthy,
                    last_check=time.time(),
                    load_time_ms=load_time_ms,
                )
            else:
                return ModelHealthStatus(
                    model_name=name,
                    is_healthy=True,
                    last_check=time.time(),
                    load_time_ms=0.0,
                    error_message="Model not loaded",
                )

        except Exception as e:
            return ModelHealthStatus(
                model_name=name,
                is_healthy=False,
                last_check=time.time(),
                error_message=str(e),
            )

    def health_check_all(self) -> Dict[str, ModelHealthStatus]:
        """Perform health check on all models."""
        results = {}
        for name in self._model_factories:
            results[name] = self.health_check(name)
        return results

    def get_loaded_models(self) -> Dict[str, Any]:
        """Get dictionary of all loaded models."""
        return dict(self._loaded_models)

    def get_memory_info(self) -> Dict[str, Any]:
        """Get memory information for the registry."""
        return self._memory_manager.get_memory_info()

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the registry.

        Returns:
            Dictionary with registry status
        """
        loaded = []
        for name, metadata in self._model_metadata.items():
            loaded.append({
                "name": name,
                "type": metadata.model_type,
                "version": metadata.version,
                "loaded": metadata.is_loaded,
                "device": metadata.device,
                "memory_mb": metadata.memory_usage_mb,
                "access_count": metadata.access_count,
            })

        return {
            "loaded_models": len(self._loaded_models),
            "registered_models": len(self._model_factories),
            "memory": self.get_memory_info(),
            "models": loaded,
        }


@contextmanager
def load_model_context(name: str, device: Optional[torch.device] = None):
    """
    Context manager for temporarily loading a model.

    Usage:
        with load_model_context("forensic_flow") as model:
            result = model.detect(frames)

    Args:
        name: Model identifier
        device: Device to load on
    """
    registry = ModelRegistry.get_instance()
    model = registry.load(name, device)
    try:
        yield model
    finally:
        pass


def get_model(name: str, device: Optional[torch.device] = None) -> Any:
    """
    Convenience function to get a loaded model.

    Args:
        name: Model identifier
        device: Device to load on

    Returns:
        Loaded model instance
    """
    registry = ModelRegistry.get_instance()
    return registry.load(name, device)


def register_model(
    name: str,
    model_factory: Callable[[], T],
    model_type: str = "detector",
    version: str = "1.0.0",
) -> None:
    """
    Register a model with the global registry.

    Args:
        name: Unique model identifier
        model_factory: Callable that returns a model instance
        model_type: Type of model
        version: Model version
    """
    registry = ModelRegistry.get_instance()
    registry.register(name, model_factory, model_type, version)
