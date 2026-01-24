"""
Prometheus Metrics Module.

Provides Prometheus metrics for monitoring and observability:
- Request latency histograms
- Counter for analyses by type
- Gauge for active analyses
- Model inference metrics
- System resource metrics
"""

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Optional

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
)

from src.config import get_config
from src.core.logging import logger


class Metrics:
    """
    Prometheus metrics collector for Argus Prime.

    Tracks:
    - Request latency and counts
    - Analysis counts by modality
    - Model inference times
    - System resources
    - Error rates
    """

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """
        Initialize metrics collector.

        Args:
            registry: Optional custom Prometheus registry
        """
        self.config = get_config().metrics
        self._registry = registry or CollectorRegistry()

        self._setup_request_metrics()
        self._setup_analysis_metrics()
        self._setup_model_metrics()
        self._setup_system_metrics()

        logger.info(f"Metrics initialized: enabled={self.config.enabled}, path={self.config.path}")

    def _setup_request_metrics(self):
        """Setup HTTP request metrics."""
        self.requests_total = Counter(
            "argus_requests_total",
            "Total number of HTTP requests",
            ["method", "endpoint", "status"],
            registry=self._registry,
        )

        self.request_duration = Histogram(
            "argus_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self._registry,
        )

        self.requests_in_progress = Gauge(
            "argus_requests_in_progress",
            "Number of HTTP requests in progress",
            ["method", "endpoint"],
            registry=self._registry,
        )

    def _setup_analysis_metrics(self):
        """Setup analysis-specific metrics."""
        self.analyses_total = Counter(
            "argus_analyses_total",
            "Total number of analyses completed",
            ["modality", "verdict"],
            registry=self._registry,
        )

        self.analyses_active = Gauge(
            "argus_analyses_active",
            "Number of analyses currently in progress",
            registry=self._registry,
        )

        self.analysis_duration = Histogram(
            "argus_analysis_duration_seconds",
            "Analysis duration in seconds",
            ["modality"],
            buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
            registry=self._registry,
        )

        self.trust_score = Histogram(
            "argus_trust_score",
            "Distribution of trust scores",
            buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            registry=self._registry,
        )

        self.file_size_bytes = Histogram(
            "argus_file_size_bytes",
            "Size of uploaded files in bytes",
            ["file_type"],
            buckets=[1024, 10240, 102400, 1048576, 10485760, 104857600, 536870912],
            registry=self._registry,
        )

    def _setup_model_metrics(self):
        """Setup model inference metrics."""
        self.model_inferences = Counter(
            "argus_model_inferences_total",
            "Total number of model inferences",
            ["model_name", "modality"],
            registry=self._registry,
        )

        self.model_inference_duration = Histogram(
            "argus_model_inference_duration_seconds",
            "Model inference duration in seconds",
            ["model_name", "modality"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self._registry,
        )

        self.model_load_duration = Histogram(
            "argus_model_load_duration_seconds",
            "Model loading duration in seconds",
            ["model_name"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
            registry=self._registry,
        )

        self.models_loaded = Gauge(
            "argus_models_loaded",
            "Number of models currently loaded",
            registry=self._registry,
        )

    def _setup_system_metrics(self):
        """Setup system resource metrics."""
        self.memory_usage = Gauge(
            "argus_memory_usage_bytes",
            "Memory usage by component",
            ["component"],
            registry=self._registry,
        )

        self.gpu_memory = Gauge(
            "argus_gpu_memory_bytes",
            "GPU memory usage",
            registry=self._registry,
        )

        self.queue_size = Gauge(
            "argus_analysis_queue_size",
            "Number of analyses waiting in queue",
            registry=self._registry,
        )

        self.info = Info(
            "argus",
            "Argus Prime system information",
            registry=self._registry,
        )
        self.info.info({
            "version": "1.0.0",
            "platform": "Argus Prime Multi-Modal Deepfake Detection",
        })

    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager to track request metrics."""
        self.requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        start = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start
            self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
            self.requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
            self.requests_in_progress.labels(method=method, endpoint=endpoint).dec()

    def track_analysis(
        self,
        modality: str,
        verdict: str,
        duration: float,
        trust_score: float,
        file_type: str,
        file_size: int,
    ):
        """Track completed analysis metrics."""
        self.analyses_total.labels(modality=modality, verdict=verdict).inc()
        self.analysis_duration.labels(modality=modality).observe(duration)
        self.trust_score.observe(trust_score)
        self.file_size_bytes.labels(file_type=file_type).observe(file_size)

    def track_model_inference(
        self,
        model_name: str,
        modality: str,
        duration: float,
        success: bool = True,
    ):
        """Track model inference metrics."""
        status = "success" if success else "error"
        self.model_inferences.labels(
            model_name=model_name, modality=modality
        ).inc()
        self.model_inference_duration.labels(
            model_name=model_name, modality=modality
        ).observe(duration)

    def track_model_load(self, model_name: str, duration: float):
        """Track model loading metrics."""
        self.model_load_duration.labels(model_name=model_name).observe(duration)

    def update_memory_usage(self, component: str, bytes_used: int):
        """Update memory usage for a component."""
        self.memory_usage.labels(component=component).set(bytes_used)

    def update_gpu_memory(self, bytes_used: int):
        """Update GPU memory usage."""
        self.gpu_memory.set(bytes_used)

    def update_queue_size(self, size: int):
        """Update analysis queue size."""
        self.queue_size.set(size)

    def update_models_loaded(self, count: int):
        """Update number of loaded models."""
        self.models_loaded.set(count)

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self._registry)

    def get_content_type(self) -> str:
        """Get content type for metrics response."""
        return CONTENT_TYPE_LATEST

    def get_status(self) -> Dict[str, Any]:
        """Get metrics status for health checks."""
        return {
            "enabled": self.config.enabled,
            "path": self.config.path,
            "port": self.config.port,
            "registry_count": len(self._registry._names_to_collectors),
        }


def get_metrics() -> Metrics:
    """Get the global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = Metrics()
    return _metrics


_metrics: Optional[Metrics] = None


def setup_metrics() -> Metrics:
    """Setup and initialize metrics."""
    global _metrics
    _metrics = Metrics()
    return _metrics


def metrics_middleware(method: str, endpoint: str):
    """Decorator for tracking request metrics."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.track_request(method, endpoint):
                return await func(*args, **kwargs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.track_request(method, endpoint):
                return func(*args, **kwargs)

        return wrapper

    return decorator
