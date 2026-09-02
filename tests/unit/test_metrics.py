"""
Unit tests for Prometheus Metrics Module.
"""

import pytest
from prometheus_client import CollectorRegistry

from src.core.metrics import Metrics, setup_metrics, get_metrics


class TestMetrics:
    """Test Metrics collector."""

    def test_initialization(self):
        """Test metrics initialization."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        assert metrics.config.enabled is True
        assert metrics.config.path == "/metrics"
        assert metrics.config.port == 9090

    def test_request_tracking(self):
        """Test request tracking context manager."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        with metrics.track_request("GET", "/api/v1/health"):
            pass

        assert metrics.requests_in_progress.labels(
            method="GET", endpoint="/api/v1/health"
        )._value.get() == 0

    def test_analysis_tracking(self):
        """Test analysis tracking."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.track_analysis(
            modality="video",
            verdict="MANIPULATED",
            duration=2.5,
            trust_score=87.5,
            file_type="video/mp4",
            file_size=10485760,
        )

        assert metrics.analyses_total.labels(
            modality="video", verdict="MANIPULATED"
        )._value.get() == 1

    def test_model_inference_tracking(self):
        """Test model inference tracking."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.track_model_inference(
            model_name="forensic_flow",
            modality="video",
            duration=0.5,
            success=True,
        )

        assert metrics.model_inferences.labels(
            model_name="forensic_flow", modality="video"
        )._value.get() == 1

    def test_model_load_tracking(self):
        """Test model load tracking."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.track_model_load("forensic_flow", duration=5.2)

        assert metrics.model_load_duration.labels(
            model_name="forensic_flow"
        )._sum.get() == 5.2

    def test_memory_usage_update(self):
        """Test memory usage tracking."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.update_memory_usage("models", 1024000)

        assert metrics.memory_usage.labels(component="models")._value.get() == 1024000

    def test_gpu_memory_update(self):
        """Test GPU memory tracking."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.update_gpu_memory(2048000000)

        assert metrics.gpu_memory._value.get() == 2048000000

    def test_queue_size_update(self):
        """Test queue size tracking."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.update_queue_size(5)

        assert metrics.queue_size._value.get() == 5

    def test_models_loaded_update(self):
        """Test loaded models count."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics.update_models_loaded(3)

        assert metrics.models_loaded._value.get() == 3

    def test_get_metrics(self):
        """Test metrics export."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        metrics_data = metrics.get_metrics()

        assert isinstance(metrics_data, bytes)
        assert len(metrics_data) > 0

    def test_get_content_type(self):
        """Test metrics content type."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        content_type = metrics.get_content_type()

        assert "text/plain" in content_type

    def test_get_status(self):
        """Test metrics status."""
        registry = CollectorRegistry()
        metrics = Metrics(registry=registry)

        status = metrics.get_status()

        assert status["enabled"] is True
        assert status["path"] == "/metrics"
        assert status["port"] == 9090
        assert "registry_count" in status


class TestMetricsGlobalInstance:
    """Test global metrics instance management."""

    def test_setup_metrics_function(self):
        """Test setup_metrics function creates instance."""
        from src.core.metrics import setup_metrics

        metrics = setup_metrics()

        assert metrics is not None
        assert isinstance(metrics, Metrics)

    def test_metrics_decorator_exists(self):
        """Test decorator is importable."""
        from src.core.metrics import metrics_middleware

        assert callable(metrics_middleware)

    def test_metrics_decorator_returns_function(self):
        """Test decorator returns a callable."""
        from src.core.metrics import metrics_middleware

        decorator = metrics_middleware("GET", "/test")
        assert callable(decorator)


class TestMetricsMiddleware:
    """Test metrics middleware decorator."""

    def test_decorator_exists(self):
        """Test decorator is importable."""
        from src.core.metrics import metrics_middleware

        assert callable(metrics_middleware)

    def test_decorator_returns_function(self):
        """Test decorator returns a callable."""
        from src.core.metrics import metrics_middleware

        decorator = metrics_middleware("GET", "/test")
        assert callable(decorator)


class TestMetricsConfigIntegration:
    """Test metrics integration with config."""

    def test_metrics_config_exists(self):
        """Test MetricsConfig is properly defined."""
        from src.config import get_config

        config = get_config()
        assert hasattr(config, "metrics")

    def test_metrics_config_defaults(self):
        """Test metrics config default values."""
        from src.config import Config

        config = Config()
        assert config.metrics.enabled is True
        assert config.metrics.port == 9090
        assert config.metrics.path == "/metrics"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
