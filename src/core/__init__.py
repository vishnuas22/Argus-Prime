"""Core module - Security, logging, metrics, and health checks."""

from .logging import setup_logging, logger
from .security import (
    SecurityCheckResult,
    MalwareScanner,
    check_file_signature,
)
from .metrics import (
    Metrics,
    get_metrics,
    setup_metrics,
    metrics_middleware,
)

__all__ = [
    "setup_logging",
    "logger",
    "SecurityCheckResult",
    "MalwareScanner",
    "check_file_signature",
    "Metrics",
    "get_metrics",
    "setup_metrics",
    "metrics_middleware",
]
