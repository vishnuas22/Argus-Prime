"""
Structured logging configuration.

Provides JSON-structured logging with consistent formatting
for production observability.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import get_config


class JSONFormatter(logging.Formatter):
    """JSON-structured log formatter for production logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields (using setattr for type checking)
        if hasattr(record, "extra_data"):
            log_data["extra"] = getattr(record, "extra_data")

        return json.dumps(log_data)


class StructuredLogger:
    """
    Structured logger with consistent field formatting.

    Usage:
        logger.info("Analysis completed", extra={"request_id": "uuid", "duration_ms": 150})
    """

    def __init__(self, name: str = "argus"):
        self.logger = logging.getLogger(name)
        self._setup_handler()

    def _setup_handler(self) -> None:
        """Configure log handlers with JSON formatting."""
        config = get_config()
        log_level = getattr(logging, config.logging.level.upper(), logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        handler.setLevel(log_level)

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(handler)
            self.logger.setLevel(log_level)

    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
    ) -> None:
        """Internal log method with structured fields."""
        extra_data = extra or {}
        self.logger.log(
            level,
            message,
            extra={"extra_data": extra_data} if extra_data else {},
            exc_info=exc_info,
        )

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self._log(logging.ERROR, message, kwargs, exc_info=True)


def setup_logging() -> StructuredLogger:
    """
    Initialize structured logging.

    Returns:
        Configured StructuredLogger instance
    """
    return StructuredLogger()


logger = StructuredLogger()
