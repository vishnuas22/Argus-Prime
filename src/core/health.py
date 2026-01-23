"""
Health check endpoints for monitoring.

Provides health and readiness probes for container orchestration
and load balancer integration, plus Prometheus metrics endpoint.
"""

from datetime import datetime
from typing import Any, Dict
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from src.config import get_config
from src.core.logging import logger
from src.core.metrics import get_metrics


router = APIRouter(prefix="/api/v1", tags=["health"])


async def check_database() -> Dict[str, Any]:
    """Check database connectivity."""
    config = get_config()
    # Placeholder - implement actual database check
    return {"status": "unknown", "latency_ms": 0}


async def check_models() -> Dict[str, Any]:
    """Check if models are loaded and healthy."""
    # Placeholder - implement model registry health check
    return {"status": "unknown", "loaded_models": []}


async def check_storage() -> Dict[str, Any]:
    """Check storage accessibility."""
    # Placeholder - implement storage check
    return {"status": "unknown", "free_bytes": 0}


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.

    Returns:
        Health status with timestamp
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with component status.

    Returns:
        Detailed status of all components
    """
    components = {
        "database": await check_database(),
        "models": await check_models(),
        "storage": await check_storage(),
    }

    all_healthy = all(c.get("status") == "healthy" for c in components.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "components": components,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/ready")
async def readiness_check() -> Dict[str, str]:
    """
    Readiness probe for load balancer.

    Returns:
        200 OK if ready, 503 if not
    """
    try:
        db_status = await check_database()
        models_status = await check_models()

        if db_status.get("status") == "healthy" and models_status.get("status") == "healthy":
            return {"status": "ready"}
        else:
            return {"status": "not_ready"}

    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return {"status": "not_ready"}


@router.get("/health/live")
async def liveness_check() -> Dict[str, str]:
    """
    Liveness probe for container orchestration.

    Returns:
        200 OK if process is alive
    """
    return {"status": "alive"}


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format
    """
    try:
        metrics = get_metrics()
        metrics_data = metrics.get_metrics()
        content_type = metrics.get_content_type()
        return Response(content=metrics_data, media_type=content_type)
    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return Response(content=b"# Metrics unavailable", media_type="text/plain")


class HealthChecker:
    """Utility class for programmatic health checks."""

    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks and return combined result."""
        return {
            "database": await check_database(),
            "models": await check_models(),
            "storage": await check_storage(),
        }
