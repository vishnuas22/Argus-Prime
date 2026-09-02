"""
Health check endpoints for monitoring.

Provides health and readiness probes for container orchestration
and load balancer integration, plus Prometheus metrics endpoint.
"""

import time
import os
import shutil
from datetime import datetime
from typing import Any, Dict
from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse
from sqlalchemy import text

from src.config import get_config
from src.core.logging import logger
from src.core.metrics import get_metrics
from src.db.database import get_database
from src.models.registry import get_model_registry


router = APIRouter(prefix="/api/v1", tags=["health"])


async def check_database() -> Dict[str, Any]:
    """
    Check database connectivity and latency.
    
    Returns:
        Dict with status, latency_ms, and connection details
    """
    try:
        db = get_database()
        start_time = time.time()
        
        async with db.get_session() as session:
            result = await session.execute(text("SELECT 1"))
            result.scalar()
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "status": "healthy",
            "latency_ms": latency_ms,
            "pool_size": db.db_config.pool_size,
            "max_overflow": db.db_config.max_overflow,
        }
    except Exception as e:
        logger.error(f"Database health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "latency_ms": 0,
            "error": str(e),
        }


async def check_models() -> Dict[str, Any]:
    """
    Check if detection models are loaded and healthy.
    
    Returns:
        Dict with status and list of loaded models
    """
    try:
        registry = get_model_registry()
        loaded_models = registry.list_models()
        
        if not loaded_models:
            return {
                "status": "degraded",
                "loaded_models": [],
                "message": "No models loaded yet",
            }
        
        return {
            "status": "healthy",
            "loaded_models": [
                {
                    "id": model_id,
                    "type": info.get("type", "unknown"),
                    "version": info.get("version", "unknown"),
                }
                for model_id, info in loaded_models.items()
            ],
            "total_count": len(loaded_models),
        }
    except Exception as e:
        logger.error(f"Model health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "loaded_models": [],
            "error": str(e),
        }


async def check_storage() -> Dict[str, Any]:
    """
    Check storage accessibility and available space.
    
    Returns:
        Dict with status, free space, and path info
    """
    try:
        config = get_config()
        temp_dir = config.storage.temp_dir
        
        os.makedirs(temp_dir, exist_ok=True)
        
        disk_stats = shutil.disk_usage(temp_dir)
        free_bytes = disk_stats.free
        total_bytes = disk_stats.total
        used_percent = (disk_stats.used / total_bytes) * 100 if total_bytes > 0 else 0
        
        test_file = os.path.join(temp_dir, ".health_check")
        try:
            with open(test_file, 'w') as f:
                f.write("health_check")
            os.remove(test_file)
            writable = True
        except Exception:
            writable = False
        
        if not writable or used_percent > 95:
            status = "unhealthy"
        elif used_percent > 80:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
            "used_percent": round(used_percent, 2),
            "path": temp_dir,
            "writable": writable,
        }
    except Exception as e:
        logger.error(f"Storage health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "free_bytes": 0,
            "error": str(e),
        }


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
