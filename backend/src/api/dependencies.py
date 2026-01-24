"""
API dependencies for FastAPI dependency injection.

Provides dependency injection for services and configuration.
All dependencies are production-ready with proper error handling.
"""

from typing import Optional
from fastapi import Depends, HTTPException

from src.config import get_config, Config
from src.core.logging import logger
from src.ingestion.analysis_service import (
    get_analysis_service as _get_analysis_service,
    AnalysisService,
)
from src.ingestion.storage_service import (
    get_storage_service as _get_storage_service,
    StorageService,
)


async def get_analysis_service() -> AnalysisService:
    """
    Get analysis service instance for dependency injection.
    
    Returns:
        AnalysisService: Production analysis service
        
    Raises:
        HTTPException: If service initialization fails
    """
    try:
        service = _get_analysis_service()
        if service is None:
            service = AnalysisService()
        return service
    except Exception as e:
        logger.error(f"Failed to get analysis service: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Analysis service unavailable"
        )


async def get_storage_service() -> StorageService:
    """
    Get storage service instance for dependency injection.
    
    Returns:
        StorageService: Production storage service
        
    Raises:
        HTTPException: If service initialization fails
    """
    try:
        service = _get_storage_service()
        if service is None:
            service = StorageService()
        return service
    except Exception as e:
        logger.error(f"Failed to get storage service: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Storage service unavailable"
        )


async def get_config_dependency() -> Config:
    """
    Get configuration for dependency injection.
    
    Returns:
        Config: Application configuration
    """
    return get_config()
