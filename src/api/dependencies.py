"""API dependencies for FastAPI dependency injection."""

from typing import Optional
from fastapi import Depends, HTTPException
from fastapi import UploadFile, File

from src.config import get_config
from src.core.logging import logger
from src.ingestion.analysis_service import get_analysis_service as _get_analysis_service
from src.ingestion.analysis_service import AnalysisService as RealAnalysisService
from src.ingestion.storage_service import get_storage_service as _get_storage_service
from src.ingestion.storage_service import StorageService as RealStorageService


class AnalysisService:
    """Stub service interface for analysis operations."""

    async def create_analysis(
        self,
        file: UploadFile,
        options: dict,
    ) -> str:
        """Create a new analysis request."""
        raise HTTPException(
            status_code=501,
            detail="Analysis service not yet implemented"
        )

    async def get_status(self, request_id: str):
        """Get status of an analysis request."""
        raise HTTPException(
            status_code=404,
            detail=f"Analysis request {request_id} not found"
        )

    async def get_result(self, request_id: str):
        """Get result of an analysis request."""
        raise HTTPException(
            status_code=404,
            detail=f"Analysis result {request_id} not found"
        )

    async def stream_progress(self, request_id: str):
        """Stream progress updates for an analysis request."""
        raise HTTPException(
            status_code=404,
            detail=f"Analysis stream {request_id} not found"
        )


class StorageService:
    """Stub service interface for file storage operations."""

    async def save_file(
        self,
        file: UploadFile,
        request_id: str,
    ) -> str:
        """Save uploaded file and return path."""
        raise HTTPException(
            status_code=501,
            detail="Storage service not yet implemented"
        )

    async def get_file_path(self, request_id: str, file_name: Optional[str] = None) -> Optional[str]:
        """Get path to stored file."""
        return None

    async def cleanup(self, request_id: str) -> None:
        """Clean up files for a request."""
        return None


async def get_analysis_service() -> RealAnalysisService:
    """Get real analysis service instance."""
    service = _get_analysis_service()
    if service is None:
        service = RealAnalysisService()
    return service


async def get_storage_service() -> RealStorageService:
    """Get real storage service instance."""
    service = _get_storage_service()
    if service is None:
        service = RealStorageService()
    return service


async def get_config_dependency():
    """Get configuration for dependency injection."""
    return get_config()
