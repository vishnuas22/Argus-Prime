"""
FastAPI router with REST endpoints.

Defines all API endpoints for file upload, analysis submission,
status polling, and result retrieval.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uuid

from .schemas import (
    AnalysisRequestSchema,
    AnalysisStatusSchema,
    ForensicReportSchema,
    Verdict,
    AnalysisStatus,
    ProgressUpdateSchema,
)
from .dependencies import (
    get_analysis_service,
    get_storage_service,
    get_config_dependency,
)
from src.core.logging import logger


router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    analysis_service=Depends(get_analysis_service),
):
    """
    Submit a file for deepfake analysis.

    Supports: video (mp4, avi, mov), audio (wav, mp3, flac),
    image (jpg, png, webp), text (txt, md, pdf).

    Returns request_id for polling/websocket connection.
    """
    logger.info(f"Received analysis request: {file.filename}")

    try:
        request_id = await analysis_service.create_analysis(
            file=file,
            options={},
        )

        return {
            "request_id": str(request_id),
            "status": AnalysisStatus.PENDING.value,
            "message": "Analysis queued. Poll /api/v1/analyze/{id}/status for updates.",
            "websocket_url": f"/api/v1/analyze/{request_id}/stream",
        }

    except Exception as e:
        logger.error(f"Analysis creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{request_id}/status", response_model=AnalysisStatusSchema)
async def get_status(
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """Get analysis progress and status."""
    status = await analysis_service.get_status(str(request_id))

    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")

    return status


@router.get("/analyze/{request_id}/result", response_model=ForensicReportSchema)
async def get_result(
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """Get complete analysis result."""
    result = await analysis_service.get_result(str(request_id))

    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")

    return result


@router.get("/analyses")
async def list_analyses(
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    analysis_service=Depends(get_analysis_service),
):
    """
    List analyses with pagination and filters.

    Args:
        status: Filter by status (pending, processing, completed, failed)
        file_type: Filter by file type (video, audio, text)
        limit: Maximum results per page (default 50)
        offset: Offset for pagination

    Returns:
        List of analyses with pagination info
    """
    result = await analysis_service.get_analyses(
        status=status,
        file_type=file_type,
        limit=limit,
        offset=offset,
    )
    return result


@router.get("/analyses/{request_id}")
async def get_analysis_details(
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """
    Get detailed analysis with all results.

    Returns complete analysis including video, audio, text results
    and trust score breakdown.
    """
    details = await analysis_service.get_analysis_details(str(request_id))

    if details is None:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return details


@router.websocket("/analyze/{request_id}/stream")
async def stream_progress(
    websocket: WebSocket,
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()

    try:
        async for update in analysis_service.stream_progress(str(request_id)):
            await websocket.send_json(update.dict())

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for request: {request_id}")
    except Exception as e:
        logger.error(f"WebSocket error for request {request_id}: {e}")
        await websocket.close()


@router.get("/health")
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "components": {
            "database": "connected",
            "models": "loaded",
            "storage": "available",
        },
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/config")
async def get_config(config=Depends(get_config_dependency)):
    """Get public configuration (non-sensitive)."""
    return {
        "max_file_size": config.storage.max_file_size,
        "supported_types": ["video", "audio", "image", "text"],
        "default_fps": config.analysis.default_fps,
    }
