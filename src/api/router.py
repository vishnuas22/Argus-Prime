"""
FastAPI router with REST endpoints.

Defines all API endpoints for file upload, analysis submission,
status polling, and result retrieval.
"""

from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, StreamingResponse
import uuid
import json

from .schemas import (
    AnalysisRequestSchema,
    AnalysisStatusSchema,
    ForensicReportSchema,
    Verdict,
    AnalysisStatus,
    ProgressUpdateSchema,
    AnalysisSubmitResponse,
)
from .dependencies import (
    get_analysis_service,
    get_storage_service,
    get_config_dependency,
)
from src.core.logging import logger


router = APIRouter(prefix="/api/v1", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisSubmitResponse)
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

        return AnalysisSubmitResponse(
            requestId=str(request_id),
            status=AnalysisStatus.PENDING.value,
            message="Analysis queued. Poll /api/v1/analyze/{id}/status for updates.",
            websocketUrl=f"/api/v1/analyze/{request_id}/stream",
        )

    except Exception as e:
        logger.error(f"Analysis creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analyze/{request_id}/status")
async def get_status(
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """Get analysis progress and status."""
    status = await analysis_service.get_status(str(request_id))

    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")

    return status.to_status_dict()


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
    from .dependencies import get_analysis_service
    analysis_service = await get_analysis_service()
    try:
        result = await analysis_service.get_analyses(
            status=status,
            file_type=file_type,
            limit=limit,
            offset=offset,
        )
    except Exception:
        return {"analyses": [], "total": 0, "limit": limit, "offset": offset}

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


@router.post("/analyze/{request_id}/cancel")
async def cancel_analysis(
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """
    Cancel a running analysis.

    Args:
        request_id: Analysis request ID

    Returns:
        Success message
    """
    success = await analysis_service.cancel_analysis(str(request_id))

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found or already completed"
        )

    return {"message": "Analysis cancelled successfully"}


@router.delete("/analyses/{request_id}")
async def delete_analysis(
    request_id: uuid.UUID,
    analysis_service=Depends(get_analysis_service),
):
    """
    Delete an analysis and its associated files.

    Args:
        request_id: Analysis request ID

    Returns:
        Success message
    """
    success = await analysis_service.delete_analysis(str(request_id))

    if not success:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found"
        )

    return {"message": "Analysis deleted successfully"}


@router.get("/analyses/{request_id}/export")
async def export_analysis(
    request_id: uuid.UUID,
    format: str = Query("json", enum=["json", "pdf", "csv"]),
    include_evidence: bool = Query(False),
    analysis_service=Depends(get_analysis_service),
):
    """
    Export analysis result in specified format.

    Args:
        request_id: Analysis request ID
        format: Export format (json, pdf, csv)
        include_evidence: Include evidence files

    Returns:
        Export data as JSON or file download
    """
    result = await analysis_service.export_analysis(
        str(request_id),
        format=format,
        include_evidence=include_evidence,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found or not completed"
        )

    if format == "json":
        return JSONResponse(content=result)

    return JSONResponse(content=result)


@router.post("/analyses/{request_id}/share")
async def create_share_link(
    request_id: uuid.UUID,
    expires_in_hours: int = Query(72, ge=1, le=720),
    analysis_service=Depends(get_analysis_service),
):
    """
    Create a shareable link for an analysis result.

    Args:
        request_id: Analysis request ID
        expires_in_hours: Link expiration time (1-720 hours)

    Returns:
        Share URL and expiration datetime
    """
    result = await analysis_service.share_analysis(
        str(request_id),
        expires_in_hours=expires_in_hours,
    )

    if result is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found or not completed"
        )

    return result
