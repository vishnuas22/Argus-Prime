"""
Analysis Service.

Orchestrates the complete analysis pipeline:
1. File storage and validation
2. Modality-specific detection (video, audio, text)
3. Multi-modal score fusion
4. Report generation
5. Progress tracking
6. Database persistence
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, AsyncGenerator

from fastapi import UploadFile

from src.config import get_config
from src.core.logging import logger
from src.analysis import (
    VideoDetector,
    AudioDetector,
    TextDetector,
    VideoAnalysisResult,
    AudioAnalysisResult,
    TextAnalysisResult,
)
from src.fusion import FusionEngine, FusionResult, TrustScorer
from src.reporting import ReportGenerator, ForensicReport
from src.ingestion import StorageService, FileMetadata, get_storage_service
from src.db import get_database, AnalysisRepository, get_analysis_repository
from src.db.models import AnalysisStatus as DBAnalysisStatus

logger = logging.getLogger(__name__)


class AnalysisStatus(str, Enum):
    """Analysis status enum."""
    PENDING = "pending"
    EXTRACTING_FRAMES = "extracting_frames"
    RUNNING_VIDEO_DETECTION = "running_video_detection"
    RUNNING_AUDIO_DETECTION = "running_audio_detection"
    RUNNING_TEXT_DETECTION = "running_text_detection"
    FUSING_SCORES = "fusing_scores"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AnalysisRequest:
    """Analysis request tracking."""
    request_id: str
    file_metadata: FileMetadata
    status: AnalysisStatus
    progress: int
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    video_result: Optional[Dict] = None
    audio_result: Optional[Dict] = None
    text_result: Optional[Dict] = None
    fusion_result: Optional[Dict] = None
    report: Optional[Dict] = None

    def to_status_dict(self) -> Dict[str, Any]:
        """Convert to status response dict."""
        return {
            "request_id": self.request_id,
            "status": self.status.value,
            "progress": self.progress,
            "current_stage": self.status.value.replace("_", " ").title(),
            "estimated_time_remaining": self._estimate_time_remaining(),
            "file_info": {
                "file_name": self.file_metadata.file_name,
                "file_type": self.file_metadata.file_type,
                "file_size": self.file_metadata.file_size,
            },
            "error_message": self.error_message,
        }

    def _estimate_time_remaining(self) -> Optional[int]:
        """Estimate remaining time in seconds."""
        if self.status == AnalysisStatus.COMPLETED:
            return 0
        elif self.status == AnalysisStatus.FAILED:
            return None

        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        remaining_percent = 100 - self.progress

        if self.progress > 0:
            estimated_total = elapsed / (self.progress / 100)
            return int(estimated_total * (remaining_percent / 100))

        return None


@dataclass
class ProgressUpdate:
    """Progress update for streaming."""
    request_id: str
    stage: str
    progress: int
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class AnalysisService:
    """
    Analysis Service.

    Orchestrates the complete analysis pipeline:
    - File storage
    - Modality-specific detection
    - Multi-modal fusion
    - Report generation
    - Progress tracking
    - Database persistence

    Attributes:
        storage: File storage service
        video_detector: Video detection orchestrator
        audio_detector: Audio detection orchestrator
        text_detector: Text detection orchestrator
        fusion_engine: Score fusion engine
        report_generator: Report generator
        db: Database connection
    """

    _instance: Optional["AnalysisService"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __new__(cls):
        """Singleton pattern for analysis service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize analysis service."""
        if hasattr(self, "_initialized"):
            return

        config = get_config()
        self.storage = get_storage_service()
        self.db = get_database()

        self.video_detector: Optional[VideoDetector] = None
        self.audio_detector: Optional[AudioDetector] = None
        self.text_detector: Optional[TextDetector] = None

        self.fusion_engine = FusionEngine()
        self.trust_scorer = TrustScorer()
        self.report_generator = ReportGenerator()

        self.requests: Dict[str, AnalysisRequest] = {}

        self._initialized = True
        logger.info("AnalysisService initialized")

    async def _get_video_detector(self) -> VideoDetector:
        """Get or create video detector."""
        if self.video_detector is None:
            self.video_detector = VideoDetector(
                fps=get_config().analysis.default_fps,
                extract_faces=True,
                generate_heatmaps=True,
                use_gpu=True,
            )
        return self.video_detector

    async def _get_audio_detector(self) -> AudioDetector:
        """Get or create audio detector."""
        if self.audio_detector is None:
            self.audio_detector = AudioDetector(
                sample_rate=16000,
                segment_duration=2,
                apply_vad=True,
                use_gpu=True,
            )
        return self.audio_detector

    async def _get_text_detector(self) -> TextDetector:
        """Get or create text detector."""
        if self.text_detector is None:
            self.text_detector = TextDetector(
                include_perplexity=True,
                include_burstiness=True,
                use_gpu=False,
            )
        return self.text_detector

    async def create_analysis(
        self,
        file: UploadFile,
        options: Dict[str, Any],
    ) -> str:
        """
        Create a new analysis request.

        Args:
            file: Uploaded file
            options: Analysis options

        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())

        try:
            file_metadata = await self.storage.save_file(file, request_id)

            request = AnalysisRequest(
                request_id=request_id,
                file_metadata=file_metadata,
                status=AnalysisStatus.PENDING,
                progress=0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            self.requests[request_id] = request

            async with self.db.get_session() as session:
                repo = AnalysisRepository(session)
                file_hash = self._compute_file_hash(file_metadata.file_path)
                await repo.create_analysis(
                    file_name=file_metadata.file_name,
                    file_hash=file_hash,
                    file_type=file_metadata.file_type,
                    file_size=file_metadata.file_size,
                    content_type=file_metadata.content_type,
                )

            asyncio.create_task(self._run_analysis(request_id, options))

            logger.info(f"Created analysis request {request_id} for {file_metadata.file_name}")
            return request_id

        except Exception as e:
            logger.error(f"Failed to create analysis: {e}")
            raise

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    async def _run_analysis(
        self,
        request_id: str,
        options: Dict[str, Any],
    ) -> None:
        """
        Run full analysis pipeline.

        Args:
            request_id: Request identifier
            options: Analysis options
        """
        request = self.requests.get(request_id)
        if request is None:
            logger.error(f"Request {request_id} not found")
            return

        file_metadata = request.file_metadata
        file_path = file_metadata.file_path
        file_type = file_metadata.file_type

        async with self.db.get_session() as session:
            repo = AnalysisRepository(session)
            await repo.update_status(request_id, DBAnalysisStatus.PROCESSING.value)

        try:
            video_result = None
            audio_result = None
            text_result = None

            if file_type == "video":
                request.status = AnalysisStatus.EXTRACTING_FRAMES
                request.progress = 5
                await self._update_request(request)

                request.status = AnalysisStatus.RUNNING_VIDEO_DETECTION
                request.progress = 15
                await self._update_request(request)

                video_detector = await self._get_video_detector()
                video_result = await video_detector.analyze(
                    file_path,
                    progress_callback=lambda p, m: self._update_progress(request_id, p, m),
                )
                request.video_result = video_result.to_dict()
                request.progress = 60

            elif file_type == "audio":
                request.status = AnalysisStatus.RUNNING_AUDIO_DETECTION
                request.progress = 20
                await self._update_request(request)

                audio_detector = await self._get_audio_detector()
                audio_result = await audio_detector.analyze(
                    file_path,
                    progress_callback=lambda p, m: self._update_progress(request_id, p, m),
                )
                request.audio_result = audio_result.to_dict()
                request.progress = 60

            elif file_type == "text":
                content = Path(file_path).read_text()
                request.status = AnalysisStatus.RUNNING_TEXT_DETECTION
                request.progress = 20
                await self._update_request(request)

                text_detector = await self._get_text_detector()
                text_result = text_detector.analyze(
                    content,
                    progress_callback=lambda p, m: self._update_progress(request_id, p, m),
                )
                request.text_result = text_result.to_dict()
                request.progress = 60

            request.status = AnalysisStatus.FUSING_SCORES
            request.progress = 70
            await self._update_request(request)

            fusion_result = self.fusion_engine.fuse(
                video_score=video_result.overall_score if video_result else None,
                audio_score=audio_result.overall_score if audio_result else None,
                text_score=text_result.overall_score if text_result else None,
                video_confidence=video_result.confidence if video_result else None,
                audio_confidence=audio_result.confidence if audio_result else None,
                text_confidence=text_result.confidence if text_result else None,
                metadata={"rppg_quality": 0.8},
            )
            request.fusion_result = fusion_result.to_dict()

            request.status = AnalysisStatus.GENERATING_REPORT
            request.progress = 85
            await self._update_request(request)

            report = self.report_generator.generate_report(
                request_id=request_id,
                file_info=file_metadata.to_dict(),
                video_result=request.video_result,
                audio_result=request.audio_result,
                text_result=request.text_result,
                fusion_result=fusion_result,
            )
            request.report = report.to_dict()

            request.status = AnalysisStatus.COMPLETED
            request.progress = 100
            request.updated_at = datetime.utcnow()

            async with self.db.get_session() as session:
                repo = AnalysisRepository(session)

                trust_result = self.trust_scorer.compute_trust_score(fusion_result)

                if video_result:
                    await repo.save_video_result(
                        analysis_id=request_id,
                        frames_analyzed=getattr(video_result, 'frames_analyzed', 0),
                        faces_detected=getattr(video_result, 'faces_detected', 0),
                        overall_score=video_result.overall_score,
                        confidence=video_result.confidence,
                        is_manipulated=video_result.is_manipulated,
                        branch_scores=getattr(video_result, 'branch_scores', None),
                        per_frame_scores=getattr(video_result, 'per_frame_scores', None),
                    )

                if audio_result:
                    await repo.save_audio_result(
                        analysis_id=request_id,
                        duration_seconds=getattr(audio_result, 'duration_seconds', 0.0),
                        segments_analyzed=getattr(audio_result, 'segments_analyzed', 0),
                        overall_score=audio_result.overall_score,
                        confidence=audio_result.confidence,
                        is_manipulated=audio_result.is_manipulated,
                        vocoder_detected=getattr(audio_result, 'vocoder_detected', None),
                        artifact_evidence=getattr(audio_result, 'artifact_evidence', None),
                        per_segment_scores=getattr(audio_result, 'per_segment_scores', None),
                    )

                if text_result:
                    await repo.save_text_result(
                        analysis_id=request_id,
                        word_count=getattr(text_result, 'word_count', 0),
                        language_detected=getattr(text_result, 'language_detected', 'unknown'),
                        overall_score=text_result.overall_score,
                        confidence=text_result.confidence,
                        is_manipulated=text_result.is_manipulated,
                        perplexity_scores=getattr(text_result, 'perplexity_scores', None),
                        burstiness_metrics=getattr(text_result, 'burstiness_metrics', None),
                        per_sentence_scores=getattr(text_result, 'per_sentence_scores', None),
                    )

                await repo.save_trust_score(
                    analysis_id=request_id,
                    overall_score=trust_result["overall_score"],
                    confidence=trust_result["confidence"],
                    confidence_level=trust_result["confidence_level"],
                    verdict=trust_result["verdict"],
                    breakdown=trust_result["breakdown"],
                    fused_score=trust_result["fused_score"],
                    weights_used=fusion_result.weights_used,
                    consistency_score=fusion_result.consistency_score,
                )

                await repo.update_status(request_id, DBAnalysisStatus.COMPLETED.value)

            logger.info(f"Analysis {request_id} completed successfully")

        except Exception as e:
            logger.error(f"Analysis {request_id} failed: {e}")
            request.status = AnalysisStatus.FAILED
            request.error_message = str(e)
            request.updated_at = datetime.utcnow()

            async with self.db.get_session() as session:
                repo = AnalysisRepository(session)
                await repo.update_status(
                    request_id,
                    DBAnalysisStatus.FAILED.value,
                    error_message=str(e),
                )

    async def _update_request(self, request: AnalysisRequest) -> None:
        """Update request in storage."""
        request.updated_at = datetime.utcnow()

    async def _update_progress(
        self,
        request_id: str,
        progress: int,
        message: str,
    ) -> None:
        """Update progress for a request."""
        request = self.requests.get(request_id)
        if request:
            request.progress = progress
            request.updated_at = datetime.utcnow()

    async def get_status(self, request_id: str) -> Optional[AnalysisRequest]:
        """Get status of an analysis request."""
        return self.requests.get(request_id)

    async def get_result(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get complete analysis result."""
        request = self.requests.get(request_id)
        if request is None:
            return None

        if request.status == AnalysisStatus.COMPLETED:
            return request.report

        if request.status == AnalysisStatus.FAILED:
            return {
                "request_id": request_id,
                "status": "failed",
                "error": request.error_message,
            }

        return None

    async def get_analyses(
        self,
        status: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Get analyses with pagination and filters from database.

        Args:
            status: Filter by status
            file_type: Filter by file type
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            Dict with analyses list and total count
        """
        async with self.db.get_session() as session:
            repo = AnalysisRepository(session)
            analyses = await repo.get_analyses(
                status=status,
                file_type=file_type,
                limit=limit,
                offset=offset,
            )
            total = await repo.count_analyses(
                status=status,
                file_type=file_type,
            )

        return {
            "analyses": [a.to_dict() for a in analyses],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    async def get_analysis_details(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed analysis with all results.

        Args:
            request_id: Analysis request ID

        Returns:
            Dict with analysis and all results, or None if not found
        """
        async with self.db.get_session() as session:
            repo = AnalysisRepository(session)
            analysis = await repo.get_analysis_with_results(request_id)

            if analysis is None:
                return None

            result = analysis.to_dict()

            if analysis.video_result:
                result["video_result"] = analysis.video_result.to_dict()
            if analysis.audio_result:
                result["audio_result"] = analysis.audio_result.to_dict()
            if analysis.text_result:
                result["text_result"] = analysis.text_result.to_dict()
            if analysis.trust_score:
                result["trust_score"] = analysis.trust_score.to_dict()

            return result

    async def stream_progress(self, request_id: str) -> AsyncGenerator[ProgressUpdate, None]:
        """Stream progress updates for a request."""
        last_progress = -1

        while True:
            request = self.requests.get(request_id)
            if request is None:
                break

            if request.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
                break

            if request.progress != last_progress:
                update = ProgressUpdate(
                    request_id=request_id,
                    stage=request.status.value,
                    progress=request.progress,
                    message=f"Processing: {request.status.value.replace('_', ' ').title()}",
                )
                yield update
                last_progress = request.progress

            await asyncio.sleep(0.5)

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        return {
            "video_detector": "ForensicFlow + DistilDIRE",
            "audio_detector": "RawNet2 + Transcrypt",
            "text_detector": "SpecDetect + PerplexityDetector",
            "fusion_engine": "Adaptive Gated Fusion",
            "requests_tracked": len(self.requests),
            "database": "connected",
        }


_analysis_service: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """Get or create analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
