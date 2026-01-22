"""
Data Access Layer - Repository Pattern.

Provides async methods for database operations.
Methods:
- create_analysis(analysis: Analysis) -> Analysis
- get_analysis(request_id: str) -> Analysis
- update_status(request_id: str, status: str) -> None
- save_result(request_id: str, result: VideoResult) -> None
- get_analyses(filters: dict, pagination: dict) -> List[Analysis]
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.logging import logger
from src.db import (
    Analysis,
    VideoResult,
    AudioResult,
    TextResult,
    TrustScore,
    AnalysisStatus,
    get_database,
)


class AnalysisRepository:
    """Repository for analysis operations."""

    def __init__(self, session: AsyncSession):
        """Initialize repository with session."""
        self.session = session

    async def create_analysis(
        self,
        file_name: str,
        file_hash: str,
        file_type: str,
        file_size: int,
        content_type: Optional[str] = None,
    ) -> Analysis:
        """
        Create new analysis record.

        Args:
            file_name: Original file name
            file_hash: MD5/SHA256 hash of file
            file_type: Type of file (video/audio/text)
            file_size: File size in bytes
            content_type: MIME type

        Returns:
            Created Analysis object
        """
        analysis = Analysis(
            file_name=file_name,
            file_hash=file_hash,
            file_type=file_type,
            file_size=file_size,
            content_type=content_type,
            status=AnalysisStatus.PENDING.value,
        )

        self.session.add(analysis)
        await self.session.flush()
        await self.session.refresh(analysis)

        logger.info(f"Created analysis record: {analysis.id}")
        return analysis

    async def get_analysis(self, request_id: str) -> Optional[Analysis]:
        """
        Get analysis by ID.

        Args:
            request_id: Analysis request ID

        Returns:
            Analysis object or None
        """
        stmt = select(Analysis).where(Analysis.id == request_id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_analysis_with_results(self, request_id: str) -> Optional[Analysis]:
        """
        Get analysis with all related results.

        Args:
            request_id: Analysis request ID

        Returns:
            Analysis object with relationships or None
        """
        stmt = (
            select(Analysis)
            .where(Analysis.id == request_id)
            .options(
                __import__('sqlalchemy.orm').selectinload(Analysis.video_result),
                __import__('sqlalchemy.orm').selectinload(Analysis.audio_result),
                __import__('sqlalchemy.orm').selectinload(Analysis.text_result),
                __import__('sqlalchemy.orm').selectinload(Analysis.trust_score),
            )
        )
        result = await self.session.execute(stmt)
        return result.unique().scalar_one_or_none()

    async def update_status(
        self,
        request_id: str,
        status: str,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Update analysis status.

        Args:
            request_id: Analysis request ID
            status: New status
            error_message: Error message if failed

        Returns:
            True if updated, False if not found
        """
        stmt = (
            update(Analysis)
            .where(Analysis.id == request_id)
            .values(
                status=status,
                error_message=error_message,
                updated_at=datetime.utcnow(),
            )
        )
        result = await self.session.execute(stmt)

        if result.rowcount > 0:
            logger.info(f"Updated analysis {request_id} status to {status}")
            return True

        return False

    async def save_video_result(
        self,
        analysis_id: str,
        frames_analyzed: int,
        faces_detected: int,
        overall_score: float,
        confidence: float,
        is_manipulated: bool,
        branch_scores: Optional[Dict] = None,
        per_frame_scores: Optional[List] = None,
    ) -> VideoResult:
        """
        Save video analysis result.

        Args:
            analysis_id: Analysis ID
            frames_analyzed: Number of frames analyzed
            faces_detected: Number of faces detected
            overall_score: Manipulation score
            confidence: Confidence level
            is_manipulated: Whether manipulation detected
            branch_scores: Per-branch scores
            per_frame_scores: Frame-level scores

        Returns:
            Created VideoResult object
        """
        result = VideoResult(
            analysis_id=analysis_id,
            frames_analyzed=frames_analyzed,
            faces_detected=faces_detected,
            overall_score=overall_score,
            confidence=confidence,
            is_manipulated=is_manipulated,
            branch_scores=branch_scores,
            per_frame_scores=per_frame_scores,
        )

        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)

        logger.info(f"Saved video result for analysis {analysis_id}")
        return result

    async def save_audio_result(
        self,
        analysis_id: str,
        duration_seconds: float,
        segments_analyzed: int,
        overall_score: float,
        confidence: float,
        is_manipulated: bool,
        vocoder_detected: Optional[str] = None,
        artifact_evidence: Optional[Dict] = None,
        per_segment_scores: Optional[List] = None,
    ) -> AudioResult:
        """Save audio analysis result."""
        result = AudioResult(
            analysis_id=analysis_id,
            duration_seconds=duration_seconds,
            segments_analyzed=segments_analyzed,
            overall_score=overall_score,
            confidence=confidence,
            is_manipulated=is_manipulated,
            vocoder_detected=vocoder_detected,
            artifact_evidence=artifact_evidence,
            per_segment_scores=per_segment_scores,
        )

        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)

        logger.info(f"Saved audio result for analysis {analysis_id}")
        return result

    async def save_text_result(
        self,
        analysis_id: str,
        word_count: int,
        language_detected: str,
        overall_score: float,
        confidence: float,
        is_manipulated: bool,
        perplexity_scores: Optional[Dict] = None,
        burstiness_metrics: Optional[Dict] = None,
        per_sentence_scores: Optional[List] = None,
    ) -> TextResult:
        """Save text analysis result."""
        result = TextResult(
            analysis_id=analysis_id,
            word_count=word_count,
            language_detected=language_detected,
            overall_score=overall_score,
            confidence=confidence,
            is_manipulated=is_manipulated,
            perplexity_scores=perplexity_scores,
            burstiness_metrics=burstiness_metrics,
            per_sentence_scores=per_sentence_scores,
        )

        self.session.add(result)
        await self.session.flush()
        await self.session.refresh(result)

        logger.info(f"Saved text result for analysis {analysis_id}")
        return result

    async def save_trust_score(
        self,
        analysis_id: str,
        overall_score: float,
        confidence: float,
        confidence_level: str,
        verdict: str,
        breakdown: Dict[str, Any],
        fused_score: float,
        weights_used: Dict[str, float],
        consistency_score: Optional[float] = None,
    ) -> TrustScore:
        """Save trust score result."""
        score = TrustScore(
            analysis_id=analysis_id,
            overall_score=overall_score,
            confidence=confidence,
            confidence_level=confidence_level,
            verdict=verdict,
            breakdown=breakdown,
            fused_score=fused_score,
            weights_used=weights_used,
            consistency_score=consistency_score,
        )

        self.session.add(score)
        await self.session.flush()
        await self.session.refresh(score)

        logger.info(f"Saved trust score for analysis {analysis_id}")
        return score

    async def get_analyses(
        self,
        status: Optional[str] = None,
        file_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Analysis]:
        """
        Get analyses with filters and pagination.

        Args:
            status: Filter by status
            file_type: Filter by file type
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of Analysis objects
        """
        stmt = select(Analysis)

        if status:
            stmt = stmt.where(Analysis.status == status)
        if file_type:
            stmt = stmt.where(Analysis.file_type == file_type)

        stmt = stmt.order_by(Analysis.created_at.desc())
        stmt = stmt.offset(offset).limit(limit)

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def count_analyses(
        self,
        status: Optional[str] = None,
        file_type: Optional[str] = None,
    ) -> int:
        """
        Count analyses with filters.

        Args:
            status: Filter by status
            file_type: Filter by file type

        Returns:
            Count of analyses
        """
        stmt = select(func.count(Analysis.id))

        if status:
            stmt = stmt.where(Analysis.status == status)
        if file_type:
            stmt = stmt.where(Analysis.file_type == file_type)

        result = await self.session.execute(stmt)
        return result.scalar() or 0


async def get_analysis_repository() -> AnalysisRepository:
    """
    Get analysis repository for dependency injection.

    Returns:
        AnalysisRepository instance
    """
    db = get_database()
    async with db.get_session() as session:
        yield AnalysisRepository(session)
