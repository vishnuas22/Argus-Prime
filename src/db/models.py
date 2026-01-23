"""
SQLAlchemy ORM Models.

Database models for analysis tracking and results storage.
Key Models:
- Analysis: request_id, file_hash, status, created_at
- VideoResult: request_id, frames_analyzed, overall_score
- AudioResult: request_id, duration, overall_score
- TextResult: request_id, word_count, overall_score
- TrustScore: request_id, score, confidence, breakdown
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

import sys
sys.path.insert(0, '/Users/Dataghost/Argus_Prime')


class AnalysisStatus(str, Enum):
    """Analysis status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType(str, Enum):
    """File type values."""
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for ORM models."""
    pass


class Analysis(Base):
    """Analysis request tracking."""
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    file_type: Mapped[str] = mapped_column(String(20), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    content_type: Mapped[str] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default=AnalysisStatus.PENDING.value,
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    video_result: Mapped[Optional["VideoResult"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )
    audio_result: Mapped[Optional["AudioResult"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )
    text_result: Mapped[Optional["TextResult"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )
    trust_score: Mapped[Optional["TrustScore"]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class VideoResult(Base):
    """Video analysis result."""
    __tablename__ = "video_results"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analyses.id"),
        nullable=False,
        unique=True,
    )
    frames_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    faces_detected: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    is_manipulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    branch_scores: Mapped[dict] = mapped_column(JSON, nullable=True)
    per_frame_scores: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    heatmap_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="video_result")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "frames_analyzed": self.frames_analyzed,
            "faces_detected": self.faces_detected,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "branch_scores": self.branch_scores,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AudioResult(Base):
    """Audio analysis result."""
    __tablename__ = "audio_results"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analyses.id"),
        nullable=False,
        unique=True,
    )
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    segments_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    is_manipulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    vocoder_detected: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    artifact_evidence: Mapped[dict] = mapped_column(JSON, nullable=True)
    per_segment_scores: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="audio_result")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "duration_seconds": self.duration_seconds,
            "segments_analyzed": self.segments_analyzed,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "vocoder_detected": self.vocoder_detected,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TextResult(Base):
    """Text analysis result."""
    __tablename__ = "text_results"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analyses.id"),
        nullable=False,
        unique=True,
    )
    word_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    language_detected: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    overall_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    is_manipulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    perplexity_scores: Mapped[dict] = mapped_column(JSON, nullable=True)
    burstiness_metrics: Mapped[dict] = mapped_column(JSON, nullable=True)
    per_sentence_scores: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="text_result")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "word_count": self.word_count,
            "language_detected": self.language_detected,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "is_manipulated": self.is_manipulated,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class TrustScore(Base):
    """Trust score and fusion result."""
    __tablename__ = "trust_scores"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analyses.id"),
        nullable=False,
        unique=True,
    )
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_level: Mapped[str] = mapped_column(String(20), nullable=False)
    verdict: Mapped[str] = mapped_column(String(30), nullable=False)
    breakdown: Mapped[dict] = mapped_column(JSON, nullable=True)
    fused_score: Mapped[float] = mapped_column(Float, nullable=False)
    weights_used: Mapped[dict] = mapped_column(JSON, nullable=True)
    consistency_score: Mapped[float] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    analysis: Mapped["Analysis"] = relationship(back_populates="trust_score")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level,
            "verdict": self.verdict,
            "breakdown": self.breakdown,
            "fused_score": self.fused_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ShareLink(Base):
    """Share link for analysis results."""
    __tablename__ = "share_links"

    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )
    analysis_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analyses.id"),
        nullable=False,
    )
    share_token: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        index=True,
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "share_token": self.share_token,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }
