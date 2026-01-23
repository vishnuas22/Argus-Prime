"""
Pydantic schemas for API request/response validation.

This module defines all data models used in API endpoints,
ensuring type safety and runtime validation.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class Verdict(str, Enum):
    """Classification verdict for analyzed content."""
    AUTHENTIC = "AUTHENTIC"
    MANIPULATED = "MANIPULATED"
    UNCERTAIN = "UNCERTAIN"


class FileType(str, Enum):
    """Supported file types for analysis."""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"


class AnalysisStatus(str, Enum):
    """Status of an analysis request."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class UploadFileSchema(BaseModel):
    """Schema for uploaded file metadata validation."""
    file_name: str = Field(..., max_length=255, description="Original file name")
    content_type: str = Field(
        ...,
        pattern=r"^(video|audio|image)/(mp4|avi|mov|wav|mp3|flac|png|jpg|jpeg|webp)$",
        description="MIME type of the file"
    )
    file_size: int = Field(
        ...,
        gt=0,
        le=500_000_000,
        description="File size in bytes (max 500MB)"
    )
    md5_hash: Optional[str] = Field(
        None,
        min_length=32,
        max_length=32,
        description="MD5 hash for integrity verification"
    )

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Ensure file name is not empty and has valid extension."""
        if not v.strip():
            raise ValueError("File name cannot be empty")
        return v.strip()


class AnalysisOptionsSchema(BaseModel):
    """Schema for analysis configuration options."""
    video: "VideoOptionsSchema" = Field(
        default_factory=lambda: VideoOptionsSchema(),
        description="Video analysis options"
    )
    audio: "AudioOptionsSchema" = Field(
        default_factory=lambda: AudioOptionsSchema(),
        description="Audio analysis options"
    )
    text: "TextOptionsSchema" = Field(
        default_factory=lambda: TextOptionsSchema(),
        description="Text analysis options"
    )


class VideoOptionsSchema(BaseModel):
    """Configuration for video analysis."""
    fps: int = Field(default=5, ge=1, le=30, description="Frames per second to extract")
    extract_faces: bool = Field(default=True, description="Extract and analyze faces")
    generate_heatmaps: bool = Field(default=True, description="Generate attention heatmaps")
    rppg_analysis: bool = Field(default=True, description="Include rPPG biological analysis")


class AudioOptionsSchema(BaseModel):
    """Configuration for audio analysis."""
    sample_rate: int = Field(default=16000, ge=8000, le=48000, description="Target sample rate")
    segment_duration: int = Field(default=2, ge=1, le=10, description="Segment duration in seconds")
    apply_vad: bool = Field(default=True, description="Apply voice activity detection")


class TextOptionsSchema(BaseModel):
    """Configuration for text analysis."""
    language: Optional[str] = Field(None, description="Force language detection (ISO code)")
    include_perplexity: bool = Field(default=True, description="Include perplexity analysis")
    include_burstiness: bool = Field(default=True, description="Include burstiness analysis")


class AnalysisRequestSchema(BaseModel):
    """Schema for analysis request."""
    file_id: uuid.UUID = Field(..., description="ID of uploaded file")
    analysis_options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis configuration options"
    )


class TrustScoreSchema(BaseModel):
    """Schema for trust score output."""
    overall_score: float = Field(..., ge=0, le=100, description="Overall trust score (0-100)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence level (0-1)")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence classification")
    breakdown: Dict[str, float] = Field(..., description="Score breakdown by modality")
    verdict: Verdict = Field(..., description="Classification verdict")


class FrameAnalysisSchema(BaseModel):
    """Schema for per-frame analysis results."""
    frame_number: int = Field(..., description="Frame number in video")
    timestamp: float = Field(..., description="Timestamp in seconds")
    face_bbox: Optional[List[float]] = Field(
        None,
        description="Face bounding box [x1, y1, x2, y2]"
    )
    manipulation_score: float = Field(..., ge=0, le=1, description="Manipulation probability")
    heatmap_path: Optional[str] = Field(None, description="Path to heatmap visualization")


class VideoResultSchema(BaseModel):
    """Schema for video analysis results."""
    frames_analyzed: int = Field(..., description="Number of frames analyzed")
    faces_detected: int = Field(..., description="Number of faces detected")
    overall_score: float = Field(..., ge=0, le=1, description="Overall manipulation score")
    per_frame_scores: List[FrameAnalysisSchema] = Field(
        default_factory=list,
        description="Per-frame analysis results"
    )
    heatmap_path: Optional[str] = Field(None, description="Path to aggregated heatmap")
    branch_scores: Dict[str, float] = Field(
        ...,
        description="Scores from each detection branch"
    )
    rppg_result: Optional[Dict[str, Any]] = Field(
        None,
        description="rPPG analysis results if enabled"
    )


class SegmentAnalysisSchema(BaseModel):
    """Schema for audio segment analysis."""
    start_time: float = Field(..., description="Segment start time in seconds")
    end_time: float = Field(..., description="Segment end time in seconds")
    manipulation_score: float = Field(..., ge=0, le=1, description="Manipulation probability")
    vocoder_class: Optional[str] = Field(None, description="Detected vocoder type")
    spectrogram_path: Optional[str] = Field(None, description="Path to spectrogram visualization")


class AudioResultSchema(BaseModel):
    """Schema for audio analysis results."""
    duration_seconds: float = Field(..., description="Total audio duration")
    segments_analyzed: int = Field(..., description="Number of segments analyzed")
    overall_score: float = Field(..., ge=0, le=1, description="Overall manipulation score")
    per_segment_scores: List[SegmentAnalysisSchema] = Field(
        default_factory=list,
        description="Per-segment analysis results"
    )
    vocoder_detected: Optional[str] = Field(None, description="Primary vocoder detected")
    artifact_evidence: Dict[str, float] = Field(
        ...,
        description="Evidence scores for different artifact types"
    )


class SentenceAnalysisSchema(BaseModel):
    """Schema for per-sentence text analysis."""
    sentence: str = Field(..., description="Original sentence text")
    start_char: int = Field(..., description="Starting character position")
    end_char: int = Field(..., description="Ending character position")
    manipulation_score: float = Field(..., ge=0, le=1, description="Manipulation probability")
    perplexity: float = Field(..., description="Perplexity score")
    burstiness: float = Field(..., description="Burstiness metric")


class TextResultSchema(BaseModel):
    """Schema for text analysis results."""
    word_count: int = Field(..., description="Total word count")
    language_detected: str = Field(..., description="Detected language")
    overall_score: float = Field(..., ge=0, le=1, description="Overall manipulation score")
    per_sentence_scores: List[SentenceAnalysisSchema] = Field(
        default_factory=list,
        description="Per-sentence analysis results"
    )
    perplexity_scores: Dict[str, float] = Field(
        ...,
        description="Perplexity scores from different models"
    )
    burstiness_metrics: Dict[str, float] = Field(
        ...,
        description="Burstiness analysis metrics"
    )


class LocalizationSchema(BaseModel):
    """Schema for manipulation localization results."""
    timestamps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Manipulation timestamps with confidence"
    )
    spatial_regions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Manipulation regions with polygons"
    )
    heatmap_overlay_path: Optional[str] = Field(
        None,
        description="Path to heatmap overlay image"
    )


class EvidenceSchema(BaseModel):
    """Schema for evidence collection."""
    heatmaps: List[str] = Field(default_factory=list, description="Paths to heatmap images")
    spectrograms: List[str] = Field(
        default_factory=list,
        description="Paths to spectrogram images"
    )
    localization: LocalizationSchema = Field(
        default_factory=LocalizationSchema,
        description="Localization of manipulations"
    )
    c2pa_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="C2PA provenance metadata"
    )
    adversarial_check: Optional[Dict[str, Any]] = Field(
        None,
        description="Adversarial defense results"
    )


class ForensicReportSchema(BaseModel):
    """Complete forensic analysis report schema."""
    report_id: uuid.UUID = Field(..., description="Unique report identifier")
    request_id: uuid.UUID = Field(..., description="Associated request identifier")
    created_at: datetime = Field(..., description="Report creation timestamp")
    file_info: Dict[str, Any] = Field(..., description="Original file information")
    executive_summary: Dict[str, Any] = Field(
        ...,
        description="Summary with verdict, score, and confidence"
    )
    trust_score: TrustScoreSchema = Field(..., description="Calculated trust scores")
    findings: Dict[str, Any] = Field(
        ...,
        description="Detailed findings per modality"
    )
    evidence: EvidenceSchema = Field(..., description="Evidence and visualizations")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended next steps"
    )
    raw_output: Optional[Dict[str, Any]] = Field(
        None,
        description="Raw model outputs for debugging"
    )


class AnalysisStatusSchema(BaseModel):
    """Schema for analysis status response."""
    request_id: uuid.UUID = Field(..., description="Request identifier")
    status: AnalysisStatus = Field(..., description="Current status")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    current_stage: Optional[str] = Field(None, description="Current processing stage")
    estimated_time_remaining: Optional[int] = Field(
        None,
        description="Estimated remaining time in seconds"
    )
    error_message: Optional[str] = Field(None, description="Error details if failed")

    class Config:
        populate_by_name = True
        alias_generator = lambda s: ''.join(
            word.capitalize() if i else word
            for i, word in enumerate(s.split('_'))
        )
        json_schema_extra = {
            "requestId": "uuid",
            "status": "pending",
            "progress": 0,
            "currentStage": "string",
            "estimatedTimeRemaining": 60,
            "errorMessage": "string or null",
        }


class AnalysisSubmitResponse(BaseModel):
    """Response schema for analysis submission."""
    request_id: str = Field(..., alias="requestId")
    status: str = Field(..., alias="status")
    message: str = Field(..., alias="message")
    websocket_url: str = Field(..., alias="websocketUrl")

    class Config:
        populate_by_name = True
        from_attributes = True
        alias_generator = lambda s: ''.join(
            word.capitalize() if i else word
            for i, word in enumerate(s.split('_'))
        )


class ProgressUpdateSchema(BaseModel):
    """Schema for WebSocket progress updates."""
    request_id: uuid.UUID = Field(..., description="Request identifier")
    stage: str = Field(..., description="Current processing stage")
    progress: int = Field(..., ge=0, le=100, description="Progress percentage")
    message: str = Field(..., description="Human-readable status message")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
