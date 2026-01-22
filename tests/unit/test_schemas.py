"""
Unit tests for core schemas and configurations.
"""

import pytest
import uuid
from datetime import datetime

from src.api.schemas import (
    Verdict,
    FileType,
    AnalysisStatus,
    ConfidenceLevel,
    UploadFileSchema,
    TrustScoreSchema,
    FrameAnalysisSchema,
    VideoResultSchema,
    AnalysisStatusSchema,
)


class TestEnums:
    """Test enum values."""

    def test_verdict_values(self):
        assert Verdict.AUTHENTIC.value == "AUTHENTIC"
        assert Verdict.MANIPULATED.value == "MANIPULATED"
        assert Verdict.UNCERTAIN.value == "UNCERTAIN"

    def test_file_type_values(self):
        assert FileType.VIDEO.value == "video"
        assert FileType.AUDIO.value == "audio"
        assert FileType.IMAGE.value == "image"
        assert FileType.TEXT.value == "text"

    def test_analysis_status_values(self):
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.PROCESSING.value == "processing"
        assert AnalysisStatus.COMPLETED.value == "completed"
        assert AnalysisStatus.FAILED.value == "failed"

    def test_confidence_level_values(self):
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


class TestUploadFileSchema:
    """Test UploadFileSchema validation."""

    def test_valid_upload_file(self):
        schema = UploadFileSchema(
            file_name="test.mp4",
            content_type="video/mp4",
            file_size=1000000,
            md5_hash="d41d8cd98f00b204e9800998ecf8427e",
        )
        assert schema.file_name == "test.mp4"
        assert schema.content_type == "video/mp4"
        assert schema.file_size == 1000000

    def test_missing_md5_hash(self):
        schema = UploadFileSchema(
            file_name="test.mp4",
            content_type="video/mp4",
            file_size=1000000,
        )
        assert schema.md5_hash is None

    def test_invalid_content_type(self):
        with pytest.raises(ValueError):
            UploadFileSchema(
                file_name="test.mp4",
                content_type="invalid/type",
                file_size=1000000,
            )

    def test_file_size_too_large(self):
        with pytest.raises(ValueError):
            UploadFileSchema(
                file_name="test.mp4",
                content_type="video/mp4",
                file_size=600_000_000,
            )

    def test_file_size_zero(self):
        with pytest.raises(ValueError):
            UploadFileSchema(
                file_name="test.mp4",
                content_type="video/mp4",
                file_size=0,
            )


class TestTrustScoreSchema:
    """Test TrustScoreSchema validation."""

    def test_valid_trust_score(self):
        schema = TrustScoreSchema(
            overall_score=87.5,
            confidence=0.89,
            confidence_level=ConfidenceLevel.HIGH,
            breakdown={"video": 0.40, "audio": 0.35, "text": 0.25},
            verdict=Verdict.MANIPULATED,
        )
        assert schema.overall_score == 87.5
        assert schema.confidence == 0.89
        assert schema.verdict == Verdict.MANIPULATED

    def test_score_above_100(self):
        with pytest.raises(ValueError):
            TrustScoreSchema(
                overall_score=150,
                confidence=0.89,
                confidence_level=ConfidenceLevel.HIGH,
                breakdown={},
                verdict=Verdict.MANIPULATED,
            )

    def test_negative_score(self):
        with pytest.raises(ValueError):
            TrustScoreSchema(
                overall_score=-10,
                confidence=0.89,
                confidence_level=ConfidenceLevel.HIGH,
                breakdown={},
                verdict=Verdict.AUTHENTIC,
            )

    def test_confidence_above_1(self):
        with pytest.raises(ValueError):
            TrustScoreSchema(
                overall_score=50,
                confidence=1.5,
                confidence_level=ConfidenceLevel.MEDIUM,
                breakdown={},
                verdict=Verdict.UNCERTAIN,
            )


class TestFrameAnalysisSchema:
    """Test FrameAnalysisSchema validation."""

    def test_valid_frame_analysis(self):
        schema = FrameAnalysisSchema(
            frame_number=45,
            timestamp=9.0,
            face_bbox=[100.0, 100.0, 200.0, 200.0],
            manipulation_score=0.87,
            heatmap_path="/heatmaps/frame_45.png",
        )
        assert schema.frame_number == 45
        assert schema.manipulation_score == 0.87

    def test_missing_bbox(self):
        schema = FrameAnalysisSchema(
            frame_number=1,
            timestamp=0.2,
            face_bbox=None,
            manipulation_score=0.5,
        )
        assert schema.face_bbox is None


class TestAnalysisStatusSchema:
    """Test AnalysisStatusSchema validation."""

    def test_valid_status(self):
        schema = AnalysisStatusSchema(
            request_id=uuid.uuid4(),
            status=AnalysisStatus.PROCESSING,
            progress=45,
            current_stage="video_detection",
            estimated_time_remaining=120,
        )
        assert schema.progress == 45
        assert schema.status == AnalysisStatus.PROCESSING

    def test_progress_bounds(self):
        with pytest.raises(ValueError):
            AnalysisStatusSchema(
                request_id=uuid.uuid4(),
                status=AnalysisStatus.PENDING,
                progress=150,
            )

        with pytest.raises(ValueError):
            AnalysisStatusSchema(
                request_id=uuid.uuid4(),
                status=AnalysisStatus.PENDING,
                progress=-10,
            )


class TestVideoResultSchema:
    """Test VideoResultSchema validation."""

    def test_valid_video_result(self):
        schema = VideoResultSchema(
            frames_analyzed=50,
            faces_detected=3,
            overall_score=0.75,
            per_frame_scores=[
                FrameAnalysisSchema(
                    frame_number=1,
                    timestamp=0.2,
                    face_bbox=[100, 100, 200, 200],
                    manipulation_score=0.6,
                )
            ],
            branch_scores={
                "global": 0.7,
                "texture": 0.8,
                "spectral": 0.75,
            },
            rppg_result={"heart_rate": 72, "quality": 0.9},
        )
        assert schema.frames_analyzed == 50
        assert len(schema.per_frame_scores) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
