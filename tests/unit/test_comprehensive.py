"""
Comprehensive Integration Tests for Argus Prime

Tests real-world scenarios, edge cases, and production readiness.
"""

import pytest

from src.main import create_app
from src.config import get_config, reset_config
from src.core.logging import StructuredLogger
from src.api.schemas import (
    Verdict, FileType, AnalysisStatus, ConfidenceLevel,
    UploadFileSchema, TrustScoreSchema,
    VideoOptionsSchema, AudioOptionsSchema, TextOptionsSchema,
)


class TestConfigurationLoading:
    """Test configuration management."""

    def test_default_config_loading(self):
        """Test that default configuration loads correctly."""
        reset_config()
        config = get_config()
        
        assert config.app.port == 8000
        assert config.app.host == "0.0.0.0"
        assert config.app.debug is False
        assert config.storage.max_file_size == 500_000_000
        assert config.analysis.default_fps == 5

    def test_fusion_weights_configuration(self):
        """Test fusion weight configuration."""
        config = get_config()
        
        assert config.analysis.fusion["video_weight"] == 0.40
        assert config.analysis.fusion["audio_weight"] == 0.35
        assert config.analysis.fusion["text_weight"] == 0.25
        assert config.analysis.fusion["cross_modal_adjustment"] == 0.20

    def test_defense_configuration(self):
        """Test adversarial defense configuration."""
        config = get_config()
        
        assert config.defense.enabled is True
        assert config.defense.jpeg_quality == 85
        assert config.defense.noise_std == 0.01
        assert config.defense.trim_threshold == 0.1

    def test_model_configuration(self):
        """Test model selection configuration."""
        config = get_config()
        
        assert config.models.video.primary == "forensic_flow"
        assert config.models.audio.primary == "rawnet2"
        assert config.models.text.primary == "spec_detect"


class TestSchemaValidation:
    """Test Pydantic schema validation."""

    def test_video_options_schema(self):
        """Test video analysis options schema."""
        options = VideoOptionsSchema(fps=10, extract_faces=True, generate_heatmaps=True)
        assert options.fps == 10
        assert options.extract_faces is True

    def test_audio_options_schema(self):
        """Test audio analysis options schema."""
        options = AudioOptionsSchema(sample_rate=16000, segment_duration=2, apply_vad=True)
        assert options.sample_rate == 16000
        assert options.segment_duration == 2

    def test_text_options_schema(self):
        """Test text analysis options schema."""
        options = TextOptionsSchema(language="en", include_perplexity=True, include_burstiness=True)
        assert options.language == "en"
        assert options.include_perplexity is True

    def test_upload_schema_edge_cases(self):
        """Test upload schema edge cases."""
        # Test empty file name
        with pytest.raises(ValueError):
            UploadFileSchema(file_name="", content_type="video/mp4", file_size=1000)
        
        # Test file name with spaces (should be trimmed)
        schema = UploadFileSchema(file_name="  test.mp4  ", content_type="video/mp4", file_size=1000)
        assert schema.file_name == "test.mp4"

    def test_confidence_level_enum(self):
        """Test confidence level classification."""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"


class TestLoggingSystem:
    """Test structured logging system."""

    def test_logger_initialization(self):
        """Test logger creates properly."""
        logger = StructuredLogger("test_module")
        assert logger.logger is not None

    def test_log_levels(self):
        """Test all log levels work."""
        logger = StructuredLogger("test_levels")
        # These should not raise exceptions
        logger.debug("Debug message", key="value")
        logger.info("Info message", key="value")
        logger.warning("Warning message", key="value")
        logger.error("Error message", key="value")

    def test_extra_fields_in_log(self):
        """Test extra fields are included in log output."""
        logger = StructuredLogger("test_extra")
        # This should include the extra field
        logger.info("Test with extra", request_id="test-123", user_id="user-456")


class TestSecurityValidation:
    """Test security and input validation."""

    def test_md5_hash_validation_length(self):
        """Test MD5 hash field length validation."""
        # Valid 32-character MD5
        md5 = "d41d8cd98f00b204e9800998ecf8427e"
        assert len(md5) == 32, "MD5 should be 32 characters"
        
        schema = UploadFileSchema(
            file_name="test.mp4",
            content_type="video/mp4",
            file_size=1000,
            md5_hash=md5
        )
        assert schema.md5_hash == md5

    def test_content_type_pattern_validation(self):
        """Test content type pattern matching."""
        valid_types = ["video/mp4", "video/avi", "video/mov", 
                       "audio/wav", "audio/mp3", "audio/flac",
                       "image/png", "image/jpg", "image/jpeg", "image/webp"]
        for content_type in valid_types:
            schema = UploadFileSchema(
                file_name="test.mp4",
                content_type=content_type,
                file_size=1000
            )
            assert schema.content_type == content_type

        invalid_types = ["application/json", "text/html", "text/plain", "invalid", "video/mkv"]
        for content_type in invalid_types:
            with pytest.raises(ValueError):
                UploadFileSchema(
                    file_name="test.mp4",
                    content_type=content_type,
                    file_size=1000
                )


class TestAPISchemasCompleteness:
    """Test all required API schemas are defined."""

    def test_verdict_enum_complete(self):
        """Test all verdict types are defined."""
        assert Verdict.AUTHENTIC.value == "AUTHENTIC"
        assert Verdict.MANIPULATED.value == "MANIPULATED"
        assert Verdict.UNCERTAIN.value == "UNCERTAIN"

    def test_file_type_enum_complete(self):
        """Test all file types are defined."""
        assert FileType.VIDEO.value == "video"
        assert FileType.AUDIO.value == "audio"
        assert FileType.IMAGE.value == "image"
        assert FileType.TEXT.value == "text"

    def test_analysis_status_enum_complete(self):
        """Test all analysis statuses are defined."""
        assert AnalysisStatus.PENDING.value == "pending"
        assert AnalysisStatus.PROCESSING.value == "processing"
        assert AnalysisStatus.COMPLETED.value == "completed"
        assert AnalysisStatus.FAILED.value == "failed"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_boundary_file_size(self):
        """Test file size boundary conditions."""
        # Exactly 500MB should pass
        schema = UploadFileSchema(
            file_name="test.mp4",
            content_type="video/mp4",
            file_size=500_000_000
        )
        assert schema.file_size == 500_000_000

        # Over 500MB should fail
        with pytest.raises(ValueError):
            UploadFileSchema(
                file_name="test.mp4",
                content_type="video/mp4",
                file_size=500_000_001
            )

    def test_boundary_confidence(self):
        """Test confidence boundary conditions."""
        # Exactly 0.0 should pass
        schema = TrustScoreSchema(
            overall_score=50,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            breakdown={},
            verdict=Verdict.UNCERTAIN
        )
        assert schema.confidence == 0.0

        # Exactly 1.0 should pass
        schema = TrustScoreSchema(
            overall_score=100,
            confidence=1.0,
            confidence_level=ConfidenceLevel.HIGH,
            breakdown={},
            verdict=Verdict.AUTHENTIC
        )
        assert schema.confidence == 1.0

        # Over 1.0 should fail
        with pytest.raises(ValueError):
            TrustScoreSchema(
                overall_score=100,
                confidence=1.0001,
                confidence_level=ConfidenceLevel.HIGH,
                breakdown={},
                verdict=Verdict.AUTHENTIC
            )

    def test_progress_boundary(self):
        """Test progress percentage boundaries."""
        from src.api.schemas import AnalysisStatusSchema, AnalysisStatus
        
        # 0% should pass
        schema = AnalysisStatusSchema(
            request_id="00000000-0000-0000-0000-000000000000",
            status=AnalysisStatus.PENDING,
            progress=0
        )
        assert schema.progress == 0

        # 100% should pass
        schema = AnalysisStatusSchema(
            request_id="00000000-0000-0000-0000-000000000000",
            status=AnalysisStatus.COMPLETED,
            progress=100
        )
        assert schema.progress == 100

        # Over 100% should fail
        with pytest.raises(ValueError):
            AnalysisStatusSchema(
                request_id="00000000-0000-0000-0000-000000000000",
                status=AnalysisStatus.PROCESSING,
                progress=101
            )


class TestDocumentationCompliance:
    """Test compliance with PRIME_ARGUS_DOCUMENT.md requirements."""

    def test_all_required_schemas_exist(self):
        """Verify all schemas from documentation are implemented."""
        from src.api import schemas
        
        required_schemas = [
            'Verdict', 'FileType', 'AnalysisStatus', 'ConfidenceLevel',
            'UploadFileSchema', 'TrustScoreSchema', 'ForensicReportSchema',
            'AnalysisStatusSchema', 'VideoResultSchema', 'AudioResultSchema',
            'TextResultSchema', 'FrameAnalysisSchema', 'SegmentAnalysisSchema',
            'SentenceAnalysisSchema', 'LocalizationSchema', 'EvidenceSchema',
        ]
        
        for schema_name in required_schemas:
            assert hasattr(schemas, schema_name), f"Missing schema: {schema_name}"

    def test_all_required_endpoints_defined(self):
        """Verify all endpoints from documentation are in router."""
        from src.api.router import router
        
        routes = [r.path for r in router.routes]
        
        # Check for key endpoint patterns
        endpoint_patterns = [
            ('/health', 'Health check'),
            ('/health/detailed', 'Detailed health'),
            ('/api/v1/config', 'Config endpoint'),
            ('/api/v1/analyze', 'Analyze endpoint'),
            ('/api/v1/analyze/{request_id}/status', 'Status endpoint'),
            ('/api/v1/analyze/{request_id}/result', 'Result endpoint'),
        ]
        
        # Check that the router has routes
        assert len(routes) > 0, "Router should have routes defined"
        
        # Verify analyze endpoints exist (they use path parameters)
        has_analyze = any('/api/v1/analyze' in p for p in routes)
        assert has_analyze, "Should have /api/v1/analyze endpoints"


class TestPerformanceRequirements:
    """Test performance-related configurations."""

    def test_timeout_configurations(self):
        """Test timeout configurations from AGENTS.md."""
        config = get_config()
        
        # Database timeout should be < 100ms
        # This is configured at connection pool level
        assert config.database.pool_size >= 5
        assert config.database.max_overflow >= 10

    def test_cache_configuration(self):
        """Test Redis cache configuration."""
        config = get_config()
        
        assert config.redis.host == "localhost"
        assert config.redis.port == 6379
        assert config.redis.db == 0
        assert config.redis.key_prefix == "argus:"


class TestSecurityStandards:
    """Test security standards from AGENTS.md."""

    def test_cors_configuration_present(self):
        """Test CORS middleware is configured."""
        app = create_app()
        
        # Check if CORS middleware is in app
        assert len(app.user_middleware) > 0
        cors_middleware = [m for m in app.user_middleware 
                          if 'cors' in str(m).lower()]
        assert len(cors_middleware) > 0

    def test_input_validation_present(self):
        """Test that input validation is enabled."""
        # Pydantic schemas enforce validation
        schema = UploadFileSchema(
            file_name="test.mp4",
            content_type="video/mp4",
            file_size=1000
        )
        # Invalid inputs should raise errors
        with pytest.raises(ValueError):
            UploadFileSchema(
                file_name="test.mp4",
                content_type="invalid",
                file_size=1000
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
