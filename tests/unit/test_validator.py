"""
Unit Tests for File Validator.

Tests comprehensive file validation including format verification,
magic byte detection, and security checks.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.ingestion.validator import FileValidator, ValidationResult


class TestFileValidator:
    """Test cases for FileValidator."""

    @pytest.fixture
    def validator(self):
        """Create a FileValidator instance."""
        return FileValidator()

    @pytest.fixture
    def temp_video_file(self):
        """Create a temporary video file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp4")  # MP4 magic bytes
            f.write(b"fake video content" * 1000)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_audio_file(self):
        """Create a temporary audio file."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF")  # WAV magic bytes
            f.write(b"\x00" * 4)
            f.write(b"WAVE")
            f.write(b"fake audio content" * 1000)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_image_file(self):
        """Create a temporary image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
            f.write(b"fake image content" * 1000)
            path = f.name
        yield path
        os.unlink(path)

    @pytest.fixture
    def temp_text_file(self):
        """Create a temporary text file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is a test text file for validation.")
            path = f.name
        yield path
        os.unlink(path)

    def test_validator_initialization(self, validator):
        """Test that validator initializes correctly."""
        assert validator is not None
        assert validator.allowed_extensions is not None
        assert validator.max_file_sizes is not None
        assert isinstance(validator.allowed_extensions, set)

    def test_detect_file_type_video(self, validator, temp_video_file):
        """Test file type detection for video."""
        file_type = validator.detect_file_type(temp_video_file)
        assert file_type == "video"

    def test_detect_file_type_audio(self, validator, temp_audio_file):
        """Test file type detection for audio."""
        file_type = validator.detect_file_type(temp_audio_file)
        assert file_type == "audio"

    def test_detect_file_type_image(self, validator, temp_image_file):
        """Test file type detection for image."""
        file_type = validator.detect_file_type(temp_image_file)
        assert file_type == "image"

    def test_detect_file_type_text(self, validator, temp_text_file):
        """Test file type detection for text."""
        file_type = validator.detect_file_type(temp_text_file)
        assert file_type == "text"

    def test_detect_file_type_unknown(self, validator):
        """Test file type detection for unknown extension."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"unknown file type")
            path = f.name
        try:
            file_type = validator.detect_file_type(path)
            assert file_type == "unknown"
        finally:
            os.unlink(path)

    def test_validate_extension_valid(self, validator):
        """Test extension validation with valid extension."""
        is_valid, error = validator._validate_extension("test.mp4")
        assert is_valid is True
        assert error is None

    def test_validate_extension_invalid(self, validator):
        """Test extension validation with invalid extension."""
        is_valid, error = validator._validate_extension("test.xyz")
        assert is_valid is False
        assert error is not None

    def test_validate_extension_empty(self, validator):
        """Test extension validation with no extension."""
        is_valid, error = validator._validate_extension("test")
        assert is_valid is False
        assert "No file extension" in error

    def test_get_file_type_from_ext_video(self, validator):
        """Test file type mapping for video extensions."""
        assert validator._get_file_type_from_ext(".mp4") == "video"
        assert validator._get_file_type_from_ext(".avi") == "video"
        assert validator._get_file_type_from_ext(".mov") == "video"

    def test_get_file_type_from_ext_audio(self, validator):
        """Test file type mapping for audio extensions."""
        assert validator._get_file_type_from_ext(".wav") == "audio"
        assert validator._get_file_type_from_ext(".mp3") == "audio"
        assert validator._get_file_type_from_ext(".flac") == "audio"

    def test_get_file_type_from_ext_image(self, validator):
        """Test file type mapping for image extensions."""
        assert validator._get_file_type_from_ext(".png") == "image"
        assert validator._get_file_type_from_ext(".jpg") == "image"
        assert validator._get_file_type_from_ext(".webp") == "image"

    def test_get_file_type_from_ext_text(self, validator):
        """Test file type mapping for text extensions."""
        assert validator._get_file_type_from_ext(".txt") == "text"
        assert validator._get_file_type_from_ext(".md") == "text"

    def test_get_file_type_from_ext_unknown(self, validator):
        """Test file type mapping for unknown extensions."""
        assert validator._get_file_type_from_ext(".xyz") == "unknown"

    def test_check_magic_bytes_valid(self, validator, temp_video_file):
        """Test magic byte validation with valid file."""
        is_valid, error, types = validator._check_magic_bytes(temp_video_file)
        assert is_valid is True
        assert error is None
        assert "mp4" in types

    def test_check_magic_bytes_wav(self, validator, temp_audio_file):
        """Test magic byte validation for WAV file."""
        is_valid, error, types = validator._check_magic_bytes(temp_audio_file)
        assert is_valid is True
        assert "wav" in types

    def test_check_magic_bytes_png(self, validator, temp_image_file):
        """Test magic byte validation for PNG file."""
        is_valid, error, types = validator._check_magic_bytes(temp_image_file)
        assert is_valid is True
        assert "png" in types

    def test_validate_file_size_valid(self, validator, temp_video_file):
        """Test file size validation with valid size."""
        is_valid, error = validator._validate_file_size(temp_video_file, "video")
        assert is_valid is True
        assert error is None

    def test_validate_file_size_exceeds(self, validator):
        """Test file size validation when exceeds limit."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00" * (600 * 1024 * 1024))  # 600MB
            path = f.name
        try:
            is_valid, error = validator._validate_file_size(path, "video")
            assert is_valid is False
            assert "exceeds maximum" in error
        finally:
            os.unlink(path)

    def test_validate_file_valid(self, validator, temp_video_file):
        """Test comprehensive file validation."""
        result = validator.validate_file_sync(
            file_name="test.mp4",
            file_path=temp_video_file,
            content_type="video/mp4",
        )
        assert result.is_valid is True
        assert result.file_type == "video"
        assert len(result.errors) == 0

    def test_validate_file_invalid_extension(self, validator, temp_video_file):
        """Test validation with invalid extension."""
        result = validator.validate_file_sync(
            file_name="test.xyz",
            file_path=temp_video_file,
            content_type="video/mp4",
        )
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_validate_file_nonexistent(self, validator):
        """Test validation of non-existent file."""
        result = validator.validate_file_sync(
            file_name="test.mp4",
            file_path="/nonexistent/path/test.mp4",
            content_type="video/mp4",
        )
        assert result.is_valid is False

    def test_validation_result_to_dict(self):
        """Test ValidationResult serialization."""
        result = ValidationResult(
            is_valid=True,
            file_type="video",
            errors=["error1"],
            warnings=["warning1"],
            metadata={"key": "value"},
        )
        as_dict = result.to_dict()
        assert as_dict["is_valid"] is True
        assert as_dict["file_type"] == "video"
        assert as_dict["errors"] == ["error1"]
        assert as_dict["warnings"] == ["warning1"]
        assert as_dict["metadata"]["key"] == "value"

    def test_detect_archive_bomb_safe(self, validator, temp_video_file):
        """Test archive bomb detection with safe file."""
        is_safe, warnings = validator._detect_archive_bomb(temp_video_file)
        assert is_safe is True

    def test_detect_archive_bomb_compressed(self, validator):
        """Test archive bomb detection with compressed file."""
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            f.write(b"PK" + b"\x00" * 100)  # ZIP magic bytes
            path = f.name
        try:
            is_safe, warnings = validator._detect_archive_bomb(path)
            assert is_safe is True
            assert len(warnings) > 0  # Should have warning about compressed file
        finally:
            os.unlink(path)

    def test_magic_bytes_not_available(self, validator):
        """Test behavior when libmagic is not available."""
        validator.magic_available = False
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"test content")
            path = f.name
        try:
            detected_mime = validator._detect_mime_type(path)
            assert detected_mime == "video/mp4"  # Fallback to extension-based detection
        finally:
            os.unlink(path)

    def test_get_file_validator_singleton(self):
        """Test that get_file_validator returns singleton."""
        from src.ingestion.validator import get_file_validator
        validator1 = get_file_validator()
        validator2 = get_file_validator()
        assert validator1 is validator2


class TestFileValidatorEdgeCases:
    """Edge case tests for FileValidator."""

    @pytest.fixture
    def validator(self):
        return FileValidator()

    def test_empty_file(self, validator):
        """Test validation of empty file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            path = f.name
        try:
            result = validator.validate_file_sync(
                file_name="test.mp4",
                file_path=path,
                content_type="video/mp4",
            )
            assert result.is_valid is False
        finally:
            os.unlink(path)

    def test_very_large_filename(self, validator):
        """Test with very large filename."""
        large_name = "a" * 500 + ".mp4"
        result = validator._validate_extension(large_name)
        assert result[0] is True  # Should be valid

    def test_case_sensitive_extension(self, validator):
        """Test that extension detection is case insensitive."""
        assert validator._get_file_type_from_ext(".MP4") == "video"
        assert validator._get_file_type_from_ext(".Mp4") == "video"

    def test_content_type_override(self, validator):
        """Test content type override."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"RIFF")  # Wrong magic bytes for MP4
            path = f.name
        try:
            result = validator._validate_mime_type(
                "test.mp4", "video/mp4", path
            )
            # Content type should match extension despite wrong magic bytes
            assert result[2] == "video/mp4"
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
