"""
Unit Tests for Content Sanitizer.

Tests C2PA metadata extraction and content sanitization.
"""

import pytest
import tempfile
import os
from pathlib import Path

from src.ingestion.sanitizer import ContentSanitizer, C2PASecurityResult, C2PAManifest


class TestContentSanitizer:
    """Test cases for ContentSanitizer."""

    @pytest.fixture
    def sanitizer(self):
        """Create a ContentSanitizer instance."""
        return ContentSanitizer()

    def test_sanitizer_initialization(self, sanitizer):
        """Test that sanitizer initializes correctly."""
        assert sanitizer is not None
        assert sanitizer.c2patool_path is not None
        assert len(sanitizer.ai_generation_indicators) > 0
        assert len(sanitizer.editing_tool_indicators) > 0

    def test_check_c2patool_availability(self, sanitizer):
        """Test c2patool availability check."""
        # c2patool may or may not be installed
        result = sanitizer._check_c2patool_available()
        assert isinstance(result, bool)

    def test_parse_manifest_valid(self, sanitizer):
        """Test manifest parsing with valid data."""
        manifest_data = {
            "claim": {
                "actor": {"name": "Test User"}
            },
            "manifest": {
                "title": "Test Document",
                "description": "A test document",
                "dateTime": {"dateTime": "2024-01-01T00:00:00Z"}
            },
            "ingredients": [],
            "assertions": []
        }
        manifest = sanitizer._parse_manifest(manifest_data)
        assert manifest.claimed_by == "Test User"
        assert manifest.title == "Test Document"
        assert manifest.description == "A test document"

    def test_parse_manifest_empty(self, sanitizer):
        """Test manifest parsing with empty data."""
        manifest = sanitizer._parse_manifest({})
        assert isinstance(manifest, C2PAManifest)
        assert manifest.claimed_by == ""

    def test_parse_manifest_partial(self, sanitizer):
        """Test manifest parsing with partial data."""
        manifest_data = {
            "manifest": {
                "title": "Partial Manifest"
            }
        }
        manifest = sanitizer._parse_manifest(manifest_data)
        assert manifest.title == "Partial Manifest"

    def test_analyze_ai_claims_no_ai(self, sanitizer):
        """Test AI claim analysis with no AI indicators."""
        manifest = C2PAManifest(
            claimed_by="John Doe",
            title="Natural Photo",
            description="A regular photo taken with a camera",
        )
        ai_generated, ai_claim, creation_tool = sanitizer._analyze_ai_claims(manifest)
        assert ai_generated is False
        assert ai_claim is None
        assert creation_tool is None

    def test_analyze_ai_claims_with_ai(self, sanitizer):
        """Test AI claim analysis with AI indicators."""
        manifest = C2PAManifest(
            claimed_by="AI Generator",
            title="AI Generated Image",
            description="This image was generated using stable diffusion",
        )
        ai_generated, ai_claim, creation_tool = sanitizer._analyze_ai_claims(manifest)
        assert ai_generated is True
        assert ai_claim is not None
        assert "stable diffusion" in ai_claim.lower() or "ai" in ai_claim.lower()

    def test_analyze_ai_claims_editing_tool(self, sanitizer):
        """Test AI claim analysis with editing tool indicators."""
        manifest = C2PAManifest(
            claimed_by="User",
            title="Edited Photo",
            description="Photo edited in photoshop",
        )
        ai_generated, ai_claim, creation_tool = sanitizer._analyze_ai_claims(manifest)
        assert creation_tool == "photoshop"

    def test_check_timestamp_consistent(self, sanitizer):
        """Test timestamp consistency check with consistent timestamps."""
        manifest = C2PAManifest(
            created_at="2024-01-01T12:00:00Z"
        )
        file_stats = {"mtime": 1704110400}  # Corresponds to 2024-01-01T12:00:00Z
        is_consistent, warnings = sanitizer._check_timestamp_consistency(manifest, file_stats)
        assert is_consistent is True
        assert len(warnings) == 0

    def test_check_timestamp_inconsistent(self, sanitizer):
        """Test timestamp consistency check with inconsistent timestamps."""
        manifest = C2PAManifest(
            created_at="2020-01-01T00:00:00Z"
        )
        file_stats = {"mtime": 1704110400}  # 2024-01-01
        is_consistent, warnings = sanitizer._check_timestamp_consistency(manifest, file_stats)
        assert is_consistent is False
        assert len(warnings) > 0

    def test_check_timestamp_no_manifest_time(self, sanitizer):
        """Test timestamp check with no manifest timestamp."""
        manifest = C2PAManifest(created_at=None)
        file_stats = {"mtime": 1704110400}
        is_consistent, warnings = sanitizer._check_timestamp_consistency(manifest, file_stats)
        assert is_consistent is True  # Should pass with warning

    def test_quick_check_existing_file(self, sanitizer):
        """Test quick check for existing file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp4")
            path = f.name
        try:
            result = sanitizer.quick_check(path)
            assert result["file_exists"] is True
            assert result["file_type"] == "video"
            assert result["file_size"] > 0
        finally:
            os.unlink(path)

    def test_quick_check_nonexistent_file(self, sanitizer):
        """Test quick check for non-existent file."""
        result = sanitizer.quick_check("/nonexistent/path.mp4")
        assert result["file_exists"] is False

    def test_get_file_type(self, sanitizer):
        """Test file type detection."""
        assert sanitizer._get_file_type(Path("/test.mp4")) == "video"
        assert sanitizer._get_file_type(Path("/test.wav")) == "audio"
        assert sanitizer._get_file_type(Path("/test.png")) == "image"
        assert sanitizer._get_file_type(Path("/test.txt")) == "other"

    def test_sanitize_content_no_file(self, sanitizer):
        """Test sanitization of non-existent file."""
        import asyncio
        result = asyncio.run(sanitizer.sanitize_content("/nonexistent.mp4"))
        assert result["warnings"] == ["File not found"]

    def test_c2pa_security_result_to_dict(self):
        """Test C2PASecurityResult serialization."""
        result = C2PASecurityResult(
            has_manifest=True,
            is_valid=True,
            manifest=C2PAManifest(
                claimed_by="Test User",
                title="Test",
            ),
            ai_generated=False,
            ai_claim=None,
            creation_tool="photoshop",
            signatures_valid=True,
            timestamp_consistent=True,
            warnings=["warning1"],
            raw_data={"key": "value"},
        )
        as_dict = result.to_dict()
        assert as_dict["has_manifest"] is True
        assert as_dict["is_valid"] is True
        assert as_dict["ai_generated"] is False
        assert as_dict["creation_tool"] == "photoshop"
        assert as_dict["warnings"] == ["warning1"]

    def test_c2pa_manifest_to_dict(self):
        """Test C2PAManifest serialization."""
        manifest = C2PAManifest(
            claimed_by="Test User",
            title="Test Document",
            description="A test",
            created_at="2024-01-01T00:00:00Z",
            ingredients=[{"id": "1"}],
            assertions=[{"type": "test"}],
        )
        as_dict = manifest.to_dict()
        assert as_dict["claimed_by"] == "Test User"
        assert as_dict["title"] == "Test Document"
        assert as_dict["ingredients"] == [{"id": "1"}]
        assert as_dict["assertions"] == [{"type": "test"}]

    def test_get_content_sanitizer_singleton(self):
        """Test that get_content_sanitizer returns singleton."""
        from src.ingestion.sanitizer import get_content_sanitizer
        sanitizer1 = get_content_sanitizer()
        sanitizer2 = get_content_sanitizer()
        assert sanitizer1 is sanitizer2


class TestContentSanitizerAsync:
    """Async test cases for ContentSanitizer."""

    @pytest.fixture
    def sanitizer(self):
        return ContentSanitizer()

    @pytest.mark.asyncio
    async def test_extract_c2pa_metadata_no_file(self, sanitizer):
        """Test C2PA extraction with non-existent file."""
        result = await sanitizer.extract_c2pa_metadata("/nonexistent/file.mp4")
        assert result.has_manifest is False
        assert len(result.warnings) > 0

    @pytest.mark.asyncio
    async def test_extract_c2pa_metadata_file_exists(self, sanitizer):
        """Test C2PA extraction with existing file."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp4")
            path = f.name
        try:
            result = await sanitizer.extract_c2pa_metadata(path)
            assert isinstance(result, C2PASecurityResult)
            # c2patool may or may not be available
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_sanitize_content_no_file(self, sanitizer):
        """Test content sanitization with non-existent file."""
        result = await sanitizer.sanitize_content("/nonexistent/file.mp4")
        assert result["warnings"] == ["File not found"]

    @pytest.mark.asyncio
    async def test_sanitize_content_with_remove_metadata(self, sanitizer):
        """Test content sanitization with metadata removal."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"\x00\x00\x00\x18ftypmp4")
            path = f.name
        try:
            result = await sanitizer.sanitize_content(path, remove_metadata=True)
            assert result["sanitized"] is True
            assert "c2pa_analysis" in result
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
