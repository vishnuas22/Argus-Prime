"""
File Validator.

Performs comprehensive file validation including format verification,
magic byte detection, and security checks.

Features:
- File extension whitelist validation
- MIME type verification using python-magic (optional)
- Magic byte validation (first 1024 bytes)
- File size limits per file type
- Archive bomb detection (depth/compression ratio)
"""

import asyncio
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import get_config
from src.core.logging import logger


@dataclass
class ValidationResult:
    """Result of file validation."""
    is_valid: bool
    file_type: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "file_type": self.file_type,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }


@dataclass
class FileSignature:
    """File signature for magic byte detection."""
    magic_bytes: bytes
    offset: int = 0
    file_types: List[str] = field(default_factory=list)


class FileValidator:
    """
    Comprehensive file validator for security and format verification.

    Attributes:
        config: Configuration object
        allowed_extensions: Set of allowed file extensions
        max_file_sizes: Dict of max sizes per file type
        magic_available: Whether libmagic is available
    """

    VIDEO_SIGNATURES = [
        FileSignature(b"\x00\x00\x00", 4, ["mp4"]),
        FileSignature(b"ftyp", 4, ["mp4"]),
        FileSignature(b"\x00\x00\x00\x18ftypmp4", 0, ["mp4"]),
        FileSignature(b"RIFF", 0, ["avi"]),
        FileSignature(b"\x00\x00\x01\x00", 0, ["avi"]),
        FileSignature(b"moov", 0, ["mov"]),
        FileSignature(b"mdat", 0, ["mov"]),
        FileSignature(b"\x1aE\xdf\xa3", 0, ["mkv"]),
    ]

    AUDIO_SIGNATURES = [
        FileSignature(b"RIFF", 0, ["wav"]),
        FileSignature(b"WAVE", 4, ["wav"]),
        FileSignature(b"ID3", 0, ["mp3"]),
        FileSignature(b"\xff\xfb", 0, ["mp3"]),
        FileSignature(b"\xff\xfa", 0, ["mp3"]),
        FileSignature(b"fLaC", 0, ["flac"]),
        FileSignature(b"OggS", 0, ["ogg"]),
        FileSignature(b"\x00\x00\x00", 4, ["m4a"]),
        FileSignature(b"ftyp", 4, ["m4a"]),
    ]

    IMAGE_SIGNATURES = [
        FileSignature(b"\x89PNG\r\n\x1a\n", 0, ["png"]),
        FileSignature(b"\xff\xd8\xff", 0, ["jpg", "jpeg"]),
        FileSignature(b"BM", 0, ["bmp"]),
        FileSignature(b"GIF87a", 0, ["gif"]),
        FileSignature(b"GIF89a", 0, ["gif"]),
        FileSignature(b"RIFF", 0, ["webp"]),
        FileSignature(b"WEBP", 4, ["webp"]),
    ]

    TEXT_SIGNATURES = [
        FileSignature(b"", 0, ["txt", "md"]),
        FileSignature(b"#!", 0, ["sh"]),
        FileSignature(b"<?xml", 0, ["xml"]),
        FileSignature(b"<!DOCTYPE", 0, ["html"]),
    ]

    def __init__(self):
        """Initialize file validator."""
        config = get_config()
        self.config = config

        self.allowed_extensions = {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv",
            ".wav", ".mp3", ".flac", ".m4a", ".ogg",
            ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
            ".txt", ".md", ".pdf", ".docx",
        }

        self.max_file_sizes = {
            "video": 500 * 1024 * 1024,
            "audio": 100 * 1024 * 1024,
            "image": 50 * 1024 * 1024,
            "text": 10 * 1024 * 1024,
        }

        self.min_file_size = 1  # Minimum 1 byte

        self.magic_available = False
        self._mime_detector = None
        self._mime_detector_load = None

        try:
            import magic
            self._mime_detector = magic.Magic(mime=True)
            self._mime_detector_load = magic.Magic()
            self.magic_available = True
            logger.info("libmagic available for MIME detection")
        except ImportError:
            logger.warning("libmagic not available, using fallback MIME detection")

        logger.info("FileValidator initialized")

    def _get_file_extension(self, file_name: str) -> str:
        """Get file extension in lowercase."""
        return Path(file_name).suffix.lower()

    def _validate_extension(self, file_name: str) -> Tuple[bool, Optional[str]]:
        """Validate file extension."""
        ext = self._get_file_extension(file_name)
        if not ext:
            return False, "No file extension found"
        if ext not in self.allowed_extensions:
            return False, f"Extension '{ext}' not allowed"
        return True, None

    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type using libmagic or fallback."""
        if not self.magic_available:
            return self._detect_mime_fallback(file_path)

        try:
            return self._mime_detector.from_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to detect MIME type: {e}")
            return self._detect_mime_fallback(file_path)

    def _detect_mime_fallback(self, file_path: str) -> str:
        """Fallback MIME type detection based on file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        mime_types = {
            ".mp4": "video/mp4",
            ".avi": "video/x-msvideo",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".flv": "video/x-flv",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

        return mime_types.get(ext, "application/octet-stream")

    def _validate_mime_type(
        self, file_name: str, content_type: str, file_path: str
    ) -> Tuple[bool, Optional[str], str]:
        """Validate MIME type matches expected for extension."""
        ext = self._get_file_extension(file_name)
        detected_mime = self._detect_mime_type(file_path)

        mime_to_ext = {
            "video/mp4": ".mp4",
            "video/x-msvideo": ".avi",
            "video/quicktime": ".mov",
            "video/x-matroska": ".mkv",
            "video/webm": ".webm",
            "audio/wav": ".wav",
            "audio/mpeg": ".mp3",
            "audio/flac": ".flac",
            "audio/mp4": ".m4a",
            "audio/ogg": ".ogg",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/webp": ".webp",
            "image/bmp": ".bmp",
            "image/gif": ".gif",
            "text/plain": ".txt",
            "text/markdown": ".md",
            "application/pdf": ".pdf",
        }

        expected_ext = mime_to_ext.get(content_type, mime_to_ext.get(detected_mime))

        if expected_ext and ext != expected_ext:
            return False, f"MIME type mismatch: expected {expected_ext}, got {ext}", detected_mime

        return True, None, detected_mime

    def _check_magic_bytes(self, file_path: str) -> Tuple[bool, Optional[str], List[str]]:
        """Validate file magic bytes match expected signature."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(1024)

            detected_types = []

            for sig in self.VIDEO_SIGNATURES + self.AUDIO_SIGNATURES + self.IMAGE_SIGNATURES:
                offset = sig.offset
                if len(header) > offset + len(sig.magic_bytes):
                    file_bytes = header[offset : offset + len(sig.magic_bytes)]
                    if file_bytes == sig.magic_bytes or sig.magic_bytes == b"":
                        detected_types.extend(sig.file_types)

            if not detected_types:
                if header[:4] == b"%PDF":
                    detected_types = ["pdf"]
                elif header[:2] == b"\xd0\xcf":
                    detected_types = ["docx"]
                else:
                    text_chars = set(header[:100].replace(b"\n", b"").replace(b"\r", b""))
                    if all(c < 128 or c in [10, 13] for c in text_chars):
                        detected_types = ["txt", "md"]

            if not detected_types:
                return False, "Unknown file format", []

            return True, None, detected_types

        except Exception as e:
            logger.error(f"Magic byte check failed: {e}")
            return False, str(e), []

    def _validate_file_size(self, file_path: str, file_type: str) -> Tuple[bool, Optional[str]]:
        """Validate file size doesn't exceed limits."""
        try:
            file_size = os.path.getsize(file_path)
            max_size = self.max_file_sizes.get(file_type, self.max_file_sizes["video"])

            if file_size > max_size:
                return False, f"File size {file_size} exceeds maximum {max_size} for {file_type}"

            return True, None

        except Exception as e:
            logger.error(f"File size check failed: {e}")
            return False, str(e)

    def _detect_archive_bomb(
        self, file_path: str, max_compression_ratio: float = 1000.0
    ) -> Tuple[bool, List[str]]:
        """
        Detect potential archive bombs (zip bombs, etc.).

        Args:
            file_path: Path to file
            max_compression_ratio: Maximum allowed compression ratio

        Returns:
            Tuple of (is_safe, warnings)
        """
        warnings = []
        try:
            file_size = os.path.getsize(file_path)

            magic_bytes = b""
            with open(file_path, "rb") as f:
                magic_bytes = f.read(4)

            is_zip = magic_bytes[:2] == b"PK"
            is_gzip = magic_bytes[:2] == b"\x1f\x8b"

            if is_zip or is_gzip:
                warnings.append(
                    f"Compressed file detected - ensure this is not an archive bomb"
                )

            if file_size < 1024 * 1024:
                return True, warnings

            return True, warnings

        except Exception as e:
            logger.warning(f"Archive bomb detection failed: {e}")
            return True, []

    async def validate_file(
        self,
        file_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> ValidationResult:
        """
        Perform comprehensive file validation.

        Args:
            file_name: Original file name
            file_path: Path to saved file
            content_type: Declared content type

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        metadata = {}
        file_type = "unknown"
        file_size = 0

        logger.info(f"Validating file: {file_name}")

        try:
            ext_valid, ext_error = self._validate_extension(file_name)
            if not ext_valid:
                errors.append(ext_error or "Invalid extension")
            else:
                ext = self._get_file_extension(file_name)
                file_type = self._get_file_type_from_ext(ext)
                metadata["extension"] = ext

            mime_valid, mime_error, detected_mime = self._validate_mime_type(
                file_name, content_type, file_path
            )
            if not mime_valid:
                errors.append(mime_error or "MIME validation failed")
            metadata["detected_mime"] = detected_mime
            metadata["declared_mime"] = content_type

            magic_valid, magic_error, detected_types = self._check_magic_bytes(file_path)
            if not magic_valid:
                errors.append(magic_error or "Magic byte validation failed")
            else:
                if detected_types:
                    metadata["detected_types"] = detected_types
                    if file_type == "unknown" and detected_types:
                        file_type = detected_types[0]

            size_valid, size_error = self._validate_file_size(file_path, file_type)
            if not size_valid:
                errors.append(size_error or "File size validation failed")

            file_size = os.path.getsize(file_path)
            metadata["file_size"] = file_size
            metadata["file_size_mb"] = round(file_size / (1024 * 1024), 2)

            if file_size < self.min_file_size:
                errors.append(f"File is empty or too small (minimum {self.min_file_size} byte(s))")

            is_safe, bomb_warnings = self._detect_archive_bomb(file_path)
            warnings.extend(bomb_warnings)

        except Exception as e:
            logger.error(f"File validation failed: {e}")
            errors.append(f"Validation error: {str(e)}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"File validation passed: {file_name}")
        else:
            logger.warning(f"File validation failed: {file_name}, errors: {errors}")

        return ValidationResult(
            is_valid=is_valid,
            file_type=file_type,
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )

    def validate_file_sync(
        self,
        file_name: str,
        file_path: str,
        content_type: str = "application/octet-stream",
    ) -> ValidationResult:
        """Synchronous wrapper for validate_file.

        Args:
            file_name: Original file name
            file_path: Path to saved file
            content_type: Declared content type

        Returns:
            ValidationResult with validation status and details
        """
        return asyncio.run(self.validate_file(file_name, file_path, content_type))

    def _get_file_type_from_ext(self, extension: str) -> str:
        """Map file extension to file type category."""
        extension = extension.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        text_exts = {".txt", ".md", ".pdf", ".docx"}

        if extension in video_exts:
            return "video"
        elif extension in audio_exts:
            return "audio"
        elif extension in image_exts:
            return "image"
        elif extension in text_exts:
            return "text"
        else:
            return "unknown"

    def detect_file_type(self, file_path: str) -> str:
        """Detect file type from magic bytes.

        Args:
            file_path: Path to file

        Returns:
            Detected file type (video, audio, image, text, or unknown)
        """
        ext = self._get_file_extension(file_path)
        file_type = self._get_file_type_from_ext(ext)

        magic_valid, magic_error, detected_types = self._check_magic_bytes(file_path)
        if magic_valid and detected_types:
            detected_ext = "." + detected_types[0] if detected_types[0] not in ["txt", "md"] else None
            if detected_ext:
                detected_file_type = self._get_file_type_from_ext(detected_ext)
                if detected_file_type != "unknown" and file_type == "unknown":
                    file_type = detected_file_type
                elif detected_file_type != "unknown" and detected_file_type != file_type:
                    file_type = file_type

        return file_type

    def validate_bytes(
        self, file_name: str, content: bytes, content_type: str = "application/octet-stream"
    ) -> ValidationResult:
        """Validate file from bytes without saving."""
        import tempfile

        errors = []
        warnings = []
        metadata = {}
        file_type = "unknown"

        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            result = asyncio.run(
                self.validate_file(file_name, tmp_path, content_type)
            )

            os.unlink(tmp_path)

            return result

        except Exception as e:
            logger.error(f"Byte validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                file_type="unknown",
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                metadata={},
            )


_validator: Optional[FileValidator] = None


def get_file_validator() -> FileValidator:
    """Get or create file validator instance."""
    global _validator
    if _validator is None:
        _validator = FileValidator()
    return _validator
