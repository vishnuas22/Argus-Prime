"""
File Storage Service with MinIO Integration.

Handles file storage operations for uploaded files using MinIO object storage.
Provides fallback to local storage when MinIO is unavailable.

Role: Object storage abstraction layer for large media files
Features:
  - MinIO primary storage with local fallback
  - Streaming uploads for large files
  - Automatic content type detection
  - File metadata tracking
  - Presigned URL generation for secure downloads
"""

import asyncio
import os
import io
import shutil
import uuid
from pathlib import Path
from typing import Optional, BinaryIO, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib
import logging

from fastapi import UploadFile, HTTPException

from src.config import get_config
from src.core.logging import logger

logger = logging.getLogger(__name__)


@dataclass
class FileMetadata:
    """Metadata for stored file."""
    request_id: str
    file_name: str
    file_path: str
    file_type: str
    file_size: int
    content_type: str
    stored_at: datetime = field(default_factory=datetime.utcnow)
    storage_backend: str = "minio"
    object_name: Optional[str] = None
    bucket_name: Optional[str] = None
    etag: Optional[str] = None
    url: Optional[str] = None
    md5_hash: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "file_name": self.file_name,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "stored_at": self.stored_at.isoformat(),
            "storage_backend": self.storage_backend,
            "object_name": self.object_name,
            "bucket_name": self.bucket_name,
            "etag": self.etag,
            "url": self.url,
            "md5_hash": self.md5_hash,
        }


class StorageService:
    """
    File Storage Service with MinIO integration.

    Handles saving, retrieving, and cleaning up uploaded files.
    Uses MinIO as primary storage with local filesystem fallback.

    Attributes:
        temp_dir: Temporary directory for local fallback
        max_file_size: Maximum allowed file size
        minio_client: MinIO client instance (lazy loaded)
        use_minio: Flag indicating if MinIO is available
    """

    def __init__(self):
        """Initialize storage service."""
        config = get_config()
        self.temp_dir = Path(config.storage.temp_dir)
        self.max_file_size = config.storage.max_file_size
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self._minio_client = None
        self._minio_available = None
        
        logger.info(f"[Storage Init] StorageService initialized: temp_dir={self.temp_dir}")

    def _get_minio_client(self):
        """Lazy load MinIO client."""
        if self._minio_client is None:
            try:
                from src.ingestion.minio_client import get_minio_client, MinioConnectionError
                self._minio_client = get_minio_client()
                self._minio_available = self._minio_client.is_healthy()
                logger.info(f"[Storage Init] MinIO client loaded, available: {self._minio_available}")
            except Exception as e:
                logger.warning(f"[Storage Init] MinIO unavailable, using local storage: {e}")
                self._minio_available = False
        return self._minio_client if self._minio_available else None

    def _is_minio_available(self) -> bool:
        """Check if MinIO is available."""
        if self._minio_available is None:
            self._get_minio_client()
        return self._minio_available

    def _determine_file_type(self, file_name: str, content_type: str) -> str:
        """Determine file type from filename and content type."""
        ext = Path(file_name).suffix.lower()

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        text_exts = {".txt", ".md", ".pdf", ".docx"}

        if ext in video_exts or "video" in content_type:
            return "video"
        elif ext in audio_exts or "audio" in content_type:
            return "audio"
        elif ext in image_exts or "image" in content_type:
            return "image"
        elif ext in text_exts or "text" in content_type:
            return "text"
        else:
            return "unknown"

    def _get_content_type(self, extension: str) -> str:
        """Get content type from file extension."""
        content_types = {
            ".mp4": "video/mp4",
            ".avi": "video/avi",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
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
            ".txt": "text/plain",
            ".md": "text/markdown",
            ".pdf": "application/pdf",
        }
        return content_types.get(extension.lower(), "application/octet-stream")

    def _compute_md5(self, data: bytes) -> str:
        """Compute MD5 hash of data."""
        return hashlib.md5(data).hexdigest()

    async def save_file(
        self,
        file: UploadFile,
        request_id: str,
    ) -> FileMetadata:
        """
        Save uploaded file to storage (MinIO or local fallback).

        Args:
            file: Uploaded file
            request_id: Request identifier

        Returns:
            FileMetadata with saved file information

        Raises:
            HTTPException: If file validation fails or storage unavailable
        """
        logger.info(f"[Upload Init] Starting upload for request {request_id}: {file.filename}")
        
        content = await file.read()
        file_size = len(content)
        
        if file_size > self.max_file_size:
            logger.warning(f"[Upload Init] File too large: {file_size} > {self.max_file_size}")
            raise HTTPException(
                status_code=413,
                detail=f"File size {file_size} exceeds maximum {self.max_file_size} bytes"
            )
        
        if file_size == 0:
            logger.warning(f"[Upload Init] Empty file received")
            raise HTTPException(
                status_code=400,
                detail="Empty file received"
            )
        
        content_type = file.content_type or self._get_content_type(Path(file.filename).suffix)
        file_type = self._determine_file_type(file.filename, content_type)
        md5_hash = self._compute_md5(content)
        
        minio_client = self._get_minio_client()
        
        if minio_client and self._minio_available:
            return await self._save_to_minio(
                content=content,
                request_id=request_id,
                file_name=file.filename,
                file_type=file_type,
                file_size=file_size,
                content_type=content_type,
                md5_hash=md5_hash,
            )
        else:
            return await self._save_to_local(
                content=content,
                request_id=request_id,
                file_name=file.filename,
                file_type=file_type,
                file_size=file_size,
                content_type=content_type,
                md5_hash=md5_hash,
            )

    async def _save_to_minio(
        self,
        content: bytes,
        request_id: str,
        file_name: str,
        file_type: str,
        file_size: int,
        content_type: str,
        md5_hash: str,
    ) -> FileMetadata:
        """Save file to MinIO object storage."""
        from src.ingestion.minio_client import MinioUploadError, MinioConnectionError
        
        try:
            minio_client = self._get_minio_client()
            
            ext = Path(file_name).suffix
            sanitized_name = f"{file_name}".replace(" ", "_")
            object_name = f"{request_id}/{sanitized_name}"
            
            logger.info(f"[MinIO Upload] Uploading {object_name} ({file_size} bytes)")
            
            result = minio_client.upload_file(
                file_data=content,
                object_name=object_name,
                content_type=content_type,
                file_size=file_size,
                metadata={
                    "request_id": request_id,
                    "original_name": file_name,
                    "file_type": file_type,
                    "md5": md5_hash,
                },
            )
            
            local_path = self.temp_dir / request_id / file_name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(content)
            
            metadata = FileMetadata(
                request_id=request_id,
                file_name=file_name,
                file_path=str(local_path),
                file_type=file_type,
                file_size=file_size,
                content_type=content_type,
                storage_backend="minio",
                object_name=result.object_name,
                bucket_name=result.bucket_name,
                etag=result.etag,
                url=result.url,
                md5_hash=md5_hash,
            )
            
            logger.info(f"[MinIO Success] Uploaded {file_name} ({file_size} bytes) for request {request_id}")
            logger.info(f"[DB Record Created] Metadata stored for {request_id}")
            
            return metadata
            
        except (MinioUploadError, MinioConnectionError) as e:
            logger.error(f"[MinIO Upload] Failed, falling back to local: {e}")
            self._minio_available = False
            return await self._save_to_local(
                content=content,
                request_id=request_id,
                file_name=file_name,
                file_type=file_type,
                file_size=file_size,
                content_type=content_type,
                md5_hash=md5_hash,
            )

    async def _save_to_local(
        self,
        content: bytes,
        request_id: str,
        file_name: str,
        file_type: str,
        file_size: int,
        content_type: str,
        md5_hash: str,
    ) -> FileMetadata:
        """Save file to local filesystem (fallback)."""
        request_dir = self.temp_dir / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = request_dir / file_name
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        metadata = FileMetadata(
            request_id=request_id,
            file_name=file_name,
            file_path=str(file_path),
            file_type=file_type,
            file_size=file_size,
            content_type=content_type,
            storage_backend="local",
            md5_hash=md5_hash,
        )
        
        logger.info(f"[Local Storage] Saved {file_name} ({file_size} bytes) for request {request_id}")
        logger.info(f"[DB Record Created] Metadata stored for {request_id}")
        
        return metadata

    def get_file_path(self, request_id: str, file_name: str = None) -> Optional[str]:
        """
        Get path to stored file.

        Args:
            request_id: Request identifier
            file_name: Optional specific file name

        Returns:
            Path to file or None if not found
        """
        request_dir = self.temp_dir / request_id
        if not request_dir.exists():
            return None

        if file_name:
            file_path = request_dir / file_name
            if file_path.exists():
                return str(file_path)
        else:
            files = list(request_dir.glob("*"))
            if files:
                return str(files[0])

        return None

    async def get_file_info(self, request_id: str) -> Optional[FileMetadata]:
        """
        Get file metadata for a request.

        Args:
            request_id: Request identifier

        Returns:
            FileMetadata or None
        """
        file_path = self.get_file_path(request_id)
        if file_path is None:
            return None

        path = Path(file_path)
        content_type = self._get_content_type(path.suffix)

        return FileMetadata(
            request_id=request_id,
            file_name=path.name,
            file_path=file_path,
            file_type=self._determine_file_type(path.name, content_type),
            file_size=path.stat().st_size,
            content_type=content_type,
        )

    def get_presigned_url(
        self,
        request_id: str,
        file_name: str,
        expires_hours: int = 24,
    ) -> Optional[str]:
        """
        Get presigned URL for secure file download.

        Args:
            request_id: Request identifier
            file_name: File name
            expires_hours: URL expiration in hours

        Returns:
            Presigned URL or None if MinIO unavailable
        """
        minio_client = self._get_minio_client()
        if not minio_client or not self._minio_available:
            return None
        
        try:
            object_name = f"{request_id}/{file_name}"
            return minio_client.get_presigned_url(
                object_name=object_name,
                expires=timedelta(hours=expires_hours),
            )
        except Exception as e:
            logger.warning(f"[Storage] Failed to get presigned URL: {e}")
            return None

    async def cleanup(self, request_id: str) -> None:
        """
        Clean up files for a request.

        Args:
            request_id: Request identifier
        """
        request_dir = self.temp_dir / request_id
        if request_dir.exists():
            shutil.rmtree(request_dir)
            logger.info(f"[Storage Cleanup] Cleaned up local files for request {request_id}")
        
        minio_client = self._get_minio_client()
        if minio_client and self._minio_available:
            try:
                objects = minio_client._client.list_objects(
                    minio_client.bucket_name,
                    prefix=f"{request_id}/",
                    recursive=True,
                )
                for obj in objects:
                    minio_client.delete_object(obj.object_name)
                logger.info(f"[Storage Cleanup] Cleaned up MinIO objects for request {request_id}")
            except Exception as e:
                logger.warning(f"[Storage Cleanup] Failed to cleanup MinIO objects: {e}")

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """
        Clean up files older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of directories cleaned up
        """
        cleaned = 0
        now = datetime.utcnow()

        for request_dir in self.temp_dir.glob("*"):
            if request_dir.is_dir():
                try:
                    mtime = datetime.fromtimestamp(request_dir.stat().st_mtime)
                    age = (now - mtime).total_seconds() / 3600

                    if age > max_age_hours:
                        shutil.rmtree(request_dir)
                        cleaned += 1
                except Exception as e:
                    logger.warning(f"[Storage Cleanup] Failed to clean {request_dir}: {e}")

        if cleaned > 0:
            logger.info(f"[Storage Cleanup] Cleaned up {cleaned} old request directories")

        return cleaned

    def is_healthy(self) -> dict:
        """
        Check storage health status.

        Returns:
            Health status dictionary
        """
        local_ok = self.temp_dir.exists() and os.access(self.temp_dir, os.W_OK)
        
        minio_ok = False
        minio_client = self._get_minio_client()
        if minio_client:
            minio_ok = minio_client.is_healthy()
        
        return {
            "local_storage": "healthy" if local_ok else "unhealthy",
            "minio_storage": "healthy" if minio_ok else "unavailable",
            "active_backend": "minio" if minio_ok else "local",
        }


_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
