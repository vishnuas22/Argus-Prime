"""
MinIO Object Storage Client.

Provides a singleton MinIO client for file storage operations.
Handles bucket creation, file uploads with streaming, and presigned URL generation.

Role: Object storage interface for large media files
Features:
  - Singleton pattern for connection pooling
  - Automatic bucket creation
  - Streaming uploads (memory-efficient)
  - Presigned URL generation for secure downloads
  - Health check capabilities
"""

import os
import io
from datetime import timedelta
from pathlib import Path
from typing import Optional, BinaryIO, Union
from dataclasses import dataclass
import hashlib
import logging

from minio import Minio
from minio.error import S3Error
from urllib3.exceptions import MaxRetryError

logger = logging.getLogger(__name__)


@dataclass
class MinioConfig:
    """MinIO configuration from environment variables."""
    endpoint: str
    access_key: str
    secret_key: str
    bucket_name: str
    secure: bool
    region: str

    @classmethod
    def from_env(cls) -> "MinioConfig":
        """Load configuration from environment variables."""
        endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
        access_key = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
        secret_key = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
        bucket_name = os.environ.get("MINIO_BUCKET", "argus-uploads")
        secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
        region = os.environ.get("MINIO_REGION", "us-east-1")
        
        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            bucket_name=bucket_name,
            secure=secure,
            region=region,
        )


@dataclass
class UploadResult:
    """Result of a successful upload operation."""
    object_name: str
    bucket_name: str
    etag: str
    size: int
    content_type: str
    url: str
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "object_name": self.object_name,
            "bucket_name": self.bucket_name,
            "etag": self.etag,
            "size": self.size,
            "content_type": self.content_type,
            "url": self.url,
        }


class MinioClientError(Exception):
    """Base exception for MinIO client errors."""
    pass


class MinioConnectionError(MinioClientError):
    """Raised when MinIO connection fails."""
    pass


class MinioBucketError(MinioClientError):
    """Raised when bucket operations fail."""
    pass


class MinioUploadError(MinioClientError):
    """Raised when upload operations fail."""
    pass


class MinioClient:
    """
    Singleton MinIO client for object storage operations.
    
    Provides thread-safe access to MinIO with automatic bucket management.
    """
    
    _instance: Optional["MinioClient"] = None
    _client: Optional[Minio] = None
    _config: Optional[MinioConfig] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize MinIO client if not already done."""
        if self._initialized:
            return
        
        self._config = MinioConfig.from_env()
        self._connect()
        self._ensure_bucket()
        self._initialized = True
        logger.info(f"[MinIO Init] MinIO client initialized: endpoint={self._config.endpoint}, bucket={self._config.bucket_name}")
    
    def _connect(self) -> None:
        """Establish connection to MinIO server."""
        try:
            self._client = Minio(
                endpoint=self._config.endpoint,
                access_key=self._config.access_key,
                secret_key=self._config.secret_key,
                secure=self._config.secure,
                region=self._config.region,
            )
            logger.info(f"[MinIO Connect] Connected to MinIO at {self._config.endpoint}")
        except Exception as e:
            logger.error(f"[MinIO Connect] Failed to connect: {e}")
            raise MinioConnectionError(f"Failed to connect to MinIO: {e}")
    
    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self._client.bucket_exists(self._config.bucket_name):
                self._client.make_bucket(self._config.bucket_name, location=self._config.region)
                logger.info(f"[MinIO Bucket] Created bucket: {self._config.bucket_name}")
            else:
                logger.info(f"[MinIO Bucket] Bucket exists: {self._config.bucket_name}")
        except S3Error as e:
            logger.error(f"[MinIO Bucket] Failed to ensure bucket: {e}")
            raise MinioBucketError(f"Failed to ensure bucket: {e}")
        except MaxRetryError as e:
            logger.error(f"[MinIO Bucket] Connection failed during bucket check: {e}")
            raise MinioConnectionError(f"MinIO connection failed: {e}")
    
    def is_healthy(self) -> bool:
        """Check if MinIO connection is healthy."""
        try:
            self._client.bucket_exists(self._config.bucket_name)
            return True
        except Exception as e:
            logger.warning(f"[MinIO Health] Health check failed: {e}")
            return False
    
    def upload_file(
        self,
        file_data: Union[BinaryIO, bytes],
        object_name: str,
        content_type: str,
        file_size: int,
        metadata: Optional[dict] = None,
    ) -> UploadResult:
        """
        Upload file to MinIO using streaming.
        
        Args:
            file_data: File data (bytes or file-like object)
            object_name: Name for the object in bucket
            content_type: MIME type of the file
            file_size: Size of the file in bytes
            metadata: Optional metadata to attach
            
        Returns:
            UploadResult with upload details
            
        Raises:
            MinioUploadError: If upload fails
        """
        try:
            if isinstance(file_data, bytes):
                file_data = io.BytesIO(file_data)
            
            result = self._client.put_object(
                bucket_name=self._config.bucket_name,
                object_name=object_name,
                data=file_data,
                length=file_size,
                content_type=content_type,
                metadata=metadata or {},
            )
            
            url = self._generate_internal_url(object_name)
            
            upload_result = UploadResult(
                object_name=object_name,
                bucket_name=self._config.bucket_name,
                etag=result.etag,
                size=file_size,
                content_type=content_type,
                url=url,
            )
            
            logger.info(f"[MinIO Success] Uploaded {object_name} ({file_size} bytes) to {self._config.bucket_name}")
            return upload_result
            
        except S3Error as e:
            logger.error(f"[MinIO Upload] S3 error during upload: {e}")
            raise MinioUploadError(f"Failed to upload file: {e}")
        except MaxRetryError as e:
            logger.error(f"[MinIO Upload] Connection failed during upload: {e}")
            raise MinioConnectionError(f"MinIO connection failed during upload: {e}")
        except Exception as e:
            logger.error(f"[MinIO Upload] Unexpected error: {e}")
            raise MinioUploadError(f"Unexpected error during upload: {e}")
    
    def upload_stream(
        self,
        stream: BinaryIO,
        object_name: str,
        content_type: str,
        metadata: Optional[dict] = None,
    ) -> UploadResult:
        """
        Upload file from stream with unknown size.
        
        Uses part upload for large files to avoid loading into memory.
        
        Args:
            stream: File-like object to upload
            object_name: Name for the object in bucket
            content_type: MIME type of the file
            metadata: Optional metadata to attach
            
        Returns:
            UploadResult with upload details
        """
        try:
            result = self._client.put_object(
                bucket_name=self._config.bucket_name,
                object_name=object_name,
                data=stream,
                length=-1,
                part_size=10 * 1024 * 1024,
                content_type=content_type,
                metadata=metadata or {},
            )
            
            stat = self._client.stat_object(self._config.bucket_name, object_name)
            url = self._generate_internal_url(object_name)
            
            upload_result = UploadResult(
                object_name=object_name,
                bucket_name=self._config.bucket_name,
                etag=result.etag,
                size=stat.size,
                content_type=content_type,
                url=url,
            )
            
            logger.info(f"[MinIO Success] Stream uploaded {object_name} ({stat.size} bytes)")
            return upload_result
            
        except S3Error as e:
            logger.error(f"[MinIO Upload] S3 error during stream upload: {e}")
            raise MinioUploadError(f"Failed to upload stream: {e}")
        except Exception as e:
            logger.error(f"[MinIO Upload] Unexpected error during stream upload: {e}")
            raise MinioUploadError(f"Unexpected error during stream upload: {e}")
    
    def get_presigned_url(
        self,
        object_name: str,
        expires: timedelta = timedelta(hours=24),
    ) -> str:
        """
        Generate presigned URL for secure download.
        
        Args:
            object_name: Name of the object
            expires: URL expiration time (default 24 hours)
            
        Returns:
            Presigned URL string
        """
        try:
            url = self._client.presigned_get_object(
                bucket_name=self._config.bucket_name,
                object_name=object_name,
                expires=expires,
            )
            logger.info(f"[MinIO URL] Generated presigned URL for {object_name}")
            return url
        except S3Error as e:
            logger.error(f"[MinIO URL] Failed to generate presigned URL: {e}")
            raise MinioClientError(f"Failed to generate presigned URL: {e}")
    
    def _generate_internal_url(self, object_name: str) -> str:
        """Generate internal URL for object."""
        protocol = "https" if self._config.secure else "http"
        return f"{protocol}://{self._config.endpoint}/{self._config.bucket_name}/{object_name}"
    
    def download_file(self, object_name: str, file_path: str) -> str:
        """
        Download file from MinIO to local path.
        
        Args:
            object_name: Name of the object to download
            file_path: Local path to save the file
            
        Returns:
            Local file path
        """
        try:
            self._client.fget_object(
                bucket_name=self._config.bucket_name,
                object_name=object_name,
                file_path=file_path,
            )
            logger.info(f"[MinIO Download] Downloaded {object_name} to {file_path}")
            return file_path
        except S3Error as e:
            logger.error(f"[MinIO Download] Failed to download: {e}")
            raise MinioClientError(f"Failed to download file: {e}")
    
    def get_object_stream(self, object_name: str) -> BinaryIO:
        """
        Get object as a stream for reading.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Response object that can be used as a stream
        """
        try:
            response = self._client.get_object(
                bucket_name=self._config.bucket_name,
                object_name=object_name,
            )
            logger.info(f"[MinIO Stream] Got stream for {object_name}")
            return response
        except S3Error as e:
            logger.error(f"[MinIO Stream] Failed to get object stream: {e}")
            raise MinioClientError(f"Failed to get object stream: {e}")
    
    def delete_object(self, object_name: str) -> bool:
        """
        Delete object from MinIO.
        
        Args:
            object_name: Name of the object to delete
            
        Returns:
            True if deleted successfully
        """
        try:
            self._client.remove_object(
                bucket_name=self._config.bucket_name,
                object_name=object_name,
            )
            logger.info(f"[MinIO Delete] Deleted {object_name}")
            return True
        except S3Error as e:
            logger.error(f"[MinIO Delete] Failed to delete: {e}")
            return False
    
    def object_exists(self, object_name: str) -> bool:
        """
        Check if object exists in bucket.
        
        Args:
            object_name: Name of the object
            
        Returns:
            True if object exists
        """
        try:
            self._client.stat_object(self._config.bucket_name, object_name)
            return True
        except S3Error:
            return False
    
    @property
    def bucket_name(self) -> str:
        """Get the configured bucket name."""
        return self._config.bucket_name
    
    @property
    def endpoint(self) -> str:
        """Get the configured endpoint."""
        return self._config.endpoint


_minio_client: Optional[MinioClient] = None


def get_minio_client() -> MinioClient:
    """Get or create MinIO client instance."""
    global _minio_client
    if _minio_client is None:
        _minio_client = MinioClient()
    return _minio_client


def reset_minio_client() -> None:
    """Reset MinIO client (for testing)."""
    global _minio_client
    _minio_client = None
    MinioClient._instance = None
    MinioClient._initialized = False
