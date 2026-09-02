"""
Configuration management module.

Provides centralized configuration loading from YAML files
with environment variable overrides following 12-factor app principles.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Application server configuration."""
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
    workers: int = Field(default=4)


class DatabaseConfig(BaseModel):
    """Database connection configuration."""
    url: str = Field(default="postgresql+asyncpg://argus:argus@localhost:5432/argus_prime")
    pool_size: int = Field(default=10)
    max_overflow: int = Field(default=20)


class RedisConfig(BaseModel):
    """Redis cache configuration."""
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    key_prefix: str = Field(default="argus:")


class StorageConfig(BaseModel):
    """File storage configuration."""
    temp_dir: str = Field(default="/tmp/argus_uploads")
    max_file_size: int = Field(default=500_000_000)
    ttl_seconds: int = Field(default=86400)


class VideoModelConfig(BaseModel):
    """Video detection model configuration."""
    primary: str = Field(default="forensic_flow")
    fallback: list[str] = Field(default_factory=lambda: ["distil_dire", "kan_detector"])
    batch_size: int = Field(default=8)


class AudioModelConfig(BaseModel):
    """Audio detection model configuration."""
    primary: str = Field(default="rawnet2")
    fallback: list[str] = Field(default_factory=lambda: ["transcrypt"])
    segment_duration: int = Field(default=2)


class TextModelConfig(BaseModel):
    """Text detection model configuration."""
    primary: str = Field(default="spec_detect")
    fallback: list[str] = Field(default_factory=lambda: ["perplexity_detector"])


class ModelsConfig(BaseModel):
    """Model configuration container."""
    video: VideoModelConfig = Field(default_factory=VideoModelConfig)
    audio: AudioModelConfig = Field(default_factory=AudioModelConfig)
    text: TextModelConfig = Field(default_factory=TextModelConfig)


class AnalysisConfig(BaseModel):
    """Analysis pipeline configuration."""
    default_fps: int = Field(default=5)
    min_face_size: int = Field(default=64)
    confidence_threshold: float = Field(default=0.5)
    fusion: Dict[str, float] = Field(
        default_factory=lambda: {
            "video_weight": 0.40,
            "audio_weight": 0.35,
            "text_weight": 0.25,
            "cross_modal_adjustment": 0.20,
        }
    )


class DefenseConfig(BaseModel):
    """Adversarial defense configuration."""
    enabled: bool = Field(default=True)
    jpeg_quality: int = Field(default=85)
    noise_std: float = Field(default=0.01)
    trim_threshold: float = Field(default=0.1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="json")
    include_timestamp: bool = Field(default=True)


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""
    enabled: bool = Field(default=True)
    port: int = Field(default=9090)
    path: str = Field(default="/metrics")


class Config(BaseModel):
    """Root configuration model."""
    app: AppConfig = Field(default_factory=AppConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    defense: DefenseConfig = Field(default_factory=DefenseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file with environment variable overrides.

    Args:
        config_path: Path to configuration file. If None, uses CONFIG_PATH
                    env var or defaults to configs/default.yaml

    Returns:
        Config object with all settings
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "configs/default.yaml")

    config_data: Dict[str, Any] = {}

    # Load from YAML if exists
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            config_data = yaml.safe_load(f) or {}

    # Environment variable overrides
    env_overrides = {
        "database.url": os.environ.get("DATABASE_URL"),
        "redis.host": os.environ.get("REDIS_HOST"),
        "redis.port": os.environ.get("REDIS_PORT"),
        "storage.temp_dir": os.environ.get("STORAGE_TEMP_DIR"),
        "app.port": os.environ.get("APP_PORT"),
        "app.debug": os.environ.get("APP_DEBUG"),
    }

    for key, value in env_overrides.items():
        if value is not None:
            # Navigate nested dict structure
            keys = key.split(".")
            target = config_data
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]

            # Convert types
            if keys[-1] == "port":
                value = int(value)
            elif keys[-1] == "debug":
                value = value.lower() in ("true", "1", "yes")

            target[keys[-1]] = value

    return Config(**config_data)


_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global configuration (for testing)."""
    global _config
    _config = None
