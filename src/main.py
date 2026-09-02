"""
Argus Prime - Multi-Modal Deepfake Detection & Forensic Platform

Application entry point and FastAPI app factory.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config, load_config
from src.core.logging import setup_logging, logger
from src.core.health import router as health_router
from src.core.security_middleware import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RateLimitConfig,
)
from src.api.router import router as analysis_router

from src.models.video.forensic_flow import register_forensic_flow
from src.models.video.distil_dire import register_distil_dire
from src.models.video.kan_detector import register_kan
from src.models.audio.rawnet2_detector import register_rawnet2
from src.models.audio.transcrypt_detector import register_transcrypt
from src.models.text.spec_detect import register_specdetect
from src.models.text.perplexity_detector import register_perplexity


def register_all_models() -> None:
    """Register all detection models with the registry."""
    logger.info("Registering detection models...")
    
    try:
        register_forensic_flow()
        logger.info("Registered: ForensicFlow (video)")
    except Exception as e:
        logger.warning(f"Failed to register ForensicFlow: {e}")
    
    try:
        register_distil_dire()
        logger.info("Registered: DistilDIRE (video)")
    except Exception as e:
        logger.warning(f"Failed to register DistilDIRE: {e}")
    
    try:
        register_kan()
        logger.info("Registered: KAN Detector (video)")
    except Exception as e:
        logger.warning(f"Failed to register KAN Detector: {e}")
    
    try:
        register_rawnet2()
        logger.info("Registered: RawNet2 (audio)")
    except Exception as e:
        logger.warning(f"Failed to register RawNet2: {e}")
    
    try:
        register_transcrypt()
        logger.info("Registered: Transcrypt (audio)")
    except Exception as e:
        logger.warning(f"Failed to register Transcrypt: {e}")
    
    try:
        register_specdetect()
        logger.info("Registered: SpecDetect (text)")
    except Exception as e:
        logger.warning(f"Failed to register SpecDetect: {e}")
    
    try:
        register_perplexity()
        logger.info("Registered: PerplexityDetector (text)")
    except Exception as e:
        logger.warning(f"Failed to register PerplexityDetector: {e}")
    
    logger.info("Model registration complete")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager.

    Handles startup and shutdown events for clean resource management.
    """
    # Startup
    logger.info("Starting Argus Prime application...")
    config = get_config()
    logger.info(f"Configuration loaded: debug={config.app.debug}")
    
    # Register all detection models
    register_all_models()
    
    logger.info("Application startup complete")

    yield

    # Shutdown
    logger.info("Shutting down Argus Prime application...")
    
    # Unload all models
    from src.models.registry import ModelRegistry
    registry = ModelRegistry.get_instance()
    registry.unload_all()
    
    logger.info("Application shutdown complete")


def create_app(config_path: str = None) -> FastAPI:
    """
    Create and configure FastAPI application.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured FastAPI application instance
    """
    # Load configuration
    if config_path:
        load_config(config_path)

    config = get_config()

    # Create FastAPI app
    app = FastAPI(
        title="Argus Prime API",
        description="Multi-Modal Deepfake Detection & Forensic Platform",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000", 
            "http://127.0.0.1:3000",
            "https://9b9aeaaf-8feb-40f0-aa90-80099146e80c.preview.emergentagent.com",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add security middleware
    app.add_middleware(
        SecurityHeadersMiddleware,
    )
    app.add_middleware(
        RateLimitMiddleware,
        config=RateLimitConfig(requests=100, window_seconds=60),
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(analysis_router)

    logger.info("FastAPI application created successfully")

    return app


# Create default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "main:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.debug,
    )
