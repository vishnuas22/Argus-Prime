# Argus Prime - Multi-Modal Deepfake Detection & Forensic Platform

## Overview

Argus Prime is a production-grade backend platform for detecting deepfakes across audio, image, video, and text modalities with forensic precision. The platform generates trust scores and explainable forensic reports to combat misinformation, identity fraud, and digital deception.

## Features

- **Multi-Modal Analysis**: Detect deepfakes in video, audio, image, and text content
- **SOTA Models**: Integrates state-of-the-art models (ForensicFlow, RawNet2, SpecDetect++)
- **Explainable AI**: Generates heatmaps, timestamps, and chain-of-thought reasoning
- **Adversarial Defense**: Multi-stage input sanitization pipeline
- **Production Ready**: Async-first architecture, type safety, comprehensive testing

## Quick Start

### Installation

```bash
# Clone the repository
cd Argus_Prime

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Server

```bash
# Development mode with auto-reload
python -m uvicorn src.main:app --reload --port 8000

# Production mode
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

- **Health Check**: `GET /health`
- **Detailed Health**: `GET /health/detailed`
- **Configuration**: `GET /api/v1/config`
- **Submit Analysis**: `POST /api/v1/analyze`
- **Check Status**: `GET /api/v1/analyze/{request_id}/status`
- **Get Result**: `GET /api/v1/analyze/{request_id}/result`
- **WebSocket Stream**: `WS /api/v1/analyze/{request_id}/stream`

### Example Usage

```bash
# Submit a file for analysis
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/v1/analyze

# Check status
curl http://localhost:8000/api/v1/analyze/{request_id}/status

# Get full report
curl http://localhost:8000/api/v1/analyze/{request_id}/result
```

## Project Structure

```
Argus_Prime/
├── src/
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration management
│   ├── api/
│   │   ├── router.py           # FastAPI endpoints
│   │   ├── schemas.py          # Pydantic models
│   │   └── dependencies.py     # Dependency injection
│   ├── core/
│   │   ├── logging.py          # Structured logging
│   │   ├── security.py         # File validation
│   │   └── health.py           # Health checks
│   ├── models/
│   │   ├── interfaces.py       # Abstract base classes
│   │   ├── video/              # Video detection models
│   │   ├── audio/              # Audio detection models
│   │   └── text/               # Text detection models
│   ├── preprocessing/          # Media preprocessing
│   ├── analysis/               # Detection orchestrators
│   ├── defense/                # Adversarial defense
│   ├── explainability/         # Heatmaps, reasoning
│   ├── fusion/                 # Score fusion
│   ├── reporting/              # Report generation
│   └── db/                     # Database layer
├── configs/
│   └── default.yaml            # Default configuration
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── requirements.txt            # Dependencies
├── pyproject.toml              # Project metadata
└── README.md                   # This file
```

## Configuration

All configuration is managed via YAML files with environment variable overrides:

```yaml
# configs/default.yaml
app:
  host: "0.0.0.0"
  port: 8000
  debug: false

database:
  url: "postgresql+asyncpg://argus:argus@localhost:5432/argus_prime"

models:
  video:
    primary: "forensic_flow"
  audio:
    primary: "rawnet2"
  text:
    primary: "spec_detect"

defense:
  enabled: true
  jpeg_quality: 85
```

Environment variables:
- `DATABASE_URL` - Database connection string
- `REDIS_HOST` - Redis host
- `APP_PORT` - Server port
- `APP_DEBUG` - Debug mode

## SOTA Model Integrations

| Modality | Model | Performance | Resource |
|----------|-------|-------------|----------|
| **Video** | ForensicFlow (Tri-Modal) | AUC 0.9752 | 6-12GB VRAM |
| **Video (Lite)** | DistilDIRE + KAN | AUC ~0.92 | 2-4GB VRAM |
| **Audio** | RawNet2 + Vocoder ID | 4.54% EER | 2-4GB VRAM |
| **Text** | SpecDetect++ (Spectral) | SOTA | 1-3GB VRAM |

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/unit/test_schemas.py -v
```

## Architecture

### Request Pipeline

```
Ingestion (0-50ms) → Preprocessing (50-200ms) → Adversarial Defense (200-300ms)
    → Multi-Model Inference (300-2000ms) → Explainability (2000-2500ms)
    → Response Generation (2500-2700ms)
```

### Key Design Principles

1. **Zero Hardcoding**: All thresholds loaded from config
2. **Async-First**: All I/O operations use AsyncIO
3. **Type Safety**: Full Pydantic validation
4. **Modularity**: Swappable model implementations

## License

MIT License
