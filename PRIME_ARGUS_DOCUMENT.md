# PRIME ARGUS DOCUMENT
## Multi-Modal Deepfake Detection & Forensic Platform - Implementation Blueprint

**Version:** 1.0.0  
**Date:** January 21, 2026  
**Status:** Production-Ready Architecture  
**Document Owner:** Lead AI Architect & Forensic Engineer

---

## TABLE OF CONTENTS

1. [Executive Summary](#1-executive-summary)
2. [Life of a Request Flow](#2-life-of-a-request-flow)
3. [Architecture & File Manifesto](#3-architecture--file-manifesto)
4. [Development Strategy](#4-development-strategy)
5. [Shared Data Schemas](#5-shared-data-schemas)
6. [Abstract Base Classes (Interfaces)](#6-abstract-base-classes-interfaces)
7. [API Router Skeleton](#7-api-router-skeleton)
8. [Adversarial Defense Architecture](#8-adversarial-defense-architecture)
9. [Performance & Optimization Guidelines](#9-performance--optimization-guidelines)
10. [Testing Strategy](#10-testing-strategy)

---

## 1. EXECUTIVE SUMMARY

### 1.1 Project Vision

Argus Prime is a production-grade, multi-modal deepfake detection platform designed to analyze audio, image, video, and text content with forensic precision. The platform generates trust scores and explainable forensic reports to combat misinformation, identity fraud, and digital deception in an era where AI-generated content is increasingly indistinguishable from authentic media.

### 1.2 Core Technology Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Backend Framework | FastAPI | 0.115+ | Async REST API with WebSocket support |
| ML Framework | PyTorch | 2.5+ | Neural network inference and training |
| Video Processing | FFmpeg + OpenCV | Latest | Frame extraction and preprocessing |
| Audio Processing | Librosa + torchaudio | Latest | Audio feature extraction |
| Database | PostgreSQL + Redis | 15+/7.0+ | Metadata storage and caching |
| API Documentation | OpenAPI 3.1 + Swagger | - | Interactive API docs |
| Containerization | Docker + Docker Compose | Latest | Environment isolation |

### 1.3 SOTA Model Integrations

| Modality | Primary Model | Performance | Resource Profile |
|----------|---------------|-------------|------------------|
| **Video** | ForensicFlow (Tri-Modal Adaptive) | AUC 0.9752 | 6-12GB VRAM |
| **Video (Lite)** | DistilDIRE + Single-layer KAN | AUC ~0.92 | 2-4GB VRAM |
| **Audio** | RawNet2 + Vocoder Artifact Detection | 4.54% EER | 2-4GB VRAM |
| **Text** | SpecDetect++ (Spectral Analysis) | SOTA | 1-3GB VRAM |
| **Multi-modal** | Gated Fusion Network (ResearchGate 2025) | AUC 0.982 | Adaptive |

### 1.4 Key Architectural Principles

1. **Zero Hardcoding**: All thresholds, model weights, and parameters loaded from config/env
2. **Async-First**: All I/O operations use AsyncIO for maximum throughput
3. **Type Safety**: Full type hinting with Pydantic validation
4. **Adversarial Defense**: Multi-stage input sanitization pipeline
5. **Explainability**: Every detection includes heatmaps, timestamps, and reasoning
6. **Modularity**: Swappable model implementations via abstract interfaces

---

## 2. LIFE OF A REQUEST FLOW

### 2.1 High-Level Request Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REQUEST LIFECYCLE TIMELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Ingestion (0-50ms)                                                         │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ upload  │───▶│ validate│───▶│ malware │───▶│ sanitize│                 │
│  │ router  │    │ schema  │    │ scan    │    │ C2PA    │                 │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                 │
│       │              │              │              │                       │
│       ▼              ▼              ▼              ▼                       │
│  [file_id]      [valid/invalid]   [clean/detected]  [metadata]            │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Preprocessing (50-200ms)                                                   │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ extract │───▶│ detect  │───▶│ crop    │───▶│ normalize│                 │
│  │ frames  │    │ faces   │    │ align   │    │ enhance │                 │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                 │
│       │              │              │              │                       │
│       ▼              ▼              ▼              ▼                       │
│  [frames]       [bboxes]      [aligned]       [tensor]                     │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Adversarial Defense (200-300ms)                                            │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ JPEG    │───▶│ gaussian│───▶│ total   │───▶│ TRIM    │                 │
│  │ compress│    │ noise   │    │ variation│    │ analyze │                 │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                 │
│       │              │              │              │                       │
│       ▼              ▼              ▼              ▼                       │
│  [robust]       [purified]    [smoothed]    [is_adversarial]              │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Multi-Model Inference (300-2000ms)                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        PARALLEL INFERENCE                            │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐        │   │
│  │  │  Video    │  │  Audio    │  │   Text    │  │  rPPG     │        │   │
│  │  │ Detector  │  │ Detector  │  │ Detector  │  │ Analysis  │        │   │
│  │  │ (400-800ms)│  │(200-400ms)│  │(50-100ms) │  │(100-200ms)│        │   │
│  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘        │   │
│  │        │              │              │              │                │   │
│  │        ▼              ▼              ▼              ▼                │   │
│  │  [0.87]         [0.92]         [0.45]         [0.78]               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                               │                                              │
│                               ▼                                              │
│                    ┌────────────────────┐                                   │
│                    │  Score Aggregation │                                   │
│                    │    (Fusion Engine) │                                   │
│                    └────────────────────┘                                   │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Explainability Generation (2000-2500ms)                                    │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐                 │
│  │ heatmap │───▶│ gradcam │───▶│ localize│───▶│ generate│                 │
│  │ Gen     │    │ Viz     │    │ regions │    │ verdict │                 │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘                 │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Response Generation (2500-2700ms)                                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                                │
│  │  Fusion │───▶│  Trust  │───▶│  Return │                                │
│  │  Report │    │  Score  │    │  JSON   │                                │
│  └─────────┘    └─────────┘    └─────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed File-by-File Flow

#### Phase 1: Ingestion (0-50ms)

**File: `router.py` (lines 45-89)**
- Receives multipart file upload via FastAPI
- Extracts file extension, MIME type, and content
- Generates unique `request_id` (UUID4)
- Stores temporary file in `/tmp/argus_uploads/{request_id}/`
- Returns `request_id` to client for polling

**File: `schemas.py` (lines 23-67)**
- Validates file against `UploadFileSchema`
- Checks: file size < 500MB, allowed extensions, MIME type validation
- Returns validation result with errors if any

**File: `security.py` (lines 12-45)**
- Scans file signature (magic bytes) for malware
- Integrates ClamAV daemon via socket
- Returns `SecurityCheckResult` with threat classification

**File: `metadata.py` (lines 15-78)**
- Extracts C2PA metadata using `c2patool` subprocess
- Parses manifest store for AI generation markers
- Returns `C2PASecurityResult` with provenance data

#### Phase 2: Preprocessing (50-200ms)

**File: `video_processor.py` (lines 45-120)**
- Extracts frames at configurable FPS (default: 5 FPS)
- Uses FFmpeg subprocess for reliable extraction
- Returns list of frame paths and metadata

**File: `face_detector.py` (lines 23-89)**
- Runs MTCNN for face detection on each frame
- Returns bounding boxes and confidence scores
- Applies face alignment using 5-point landmarks

**File: `image_preprocessor.py` (lines 34-95)**
- Crops faces to 224x224 (model input size)
- Applies histogram equalization for quality normalization
- Converts to tensor with ImageNet/CLIP normalization

**File: `audio_processor.py` (lines 28-110)**
- Normalizes audio to 16kHz, mono, 16-bit PCM
- Applies Voice Activity Detection (VAD) using WebRTC VAD
- Segments into 2-second windows with 50% overlap
- Generates Mel-spectrograms for RawNet2 input

**File: `text_processor.py` (lines 18-76)**
- Tokenizes text using HuggingFace transformers tokenizer
- Calculates perplexity using multiple LLMs (GPT-2, LLaMA 3B)
- Analyzes burstiness (variance in sentence length/structure)

#### Phase 3: Adversarial Defense (200-300ms)

**File: `adversarial_defender.py` (lines 34-145)**
- Stage 1: JPEG compression (quality=85, breaks pixel-level attacks)
- Stage 2: Gaussian noise injection (σ=0.01, breaks gradient attacks)
- Stage 3: Total variation minimization (breaks diffusion-based attacks)
- Stage 4: TRIM analysis (prediction entropy + KL divergence)
- Returns `AdversarialCheckResult` with confidence scores

#### Phase 4: Multi-Model Inference (300-2000ms)

**File: `video_detector.py` (lines 45-180)**
- **Branch A (Global Visual)**: ConvNeXt-tiny backbone
  - Input: Full-frame tensors (B, 3, 224, 224)
  - Output: Global score (0-1) + attention weights
- **Branch B (Texture)**: Swin Transformer-tiny
  - Input: Face crops (B, 3, 224, 224)
  - Output: Texture anomaly score + patch-level attention
- **Branch C (Spectral)**: CNN with channel attention
  - Input: FFT of face regions (B, 3, 224, 224)
  - Output: Spectral noise pattern score
- **Temporal Pooling**: Attention-weighted frame selection
- **Adaptive Fusion**: Gated fusion with learned weights
- Returns `VideoAnalysisResult` with per-branch scores and heatmap

**File: `audio_detector.py` (lines 38-156)**
- **Vocoder ID**: RawNet2 multi-task head
  - Input: Raw audio waveforms (B, 1, 16000)
  - Output: Vocoder class probabilities (6 classes)
- **Spectral Analysis**: Mel-spectrogram + 2D CNN
  - Input: Mel-spectrograms (B, 1, 256, 256)
  - Output: Artifact probability
- **Fusion**: Weighted average (vocoder_id: 0.6, spectral: 0.4)
- Returns `AudioAnalysisResult` with artifact evidence

**File: `text_detector.py` (lines 25-98)**
- **Spectral Analysis**: DFT of token probabilities
  - Input: Token sequences (B, L)
  - Output: Generation probability via frequency analysis
- **Perplexity Ensemble**: Multiple LLMs for robustness
  - Input: Text tokens
  - Output: Ensemble perplexity score
- **Burstiness Analysis**: Statistical variance analysis
  - Output: Burstiness anomaly score
- Returns `TextAnalysisResult` with per-sentence scores

**File: `rppg_analyzer.py` (lines 32-124)**
- **G-PPG**: Green channel pulse extraction
  - Input: Face crops (B, H, W, 3)
  - Output: Heart rate (BPM) + quality score
- **C-PPG**: Chrominance-based pulse extraction
  - Input: Face crops (B, H, W, 3)
  - Output: Heart rate (BPM) + quality score
- **Consistency Check**: Temporal and spatial coherence
  - Output: rPPG confidence (0-1)
- Returns `rPPGResult` with biological signal analysis

#### Phase 5: Explainability Generation (2000-2500ms)

**File: `heatmap_generator.py` (lines 28-112)**
- **Grad-CAM**: Compute gradients for target class
  - Input: Model logits + target layer
  - Output: Class activation maps
- **DeCLIP**: CLIP-based localization
  - Input: Face crops + CLIP model
  - Output: Manipulation probability heatmap
- Returns heatmap overlays and visualization data

**File: `localizer.py` (lines 18-87)**
- **Temporal**: Frame-level timestamps of manipulation
  - Output: List of (start_frame, end_frame, confidence)
- **Spatial**: Pixel-level mask generation
  - Output: Binary mask and polygon coordinates
- Returns `LocalizationResult` with region coordinates

**File: `reasoning_engine.py` (lines 45-178)**
- **FakeHunter Integration**: Chain-of-thought reasoning
  - Input: All detection results + heatmaps
  - Output: Step-by-step reasoning trace
- **Verdict Generation**: Structured JSON with:
  - What: Manipulation type
  - Where: Timestamp/region
  - Why: Evidence list with confidence
- Returns `ReasoningResult` with explainable verdict

#### Phase 6: Response Generation (2500-2700ms)

**File: `fusion_engine.py` (lines 34-198)**
- **Modality Weights**: Load from config (video: 0.40, audio: 0.35, text: 0.25)
- **Cross-Modal Consistency**: Check for contradictions
  - If video_score > 0.8 and audio_score < 0.3: flag inconsistency
- **Adaptive Adjustment**: Apply ±20% based on consistency
- **Final Score**: Weighted fusion → normalized trust score (0-100)
- Returns `TrustScore` with breakdown

**File: `report_generator.py` (lines 28-145)**
- **Executive Summary**: Trust score, confidence, verdict
- **Detailed Findings**: Per-modality analysis
- **Evidence Section**: Heatmaps, timestamps, spectrograms
- **Recommendations**: Verification steps
- Returns `ForensicReport` as structured JSON

**File: `router.py` (lines 120-180)**
- Polling endpoint returns `AnalysisStatus` with:
  - Status: pending/processing/completed/failed
  - Progress: percentage (0-100)
  - Result: `ForensicReport` when complete
- WebSocket endpoint for real-time progress streaming

---

## 3. ARCHITECTURE & FILE MANIFESTO

### 3.1 Directory Structure

```
argus_prime/
├── src/
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   ├── config.py                  # Configuration loader
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── router.py              # FastAPI router + endpoints
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── dependencies.py        # FastAPI dependency injection
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── security.py            # Malware scanning, auth
│   │   ├── logging.py             # Structured logging setup
│   │   ├── metrics.py             # Prometheus metrics
│   │   └── health.py              # Health check endpoints
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── router.py              # File upload endpoint
│   │   ├── validator.py           # Schema validation
│   │   ├── sanitizer.py           # C2PA + metadata extraction
│   │   └── storage.py             # Temp file management
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── video_processor.py     # Frame extraction
│   │   ├── audio_processor.py     # Audio normalization
│   │   ├── text_processor.py      # Tokenization
│   │   ├── face_detector.py       # MTCNN face detection
│   │   ├── face_aligner.py        # 5-point landmark alignment
│   │   └── preprocessor.py        # Unified preprocessing facade
│   │
│   ├── defense/
│   │   ├── __init__.py
│   │   ├── adversarial_defender.py  # Multi-stage defense
│   │   ├── trim_analyzer.py         # TRIM detection
│   │   └── purifier.py              # DG-GAN purification
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── interfaces.py           # Abstract base classes
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   ├── forensic_flow.py     # Primary video detector
│   │   │   ├── distil_dire.py       # Lightweight fallback
│   │   │   └── kan_detector.py      # CPU-only detector
│   │   ├── audio/
│   │   │   ├── __init__.py
│   │   │   ├── rawnet2_detector.py  # Primary audio detector
│   │   │   └── transcrypt_detector.py # Lightweight fallback
│   │   ├── text/
│   │   │   ├── __init__.py
│   │   │   ├── spec_detect.py       # Spectral text detector
│   │   │   └── perplexity_detector.py # LLM-based detector
│   │   └── registry.py              # Model loader/cacher
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── video_detector.py       # Video analysis orchestrator
│   │   ├── audio_detector.py       # Audio analysis orchestrator
│   │   ├── text_detector.py        # Text analysis orchestrator
│   │   ├── rppg_analyzer.py        # rPPG biological signals
│   │   └── crossmodal.py           # Cross-modal consistency
│   │
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── heatmap_gen.py          # Grad-CAM + DeCLIP
│   │   ├── localizer.py            # Temporal/spatial localization
│   │   └── reasoning.py            # FakeHunter chain-of-thought
│   │
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── fusion_engine.py        # Weighted score fusion
│   │   └── trust_scorer.py         # Final trust score calculation
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── report_gen.py           # Report generation
│   │   ├── verdict_gen.py          # Verdict reasoning
│   │   └── templates/              # Report HTML templates
│   │
│   └── db/
│       ├── __init__.py
│       ├── database.py             # PostgreSQL connection
│       ├── models.py               # SQLAlchemy ORM models
│       ├── repositories.py         # Data access layer
│       └── migrations/             # Alembic migrations
│
├── tests/
│   ├── __init__.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── models/                         # Downloaded model weights
├── configs/
│   ├── default.yaml
│   ├── production.yaml
│   └── development.yaml
│
├── scripts/
│   ├── download_models.py
│   └── run_migrations.sh
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 3.2 File Manifesto (Detailed)

#### 3.2.1 Core Infrastructure Files

---

**File: `main.py`**
```
Role: Application entry point and FastAPI app factory
SOTA Algorithm: FastAPI 0.115+ with async lifespan management
Integration:
  - Imports: config, router, logging, health
  - Input: None (CLI entry point)
  - Output: FastAPI app instance
Why: Minimal startup overhead, clean async cleanup
```

**File: `config.py`**
```
Role: Centralized configuration management via YAML + environment variables
SOTA Algorithm: Pydantic-settings with yaml.load() for config files
Integration:
  - Imports: os, yaml, pydantic
  - Input: CONFIG_PATH env var, config/*.yaml files
  - Output: Config object with typed attributes (models, thresholds, paths)
Why: Zero hardcoding, environment-specific configs, 12-factor app compliance
```

**File: `api/router.py`**
```
Role: FastAPI router with all REST endpoints
SOTA Algorithm: FastAPI dependency injection + async responses
Integration:
  - Imports: schemas, dependencies, analysis services
  - Input: HTTP requests with file uploads
  - Output: JSON responses with analysis results
Endpoints:
  - POST /api/v1/analyze - Main analysis endpoint
  - GET /api/v1/analyze/{id}/status - Polling status
  - GET /api/v1/analyze/{id}/result - Get full report
  - WS /api/v1/analyze/{id}/stream - WebSocket progress
  - GET /api/v1/health - Health check
Why: Async-first for high concurrency, REST + WebSocket for flexibility
```

**File: `api/schemas.py`**
```
Role: Pydantic models for request/response validation
SOTA Algorithm: Pydantic v2 with strict mode and discriminator unions
Integration:
  - Imports: pydantic, datetime, uuid
  - Input: Raw HTTP request data
  - Output: Validated Pydantic models
Schema Examples:
  - UploadFileSchema: file_name, content_type, size, md5_hash
  - AnalysisRequestSchema: file_id, analysis_options
  - TrustScoreSchema: overall_score, confidence, breakdown
  - ForensicReportSchema: executive_summary, findings, evidence
Why: Runtime validation, automatic OpenAPI docs, type safety
```

---

#### 3.2.2 Ingestion Layer

---

**File: `ingestion/router.py`**
```
Role: File upload endpoint with chunked transfer support
SOTA Algorithm: FastAPI UploadFile with streaming to disk
Integration:
  - Imports: validator, sanitizer, storage
  - Input: Multipart file upload (video/audio/image/text)
  - Output: request_id, temporary file path
Features:
  - Chunked upload for large files (>100MB)
  - Progress tracking via Redis
  - Automatic cleanup on completion/failure
Why: Handles 500MB+ files without memory overflow
```

**File: `ingestion/validator.py`**
```
Role: File format and schema validation
SOTA Algorithm: python-magic (libmagic) for MIME detection + custom validators
Integration:
  - Imports: schemas, magic, pathlib
  - Input: Uploaded file bytes + metadata
  - Output: ValidationResult with is_valid, errors, warnings
Checks:
  - File extension whitelist
  - MIME type verification
  - Magic byte validation (first 1024 bytes)
  - File size limits (configurable per type)
  - Archive bomb detection (depth/compression ratio)
Why: First line of defense against malformed/malicious uploads
```

**File: `ingestion/sanitizer.py`**
```
Role: C2PA metadata extraction and content sanitization
SOTA Algorithm: c2patool (contentauth/c2pa-rs) subprocess + metadata parsing
Integration:
  - Imports: subprocess, json, c2patool_path
  - Input: File path
  - Output: C2PASecurityResult with manifest, signatures, AI markers
Features:
  - Extract embedded manifests (JPEG/PNG/MP4)
  - Validate cryptographic signatures
  - Detect AI generation claims in manifest
  - Extract creation tool information
  - Check timestamp consistency
Why: Provenance verification, chain of custody, admissibility support
```

**File: `ingestion/storage.py`**
```
Role: Temporary file storage with automatic lifecycle management
SOTA Algorithm: asyncio.Lock + Redis TTL-based cleanup
Integration:
  - Imports: pathlib, asyncio, redis
  - Input: File bytes, request_id
  - Output: File path, expiry timestamp
Features:
  - Organized directory structure: /tmp/argus/{request_id}/{original_name}
  - Automatic cleanup after TTL (configurable, default 24h)
  - Cleanup task runs every hour
  - Thread-safe concurrent access
Why: Prevents disk exhaustion, clean separation of uploads
```

---

#### 3.2.3 Preprocessing Layer

---

**File: `preprocessing/video_processor.py`**
```
Role: Video frame extraction with configurable FPS and quality
SOTA Algorithm: FFmpeg subprocess with optimized flags
Integration:
  - Imports: subprocess, asyncio, pathlib, ffmpeg_path
  - Input: Video file path, fps (1-30), quality_level
  - Output: List[Frame] with path, timestamp, frame_number
FFmpeg Command:
  ffmpeg -i {input} -vf "fps={fps},scale=640:360" \
    -q:v 2 -f image2 {output_dir}/frame_%06d.jpg
Features:
  - Keyframe extraction mode (I-frames only) for speed
  - Scene detection for smart frame selection
  - GPU-accelerated decode (NVENC) when available
Why: FFmpeg is most reliable, handles all formats, GPU acceleration
```

**File: `preprocessing/face_detector.py`**
```
Role: Multi-face detection with confidence scoring
SOTA Algorithm: MTCNN (Multi-task Cascaded CNN) with PyTorch implementation
Integration:
  - Imports: torch, torchvision, numpy, mtcnn_model_path
  - Input: Frame image (H, W, 3) numpy array
  - Output: List[FaceBox] with bbox, landmarks, confidence
Model Architecture:
  - P-Net: Proposal network (12x12) - fast initial detection
  - R-Net: Refine network (24x24) - accurate filtering
  - O-Net: Output network (48x48) - landmark detection
Features:
  - Batch processing for multiple frames
  - Configurable detection thresholds
  - GPU/CPU auto-detection
Why: SOTA for face detection, 5-point landmarks for alignment
```

**File: `preprocessing/face_aligner.py`**
```
Role: Face alignment using 5-point landmarks
SOTA Algorithm: Similarity transform with reference landmarks
Integration:
  - Imports: numpy, cv2, scipy
  - Input: Face image + 5 landmarks (eyes, nose, mouth corners)
  - Output: Aligned face (224, 224, 3) numpy array
Process:
  - Compute similarity transform matrix
  - Warp to reference template (Based on FFHQ/CelebA)
  - Crop to 224x224
  - Apply histogram equalization
Why: Standardized input for all models, improves accuracy
```

**File: `preprocessing/audio_processor.py`**
```
Role: Audio normalization, VAD, segmentation, and spectrogram generation
SOTA Algorithm: Librosa + webrtcvad + torchaudio
Integration:
  - Imports: librosa, scipy, numpy, webrtcvad
  - Input: Audio file path
  - Output: AudioAnalysisBundle with segments, spectrograms, VAD timestamps
Pipeline:
  1. FFmpeg convert to 16kHz, mono, 16-bit PCM
  2. WebRTC VAD for voice activity detection (mode 2)
  3. Segment into 2s windows with 50% overlap
  4. Generate Mel-spectrograms (n_mels=256, n_fft=512)
  5. Store as torch tensors for model input
Why: Complete audio preprocessing pipeline, VAD filters silence
```

**File: `preprocessing/text_processor.py`**
```
Role: Text tokenization, perplexity, and burstiness analysis
SOTA Algorithm: HuggingFace transformers tokenizers + multiple LLMs
Integration:
  - Imports: transformers, torch, numpy
  - Input: Raw text string
  - Output: TextAnalysisBundle with tokens, perplexities, burstiness
Pipeline:
  1. Language detection (fasttext-langdetect)
  2. Tokenize with multilingual model (bert-base-multilingual-cased)
  3. Calculate perplexity using:
     - GPT-2 (English)
     - XGLM-1.7B (Multilingual)
     - Ensemble average
  4. Burstiness analysis:
     - Sentence length variance
     - Word frequency distribution
     - Repetition patterns
Why: Multiple LLMs prevent single-model bias, burstiness is strong AI signal
```

---

#### 3.2.4 Adversarial Defense Layer

---

**File: `defense/adversarial_defender.py`**
```
Role: Multi-stage input sanitization against adversarial attacks
SOTA Algorithm: Multi-stage purification pipeline (JPEG + Noise + TVM + TRIM)
Integration:
  - Imports: numpy, cv2, torch, scipy
  - Input: Raw input tensor (video frames, audio, or image)
  - Output: Purified tensor + adversarial_check_result
Pipeline Stages:
  Stage 1: JPEG Compression (quality=85, breaks pixel-level attacks)
  Stage 2: Gaussian Noise (σ=0.01, breaks gradient-based attacks)
  Stage 3: Total Variation Minimization (breaks diffusion-based attacks)
  Stage 4: TRIM Analysis (prediction entropy + KL divergence)
Why: Traditional AT fails for AIGI; multi-stage defense covers attack vectors
```

**File: `defense/trim_analyzer.py`**
```
Role: Training-free robust detection of adversarial examples
SOTA Algorithm: TRIM (Training-free Robust detection via Information Maximization)
Integration:
  - Imports: torch, numpy, scipy.stats
  - Input: Model predictions + input features
  - Output: TRIMResult with is_adversarial, entropy_score, kl_divergence
Theory:
  - Adversarial examples shift prediction entropy
  - KL divergence between clean and perturbed predictions
  - No retraining required
Why: Training-free, defends against FGSM/BIM/DeepFool/PGD attacks
```

**File: `defense/purifier.py`**
```
Role: Generator-based adversarial purification
SOTA Algorithm: DG-GAN (Defense Generator - Generative Adversarial Network)
Integration:
  - Imports: torch, nn, dggan_model_path
  - Input: Potentially adversarial input
  - Output: Purified input
Architecture:
  - Encoder: Maps input to latent space
  - Generator: Reconstructs from latent
  - Discriminator: Distinguishes clean vs adversarial
  - Training: Purify → Detect loop
Why: Outperforms AT on iterative attacks, adaptive purification
```

---

#### 3.2.5 Model Layer

---

**File: `models/interfaces.py`**
```
Role: Abstract base classes defining model contracts
SOTA Algorithm: ABC (Abstract Base Classes) with Protocol
Integration:
  - Imports: abc, typing, torch
  - Input: N/A (defines interface only)
  - Output: N/A (interface definition)
Key Interfaces:
  - VideoDetector: detect(frames: List[Tensor]) -> VideoResult
  - AudioDetector: detect(waveform: Tensor) -> AudioResult
  - TextDetector: detect(text: str) -> TextResult
  - BaseModel: load(), predict(), get_heatmap()
Why: Enables swappable implementations, testable via mocking
```

**File: `models/registry.py`**
```
Role: Centralized model loading, caching, and lifecycle management
SOTA Algorithm: Singleton pattern with LRU cache + lazy loading
Integration:
  - Imports: torch, functools, pathlib
  - Input: Model name, device (cuda/cpu)
  - Output: Loaded model instance
Features:
  - Lazy loading (load only when first accessed)
  - GPU memory caching with eviction policy
  - Automatic device placement (cuda/mps/cpu)
  - Model versioning support
  - Health checks and reload on failure
Why: Prevents OOM, manages GPU memory, supports hot-reload
```

**File: `models/video/forensic_flow.py`**
```
Role: Primary SOTA video deepfake detector
SOTA Algorithm: ForensicFlow (CVPR 2025) - Tri-Modal Adaptive Network
Integration:
  - Imports: torch, nn, convnext, swin_transformer
  - Input: List of face-aligned frames (B, T, 3, 224, 224)
  - Output: ForensicFlowResult with scores, heatmaps, branch_outputs
Architecture:
  Branch A: ConvNeXt-tiny (global visual inconsistencies)
    - Input: Full-frame tensors
    - Output: Global score ∈ [0, 1]
  Branch B: Swin Transformer-tiny (texture anomalies)
    - Input: Face crops
    - Output: Texture score ∈ [0, 1]
  Branch C: CNN + Channel Attention (spectral noise)
    - Input: FFT of face regions
    - Output: Spectral score ∈ [0, 1]
  Temporal Pooling: Attention-weighted frame selection
  Adaptive Fusion: Gated fusion with learned weights
Performance: AUC 0.9752, F1 0.9408, Accuracy 0.9208 (CelebDF-v2)
Why: Tri-modal fusion captures artifacts single-branch networks miss
```

**File: `models/video/distil_dire.py`**
```
Role: Lightweight video/image detector for resource-constrained environments
SOTA Algorithm: DistilDIRE (3.2x faster than DIRE, 52.4MB)
Integration:
  - Imports: torch, dIRE_path, torchvision
  - Input: Face-aligned images (B, 3, 224, 224)
  - Output: DIREResult with fake_probability, reconstruction_error
Theory: DIRE (DIffusion Reconstruction Error) measures reconstruction
        error from diffusion model; fake images have higher error.
Why: 3.2x faster than full DIRE, excellent for CPU/low-GPU deployment
```

**File: `models/audio/rawnet2_detector.py`**
```
Role: Primary audio deepfake detector via vocoder artifact identification
SOTA Algorithm: RawNet2 multi-task with vocoder ID pretext task
Integration:
  - Imports: torch, torchaudio, rawnet2_path
  - Input: Raw waveform (B, 1, 16000) for 1-second audio
  - Output: RawNet2Result with vocoder_class, artifact_score, probability
Architecture:
  - Initial block: Sinc-based feature extraction
  - Residual blocks: GRUs for temporal modeling
  - Multi-task head:
    - Primary: Binary real/fake classification
    - Pretext: Vocoder identification (6 classes: HiFi-GAN, WaveGlow, etc.)
Performance: 4.54% EER on ASVspoof 2021
Why: Vocoder artifacts are most reliable audio forgery signal
```

**File: `models/audio/transcrypt_detector.py`**
```
Role: Lightweight audio detector for real-time applications
SOTA Algorithm: Transcend (TrueMedia) - Wav2Vec2 fine-tune
Integration:
  - Imports: torch, transformers, transcrypt_path
  - Input: Audio waveform (B, 16000)
  - Output: TranscryptResult with fake_probability
Architecture: Wav2Vec2-Base fine-tuned on synthetic speech dataset
Why: Small model size, fast inference, battle-tested by TrueMedia
```

**File: `models/text/spec_detect.py`**
```
Role: Primary text deepfake detector via spectral analysis
SOTA Algorithm: SpecDetect++ (2025) - DFT of token probabilities
Integration:
  - Imports: torch, numpy, scipy.fft
  - Input: Tokenized text
  - Output: SpecDetectResult with generation_probability, spectral_features
Theory:
  - AI-generated text has characteristic frequency patterns
  - DFT of token probabilities reveals generation artifacts
  - Nearly 2x faster than neural approaches
Why: Fast, reliable, works without GPU
```

**File: `models/text/perplexity_detector.py`**
```
Role: LLM-based text detection via perplexity and burstiness ensemble
SOTA Algorithm: Ensemble of GPT-2, XGLM-1.7B, and Mistral-7B
Integration:
  - Imports: transformers, torch
  - Input: Tokenized text
  - Output: PerplexityResult with ensemble_score, per_model_scores, burstiness
Features:
  - Perplexity from 3 LLMs (prevent single-model bias)
  - Burstiness analysis (variance metrics)
  - Sentence-level suspiciousness scores
Why: Perplexity is strong signal, ensemble improves robustness
```

---

#### 3.2.6 Analysis Layer

---

**File: `analysis/video_detector.py`**
```
Role: Orchestrates video deepfake detection pipeline
SOTA Algorithm: Pipeline orchestrator with async execution
Integration:
  - Imports: video_processor, face_detector, face_aligner,
            forensic_flow, adversarial_defender, heatmap_gen
  - Input: Video file path, analysis_options
  - Output: VideoAnalysisResult with overall_score, frames_analysis, heatmaps
Pipeline:
  1. Extract frames at 5 FPS
  2. Detect and align faces
  3. Run adversarial defense
  4. Run ForensicFlow inference
  5. Generate heatmaps (Grad-CAM)
  6. Aggregate per-frame scores
  7. Return comprehensive result
Why: Encapsulates video-specific logic, reusable components
```

**File: `analysis/audio_detector.py`**
```
Role: Orchestrates audio deepfake detection pipeline
SOTA Algorithm: Pipeline orchestrator with VAD-based segmentation
Integration:
  - Imports: audio_processor, rawnet2_detector, transcrypt_detector
  - Input: Audio file path, analysis_options
  - Output: AudioAnalysisResult with overall_score, segments_analysis
Pipeline:
  1. Normalize audio (16kHz, mono)
  2. Apply VAD to find speech segments
  3. Segment into 2-second windows
  4. Run RawNet2 on each segment
  5. Run Transcrypt as secondary
  6. Fuse scores with confidence weighting
  7. Return comprehensive result
Why: VAD filters non-speech, improves accuracy and speed
```

**File: `analysis/text_detector.py`**
```
Role: Orchestrates text deepfake detection pipeline
SOTA Algorithm: Pipeline orchestrator with sentence-level analysis
Integration:
  - Imports: text_processor, spec_detect, perplexity_detector
  - Input: Text string, analysis_options
  - Output: TextAnalysisResult with overall_score, sentence_scores
Pipeline:
  1. Detect language
  2. Tokenize text
  3. Run SpecDetect++ (spectral analysis)
  4. Run PerplexityDetector (LLM ensemble)
  5. Analyze burstiness
  6. Score per sentence
  7. Aggregate to document level
  8. Return comprehensive result
Why: Sentence-level granularity enables localization
```

**File: `analysis/rppg_analyzer.py`**
```
Role: Extracts and validates biological signals from video
SOTA Algorithm: G-PPG (green channel) + C-PPG (chrominance) fusion
Integration:
  - Imports: numpy, scipy, cv2
  - Input: Face-aligned video frames
  - Output: rPPGResult with heart_rate, quality_score, consistency
Algorithm:
  G-PPG:
    - Extract green channel signal from facial ROI
    - Apply bandpass filter (0.7-4.0 Hz for HR)
    - FFT to extract dominant frequency
  C-PPG:
    - Compute chrominance signals (R-G, 2B-G-R)
    - Independent component analysis (ICA)
    - Select best signal via SNR
  Fusion: Weighted average with quality-based weighting
Why: Biological signals (heart rate) are hard to fake, adds robustness
```

**File: `analysis/crossmodal.py`**
```
Role: Cross-modal consistency checking and contradiction detection
SOTA Algorithm: Rule-based + ML hybrid for consistency verification
Integration:
  - Imports: numpy, torch
  - Input: VideoResult, AudioResult, TextResult
  - Output: CrossModalResult with consistency_score, contradictions
Checks:
  - Lip-sync: Mouth movements vs audio (if both present)
  - Temporal: Frame timestamps vs audio segments
  - Content: Text vs video/audio context mismatch
  - Biological: rPPG consistency across frames
Output:
  - consistency_score ∈ [0, 1]
  - List of contradictions with confidence
  - Recommendations for manual review
Why: Cross-modal checks catch inconsistencies single-modality misses
```

---

#### 3.2.7 Explainability Layer

---

**File: `explainability/heatmap_gen.py`**
```
Role: Generates interpretable heatmaps for visual deepfake evidence
SOTA Algorithm: Grad-CAM + DeCLIP for multi-view explanation
Integration:
  - Imports: torch, cv2, numpy, clip
  - Input: Model logits, target layer, input image
  - Output: HeatmapResult with heatmap_overlay, regions_of_interest
Methods:
  Grad-CAM:
    - Compute gradients of target class wrt target layer
    - Global average pooling of gradients
    - Apply ReLU to get positive contributions
    - Upsample to input size
  DeCLIP:
    - Use CLIP image encoder features
    - Compare with text "manipulated face" embedding
    - Generate attention map
Why: Visual evidence increases human trust, enables manual verification
```

**File: `explainability/localizer.py`**
```
Role: Localizes manipulation regions in time and space
SOTA Algorithm: Attention-based temporal localization + mask prediction
Integration:
  - Imports: torch, numpy, scipy
  - Input: VideoResult, AudioResult, heatmaps
  - Output: LocalizationResult with timestamps, spatial_masks
Temporal Localization:
  - Frame-level confidence scores from ForensicFlow
  - Peak detection for manipulation timestamps
  - Smoothing via Savitzky-Golay filter
Spatial Localization:
  - Heatmap thresholding (Otsu's method)
  - Contour detection for region boundaries
  - Polygon simplification for clean masks
Why: Localization enables focused manual review, legal evidence
```

**File: `explainability/reasoning.py`**
```
Role: Generates chain-of-thought reasoning for verdicts
SOTA Algorithm: FakeHunter integration - structured reasoning engine
Integration:
  - Imports: json, templates, analysis_results
  - Input: All detection results, heatmaps, localization
  - Output: ReasoningResult with reasoning_trace, verdict, evidence_chain
Reasoning Structure:
  1. Observation: "Face shows inconsistent texture patterns"
  2. Thought: "Texture branch detected 0.87 anomaly score"
  3. Action: "Cross-check with spectral branch"
  4. Evidence: "Spectral branch confirms 0.91 noise pattern"
  5. Conclusion: "Manipulation detected with 89% confidence"
Verdict Output:
  {
    "verdict": "MANIPULATED",
    "confidence": 0.89,
    "what": "Face swapping via FaceSwap method",
    "where": "Frames 45-78, facial boundary region",
    "why": [
      "Texture anomaly: 0.87 (ConvNeXt branch)",
      "Spectral noise pattern: 0.91 (CNN branch)",
      "rPPG inconsistency: 0.23 (biological signal)"
    ]
  }
Why: Explainable AI is critical for legal/adversarial use cases
```

---

#### 3.2.8 Fusion Layer

---

**File: `fusion/fusion_engine.py`**
```
Role: Multi-modal score fusion with adaptive weighting
SOTA Algorithm: Gated Fusion Network (ResearchGate 2025)
Integration:
  - Imports: torch, numpy, config
  - Input: VideoResult, AudioResult, TextResult, rPPGResult
  - Output: FusionResult with fused_scores, weights_used
Algorithm:
  Base Weights (configurable):
    - Video: 0.40
    - Audio: 0.35
    - Text: 0.25
  Cross-Modal Consistency Check:
    - If |video_score - audio_score| > 0.3: flag inconsistency
    - If rPPG confidence < 0.5: reduce video weight by 20%
  Adaptive Fusion:
    - consistency_adjustment = ±20% based on checks
    - final_score = Σ(weight_i × score_i) × consistency_factor
  Confidence Calibration:
    - Apply temperature scaling to raw scores
    - Return calibrated confidence interval
Performance: AUC 0.982 on multi-dataset benchmark
Why: Adaptive weighting improves robustness to missing modalities
```

**File: `fusion/trust_scorer.py`**
```
Role: Computes final trust score and confidence level
SOTA Algorithm: Score normalization + confidence calibration
Integration:
  - Imports: scipy.stats, config
  - Input: FusionResult, analysis_metadata
  - Output: TrustScore with overall_score, confidence, breakdown
Calculation:
  1. Raw Score: Weighted fusion result (0-1)
  2. Trust Score: 100 × raw_score
  3. Confidence: Based on:
     - Model agreement (higher agreement = higher confidence)
     - rPPG quality (higher quality = higher confidence)
     - Adversarial check (clean = higher confidence)
  4. Breakdown:
     - Video contribution: 40%
     - Audio contribution: 35%
     - Text contribution: 25%
Why: Trust score provides actionable metric, confidence enables risk-based routing
```

---

#### 3.2.9 Reporting Layer

---

**File: `reporting/report_gen.py`**
```
Role: Generates comprehensive forensic reports
SOTA Algorithm: Template-based JSON report generation
Integration:
  - Imports: json, jinja2, datetime, uuid
  - Input: AnalysisResult bundle
  - Output: ForensicReport (JSON)
Report Structure:
  {
    "report_id": "uuid",
    "timestamp": "ISO8601",
    "file_info": {...},
    "executive_summary": {
      "verdict": "MANIPULATED",
      "trust_score": 87,
      "confidence": 0.89
    },
    "findings": {
      "video": {...},
      "audio": {...},
      "text": {...}
    },
    "evidence": {
      "heatmaps": [...],
      "timestamps": [...],
      "spectrograms": [...]
    },
    "recommendations": [...]
  }
Why: Structured reports enable automation, legal review, audit trails
```

**File: `reporting/verdict_gen.py`**
```
Role: Generates human-readable verdict with explanations
SOTA Algorithm: Template-based natural language generation
Integration:
  - Imports: templates, reasoning_result
  - Input: ReasoningResult, TrustScore
  - Output: VerdictDocument
Verdict Template:
  "Based on our multi-modal analysis, this content has been
   MANIPULATED with a confidence of {confidence:.1%}.
   
   KEY FINDINGS:
   {finding_1}
   {finding_2}
   
   EVIDENCE:
   {evidence_summary}
   
   RECOMMENDATION: {recommendation}"
Why: Human-readable output for non-technical stakeholders
```

---

#### 3.2.10 Database Layer

---

**File: `db/database.py`**
```
Role: PostgreSQL connection management with async support
SOTA Algorithm: SQLAlchemy 2.0 async engine + connection pooling
Integration:
  - Imports: sqlalchemy, asyncpg
  - Input: Database URL from config
  - Output: Async engine and session factory
Features:
  - Connection pooling (min=5, max=20)
  - Automatic reconnection on failure
  - Transaction management with context managers
  - Migration support via Alembic
Why: Async SQLAlchemy for high-concurrency, connection pooling for performance
```

**File: `db/models.py`**
```
Role: SQLAlchemy ORM models for database tables
SOTA Algorithm: Declarative mapping with relationships
Integration:
  - Imports: sqlalchemy, uuid, datetime
  - Input: N/A (model definitions)
  - Output: ORM classes
Key Models:
  - Analysis: request_id, file_hash, status, created_at
  - VideoResult: request_id, frames_analyzed, overall_score
  - AudioResult: request_id, duration, overall_score
  - TextResult: request_id, word_count, overall_score
  - TrustScore: request_id, score, confidence, breakdown
Why: Type-safe database operations, automatic migrations
```

**File: `db/repositories.py`**
```
Role: Data access layer with async methods
SOTA Algorithm: Repository pattern with unit of work
Integration:
  - Imports: models, database, typing
  - Input: Domain objects or IDs
  - Output: Domain objects or None
Methods:
  - create_analysis(analysis: Analysis) -> Analysis
  - get_analysis(request_id: str) -> Analysis
  - update_status(request_id: str, status: str) -> None
  - save_result(request_id: str, result: VideoResult) -> None
  - get_analyses(filters: dict, pagination: dict) -> List[Analysis]
Why: Separates data access from business logic, testable with mocks
```

---

## 4. DEVELOPMENT STRATEGY

### 4.1 Skeleton & Interface Approach

Instead of building complete files in isolation, follow this sequence:

**Phase 1: Schema & Interfaces (Days 1-3)**

```
1. Create schemas.py with all Pydantic models
   - UploadFileSchema
   - AnalysisRequestSchema
   - VideoAnalysisSchema
   - AudioAnalysisSchema
   - TextAnalysisSchema
   - TrustScoreSchema
   - ForensicReportSchema

2. Create interfaces.py with ABCs
   - BaseDetector (load, predict, get_heatmap)
   - VideoDetector (extends BaseDetector)
   - AudioDetector (extends BaseDetector)
   - TextDetector (extends BaseDetector)
   - Preprocessor (process, validate)

3. Create router.py skeleton
   - All endpoints with minimal docstrings
   - Type hints pointing to schemas
   - Stub implementations (raise NotImplementedError)
```

**Phase 2: Core Infrastructure (Days 4-7)**

```
1. Implement config.py
   - YAML config loading
   - Environment variable overrides
   - Type validation

2. Implement database layer
   - database.py (connection management)
   - models.py (ORM definitions)
   - repositories.py (data access)

3. Implement storage.py
   - File storage with TTL cleanup
   - Concurrent access safety
```

**Phase 3: Preprocessing Pipeline (Days 8-14)**

```
1. video_processor.py
   - FFmpeg wrapper for frame extraction
   - Scene detection integration

2. face_detector.py
   - MTCNN integration
   - Batch processing

3. audio_processor.py
   - Librosa pipeline
   - VAD integration

4. text_processor.py
   - Tokenizer integration
   - Perplexity calculation
```

**Phase 4: Model Integration (Days 15-25)**

```
1. Implement registry.py
   - Lazy loading mechanism
   - GPU memory management

2. Implement ForensicFlow (video)
   - ConvNeXt branch
   - Swin Transformer branch
   - Spectral branch
   - Fusion layer

3. Implement RawNet2 (audio)
   - Vocoder identification
   - Artifact detection

4. Implement SpecDetect++ (text)
   - Spectral analysis
   - Ensemble perplexity
```

**Phase 5: Analysis Orchestrators (Days 26-32)**

```
1. video_detector.py
   - Pipeline orchestration
   - Heatmap generation

2. audio_detector.py
   - VAD-based segmentation
   - Score fusion

3. text_detector.py
   - Sentence-level analysis
   - Burstiness calculation

4. rppg_analyzer.py
   - G-PPG + C-PPG
   - Consistency checking
```

**Phase 6: Fusion & Reporting (Days 33-38)**

```
1. fusion_engine.py
   - Gated fusion
   - Cross-modal checks

2. trust_scorer.py
   - Score normalization
   - Confidence calibration

3. report_gen.py
   - JSON report generation
   - Template rendering

4. reasoning.py
   - Chain-of-thought generation
   - Verdict structuring
```

**Phase 7: Defense & Polish (Days 39-45)**

```
1. adversarial_defender.py
   - Multi-stage pipeline
   - TRIM integration

2. integration testing
   - End-to-end request flow
   - Performance benchmarking

3. API documentation
   - OpenAPI schema refinement
   - Example responses
```

### 4.2 Recommended Build Order

| Priority | File | Reason |
|----------|------|--------|
| P0 | schemas.py | All other files depend on schemas |
| P0 | interfaces.py | Models depend on interfaces |
| P0 | config.py | Configuration needed everywhere |
| P1 | router.py | API contract for frontend |
| P1 | registry.py | Model loading for all detectors |
| P2 | video_processor.py | Video analysis depends on this |
| P2 | face_detector.py | Video analysis depends on this |
| P2 | audio_processor.py | Audio analysis depends on this |
| P2 | text_processor.py | Text analysis depends on this |
| P3 | forensic_flow.py | Core video detection |
| P3 | rawnet2_detector.py | Core audio detection |
| P3 | spec_detect.py | Core text detection |
| P4 | video_detector.py | Orchestrates video pipeline |
| P4 | audio_detector.py | Orchestrates audio pipeline |
| P4 | text_detector.py | Orchestrates text pipeline |
| P5 | fusion_engine.py | Final scoring |
| P5 | report_gen.py | User-facing output |
| P6 | adversarial_defender.py | Security hardening |
| P6 | reasoning.py | Explainability |

### 4.3 Testing Strategy per File

| File | Test Type | Mock Dependencies |
|------|-----------|-------------------|
| schemas.py | Unit | None (pure Pydantic) |
| interfaces.py | Unit | None (ABC definitions) |
| config.py | Unit | None (config files) |
| router.py | Integration | Mock analysis services |
| registry.py | Unit | Mock model weights |
| video_processor.py | Integration | Real FFmpeg |
| face_detector.py | Integration | Real MTCNN model |
| forensic_flow.py | Integration | Real model weights |
| video_detector.py | Integration | All video dependencies |
| fusion_engine.py | Unit | Mock detection results |
| report_gen.py | Unit | Mock analysis results |

---

## 5. SHARED DATA SCHEMAS

### 5.1 Core Schemas

```python
# src/api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
import uuid

class Verdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    MANIPULATED = "MANIPULATED"
    UNCERTAIN = "UNCERTAIN"

class FileType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ConfidenceLevel(str, Enum):
    HIGH = "high"      # > 90%
    MEDIUM = "medium"  # 70-90%
    LOW = "low"        # < 70%

# === Request Schemas ===

class UploadFileSchema(BaseModel):
    file_name: str = Field(..., max_length=255)
    content_type: str = Field(..., pattern=r"^(video|audio|image|text)/.*")
    file_size: int = Field(..., gt=0, le=500_000_000)  # 500MB max
    md5_hash: Optional[str] = Field(None, min_length=32, max_length=32)

class AnalysisRequestSchema(BaseModel):
    file_id: uuid.UUID
    analysis_options: Dict[str, Any] = Field(default_factory=dict)

class AnalysisOptionsSchema(BaseModel):
    video: VideoOptionsSchema = Field(default_factory=VideoOptionsSchema)
    audio: AudioOptionsSchema = Field(default_factory=AudioOptionsSchema)
    text: TextOptionsSchema = Field(default_factory=TextOptionsSchema)

class VideoOptionsSchema(BaseModel):
    fps: int = Field(default=5, ge=1, le=30)
    extract_faces: bool = True
    generate_heatmaps: bool = True
    rppg_analysis: bool = True

class AudioOptionsSchema(BaseModel):
    sample_rate: int = Field(default=16000)
    segment_duration: int = Field(default=2, ge=1, le=10)
    apply_vad: bool = True

class TextOptionsSchema(BaseModel):
    language: Optional[str] = None
    include_perplexity: bool = True
    include_burstiness: bool = True

# === Response Schemas ===

class TrustScoreSchema(BaseModel):
    overall_score: float = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0, le=1)
    confidence_level: ConfidenceLevel
    breakdown: Dict[str, float]  # modality -> contribution
    verdict: Verdict

class FrameAnalysisSchema(BaseModel):
    frame_number: int
    timestamp: float
    face_bbox: Optional[List[float]]  # [x1, y1, x2, y2]
    manipulation_score: float
    heatmap_path: Optional[str]

class VideoResultSchema(BaseModel):
    frames_analyzed: int
    faces_detected: int
    overall_score: float
    per_frame_scores: List[FrameAnalysisSchema]
    heatmap_path: Optional[str]
    branch_scores: Dict[str, float]  # global, texture, spectral
    rppg_result: Optional[Dict[str, Any]]

class SegmentAnalysisSchema(BaseModel):
    start_time: float
    end_time: float
    manipulation_score: float
    vocoder_class: Optional[str]
    spectrogram_path: Optional[str]

class AudioResultSchema(BaseModel):
    duration_seconds: float
    segments_analyzed: int
    overall_score: float
    per_segment_scores: List[SegmentAnalysisSchema]
    vocoder_detected: Optional[str]
    artifact_evidence: Dict[str, float]

class SentenceAnalysisSchema(BaseModel):
    sentence: str
    start_char: int
    end_char: int
    manipulation_score: float
    perplexity: float
    burstiness: float

class TextResultSchema(BaseModel):
    word_count: int
    language_detected: str
    overall_score: float
    per_sentence_scores: List[SentenceAnalysisSchema]
    perplexity_scores: Dict[str, float]  # model -> score
    burstiness_metrics: Dict[str, float]

class LocalizationSchema(BaseModel):
    timestamps: List[Dict[str, Any]]  # start, end, confidence
    spatial_regions: List[Dict[str, Any]]  # polygon, score
    heatmap_overlay_path: Optional[str]

class EvidenceSchema(BaseModel):
    heatmaps: List[str]
    spectrograms: List[str]
    localization: LocalizationSchema
    c2pa_metadata: Optional[Dict[str, Any]]
    adversarial_check: Optional[Dict[str, Any]]

class ForensicReportSchema(BaseModel):
    report_id: uuid.UUID
    request_id: uuid.UUID
    created_at: datetime
    file_info: Dict[str, Any]
    executive_summary: Dict[str, Any]  # verdict, score, confidence
    trust_score: TrustScoreSchema
    findings: Dict[str, Any]  # video, audio, text results
    evidence: EvidenceSchema
    recommendations: List[str]
    raw_output: Optional[Dict[str, Any]]

# === Status Schemas ===

class AnalysisStatusSchema(BaseModel):
    request_id: uuid.UUID
    status: AnalysisStatus
    progress: int = Field(..., ge=0, le=100)
    current_stage: Optional[str]
    estimated_time_remaining: Optional[int]  # seconds
    error_message: Optional[str]

# === WebSocket Schemas ===

class ProgressUpdateSchema(BaseModel):
    request_id: uuid.UUID
    stage: str
    progress: int
    message: str
    timestamp: datetime
```

### 5.2 Internal Data Schemas

```python
# src/core/schemas.py (internal use)

from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class Frame:
    path: str
    frame_number: int
    timestamp: float
    width: int
    height: int

@dataclass
class FaceBox:
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    landmarks: Tuple[Tuple[float, float], ...]  # 5 points
    confidence: float
    aligned_image: np.ndarray

@dataclass
class AudioSegment:
    waveform: np.ndarray  # (samples,)
    sample_rate: int
    start_time: float
    end_time: float
    is_speech: bool
    spectrogram: np.ndarray

@dataclass
class TextTokens:
    tokens: List[str]
    token_ids: List[int]
    attention_mask: List[int]
    language: str
```

---

## 6. ABSTRACT BASE CLASSES (INTERFACES)

```python
# src/models/interfaces.py

from abc import ABC, abstractmethod
from typing import Protocol, TypeVar, Generic
from dataclasses import dataclass
import torch
from enum import Enum

T = TypeVar("T")
R = TypeVar("R")

class ModalityType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    TEXT = "text"
    IMAGE = "image"

@dataclass
class DetectionResult:
    modality: ModalityType
    overall_score: float
    confidence: float
    is_manipulated: bool
    metadata: dict
    heatmap: Optional[torch.Tensor] = None

class BaseModel(ABC):
    @abstractmethod
    def load(self, device: torch.device) -> None:
        """Load model weights to device."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Unload model from memory."""
        pass

    @abstractmethod
    def get_model_info(self) -> dict:
        """Return model metadata."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Check if model is loaded and functional."""
        pass

class BaseDetector(BaseModel, ABC):
    @abstractmethod
    def detect(self, input_data: T) -> DetectionResult:
        """Run detection on input data."""
        pass

    @abstractmethod
    def get_heatmap(self, input_data: T) -> torch.Tensor:
        """Generate attention heatmap for explainability."""
        pass

class VideoDetector(Protocol):
    def detect(self, frames: list[torch.Tensor]) -> DetectionResult:
        """Detect deepfake in video frames."""
        ...

    def get_heatmap(self, frame: torch.Tensor) -> torch.Tensor:
        """Generate frame-level heatmap."""
        ...

class AudioDetector(Protocol):
    def detect(self, waveform: torch.Tensor, sample_rate: int) -> DetectionResult:
        """Detect synthetic audio."""
        ...

    def identify_vocoder(self, waveform: torch.Tensor) -> str:
        """Identify vocoder used for synthesis."""
        ...

class TextDetector(Protocol):
    def detect(self, text: str, tokens: dict) -> DetectionResult:
        """Detect AI-generated text."""
        ...

    def get_perplexity(self, text: str) -> float:
        """Calculate perplexity score."""
        ...

    def analyze_burstiness(self, text: str) -> dict:
        """Analyze burstiness metrics."""
        ...

class Preprocessor(Protocol):
    def process(self, input_data: T) -> R:
        """Preprocess input for model."""
        ...

    def validate(self, input_data: T) -> tuple[bool, str]:
        """Validate input is processable."""
        ...

class FusionEngine(Protocol):
    def fuse(self, results: list[DetectionResult]) -> dict:
        """Fuse multi-modal results."""
        ...

    def calculate_trust_score(self, fused: dict) -> dict:
        """Calculate final trust score."""
        ...
```

---

## 7. API ROUTER SKELETON

```python
# src/api/router.py

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from typing import Optional
import uuid
import logging

from .schemas import (
    AnalysisRequestSchema,
    AnalysisStatusSchema,
    ForensicReportSchema,
    Verdict,
    AnalysisStatus,
)
from .dependencies import (
    get_analysis_service,
    get_storage_service,
    get_config,
)
from src.core.logging import logger

router = APIRouter(prefix="/api/v1", tags=["analysis"])

@router.post("/analyze", response_model=dict)
async def analyze(
    file: UploadFile = File(...),
    analysis_service = Depends(get_analysis_service),
):
    """
    Submit a file for deepfake analysis.
    
    Supports: video (mp4, avi, mov), audio (wav, mp3, flac),
    image (jpg, png, webp), text (txt, md, pdf).
    
    Returns request_id for polling/websocket connection.
    """
    logger.info(f"Received analysis request: {file.filename}")
    try:
        request_id = await analysis_service.create_analysis(
            file=file,
            options={},
        )
        return {
            "request_id": str(request_id),
            "status": AnalysisStatus.PENDING.value,
            "message": "Analysis queued. Poll /api/v1/analyze/{id}/status for updates.",
        }
    except Exception as e:
        logger.error(f"Analysis creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analyze/{request_id}/status", response_model=AnalysisStatusSchema)
async def get_status(
    request_id: uuid.UUID,
    analysis_service = Depends(get_analysis_service),
):
    """Get analysis progress and status."""
    status = await analysis_service.get_status(request_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return status

@router.get("/analyze/{request_id}/result", response_model=ForensicReportSchema)
async def get_result(
    request_id: uuid.UUID,
    analysis_service = Depends(get_analysis_service),
):
    """Get complete analysis result."""
    result = await analysis_service.get_result(request_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Result not found")
    return result

@router.websocket("/analyze/{request_id}/stream")
async def stream_progress(
    websocket: WebSocket,
    request_id: uuid.UUID,
    analysis_service = Depends(get_analysis_service),
):
    """WebSocket endpoint for real-time progress updates."""
    await websocket.accept()
    try:
        async for update in analysis_service.stream_progress(request_id):
            await websocket.send_json(update.dict())
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

@router.get("/health")
async def health_check():
    """System health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

@router.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with component status."""
    return {
        "status": "healthy",
        "components": {
            "database": "connected",
            "models": "loaded",
            "storage": "available",
        },
    }
```

---

## 8. ADVERSARIAL DEFENSE ARCHITECTURE

### 8.1 Defense Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL DEFENSE PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Input: Raw Media (Video/Audio/Image)                                   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Stage 1: JPEG Compression (Quality=85)                            │  │
│  │ Breaks: Pixel-level perturbations, gradient-based attacks         │  │
│  │ Mechanism: Non-linear quantization removes high-frequency noise   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Stage 2: Gaussian Noise Injection (σ=0.01)                        │  │
│  │ Breaks: FGSM, BIM, PGD attacks                                    │  │
│  │ Mechanism: Random noise disrupts adversarial gradients            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Stage 3: Total Variation Minimization                             │  │
│  │ Breaks: Diffusion-based attacks, smooth perturbations             │  │
│  │ Mechanism: Removes spurious high-frequency variations             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Stage 4: TRIM Analysis                                            │  │
│  │ Detects: Adversarial examples (training-free)                     │  │
│  │ Mechanism: Prediction entropy + KL divergence                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                    │                                    │
│                                    ▼                                    │
│  Output: Purified Media + Adversarial Check Result                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Implementation Details

```python
# src/defense/adversarial_defender.py

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional
from config import get_config

@dataclass
class AdversarialCheckResult:
    is_adversarial: bool
    confidence: float
    entropy_score: float
    kl_divergence: float
    stages_passed: list[str]
    purification_applied: bool

class AdversarialDefender:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.trim_threshold = self.config.get("trim_threshold", 0.1)
        self.jpeg_quality = self.config.get("jpeg_quality", 85)
        self.noise_std = self.config.get("noise_std", 0.01)

    def purify(self, input_tensor: torch.Tensor) -> Tuple[torch.Tensor, AdversarialCheckResult]:
        """
        Multi-stage adversarial purification pipeline.
        
        Args:
            input_tensor: (C, H, W) or (B, C, H, W) image tensor
            
        Returns:
            purified_tensor: Same shape as input
            result: AdversarialCheckResult
        """
        stages_passed = []
        
        # Convert to numpy for image operations
        if input_tensor.dim() == 3:
            is_batch = False
            input_np = input_tensor.cpu().numpy()
            input_np = np.transpose(input_np, (1, 2, 0))  # CHW -> HWC
        else:
            is_batch = True
            input_np = input_tensor.cpu().numpy()
            # Keep as NCHW for batch processing
        
        # Stage 1: JPEG Compression
        purified_np = self._jpeg_compress(input_np, quality=self.jpeg_quality)
        stages_passed.append("jpeg_compression")
        
        # Stage 2: Gaussian Noise
        purified_np = self._add_gaussian_noise(purified_np, std=self.noise_std)
        stages_passed.append("gaussian_noise")
        
        # Stage 3: Total Variation Minimization
        purified_np = self._total_variation_denoise(purified_np)
        stages_passed.append("tvm_denoising")
        
        # Convert back to tensor
        if is_batch:
            purified_np = np.transpose(purified_np, (0, 3, 1, 2))  # NHWC -> NCHW
        else:
            purified_np = np.transpose(purified_np, (2, 0, 1))  # HWC -> CHW
        
        purified_tensor = torch.from_numpy(purified_np).float().to(input_tensor.device)
        
        # Stage 4: TRIM Analysis
        trim_result = self._trim_analysis(input_tensor, purified_tensor)
        stages_passed.append("trim_analysis")
        
        result = AdversarialCheckResult(
            is_adversarial=trim_result.is_adversarial,
            confidence=trim_result.confidence,
            entropy_score=trim_result.entropy_score,
            kl_divergence=trim_result.kl_divergence,
            stages_passed=stages_passed,
            purification_applied=True,
        )
        
        return purified_tensor, result

    def _jpeg_compress(self, image_np: np.ndarray, quality: int) -> np.ndarray:
        """Apply JPEG compression to break pixel-level attacks."""
        if image_np.shape[-1] == 3:
            # Encode to JPEG
            _, encoded = cv2.imencode(
                ".jpg", 
                (image_np * 255).astype(np.uint8),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            # Decode back
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            return decoded.astype(np.float32) / 255.0
        return image_np

    def _add_gaussian_noise(self, image_np: np.ndarray, std: float) -> np.ndarray:
        """Add Gaussian noise to disrupt gradient-based attacks."""
        noise = np.random.normal(0, std, image_np.shape).astype(np.float32)
        return np.clip(image_np + noise, 0, 1)

    def _total_variation_denoise(self, image_np: np.ndarray) -> np.ndarray:
        """Apply total variation minimization."""
        # Use cv2 fastNlMeansDenoising as approximation
        if image_np.shape[-1] == 3:
            denoised = cv2.fastNlMeansDenoisingColored(
                (image_np * 255).astype(np.uint8),
                None,
                h=10,
                hColor=10,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return denoised.astype(np.float32) / 255.0
        return image_np

    def _trim_analysis(
        self, 
        original: torch.Tensor, 
        purified: torch.Tensor
    ) -> AdversarialCheckResult:
        """
        TRIM: Training-free Robust detection via Information Maximization.
        
        Detects adversarial examples by analyzing prediction changes
        after purification.
        """
        # This is a simplified version - full implementation requires
        # model access and probability distribution analysis
        entropy_score = self._compute_entropy(original, purified)
        kl_divergence = self._compute_kl_divergence(original, purified)
        
        is_adversarial = (
            entropy_score > self.trim_threshold or
            kl_divergence > self.trim_threshold
        )
        
        return AdversarialCheckResult(
            is_adversarial=is_adversarial,
            confidence=max(entropy_score, kl_divergence),
            entropy_score=entropy_score,
            kl_divergence=kl_divergence,
            stages_passed=[],
            purification_applied=False,
        )

    def _compute_entropy(
        self, 
        original: torch.Tensor, 
        purified: torch.Tensor
    ) -> float:
        """Compute prediction entropy difference."""
        # Placeholder - actual implementation uses model predictions
        return 0.05  # Low entropy = clean

    def _compute_kl_divergence(
        self, 
        original: torch.Tensor, 
        purified: torch.Tensor
    ) -> float:
        """Compute KL divergence between predictions."""
        # Placeholder - actual implementation uses probability distributions
        return 0.03  # Low KL = clean
```

---

## 9. PERFORMANCE & OPTIMIZATION GUIDELINES

### 9.1 Inference Time Targets

| Component | CPU Target | GPU Target | Optimization Strategy |
|-----------|------------|------------|----------------------|
| Video (10s, 5 FPS) | < 5s | < 1s | Keyframe extraction, batch inference |
| Audio (10s) | < 500ms | < 100ms | VAD segmentation, waveform caching |
| Text (1000 words) | < 200ms | < 50ms | Token batching, model quantization |
| rPPG Analysis | < 300ms | < 50ms | Efficient ROI selection |
| Fusion & Reporting | < 100ms | < 100ms | Pre-computed templates |
| **Total Request** | < 6s | < 1.5s | Parallel modal inference |

### 9.2 Memory Management

```python
# src/models/registry.py

import torch
import gc
from functools import lru_cache
from typing import Dict, Optional
from contextlib import contextmanager

class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, torch.nn.Module] = {}
        self._device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._max_gpu_memory: float = 0.8  # Use 80% of available VRAM
    
    @contextmanager
    def get_model(self, model_name: str):
        """Context manager for temporary model access."""
        try:
            if model_name not in self._models:
                self._load_model(model_name)
            yield self._models[model_name]
        finally:
            # Cleanup after each request
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _load_model(self, model_name: str):
        """Load model with memory management."""
        if model_name == "forensic_flow":
            from src.models.video.forensic_flow import ForensicFlow
            model = ForensicFlow.load(self._device)
        # ... other models
        
        # Check memory before loading
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            available = torch.cuda.get_device_properties(0).total_memory - reserved
            
            if available < model.get_memory_requirement():
                # Evict least recently used model
                self._evict_lru_model()
        
        self._models[model_name] = model
    
    def _evict_lru_model(self):
        """Evict least recently used model to free memory."""
        # Implementation with access tracking
        pass
```

### 9.3 Batch Processing Optimization

```python
# src/models/video/forensic_flow.py

class ForensicFlow:
    def batch_inference(
        self, 
        frames: list[torch.Tensor],
        batch_size: int = 8,
    ) -> list[dict]:
        """
        Process frames in batches for memory efficiency.
        
        Args:
            frames: List of frame tensors (C, H, W)
            batch_size: Number of frames per batch
            
        Returns:
            List of detection results per frame
        """
        results = []
        
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            batch = torch.stack(batch_frames)  # (B, C, H, W)
            
            with torch.no_grad():
                batch_results = self._forward(batch)
                results.extend(batch_results)
            
            # Clear GPU memory between batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
```

---

## 10. TESTING STRATEGY

### 10.1 Test Pyramid

```
         ┌─────────────────┐
         │   E2E Tests     │  10% - Full pipeline tests
         │  (5-10 tests)   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ Integration     │  30% - Component interaction
         │ (20-30 tests)   │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │   Unit Tests    │  60% - Individual functions/classes
         │ (60-80 tests)   │
         └─────────────────┘
```

### 10.2 Test Categories

| Category | Coverage Target | Tools |
|----------|-----------------|-------|
| Unit Tests | 80% | pytest, pytest-cov, unittest.mock |
| Integration Tests | 70% | pytest-asyncio, testcontainers |
| E2E Tests | 100% critical paths | pytest, requests |
| Performance Tests | N/A | Locust, pytest-benchmark |
| Adversarial Tests | 100% defense paths | Custom attack generation |

### 10.3 Sample Unit Test

```python
# tests/unit/test_schemas.py

import pytest
from src.api.schemas import TrustScoreSchema, Verdict, ConfidenceLevel

def test_trust_score_schema_validation():
    """Test TrustScoreSchema with valid data."""
    score = TrustScoreSchema(
        overall_score=87.5,
        confidence=0.89,
        confidence_level=ConfidenceLevel.HIGH,
        breakdown={"video": 0.40, "audio": 0.35, "text": 0.25},
        verdict=Verdict.MANIPULATED,
    )
    assert score.overall_score == 87.5
    assert score.confidence == 0.89
    assert score.verdict == Verdict.MANIPULATED

def test_trust_score_schema_invalid_score():
    """Test TrustScoreSchema rejects invalid scores."""
    with pytest.raises(ValueError):
        TrustScoreSchema(
            overall_score=150,  # > 100
            confidence=0.89,
            confidence_level=ConfidenceLevel.HIGH,
            breakdown={},
            verdict=Verdict.MANIPULATED,
        )

def test_verdict_enum_values():
    """Test Verdict enum has expected values."""
    assert Verdict.AUTHENTIC.value == "AUTHENTIC"
    assert Verdict.MANIPULATED.value == "MANIPULATED"
    assert Verdict.UNCERTAIN.value == "UNCERTAIN"
```

### 10.4 Sample Integration Test

```python
# tests/integration/test_analysis_pipeline.py

import pytest
from pathlib import Path
import asyncio
from src.analysis.video_detector import VideoDetector
from src.preprocessing.video_processor import VideoProcessor

@pytest.mark.asyncio
async def test_full_video_pipeline(test_video_path: Path):
    """Test end-to-end video analysis pipeline."""
    # Initialize components
    processor = VideoProcessor()
    detector = VideoDetector()
    
    # Process video
    frames = await processor.extract_frames(
        video_path=test_video_path,
        fps=5,
    )
    
    # Detect deepfakes
    result = detector.detect(frames)
    
    # Verify result structure
    assert result.overall_score >= 0
    assert result.overall_score <= 1
    assert result.confidence > 0
    assert "video" in result.metadata
```

---

## APPENDIX A: DEPENDENCY REFERENCE

### A.1 Core Dependencies

```
fastapi==0.115.0
uvicorn[standard]==0.30.0
pydantic==2.9.0
pydantic-settings==2.4.0
python-multipart==0.0.9
pyyaml==6.0.2
torch==2.5.0
torchvision==0.20.0
torchaudio==2.5.0
numpy==1.26.4
opencv-python==4.10.0
ffmpeg-python==0.2.0
librosa==0.10.2
scipy==1.14.0
transformers==4.45.0
scikit-learn==1.5.0
sqlalchemy[asyncio]==2.0.0
asyncpg==0.30.0
redis==5.0.0
aioredis==2.0.0
```

### A.2 Development Dependencies

```
pytest==8.3.0
pytest-asyncio==0.24.0
pytest-cov==6.0.0
black==24.8.0
ruff==0.6.0
mypy==1.12.0
pre-commit==4.0.0
httpx==0.27.0
testcontainers==3.0.0
```

---

## APPENDIX B: CONFIGURATION REFERENCE

### B.1 Default Configuration (configs/default.yaml)

```yaml
# Application Settings
app:
  host: "0.0.0.0"
  port: 8000
  debug: false
  workers: 4

# Database Configuration
database:
  url: "postgresql+asyncpg://argus:argus@localhost:5432/argus_prime"
  pool_size: 10
  max_overflow: 20

# Redis Configuration
redis:
  host: "localhost"
  port: 6379
  db: 0
  key_prefix: "argus:"

# Storage Configuration
storage:
  temp_dir: "/tmp/argus_uploads"
  max_file_size: 500_000_000  # 500MB
  ttl_seconds: 86400  # 24 hours

# Model Configuration
models:
  video:
    primary: "forensic_flow"
    fallback: ["distil_dire", "kan_detector"]
    batch_size: 8
  audio:
    primary: "rawnet2"
    fallback: ["transcrypt"]
    segment_duration: 2
  text:
    primary: "spec_detect"
    fallback: ["perplexity_detector"]

# Analysis Configuration
analysis:
  default_fps: 5
  min_face_size: 64
  confidence_threshold: 0.5
  fusion:
    video_weight: 0.40
    audio_weight: 0.35
    text_weight: 0.25
    cross_modal_adjustment: 0.20

# Defense Configuration
defense:
  enabled: true
  jpeg_quality: 85
  noise_std: 0.01
  trim_threshold: 0.1

# Logging Configuration
logging:
  level: "INFO"
  format: "json"
  include_timestamp: true

# Metrics Configuration
metrics:
  enabled: true
  port: 9090
  path: "/metrics"
```

---

## APPENDIX C: GLOSSARY

| Term | Definition |
|------|------------|
| **rPPG** | Remote Photoplethysmography - extracting heart rate from video |
| **C2PA** | Coalition for Content Provenance and Authenticity |
| **TRIM** | Training-free Robust detection via Information Maximization |
| **Grad-CAM** | Gradient-weighted Class Activation Mapping |
| **RawNet2** | Neural vocoder artifact detection model |
| **DIRE** | Diffusion Reconstruction Error |
| **VAD** | Voice Activity Detection |
| **AUC** | Area Under the Curve (ROC analysis) |
| **EER** | Equal Error Rate (security metric) |

---

## DOCUMENT VERSION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2026-01-21 | Lead AI Architect | Initial implementation blueprint |

---

**END OF DOCUMENT**
