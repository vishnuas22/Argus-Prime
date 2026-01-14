# 🛡️ ARGUS CORE
## Multi-Modal Deepfake Detection & Forensic Analysis Platform
### Implementation Blueprint v1.0

**Project:** Argus Core  
**Priority:** CRITICAL (Production-Grade Architecture)  
**Date:** August 2025  
**Compliance:** AGENTS.md Clean Architecture Principles  

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [SOTA Research Verification](#2-sota-research-verification)
3. [Section 1: Life of a Request Flow](#3-section-1-life-of-a-request-flow)
4. [Section 2: Architecture & File Manifesto](#4-section-2-architecture--file-manifesto)
5. [Section 3: Development Strategy](#5-section-3-development-strategy)
6. [API Contract Specifications](#6-api-contract-specifications)
7. [Configuration & Environment](#7-configuration--environment)
8. [Appendix: Model Registry](#8-appendix-model-registry)

---

# 1. Executive Summary

## Mission Statement

Argus Core is a production-grade, multi-modal deepfake detection backend that processes video, audio, image, and text inputs to generate forensic trust scores with explainable AI artifacts. The system adheres to AGENTS.md clean architecture principles with Interface-Driven Development (IDD) methodology.

## Core Metrics

| Metric | Target | SOTA Baseline |
|--------|--------|---------------|
| Image Detection Accuracy | >99% | 99.89% (EfficientNetV2-B2) |
| Audio Detection EER | <1% | 0.22% (Wav2Vec 2.0 + AASIST) |
| Video Detection AUC | >0.95 | 0.98 (FakeVLM) |
| Inference Latency (Image) | <300ms | ~200ms |
| Inference Latency (30s Video) | <20s | ~15s |
| Adversarial Robustness | >80% | Baseline: 50% (pre-defense) |

## Architecture Philosophy

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTERFACE-DRIVEN DEVELOPMENT                  │
│                                                                  │
│   1. Define Schemas (Pydantic) → Data Contracts                 │
│   2. Define Interfaces (ABC) → Behavior Contracts               │
│   3. Implement Router Skeleton → API Surface                    │
│   4. Implement Core Logic → Business Rules                      │
│   5. Integration & Testing → Validation                         │
└─────────────────────────────────────────────────────────────────┘
```

---

# 2. SOTA Research Verification

## 2.1 Video Detection: Spatial-Temporal Analysis

### Recommended Architecture: EfficientNetV2 + Temporal Pooling

| Model | Architecture | Performance | Trade-off |
|-------|--------------|-------------|-----------|
| **EfficientNetV2-B2** | CNN (Spatial) | 99.89% accuracy | Best accuracy/efficiency ratio |
| **Xception** | CNN (Spatial) | ~95% on FF++ | Legacy, lower performance |
| **ViT Adapter** | Transformer | Good generalization | Higher compute, slower |
| **Hybrid (Ours)** | EfficientNetV2 + GRU | Spatial + Temporal | Recommended |

**Decision:** Use **EfficientNetV2-B2** for frame-level spatial features with **Bidirectional GRU** for temporal consistency. This outperforms pure ViT on efficiency while maintaining SOTA accuracy.

### Biological Signal (rPPG) Status: ⚠️ SUPPLEMENTARY ONLY

Research confirms advanced deepfakes replicate heart rate signals within 1.80-3.24 bpm of ground truth. **rPPG is NOT reliable as primary detection** but useful as tie-breaker signal (5% weight in Trust Score).

## 2.2 Audio Detection: Wav2Vec 2.0 + AASIST

### Recommended Architecture: SSL Frontend + Lightweight Backend

| Model | Frontend | Backend | EER | Notes |
|-------|----------|---------|-----|-------|
| **Wav2Vec 2.0 + AASIST** | SSL (12 layers) | Graph Attention | 0.22% | **SOTA** |
| **HuBERT + SpecRNet** | SSL | Spectral Network | 0.25% | Alternative |
| **MFCC + CNN** | Handcrafted | CNN | ~5% | Legacy |
| **YAMNet** | Pre-trained | ANN | 3% | Fast, lower accuracy |

**Decision:** Use **Wav2Vec 2.0 (XLS-R variant)** with fine-tuned layers + **AASIST backend** for production. Fallback to YAMNet for CPU-only environments.

## 2.3 Adversarial Defense Strategy

### Critical Finding: 45-50% Detection Drop in Real-World

Current detectors show **45-50% effectiveness reduction** from lab to real-world. Defense requires multi-layer approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ADVERSARIAL DEFENSE LAYERS                    │
├─────────────────────────────────────────────────────────────────┤
│ Layer 1: INPUT SANITIZATION                                      │
│   • JPEG Quality Randomization (70-95%)                         │
│   • Random Crop & Resize (±5%)                                  │
│   • Gaussian Noise Injection (σ=0.01)                           │
│   • Frequency Domain Filtering                                  │
├─────────────────────────────────────────────────────────────────┤
│ Layer 2: MODEL ENSEMBLE                                          │
│   • Multi-model voting (3+ detectors)                           │
│   • Confidence threshold gating                                 │
│   • Disagreement flagging → human review                        │
├─────────────────────────────────────────────────────────────────┤
│ Layer 3: PROCEDURAL RESILIENCE                                   │
│   • Chain-of-custody logging                                    │
│   • C2PA credential verification                                │
│   • Hash-based integrity checks                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

# 3. Section 1: Life of a Request Flow

## 3.1 Complete Request Lifecycle

```
USER REQUEST: "Analyze this MP4 file"
│
│ [T+0ms] ──────────────────────────────────────────────────────────
▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: INGESTION (router.py)                                  │
│ Files: router.py → validate.py → storage.py                     │
│ Duration: ~50-200ms                                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Receive multipart/form-data upload                           │
│ 2. Validate file signature (magic bytes)                        │
│ 3. Check file size limits (config-driven)                       │
│ 4. Generate analysis_id (UUID)                                  │
│ 5. Store raw file to object storage (MinIO)                     │
│ 6. Create analysis record in MongoDB                            │
│ 7. Dispatch to task queue (Celery/Redis)                        │
│ 8. Return immediate response with analysis_id                   │
└─────────────────────────────────────────────────────────────────┘
│
│ [T+200ms] ─────────────────────────────────────────────────────────
▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: PREPROCESSING (preprocess.py)                          │
│ Files: preprocess.py → extract.py → sanitize.py                 │
│ Duration: ~2-5s (for 30s video)                                 │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load file from object storage                                │
│ 2. Detect media type (video/audio/image)                        │
│ 3. ADVERSARIAL SANITIZATION:                                    │
│    - Apply random JPEG compression                              │
│    - Add controlled Gaussian noise                              │
│    - Normalize color/brightness                                 │
│ 4. EXTRACTION:                                                  │
│    Video: Extract frames (1 FPS) + audio track                  │
│    Audio: Resample to 16kHz mono                                │
│    Image: Resize to model input size                            │
│ 5. FACE DETECTION (for video/image):                            │
│    - MTCNN face detection                                       │
│    - Face alignment & cropping                                  │
│    - Store face coordinates                                     │
│ 6. Generate preprocessed artifacts                              │
│ 7. Update analysis status: "preprocessing_complete"             │
└─────────────────────────────────────────────────────────────────┘
│
│ [T+5s] ────────────────────────────────────────────────────────────
▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: MULTI-MODEL INFERENCE (orchestrate.py)                 │
│ Files: orchestrate.py → video.py → audio.py → image.py          │
│ Duration: ~10-15s (parallel execution)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ VIDEO BRANCH │  │ AUDIO BRANCH │  │ METADATA     │          │
│  │ (video.py)   │  │ (audio.py)   │  │ (forensic.py)│          │
│  │              │  │              │  │              │          │
│  │ EfficientNet │  │ Wav2Vec 2.0  │  │ C2PA Check   │          │
│  │ + GRU        │  │ + AASIST     │  │ EXIF Parse   │          │
│  │ + Grad-CAM   │  │ + Spectro    │  │ Hash Verify  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              PARALLEL EXECUTION (asyncio.gather)          │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│ Output per branch:                                              │
│ - confidence: float (0.0-1.0)                                   │
│ - artifacts: List[ArtifactDetail]                               │
│ - heatmaps: Optional[bytes] (base64)                            │
│ - explanations: List[str]                                       │
└─────────────────────────────────────────────────────────────────┘
│
│ [T+18s] ───────────────────────────────────────────────────────────
▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: AGGREGATION (aggregate.py)                             │
│ Files: aggregate.py → score.py → explain.py                     │
│ Duration: ~100-500ms                                            │
├─────────────────────────────────────────────────────────────────┤
│ 1. Collect results from all inference branches                  │
│ 2. TRUST SCORE CALCULATION:                                     │
│    ┌─────────────────────────────────────────────────────────┐ │
│    │ trust_score = Σ(weight_i × (1 - fake_confidence_i))    │ │
│    │                                                         │ │
│    │ Weights (configurable):                                 │ │
│    │   - visual:   0.30 (EfficientNetV2)                    │ │
│    │   - audio:    0.30 (Wav2Vec 2.0)                       │ │
│    │   - temporal: 0.15 (GRU consistency)                   │ │
│    │   - metadata: 0.15 (C2PA/EXIF)                         │ │
│    │   - rppg:     0.05 (biological - supplementary)        │ │
│    │   - ensemble: 0.05 (model agreement bonus)             │ │
│    └─────────────────────────────────────────────────────────┘ │
│ 3. ENSEMBLE VOTING:                                             │
│    - If >2 models agree: high confidence                        │
│    - If disagreement: flag for human review                     │
│ 4. EXPLANATION GENERATION:                                      │
│    - Natural language artifact descriptions                     │
│    - Heatmap overlay composition                                │
│    - Frame-by-frame anomaly timeline                            │
│ 5. Update analysis status: "complete"                           │
└─────────────────────────────────────────────────────────────────┘
│
│ [T+19s] ───────────────────────────────────────────────────────────
▼
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: RESPONSE (report.py)                                   │
│ Files: report.py → serialize.py                                 │
│ Duration: ~50-100ms                                             │
├─────────────────────────────────────────────────────────────────┤
│ 1. Format final JSON response                                   │
│ 2. Store forensic report in MongoDB                             │
│ 3. Cache result in Redis (TTL: 24h)                             │
│ 4. Notify client via WebSocket (if subscribed)                  │
│ 5. Return AnalysisResult schema                                 │
└─────────────────────────────────────────────────────────────────┘
│
▼
┌─────────────────────────────────────────────────────────────────┐
│ FINAL JSON RESPONSE                                              │
├─────────────────────────────────────────────────────────────────┤
│ {                                                                │
│   "analysis_id": "550e8400-e29b-41d4-a716-446655440000",        │
│   "status": "complete",                                          │
│   "trust_score": 23.5,                                          │
│   "verdict": "HIGH_MANIPULATION_PROBABILITY",                   │
│   "modality_scores": {                                          │
│     "visual": {"score": 0.94, "confidence": "high"},            │
│     "audio": {"score": 0.87, "confidence": "high"},             │
│     "metadata": {"score": 0.15, "confidence": "medium"}         │
│   },                                                             │
│   "artifacts": [...],                                           │
│   "explanations": [...],                                        │
│   "heatmap_url": "/api/v1/heatmaps/550e8400...",               │
│   "processing_time_ms": 19234                                   │
│ }                                                                │
└─────────────────────────────────────────────────────────────────┘
```

## 3.2 File Sequence Diagram

```
Timestamp  │ File              │ Function                │ Output
───────────┼───────────────────┼─────────────────────────┼──────────────────────
T+0ms      │ router.py         │ analyze_media()         │ AnalysisRequest
T+5ms      │ validate.py       │ validate_upload()       │ ValidatedFile
T+20ms     │ storage.py        │ store_raw()             │ StorageRef
T+50ms     │ repository.py     │ create_analysis()       │ AnalysisRecord
T+100ms    │ tasks.py          │ dispatch_analysis()     │ TaskID
T+200ms    │ preprocess.py     │ preprocess_media()      │ PreprocessedData
T+500ms    │ extract.py        │ extract_frames()        │ List[Frame]
T+1000ms   │ sanitize.py       │ apply_defenses()        │ SanitizedData
T+2000ms   │ detect.py         │ detect_faces()          │ List[FaceRegion]
T+5000ms   │ orchestrate.py    │ run_inference()         │ InferenceJob
T+5100ms   │ video.py          │ analyze_video()         │ VideoResult
T+5100ms   │ audio.py          │ analyze_audio()         │ AudioResult (parallel)
T+5100ms   │ forensic.py       │ analyze_metadata()      │ MetadataResult (parallel)
T+15000ms  │ aggregate.py      │ aggregate_results()     │ AggregatedResult
T+15500ms  │ score.py          │ calculate_trust()       │ TrustScore
T+16000ms  │ explain.py        │ generate_explanation()  │ Explanation
T+18000ms  │ report.py         │ build_report()          │ ForensicReport
T+19000ms  │ serialize.py      │ to_response()           │ AnalysisResponse
```

---

# 4. Section 2: Architecture & File Manifesto

## 4.1 Directory Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration loader
│   │
│   ├── api/                    # API Layer
│   │   ├── __init__.py
│   │   ├── router.py           # Main API router
│   │   ├── deps.py             # Dependency injection
│   │   └── middleware.py       # Request/response middleware
│   │
│   ├── schemas/                # Pydantic Data Models
│   │   ├── __init__.py
│   │   ├── base.py             # Base schemas
│   │   ├── request.py          # Request schemas
│   │   ├── response.py         # Response schemas
│   │   ├── analysis.py         # Analysis domain schemas
│   │   └── artifacts.py        # Artifact schemas
│   │
│   ├── interfaces/             # Abstract Base Classes
│   │   ├── __init__.py
│   │   ├── detector.py         # IDetector interface
│   │   ├── preprocessor.py     # IPreprocessor interface
│   │   ├── storage.py          # IStorage interface
│   │   └── repository.py       # IRepository interface
│   │
│   ├── core/                   # Business Logic
│   │   ├── __init__.py
│   │   ├── orchestrate.py      # Analysis orchestrator
│   │   ├── aggregate.py        # Result aggregation
│   │   ├── score.py            # Trust score calculation
│   │   └── explain.py          # XAI explanation generator
│   │
│   ├── detectors/              # Detection Implementations
│   │   ├── __init__.py
│   │   ├── video.py            # Video deepfake detector
│   │   ├── audio.py            # Audio deepfake detector
│   │   ├── image.py            # Image deepfake detector
│   │   ├── text.py             # AI text detector
│   │   └── forensic.py         # Metadata/C2PA analyzer
│   │
│   ├── preprocessing/          # Preprocessing Pipeline
│   │   ├── __init__.py
│   │   ├── preprocess.py       # Main preprocessor
│   │   ├── extract.py          # Media extraction
│   │   ├── sanitize.py         # Adversarial defense
│   │   ├── detect.py           # Face detection
│   │   └── validate.py         # Input validation
│   │
│   ├── models/                 # ML Model Wrappers
│   │   ├── __init__.py
│   │   ├── efficient.py        # EfficientNetV2 wrapper
│   │   ├── wav2vec.py          # Wav2Vec 2.0 wrapper
│   │   ├── temporal.py         # Temporal GRU wrapper
│   │   └── registry.py         # Model registry
│   │
│   ├── infrastructure/         # External Services
│   │   ├── __init__.py
│   │   ├── storage.py          # MinIO/S3 client
│   │   ├── cache.py            # Redis client
│   │   ├── database.py         # MongoDB client
│   │   └── queue.py            # Celery task queue
│   │
│   ├── services/               # Business Services
│   │   ├── __init__.py
│   │   ├── analysis.py         # Analysis service
│   │   ├── report.py           # Report generator
│   │   └── export.py           # Export service (PDF/JSON)
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── logging.py          # Structured logging
│       ├── metrics.py          # Prometheus metrics
│       └── errors.py           # Custom exceptions
│
├── tests/                      # Test Suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## 4.2 File Manifesto

---

### 📁 `app/main.py`

| Attribute | Value |
|-----------|-------|
| **File** | `main.py` |
| **Role** | Application entry point. Initializes FastAPI app, mounts routers, configures middleware, and manages lifecycle events. |
| **SOTA Algorithm** | N/A (Configuration only) |
| **Integration** | Imports: `router.py`, `config.py`, `middleware.py`, `database.py` |
| **Inputs** | Environment variables via `config.py` |
| **Outputs** | Configured FastAPI application instance |
| **Why this approach?** | Single entry point follows 12-factor app principles. Lifecycle hooks ensure graceful startup/shutdown of DB connections and model loading. |

```python
# Signature
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle - load models, connect DBs."""
    ...
```

---

### 📁 `app/config.py`

| Attribute | Value |
|-----------|-------|
| **File** | `config.py` |
| **Role** | Centralized configuration management. Loads all settings from environment variables with validation. Zero hardcoding. |
| **SOTA Algorithm** | Pydantic Settings with validation |
| **Integration** | Imported by: ALL files |
| **Inputs** | `.env` file, environment variables |
| **Outputs** | `Settings` singleton instance |
| **Why this approach?** | Type-safe configuration prevents runtime errors. Environment-based config enables easy deployment across dev/staging/prod. |

```python
# Schema
class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    
    # Detection Thresholds (NO HARDCODING)
    trust_score_weights: dict[str, float]  # Loaded from config
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    
    # Model Paths
    efficientnet_path: str
    wav2vec_path: str
    
    # Timeouts (per AGENTS.md)
    db_timeout_ms: int = 100
    ai_timeout_s: int = 60
    cache_timeout_ms: int = 50
```

---

### 📁 `app/api/router.py`

| Attribute | Value |
|-----------|-------|
| **File** | `router.py` |
| **Role** | Main API router. Defines all REST endpoints for analysis submission, status polling, and result retrieval. |
| **SOTA Algorithm** | FastAPI async routing with dependency injection |
| **Integration** | Imports: `schemas/*`, `services/analysis.py`, `deps.py` |
| **Inputs** | `AnalysisRequest` (file upload + options) |
| **Outputs** | `AnalysisResponse`, `AnalysisStatus`, `ForensicReport` |
| **Why this approach?** | Async handlers maximize throughput for I/O-bound upload operations. Dependency injection enables testability. |

```python
# Endpoints
@router.post("/api/v1/analyze", response_model=AnalysisSubmitResponse)
async def submit_analysis(
    file: UploadFile,
    options: AnalysisOptions = Depends(),
    service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisSubmitResponse:
    """Submit media for deepfake analysis."""
    ...

@router.get("/api/v1/analysis/{analysis_id}", response_model=AnalysisResult)
async def get_analysis(
    analysis_id: UUID,
    service: AnalysisService = Depends(get_analysis_service)
) -> AnalysisResult:
    """Retrieve analysis results."""
    ...

@router.get("/api/v1/analysis/{analysis_id}/heatmap")
async def get_heatmap(analysis_id: UUID) -> StreamingResponse:
    """Stream heatmap visualization."""
    ...
```

---

### 📁 `app/schemas/base.py`

| Attribute | Value |
|-----------|-------|
| **File** | `base.py` |
| **Role** | Base Pydantic models with common configurations. Defines shared fields and serialization settings. |
| **SOTA Algorithm** | Pydantic v2 with model_config |
| **Integration** | Inherited by: ALL schema files |
| **Inputs** | N/A (Base class) |
| **Outputs** | `BaseSchema`, `TimestampMixin` |
| **Why this approach?** | DRY principle - common config in one place. Ensures consistent JSON serialization across all schemas. |

```python
# Schema
class BaseSchema(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_schema_extra={"example": {}}
    )

class TimestampMixin(BaseModel):
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
```

---

### 📁 `app/schemas/analysis.py`

| Attribute | Value |
|-----------|-------|
| **File** | `analysis.py` |
| **Role** | Core analysis domain schemas. Defines data structures for detection results, trust scores, and artifacts. |
| **SOTA Algorithm** | Discriminated unions for modality-specific results |
| **Integration** | Used by: `orchestrate.py`, `aggregate.py`, `report.py` |
| **Inputs** | Raw detection outputs |
| **Outputs** | `ModalityResult`, `TrustScore`, `ArtifactDetail`, `Explanation` |
| **Why this approach?** | Strict typing enables compile-time validation. Discriminated unions handle polymorphic modality results elegantly. |

```python
# Schema
class ModalityType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    METADATA = "metadata"

class DetectionResult(BaseSchema):
    modality: ModalityType
    fake_confidence: float = Field(ge=0.0, le=1.0)
    model_name: str
    model_version: str
    inference_time_ms: int
    artifacts: list[ArtifactDetail]
    explanations: list[str]
    heatmap_b64: Optional[str] = None

class TrustScore(BaseSchema):
    score: float = Field(ge=0.0, le=100.0)
    verdict: Literal["AUTHENTIC", "LIKELY_AUTHENTIC", "UNCERTAIN", "LIKELY_MANIPULATED", "MANIPULATED"]
    confidence: Literal["low", "medium", "high"]
    component_scores: dict[ModalityType, float]
    weights_used: dict[ModalityType, float]

class ArtifactDetail(BaseSchema):
    artifact_type: str  # e.g., "lip_sync_mismatch", "vocoder_signature"
    location: Optional[dict]  # e.g., {"frame": 120, "bbox": [x,y,w,h]}
    severity: Literal["low", "medium", "high", "critical"]
    description: str
```

---

### 📁 `app/interfaces/detector.py`

| Attribute | Value |
|-----------|-------|
| **File** | `detector.py` |
| **Role** | Abstract Base Class defining the contract for all detector implementations. Ensures consistent interface across modalities. |
| **SOTA Algorithm** | ABC + Protocol pattern |
| **Integration** | Implemented by: `video.py`, `audio.py`, `image.py`, `text.py`, `forensic.py` |
| **Inputs** | `PreprocessedData` |
| **Outputs** | `DetectionResult` |
| **Why this approach?** | Interface segregation enables swapping detector implementations without changing orchestration logic. Critical for A/B testing models. |

```python
# Interface
class IDetector(ABC):
    """Abstract interface for all deepfake detectors."""
    
    @property
    @abstractmethod
    def modality(self) -> ModalityType:
        """Return the modality this detector handles."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
    
    @abstractmethod
    async def detect(
        self, 
        data: PreprocessedData,
        options: DetectionOptions
    ) -> DetectionResult:
        """
        Run detection on preprocessed data.
        
        Args:
            data: Preprocessed media data
            options: Detection configuration
            
        Returns:
            DetectionResult with confidence and artifacts
            
        Raises:
            DetectionError: If inference fails
            TimeoutError: If inference exceeds timeout
        """
        ...
    
    @abstractmethod
    async def generate_heatmap(
        self,
        data: PreprocessedData,
        result: DetectionResult
    ) -> Optional[bytes]:
        """Generate visual explanation heatmap."""
        ...
```

---

### 📁 `app/detectors/video.py`

| Attribute | Value |
|-----------|-------|
| **File** | `video.py` |
| **Role** | Video deepfake detection using EfficientNetV2 for spatial features and Bidirectional GRU for temporal consistency. Includes Grad-CAM heatmap generation. |
| **SOTA Algorithm** | **EfficientNetV2-B2** (99.89% accuracy) + **Bidirectional GRU** (temporal) + **Grad-CAM** (XAI) |
| **Integration** | Implements: `IDetector`. Imports: `efficient.py`, `temporal.py`. Called by: `orchestrate.py` |
| **Inputs** | `PreprocessedData` containing extracted frames (List[np.ndarray]) and face regions |
| **Outputs** | `DetectionResult` with frame-level confidences, artifacts, and heatmap |
| **Why this approach?** | EfficientNetV2 provides SOTA accuracy (99.89%) with efficient compute. GRU captures temporal inconsistencies (blink patterns, expression transitions) that frame-level CNNs miss. Grad-CAM provides legally-required explainability. |

```python
# Implementation signature
class VideoDetector(IDetector):
    """Video deepfake detector using EfficientNetV2 + GRU."""
    
    def __init__(
        self,
        spatial_model: EfficientNetWrapper,
        temporal_model: TemporalGRU,
        device: torch.device,
        config: VideoDetectorConfig
    ):
        self._spatial = spatial_model
        self._temporal = temporal_model
        self._device = device
        self._config = config
    
    @property
    def modality(self) -> ModalityType:
        return ModalityType.VIDEO
    
    async def detect(
        self, 
        data: PreprocessedData,
        options: DetectionOptions
    ) -> DetectionResult:
        """
        Process:
        1. Extract face crops from frames
        2. Run EfficientNetV2 on each frame → spatial features
        3. Sequence features through BiGRU → temporal score
        4. Aggregate frame scores with attention weights
        5. Generate artifact list from anomaly frames
        """
        ...
```

---

### 📁 `app/detectors/audio.py`

| Attribute | Value |
|-----------|-------|
| **File** | `audio.py` |
| **Role** | Audio deepfake detection using Wav2Vec 2.0 self-supervised features with AASIST graph attention backend. Detects vocoder artifacts and synthetic speech patterns. |
| **SOTA Algorithm** | **Wav2Vec 2.0 XLS-R** (SSL frontend, 0.22% EER) + **AASIST** (Graph Attention Network backend) |
| **Integration** | Implements: `IDetector`. Imports: `wav2vec.py`. Called by: `orchestrate.py` |
| **Inputs** | `PreprocessedData` containing audio waveform (np.ndarray, 16kHz mono) |
| **Outputs** | `DetectionResult` with confidence, spectrogram artifacts, and frequency analysis |
| **Why this approach?** | Wav2Vec 2.0 captures rich contextual audio representations without handcrafted features (outperforms MFCC by 20x). AASIST's graph attention models spectro-temporal dependencies crucial for detecting TTS/VC artifacts. |

```python
# Implementation signature
class AudioDetector(IDetector):
    """Audio deepfake detector using Wav2Vec 2.0 + AASIST."""
    
    def __init__(
        self,
        wav2vec: Wav2VecWrapper,
        aasist_backend: nn.Module,
        device: torch.device,
        config: AudioDetectorConfig
    ):
        self._wav2vec = wav2vec
        self._backend = aasist_backend
        self._device = device
        self._config = config
    
    async def detect(
        self, 
        data: PreprocessedData,
        options: DetectionOptions
    ) -> DetectionResult:
        """
        Process:
        1. Normalize audio to [-1, 1] range
        2. Extract Wav2Vec 2.0 features (fine-tuned layers 4-12)
        3. Pass through AASIST graph attention
        4. Analyze frequency bands for vocoder signatures
        5. Generate spectrogram visualization
        """
        ...
```

---

### 📁 `app/detectors/forensic.py`

| Attribute | Value |
|-----------|-------|
| **File** | `forensic.py` |
| **Role** | Metadata forensic analysis. Validates C2PA content credentials, extracts/verifies EXIF data, and checks file integrity via hash chains. |
| **SOTA Algorithm** | **C2PA v2.3** specification + **EXIF analysis** + **Cryptographic hash verification** |
| **Integration** | Implements: `IDetector`. Uses: `c2pa-python`. Called by: `orchestrate.py` |
| **Inputs** | `PreprocessedData` containing raw file bytes and extracted metadata |
| **Outputs** | `DetectionResult` with provenance chain, credential status, and integrity flags |
| **Why this approach?** | C2PA is the industry standard for content authenticity (Adobe, Microsoft, Google adoption). Metadata analysis catches amateur manipulations that bypass pixel-level detectors. Provides legal-admissible chain of custody. |

```python
# Implementation signature
class ForensicAnalyzer(IDetector):
    """Metadata and provenance forensic analyzer."""
    
    async def detect(
        self, 
        data: PreprocessedData,
        options: DetectionOptions
    ) -> DetectionResult:
        """
        Process:
        1. Attempt C2PA manifest extraction
        2. Validate cryptographic signatures
        3. Parse EXIF metadata
        4. Check for timestamp inconsistencies
        5. Verify file hash integrity
        6. Analyze compression artifacts
        """
        ...
    
    async def _verify_c2pa(self, file_bytes: bytes) -> C2PAResult:
        """Verify C2PA content credentials."""
        ...
    
    async def _analyze_exif(self, file_bytes: bytes) -> EXIFResult:
        """Extract and analyze EXIF metadata."""
        ...
```

---

### 📁 `app/preprocessing/sanitize.py`

| Attribute | Value |
|-----------|-------|
| **File** | `sanitize.py` |
| **Role** | Adversarial defense preprocessing. Applies input sanitization to neutralize adversarial perturbations before detection. |
| **SOTA Algorithm** | **JPEG Compression Defense** + **Gaussian Noise Injection** + **Frequency Domain Filtering** (per AADD-2025 challenge findings) |
| **Integration** | Called by: `preprocess.py`. Outputs to: `detectors/*` |
| **Inputs** | Raw media bytes/arrays |
| **Outputs** | Sanitized media ready for detection |
| **Why this approach?** | Research shows 45-50% detection drop due to adversarial attacks. Multi-layer sanitization (compression + noise + filtering) provides defense-in-depth without significant accuracy loss on clean inputs. |

```python
# Implementation signature
class AdversarialSanitizer:
    """Defense against adversarial attacks on deepfake detectors."""
    
    def __init__(self, config: SanitizerConfig):
        self._jpeg_quality_range = config.jpeg_quality_range  # (70, 95)
        self._noise_sigma = config.noise_sigma  # 0.01
        self._filter_cutoff = config.filter_cutoff  # High-freq cutoff
    
    async def sanitize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sanitization pipeline:
        1. Random JPEG recompression (quality 70-95%)
        2. Gaussian noise injection (σ=0.01)
        3. High-frequency filtering (removes perturbations)
        4. Color normalization
        """
        ...
    
    async def sanitize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio sanitization:
        1. Resampling to canonical rate
        2. Low-pass filtering (removes ultrasonic artifacts)
        3. Dynamic range compression
        """
        ...
```

---

### 📁 `app/core/orchestrate.py`

| Attribute | Value |
|-----------|-------|
| **File** | `orchestrate.py` |
| **Role** | Analysis orchestrator. Coordinates parallel execution of all detector branches and manages the inference pipeline. |
| **SOTA Algorithm** | **AsyncIO parallel execution** with timeout handling and circuit breaker pattern |
| **Integration** | Imports: ALL detectors, `aggregate.py`. Called by: `tasks.py` (Celery) |
| **Inputs** | `PreprocessedData`, `AnalysisOptions` |
| **Outputs** | `List[DetectionResult]` from all modalities |
| **Why this approach?** | Parallel execution reduces total latency from ~45s (sequential) to ~15s. Circuit breaker prevents cascade failures if one detector hangs. Timeout handling ensures SLA compliance. |

```python
# Implementation signature
class AnalysisOrchestrator:
    """Orchestrates multi-modal deepfake detection."""
    
    def __init__(
        self,
        detectors: dict[ModalityType, IDetector],
        config: OrchestratorConfig
    ):
        self._detectors = detectors
        self._config = config
        self._circuit_breaker = CircuitBreaker(config.circuit_breaker)
    
    async def run_analysis(
        self,
        data: PreprocessedData,
        options: AnalysisOptions
    ) -> list[DetectionResult]:
        """
        Execute all relevant detectors in parallel.
        
        Flow:
        1. Determine applicable modalities from input type
        2. Create detection tasks with timeouts
        3. Execute via asyncio.gather (return_exceptions=True)
        4. Handle partial failures gracefully
        5. Return successful results + error markers
        """
        tasks = []
        for modality in self._get_applicable_modalities(data):
            detector = self._detectors[modality]
            task = self._run_with_timeout(
                detector.detect(data, options),
                timeout=self._config.ai_timeout_s
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self._process_results(results)
```

---

### 📁 `app/core/aggregate.py`

| Attribute | Value |
|-----------|-------|
| **File** | `aggregate.py` |
| **Role** | Result aggregation and ensemble voting. Combines multi-modal detection results into unified trust score with confidence estimation. |
| **SOTA Algorithm** | **Weighted ensemble** with **Bayesian confidence estimation** and **disagreement detection** |
| **Integration** | Imports: `score.py`, `explain.py`. Called by: `orchestrate.py` |
| **Inputs** | `List[DetectionResult]` from all detectors |
| **Outputs** | `AggregatedResult` with trust score and combined artifacts |
| **Why this approach?** | Ensemble voting provides robustness - no single model failure compromises results. Bayesian confidence accounts for model uncertainty. Disagreement flagging triggers human review for edge cases. |

```python
# Implementation signature
class ResultAggregator:
    """Aggregates multi-modal detection results."""
    
    def __init__(self, config: AggregatorConfig):
        self._weights = config.modality_weights  # From config, not hardcoded
        self._score_calculator = TrustScoreCalculator(config)
        self._explainer = ExplanationGenerator(config)
    
    async def aggregate(
        self,
        results: list[DetectionResult]
    ) -> AggregatedResult:
        """
        Aggregate detection results:
        1. Filter successful results
        2. Calculate weighted trust score
        3. Perform ensemble voting
        4. Detect model disagreements
        5. Generate unified explanation
        6. Compose final report
        """
        ...
    
    def _detect_disagreement(
        self, 
        results: list[DetectionResult]
    ) -> bool:
        """Flag if models significantly disagree."""
        confidences = [r.fake_confidence for r in results]
        # Flag if variance exceeds threshold
        return np.var(confidences) > self._config.disagreement_threshold
```

---

### 📁 `app/core/score.py`

| Attribute | Value |
|-----------|-------|
| **File** | `score.py` |
| **Role** | Trust score calculation engine. Computes weighted authenticity score from detection results using configurable weights. |
| **SOTA Algorithm** | **Weighted linear combination** with **dynamic weight adjustment** based on input modality availability |
| **Integration** | Called by: `aggregate.py`. Uses: config weights |
| **Inputs** | `List[DetectionResult]`, weight configuration |
| **Outputs** | `TrustScore` object (0-100 scale) |
| **Why this approach?** | Linear combination is interpretable and legally defensible. Dynamic weights handle missing modalities gracefully. All weights loaded from config - zero hardcoding enables easy tuning. |

```python
# Implementation signature
class TrustScoreCalculator:
    """Calculates forensic trust score."""
    
    def __init__(self, config: ScoreConfig):
        # ALL weights from config - NO HARDCODING
        self._base_weights = config.base_weights
        self._verdict_thresholds = config.verdict_thresholds
    
    def calculate(
        self,
        results: list[DetectionResult]
    ) -> TrustScore:
        """
        Calculate trust score:
        
        Formula:
        trust_score = Σ(weight_i × (1 - fake_confidence_i)) × 100
        
        Where weights are normalized to sum to 1.0 based on
        available modalities.
        """
        available_modalities = {r.modality for r in results}
        weights = self._normalize_weights(available_modalities)
        
        score = sum(
            weights[r.modality] * (1 - r.fake_confidence)
            for r in results
        ) * 100
        
        return TrustScore(
            score=round(score, 2),
            verdict=self._get_verdict(score),
            confidence=self._estimate_confidence(results),
            component_scores={r.modality: r.fake_confidence for r in results},
            weights_used=weights
        )
```

---

### 📁 `app/core/explain.py`

| Attribute | Value |
|-----------|-------|
| **File** | `explain.py` |
| **Role** | Explainable AI report generator. Produces human-readable explanations of detection results with visual artifacts. |
| **SOTA Algorithm** | **Template-based NLG** + **Grad-CAM overlay composition** + **Timeline visualization** |
| **Integration** | Called by: `aggregate.py`. Outputs to: `report.py` |
| **Inputs** | `List[DetectionResult]` with heatmaps and artifacts |
| **Outputs** | `Explanation` with text descriptions and visualization URLs |
| **Why this approach?** | Explainability is critical for legal admissibility and user trust. Template-based NLG ensures consistent, professional language. Grad-CAM provides visual evidence of detection rationale. |

```python
# Implementation signature
class ExplanationGenerator:
    """Generates human-readable XAI explanations."""
    
    def __init__(self, config: ExplainConfig):
        self._templates = config.explanation_templates
        self._artifact_descriptions = config.artifact_descriptions
    
    async def generate(
        self,
        results: list[DetectionResult],
        trust_score: TrustScore
    ) -> Explanation:
        """
        Generate explanation:
        1. Select template based on verdict
        2. Populate with specific artifacts
        3. Compose heatmap overlays
        4. Generate timeline of anomalies
        5. Format for frontend rendering
        """
        ...
    
    def _describe_artifact(self, artifact: ArtifactDetail) -> str:
        """Generate natural language description of artifact."""
        template = self._artifact_descriptions[artifact.artifact_type]
        return template.format(**artifact.model_dump())
```

---

### 📁 `app/models/efficient.py`

| Attribute | Value |
|-----------|-------|
| **File** | `efficient.py` |
| **Role** | EfficientNetV2-B2 model wrapper. Handles model loading, inference, and Grad-CAM extraction for spatial deepfake detection. |
| **SOTA Algorithm** | **EfficientNetV2-B2** (timm) + **Grad-CAM** (pytorch-grad-cam) |
| **Integration** | Used by: `video.py`, `image.py`. Loaded via: `registry.py` |
| **Inputs** | Batch of face crops (torch.Tensor, shape [B, 3, 224, 224]) |
| **Outputs** | Fake confidence (float), feature embeddings, Grad-CAM heatmap |
| **Why this approach?** | EfficientNetV2-B2 achieves 99.89% accuracy with reasonable compute. Pre-trained weights enable transfer learning. Grad-CAM provides legally-required visual explanations. |

```python
# Implementation signature
class EfficientNetWrapper:
    """EfficientNetV2-B2 wrapper for deepfake detection."""
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        config: ModelConfig
    ):
        self._model = self._load_model(model_path)
        self._model.to(device)
        self._model.eval()
        self._grad_cam = GradCAM(
            model=self._model,
            target_layers=[self._model.conv_head]
        )
    
    @torch.inference_mode()
    async def predict(
        self,
        images: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run inference:
        Returns: (confidences, embeddings)
        """
        ...
    
    def generate_gradcam(
        self,
        image: torch.Tensor
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap for single image."""
        ...
```

---

### 📁 `app/models/wav2vec.py`

| Attribute | Value |
|-----------|-------|
| **File** | `wav2vec.py` |
| **Role** | Wav2Vec 2.0 model wrapper. Extracts self-supervised audio features for downstream spoofing detection. |
| **SOTA Algorithm** | **Wav2Vec 2.0 XLS-R** (HuggingFace transformers) with **fine-tuned layers 4-12** |
| **Integration** | Used by: `audio.py`. Loaded via: `registry.py` |
| **Inputs** | Audio waveform (torch.Tensor, 16kHz mono) |
| **Outputs** | SSL feature embeddings (torch.Tensor) |
| **Why this approach?** | Wav2Vec 2.0 achieves 0.22% EER, outperforming handcrafted features (MFCC: ~5% EER) by 20x. XLS-R variant provides multilingual robustness. Fine-tuning specific layers optimizes for spoofing task. |

```python
# Implementation signature
class Wav2VecWrapper:
    """Wav2Vec 2.0 wrapper for audio feature extraction."""
    
    def __init__(
        self,
        model_name: str,
        device: torch.device,
        config: ModelConfig
    ):
        self._processor = Wav2Vec2Processor.from_pretrained(model_name)
        self._model = Wav2Vec2Model.from_pretrained(model_name)
        self._model.to(device)
        self._model.eval()
        
        # Freeze layers 1-3, fine-tune 4-12
        self._freeze_layers(config.freeze_layers)
    
    @torch.inference_mode()
    async def extract_features(
        self,
        waveform: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract SSL features:
        1. Normalize waveform
        2. Process through Wav2Vec 2.0
        3. Extract hidden states from layers 4-12
        4. Return concatenated features
        """
        ...
```

---

### 📁 `app/infrastructure/database.py`

| Attribute | Value |
|-----------|-------|
| **File** | `database.py` |
| **Role** | MongoDB async client with connection pooling. Handles all database operations with timeout enforcement. |
| **SOTA Algorithm** | **Motor async driver** + **Connection pooling** + **Retry with exponential backoff** |
| **Integration** | Used by: `repository.py`. Configured via: `config.py` |
| **Inputs** | Connection string from environment |
| **Outputs** | AsyncIOMotorClient instance |
| **Why this approach?** | Motor provides native async MongoDB access. Connection pooling prevents connection exhaustion. Timeout enforcement (<100ms per AGENTS.md) ensures responsive API. |

```python
# Implementation signature
class DatabaseClient:
    """Async MongoDB client with connection pooling."""
    
    _instance: Optional["DatabaseClient"] = None
    
    def __init__(self, config: DatabaseConfig):
        self._client = AsyncIOMotorClient(
            config.mongo_url,
            maxPoolSize=config.pool_size,
            serverSelectionTimeoutMS=config.timeout_ms
        )
        self._db = self._client[config.database_name]
    
    @classmethod
    async def get_instance(cls) -> "DatabaseClient":
        """Singleton pattern for connection reuse."""
        ...
    
    async def execute_with_timeout(
        self,
        operation: Callable,
        timeout_ms: int = 100  # Per AGENTS.md
    ) -> Any:
        """Execute operation with timeout enforcement."""
        ...
```

---

### 📁 `app/utils/errors.py`

| Attribute | Value |
|-----------|-------|
| **File** | `errors.py` |
| **Role** | Custom exception hierarchy. Defines typed exceptions for all failure modes with structured error responses. |
| **SOTA Algorithm** | **Exception chaining** + **Error codes** + **Structured logging context** |
| **Integration** | Used by: ALL modules. Caught by: `middleware.py` |
| **Inputs** | Error context from any module |
| **Outputs** | Typed exceptions with error codes and context |
| **Why this approach?** | Typed exceptions enable precise error handling. Error codes support i18n and client-side handling. Structured context aids debugging and monitoring. |

```python
# Implementation
class ArgusError(Exception):
    """Base exception for Argus Core."""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        context: Optional[dict] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}

class DetectionError(ArgusError):
    """Raised when detection inference fails."""
    pass

class ValidationError(ArgusError):
    """Raised when input validation fails."""
    pass

class TimeoutError(ArgusError):
    """Raised when operation exceeds timeout."""
    pass

class AdversarialInputError(ArgusError):
    """Raised when adversarial input is detected."""
    pass
```

---

# 5. Section 3: Development Strategy

## 5.1 Interface-Driven Development (IDD) Approach

Replace the traditional "build one file fully" method with a structured skeleton-first approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT PHASES                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 1: CONTRACTS (Week 1)                                    │
│  ════════════════════════════                                   │
│  ☐ Define ALL Pydantic schemas (schemas/*.py)                   │
│  ☐ Define ALL Abstract Base Classes (interfaces/*.py)           │
│  ☐ Create stub implementations returning mock data              │
│  ☐ Verify schema compatibility across modules                   │
│                                                                  │
│  Deliverable: Complete type system, all files compile           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 2: API SKELETON (Week 2)                                 │
│  ═══════════════════════════════                                │
│  ☐ Implement router.py with all endpoints                       │
│  ☐ Wire dependency injection (deps.py)                          │
│  ☐ Add middleware (auth, logging, error handling)               │
│  ☐ Create health check and OpenAPI documentation                │
│                                                                  │
│  Deliverable: API responds to all endpoints (mock data)         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 3: INFRASTRUCTURE (Week 3)                               │
│  ═════════════════════════════════                              │
│  ☐ Implement database.py (MongoDB connection)                   │
│  ☐ Implement storage.py (MinIO/S3 client)                       │
│  ☐ Implement cache.py (Redis client)                            │
│  ☐ Implement queue.py (Celery tasks)                            │
│                                                                  │
│  Deliverable: Full infrastructure, persistent storage working   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 4: PREPROCESSING (Week 4)                                │
│  ═══════════════════════════════                                │
│  ☐ Implement validate.py (input validation)                     │
│  ☐ Implement extract.py (frame/audio extraction)                │
│  ☐ Implement sanitize.py (adversarial defense)                  │
│  ☐ Implement detect.py (face detection)                         │
│                                                                  │
│  Deliverable: Full preprocessing pipeline                       │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 5: DETECTORS (Weeks 5-7)                                 │
│  ═══════════════════════════════                                │
│  Week 5:                                                        │
│    ☐ Implement video.py (EfficientNetV2 + GRU)                 │
│    ☐ Implement image.py (reuse spatial model)                   │
│  Week 6:                                                        │
│    ☐ Implement audio.py (Wav2Vec 2.0 + AASIST)                 │
│    ☐ Implement text.py (perplexity analysis)                    │
│  Week 7:                                                        │
│    ☐ Implement forensic.py (C2PA + EXIF)                       │
│                                                                  │
│  Deliverable: All detectors functional                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 6: CORE LOGIC (Week 8)                                   │
│  ═════════════════════════════                                  │
│  ☐ Implement orchestrate.py (parallel execution)                │
│  ☐ Implement aggregate.py (ensemble voting)                     │
│  ☐ Implement score.py (trust calculation)                       │
│  ☐ Implement explain.py (XAI generation)                        │
│                                                                  │
│  Deliverable: End-to-end analysis pipeline                      │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  PHASE 7: INTEGRATION & TESTING (Weeks 9-10)                    │
│  ═══════════════════════════════════════════                    │
│  ☐ Integration testing on DeepfakeBench                         │
│  ☐ Adversarial robustness testing                               │
│  ☐ Performance optimization                                     │
│  ☐ Load testing                                                 │
│                                                                  │
│  Deliverable: Production-ready system                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 5.2 Shared Data Schemas (schemas.py)

All modules MUST agree on data structures before implementation begins:

```python
# app/schemas/__init__.py - Central schema exports

from .base import BaseSchema, TimestampMixin
from .request import (
    AnalysisRequest,
    AnalysisOptions,
    DetectionOptions,
)
from .response import (
    AnalysisSubmitResponse,
    AnalysisStatusResponse,
    AnalysisResult,
    ErrorResponse,
)
from .analysis import (
    ModalityType,
    DetectionResult,
    TrustScore,
    ArtifactDetail,
    Explanation,
    ForensicReport,
)
from .artifacts import (
    VideoArtifact,
    AudioArtifact,
    ImageArtifact,
    MetadataArtifact,
)
from .internal import (
    PreprocessedData,
    FrameData,
    AudioData,
    FaceRegion,
)

__all__ = [
    # Base
    "BaseSchema",
    "TimestampMixin",
    # Request
    "AnalysisRequest",
    "AnalysisOptions",
    "DetectionOptions",
    # Response
    "AnalysisSubmitResponse",
    "AnalysisStatusResponse",
    "AnalysisResult",
    "ErrorResponse",
    # Analysis
    "ModalityType",
    "DetectionResult",
    "TrustScore",
    "ArtifactDetail",
    "Explanation",
    "ForensicReport",
    # Artifacts
    "VideoArtifact",
    "AudioArtifact",
    "ImageArtifact",
    "MetadataArtifact",
    # Internal
    "PreprocessedData",
    "FrameData",
    "AudioData",
    "FaceRegion",
]
```

## 5.3 Abstract Base Classes (interfaces.py)

Define behavior contracts before implementation:

```python
# app/interfaces/__init__.py - Central interface exports

from .detector import IDetector
from .preprocessor import IPreprocessor
from .storage import IStorage, IObjectStorage, ICacheStorage
from .repository import IRepository, IAnalysisRepository

__all__ = [
    "IDetector",
    "IPreprocessor",
    "IStorage",
    "IObjectStorage",
    "ICacheStorage",
    "IRepository",
    "IAnalysisRepository",
]
```

## 5.4 Dependency Injection Setup

```python
# app/api/deps.py - Dependency injection container

from functools import lru_cache
from typing import AsyncIterator

from app.config import Settings, get_settings
from app.services.analysis import AnalysisService
from app.infrastructure.database import DatabaseClient
from app.infrastructure.storage import ObjectStorage
from app.infrastructure.cache import CacheClient
from app.detectors import VideoDetector, AudioDetector, ForensicAnalyzer
from app.core.orchestrate import AnalysisOrchestrator

@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()

async def get_database() -> AsyncIterator[DatabaseClient]:
    """Database client with connection management."""
    client = await DatabaseClient.get_instance()
    try:
        yield client
    finally:
        pass  # Connection pooling handles cleanup

async def get_analysis_service(
    settings: Settings = Depends(get_settings),
    db: DatabaseClient = Depends(get_database),
) -> AnalysisService:
    """Construct analysis service with all dependencies."""
    # Load detectors
    video_detector = VideoDetector.from_config(settings)
    audio_detector = AudioDetector.from_config(settings)
    forensic_analyzer = ForensicAnalyzer.from_config(settings)
    
    # Build orchestrator
    orchestrator = AnalysisOrchestrator(
        detectors={
            ModalityType.VIDEO: video_detector,
            ModalityType.AUDIO: audio_detector,
            ModalityType.METADATA: forensic_analyzer,
        },
        config=settings.orchestrator_config,
    )
    
    return AnalysisService(
        orchestrator=orchestrator,
        repository=AnalysisRepository(db),
        config=settings,
    )
```

---

# 6. API Contract Specifications

## 6.1 Endpoint Summary

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| `POST` | `/api/v1/analyze` | Submit media for analysis | `AnalysisSubmitResponse` |
| `GET` | `/api/v1/analysis/{id}` | Get analysis results | `AnalysisResult` |
| `GET` | `/api/v1/analysis/{id}/status` | Get analysis status | `AnalysisStatusResponse` |
| `GET` | `/api/v1/analysis/{id}/heatmap` | Get heatmap image | `StreamingResponse` |
| `GET` | `/api/v1/analysis/{id}/report` | Get full forensic report | `ForensicReport` |
| `POST` | `/api/v1/analysis/{id}/export` | Export report (PDF/JSON) | `FileResponse` |
| `GET` | `/api/v1/health` | Health check | `HealthResponse` |
| `GET` | `/api/v1/models` | List available models | `ModelListResponse` |

## 6.2 Request/Response Schemas

```python
# POST /api/v1/analyze

# Request (multipart/form-data)
class AnalysisRequest:
    file: UploadFile  # Required
    options: Optional[AnalysisOptions]

class AnalysisOptions(BaseSchema):
    modalities: list[ModalityType] = ["video", "audio", "metadata"]
    generate_heatmap: bool = True
    generate_report: bool = True
    priority: Literal["low", "normal", "high"] = "normal"

# Response
class AnalysisSubmitResponse(BaseSchema):
    analysis_id: UUID
    status: Literal["queued", "processing"]
    estimated_time_s: int
    webhook_url: Optional[str]  # For async notification
```

```python
# GET /api/v1/analysis/{id}

class AnalysisResult(BaseSchema):
    analysis_id: UUID
    status: Literal["queued", "processing", "complete", "failed"]
    trust_score: Optional[TrustScore]
    modality_results: Optional[dict[ModalityType, DetectionResult]]
    artifacts: Optional[list[ArtifactDetail]]
    explanations: Optional[list[str]]
    heatmap_url: Optional[str]
    processing_time_ms: Optional[int]
    created_at: datetime
    completed_at: Optional[datetime]
    error: Optional[ErrorDetail]
```

---

# 7. Configuration & Environment

## 7.1 Environment Variables

```bash
# .env.example

# === API Configuration ===
API_HOST=0.0.0.0
API_PORT=8001
API_WORKERS=4
DEBUG=false

# === Database ===
MONGO_URL=mongodb://localhost:27017
MONGO_DATABASE=argus_core
MONGO_TIMEOUT_MS=100

# === Cache ===
REDIS_URL=redis://localhost:6379
REDIS_TIMEOUT_MS=50

# === Object Storage ===
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=argus-media

# === Model Paths ===
EFFICIENTNET_PATH=/models/efficientnetv2_b2_deepfake.pt
WAV2VEC_PATH=/models/wav2vec2_xlsr_aasist.pt
TEMPORAL_GRU_PATH=/models/temporal_gru.pt

# === Detection Thresholds (NO HARDCODING) ===
TRUST_SCORE_WEIGHTS='{"visual": 0.30, "audio": 0.30, "temporal": 0.15, "metadata": 0.15, "rppg": 0.05, "ensemble": 0.05}'
CONFIDENCE_THRESHOLD=0.7
DISAGREEMENT_THRESHOLD=0.3

# === Timeouts (per AGENTS.md) ===
DB_TIMEOUT_MS=100
AI_TIMEOUT_S=60
CACHE_TIMEOUT_MS=50

# === Adversarial Defense ===
SANITIZE_JPEG_QUALITY_MIN=70
SANITIZE_JPEG_QUALITY_MAX=95
SANITIZE_NOISE_SIGMA=0.01

# === Observability ===
LOG_LEVEL=INFO
PROMETHEUS_PORT=9090
OTEL_EXPORTER_ENDPOINT=http://localhost:4317
```

## 7.2 Config Schema

```python
# app/config.py

from pydantic_settings import BaseSettings
from pydantic import Field
import json

class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    api_workers: int = 4
    debug: bool = False
    
    # Database
    mongo_url: str
    mongo_database: str = "argus_core"
    mongo_timeout_ms: int = Field(default=100, le=500)
    
    # Cache
    redis_url: str
    redis_timeout_ms: int = Field(default=50, le=200)
    
    # Models
    efficientnet_path: str
    wav2vec_path: str
    temporal_gru_path: str
    
    # Detection (loaded from JSON string)
    trust_score_weights: dict[str, float] = Field(
        default_factory=lambda: json.loads(
            '{"visual": 0.30, "audio": 0.30, "temporal": 0.15, '
            '"metadata": 0.15, "rppg": 0.05, "ensemble": 0.05}'
        )
    )
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    disagreement_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Timeouts
    db_timeout_ms: int = Field(default=100, le=500)
    ai_timeout_s: int = Field(default=60, ge=30, le=120)
    cache_timeout_ms: int = Field(default=50, le=200)
    
    # Adversarial Defense
    sanitize_jpeg_quality_min: int = Field(default=70, ge=50, le=100)
    sanitize_jpeg_quality_max: int = Field(default=95, ge=50, le=100)
    sanitize_noise_sigma: float = Field(default=0.01, ge=0.0, le=0.1)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
```

---

# 8. Appendix: Model Registry

## 8.1 Production Models

| Model ID | Architecture | Modality | Size | Accuracy | Latency |
|----------|--------------|----------|------|----------|---------|
| `efficientnetv2_b2_v1` | EfficientNetV2-B2 | Image/Video | 45MB | 99.89% | 50ms/image |
| `wav2vec2_xlsr_aasist_v1` | Wav2Vec 2.0 + AASIST | Audio | 380MB | 0.22% EER | 200ms/s |
| `temporal_gru_v1` | Bidirectional GRU | Video | 12MB | N/A (aux) | 10ms/seq |
| `mtcnn_v1` | Multi-task CNN | Face Detection | 8MB | 99.1% | 30ms/image |

## 8.2 Model Loading Strategy

```python
# app/models/registry.py

from typing import Dict, Type
from app.interfaces.detector import IDetector
from app.models.efficient import EfficientNetWrapper
from app.models.wav2vec import Wav2VecWrapper

class ModelRegistry:
    """Centralized model loading and versioning."""
    
    _models: Dict[str, nn.Module] = {}
    _versions: Dict[str, str] = {}
    
    @classmethod
    async def load_model(
        cls,
        model_id: str,
        device: torch.device
    ) -> nn.Module:
        """Load model with caching."""
        if model_id in cls._models:
            return cls._models[model_id]
        
        # Load based on model type
        if model_id.startswith("efficientnet"):
            model = await cls._load_efficientnet(model_id, device)
        elif model_id.startswith("wav2vec"):
            model = await cls._load_wav2vec(model_id, device)
        else:
            raise ValueError(f"Unknown model: {model_id}")
        
        cls._models[model_id] = model
        return model
    
    @classmethod
    def get_version(cls, model_id: str) -> str:
        """Get model version for logging."""
        return cls._versions.get(model_id, "unknown")
```

---

# Document Status

| Item | Status |
|------|--------|
| SOTA Research | ✅ Complete |
| Life of Request Flow | ✅ Complete |
| File Manifesto | ✅ Complete (17 core files documented) |
| Development Strategy | ✅ Complete |
| API Contracts | ✅ Complete |
| Configuration | ✅ Complete |
| AGENTS.md Compliance | ✅ Verified |

---

**Document Version:** 1.0  
**Last Updated:** August 2025  
**Next Review:** Upon implementation start  
**Approval Required:** Project Lead sign-off before Phase 2

---

*End of Argus Core Implementation Blueprint*

