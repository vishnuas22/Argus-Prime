# Weight and Third-Party Notices

**Verified:** 2026-11-20
**Scope:** every third-party model weight, dataset, and vendored component that
Argus-Prime downloads, loads, or ships.
**Read this before any commercial deployment.** Several entries are
**non-commercial (NC)** and are excluded from a commercial product.

Legend: ✅ commercial-safe · ⚠️ research/internal only · ❌ prohibited commercially

---

## 1. Vendored / bundled in this repository

| Component | Location | Licence | Commercial | Notes |
|---|---|---|---|---|
| **YuNet** face detector | `tools/models/yunet.onnx` (232,589 B) | **MIT** | ✅ | Source: `huggingface.co/opencv/face_detection_yunet`. sha256 `8f2383e4dd3cfbb4…`. |
| **SFace** face recogniser | `tools/models/face_recognition_sface_2021dec.onnx` (38.7 MB) | **MIT** (OpenCV Zoo) | ✅ | Source: `huggingface.co/opencv/face_recognition_sface`. |
| **ArcFace** w600k_r50 | `tools/models/arcface_w600k_r50.onnx` (174 MB) | **⚠️ non-commercial** (insightface) | ❌ | Extracted from `buffalo_l.zip` (insightface v0.7 release). The insightface project is licensed for **research/non-commercial use**. **This blocks commercial video deployment** until swapped for an MIT-class embedder. |
| **MediaPipe Face Landmarker** | `integrations/mediapipe/face_landmarker.task` (3.76 MB) | **Apache-2.0** | ✅ | Google MediaPipe. |
| **SyncNet** (vendored code) | `integrations/syncnet_python/` | **MIT** | ✅ | Joon Son Chung, 2016-present. |
| **SyncNet** model weights | `integrations/syncnet_python/data/syncnet_v2.model` | **⚠️ research** | ❌ | Not redistributed here; fetched by `download_model.sh`. Confirm terms before commercial use. |
| **Wav2Lip** (vendored code) | `integrations/Wav2Lip/` | **⚠️ research-only** | ❌ | The Wav2Lip repository is released for **non-commercial research**. Present as a vendored dependency; **not** part of the detection path. |

## 2. Downloaded at build time (not redistributed)

| Component | Used by | Licence | Commercial | Notes |
|---|---|---|---|---|
| **wav2vec2 XLS-R-300M** | audio probe | **Apache-2.0** | ✅ | `facebook/wav2vec2-xls-r-300m` (HuggingFace). |
| **WavLM Base+** | audio fused court | **MIT** | ✅ | `microsoft/wavlm-base-plus` (HuggingFace). |
| **CLIP ViT-L/14** | image face probes | **MIT** | ✅ | `openai/clip-vit-large-patch14`. |
| **CLIP ViT-B/14** | image probes | **MIT** | ✅ | `openai/clip-vit-base-patch14`. |
| **DINOv2 ViT-S/14** | image realness probe | **Apache-2.0** | ✅ | `facebook/dinov2-small`. |

## 3. Optional / research-mode channels (NOT in the default serve path)

These are documented because the codebase contains dormant or flag-gated paths to
them. **None are required for a commercial image/video deployment.**

| Component | Licence | Commercial | Status |
|---|---|---|---|
| **SAFE** (KDD 2025) | Apache-2.0 | ✅ | optional image channel |
| **UnivFD** (CVPR 2023) | MIT | ✅ | optional image channel |
| **Detectra-v1 / CommFor** | MIT | ✅ | optional web-hardened channel |
| **Smaug** (Abacus AI) | Apache-2.0 (Gemma base terms apply) | ✅ | optional GPU channel |
| **WaveSP-Net** (ICASSP 2026) | **MIT (code)** — verify checkpoint at fetch | ✅ (verify) | optional audio head |
| **AntiDeepfake W2V-Large** | **CC BY-NC-SA 4.0** | ❌ | research only |
| **P3** (Wang & Yamagishi 2023) | BSD-3 per Zenodo record — **verify at download** | ⚠️ verify | fallback audio baseline |
| **SBI** (CVPR 2022) | **academic non-commercial** | ❌ | research only; commercial needs agreement |
| **TALL++ / DeepfakeBench** weights | **CC BY-NC 4.0** | ❌ | research only |
| **GenConViT** weights | **CC BY-NC 4.0** | ❌ | research only |

## 4. Datasets

Datasets are **not redistributed**. Per-dataset licence receipts live in
`artifacts/p8_data_coverage/license_ledger.md` and the individual
`data/raw/<dataset>/MANIFEST.json` / `.zenodo.json` files.

Key constraints:

- **Deepfake-Eval-2024** — CC BY-SA 4.0, **gated until May 2027**. Research use;
  compliance review required before any product use.
- **GenBuster++ / GenVidBench** — see dataset card; "see dataset card" in
  `data/raw/genvidbench/MANIFEST.json`. **Confirm terms before commercial use.**
- **OpenFake** — CC BY-NC. Research only; do not ship weights trained on it in a
  commercial product.
- **Celeb-DF, FaceForensics++, DFD, WildDeepfake, LAV-DF, DFDC** — research
  licences; confirm each before commercial use.

---

## 5. Commercial posture (summary)

**Commercial-safe today (image path only):**
YuNet ✅ · SFace ✅ · CLIP ✅ · DINOv2 ✅ · MediaPipe ✅ · SyncNet code ✅ ·
wav2vec2 XLS-R ✅ · WavLM ✅

**Blocks commercial use:**
- **ArcFace (insightface)** — NC. Blocks the video face-identity path.
- **Wav2Lip** — research-only. Not in the detection path, but present as a vendored dependency.
- **SyncNet weights** — confirm terms.

**Recommendation:** if commercial deployment is intended, (a) swap ArcFace for an
MIT-class embedder or negotiate with insightface, and (b) confirm SyncNet weight
terms. Everything else in the serve path is MIT/Apache-2.0.

---

## 6. How this file was produced

Derived from:
- `deepfake/scripts/fetch_face_models.py` (URLs + sha256 for ArcFace/SFace/YuNet)
- `artifacts/p8_data_coverage/license_ledger.md` (DeepAction v1)
- vendored `integrations/*/LICENSE*` files
- HuggingFace model cards for wav2vec2 / WavLM / CLIP / DINOv2

**Not independently re-verified:** the licence text of each HuggingFace model
card was read at build time, not re-fetched on 2026-11-20. Re-verify at any
commercial commitment.