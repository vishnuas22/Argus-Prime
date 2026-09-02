# Argus Deepfake Detector — Master Execution Plan

Scope (user-confirmed): ALL synthetic media (face-swap/reenactment, voice clone/TTS, AIGC imagery) · batch-first runtime · multilingual audio · gated datasets approved · greenfield inside Argus_Prime.
Hardware: MacBook Pro M1 Max (local dev/training); cloud GPU optional later. Core strategy that makes this work locally: **frozen foundation backbones + tiny trainable heads + offline feature caching** (the field's own winning recipe).

## Research basis (verified via fan-out)

- Image SOTA: frozen CLIP ViT + adapter (5.7M params) = CVPR'25 Forensics Adapter, ~87 AUC DFDC cross-dataset; LN-tuning CLIP = 96.6 AUC Celeb-DF. Legacy supervised backbones collapse on modern generators (<60 AUC in-the-wild study) → must train against modern/self-generated fakes.
- Audio SOTA: frozen wav2vec2-XLS-R + light backend; freezing beat finetuning 27.58→7.66 EER (ASVspoof5); RawBoost + codec augmentation decisive; multilingual satisfied by XLS-R.
- AV/video SOTA: two-stage correspondence pretraining (AVFF 99.1 AUC FakeAVCeleb), identity-consistency frontier (Referee). Sync-mismatch (phoneme↔viseme) is a cheap strong signal.
- Metric lesson: field standard is AUC/EER + cross-dataset tables; accuracy@0.5 understates. Baseline audit may already recover points.

## Repo structure (greenfield)

```
deepfake/
├── pyproject.toml            # python 3.11+, torch(MPS), transformers, facenet-pytorch/insightface, librosa, soundfile, scikit-learn, onnxruntime, coremltools
├── configs/{face,audio,image_aigc,video,fusion}/*.yaml
├── src/argus_df/
│   ├── data/                 # face_crop.py audio_prep.py video_prep.py pseudo_fakes.py registry.py
│   ├── models/               # face_clip_adapter.py image_aigc_head.py audio_w2v_head.py temporal.py fusion.py
│   ├── train/                # trainer.py (MPS, bf16, grad-accum) augment.py losses.py
│   ├── eval/                 # metrics.py (AUC,EER,Brier) protocol.py (cross-dataset)
│   └── api/                  # cli.py server.py(FastAPI, phase P8)
├── scripts/download_*.sh  scripts/export_coreml.py
└── tests/
data/{raw,processed,caches}/   # gitignored
artifacts/{checkpoints,reports}/
```

## The local-training unlock: feature caching

Frozen backbones are inference-only → precompute penultimate features once into `data/caches/` (NPZ/Parquet + index). Head training then runs on cached tensors at CPU speed on any machine. Backbone choice per module:
- FACE: `openai/clip-vit-large-patch14` (dev on `ViT-B/16` laion for iteration speed)
- IMAGE-AIGC: same CLIP encoder (UnivFD-style linear probe)
- AUDIO: `facebook/wav2vec2-xls-r-300m` frozen (multilingual)

## Modules & targets

| Module | Head (trainable) | Train data | Verify gate |
|---|---|---|---|
| FACE | adapter ~5M (patch-wise contrastive + CE, Forensics-Adapter-lite) | FF++ c23 + SBI pseudo-fakes + JPEG/scale aug | in-domain FF++ AUC ≥0.98; Celeb-DF cross ≥0.85 |
| IMAGE-AIGC | linear/MLP probe | GenImage subset (SD-class) + CiFAKE held-out | GenImage val AUC ≥0.90; CiFAKE cross ≥0.80 |
| AUDIO | MHA pooling + MLP ~2M on cached XLS-R feats | ASVspoof2019 LA + WaveFake + codec/RawBoost aug | 19LA eval EER ≤6%; 21DF cross ≤15% (stretch ≤10%) |
| VIDEO | temporal agg (mean/std + small transformer over frame feats) + SyncNet-style sync score | WildDeepfake / LAV-DF (after grant) | beats single-frame baseline by ≥2 AUC |
| FUSION | Platt-calibrated logistic stack on [s_face,s_audio,s_sync,s_aigc] (+tiny transformer later) | score dumps from val splits | fusion ≥ best single modality on multimodal sets |

Pseudo-fake engine (`pseudo_fakes.py`): SBI self-blending, inswapper/FaceFusion swaps, frame-drop/repeat temporal fakes, local TTS clones (Piper/coqui) — unlimited modern hard positives, generated on-device.

## Phased roadmap (each closes only with its verify gate green)

- P0 Scaffold + EVAL HARNESS FIRST: metrics.py/protocol.py + audit script running existing user models under correct protocol → deliverable: honest AUC/EER baseline report. Gate: harness reproduces a published reference number within ±2 AUC.
- P1 Data infra: download scripts (FF++ form, Celeb-DF, WildDeepfake, ASVspoof2019/2021, WaveFake, GenImage subset, CiFAKE) + registration checklist tracker (FakeAVCeleb, DFD, LAV-DF, KoDF) + manifests with checksums.
- P2 Preprocessing + caching: face crops (contact-sheet visual QA), audio chunks 4s/16k, frame sampling; caches built. Gate: cache coverage = manifest coverage; spot-check QA sheets reviewed.
- P3 FACE v1 (ViT-B dev → ViT-L final). P4 AUDIO v1. P5 IMAGE-AIGC v1. Each: train on cached feats, full eval table, checkpoint + report artifact.
- P6 VIDEO + FUSION v1.
- P7 ROBUSTNESS ROUND: re-train/augment with JPEG q50/q75, downscale, Gaussian noise, mp3/opus re-encode, GPEN-style enhancement; publish before/after deltas (in-the-wild lesson).
- P8 Packaging: `argus-detect <file>` CLI returning per-modality scores + calibrated verdict + cue explanation; FastAPI batch endpoint; CoreML/ONNX export of heads for Apple Silicon speed.

## Risks & mitigations

- MPS quirks/slow ViT-L training → ViT-B iteration, feature caching, grad accumulation; escalate heavy runs to rented GPU only if a gate blocks.
- Gated-dataset latency (FakeAVCeleb etc.) → pipeline never blocks: P3–P5 run on instantly available sets; gated sets slot in when grants land.
- Scope breadth ("all synthetic media") → modules are independent; each ships value alone; fusion composes them.
- Multilingual audio data scarcity beyond EN/JA → XLS-R transfer + synthetic TTS across languages via local TTS engines; expand later.

## First execution slice (on approval)

P0 fully: scaffold pyproject + package skeleton + metrics/protocol/harness + baseline-audit script wired to whatever model checkpoints the user has (path needed), plus P1 download scripts for the instant-download sets. Then stop → review gates together.
