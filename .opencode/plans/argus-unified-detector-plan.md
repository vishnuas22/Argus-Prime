# ARGUS UNIFIED — Media-Agnostic Authenticity Engine (merged master plan)

Goal (user-confirmed): take the best of both architectures into one system. Reliable against ANY media, ANY generator — including ones that don't exist yet. Benchmark existing code BEFORE building new modules.

## Sources being merged

FROM Argus_Master (vishnuas22/Argus_Master) — adopted wholesale:
- 4-tier "evidence court" spine: C2PA fast-path → degradation triage d → parallel panel → conformal court
- Module contract {evidence_score, reliability_score(d), confidence, artifacts[]}
- Triage-conditioned reliability r_m(d); resize-discount rules
- Conformal abstention (upgrade stub → MAPIE split-conformal once labels exist)
- Evidence-court verdict schema + template-grounded explanations (no LLM hallucination)
- Eval doctrine: leave-one-generator-out × laundering ladder
- Product shell: FastAPI backend, React UI, registry/crash-isolation tests

FROM deepfake-detector plan (mine) — adopted wholesale:
- Learned specialist heads on frozen foundation backbones (the accuracy ceiling raisers):
  FACE: frozen CLIP ViT-L + ~5M adapter · IMAGE-AIGC: CLIP linear probe (UnivFD-style)
  AUDIO: frozen wav2vec2-XLS-R + pooled head (multilingual) · VIDEO: temporal agg + AV-sync score
- Pseudo-fake engine (SBI self-blending, inswapper swaps, local TTS clones) = modern-artifact supply
- Offline feature caching → trains heads on M1 Max at CPU speed
- Numeric gates: cross-dataset AUC/EER per module + fused; per-generator breakdowns
- Trained LightGBM stacking replaces WeightedVotingJudge (same pluggable interface, zero API change)

## Revised phase order (benchmark-first per user decision)

P0′ SCORE THE EXISTING SYSTEM (images only, as built)
- Assemble scoring corpus: real (COCO subset / OpenImages) + fake (GenImage SD-class subset,
  CiFAKE held-out generators, locally generated SDXL/Flux samples)
- Harness wraps their FastAPI/pipeline: per-module + fused AUC/EER, calibration (Brier/ECE),
  abstention rate vs alpha, runtime/image on M1 CPU
- Laundering ladder v0: JPEG q90/q75/q50, resize 0.75×/0.5×, screenshot-sim, combos
- DELIVERABLE: baseline report card — the first quantitative truth about Argus_Master.
  GATE: report exists w/ per-module numbers + fused number + ladder degradation curves.

P1 DATA INFRA (multimodal): download scripts + manifests (FF++, Celeb-DF, WildDeepfake,
ASVspoof19/21, WaveFake, GenImage, CiFAKE) + registration tracker (FakeAVCeleb, DFD, LAV-DF)

P2 PREPROCESS + CACHE: face crops (contact-sheet QA), 16k audio chunks, frame sampling;
foundation-feature caches built (CLIP, XLS-R)

P3 FACE HEAD v1 → joins panel as module `learned_face_probe` (contract-compliant).
Gate: FF++ in-domain ≥0.98 AUC; Celeb-DF cross ≥0.85; beats DINOv2-kNN realness probe
on localized-manipulation slice (predicted failure mode of one-class).

P4 AUDIO BRANCH v1: standalone scores + panel member for video pipeline.
Gate: ASVspoof19LA eval EER ≤6%; cross-21DF ≤15%.

P5 IMAGE-AIGC PROBE v1 (supervised complement to their one-class realness probe).
Gate: GenImage val ≥0.90 AUC; LOGO across generator families.

P6 FUSION UPGRADE: voting-v0 → trained LightGBM stacker on score dumps (evidence ×
reliability features) + isotonic + MAPIE conformal. Gate: fused ≥ best single module
on every eval slice; coverage/error curve at α=0.1 reported.

P7 ROBUSTNESS ROUND: heads retrained w/ augmentation suite; r_m(d) curves fit per
module from ladder runs (their novel idea, now with learned modules included).

P8 PACKAGING: argus-detect CLI + batch API + CoreML export; React UI extended to
audio/video verdicts.

## Standing principles
- No module enters the panel without a measured report card.
- Every claim traces to an artifact (their XAI rule).
- Escalate over guess (ours). Memory compounds after every phase (memory/).
- Security note: rotate/remove committed backend/.env secrets before any deployment.
