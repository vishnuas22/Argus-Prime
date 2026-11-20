# TASK-V1 — Video Modality Upgrade (SBI + Generative Channel + Wild Eval)

**Phase:** V1 · **Window:** weeks 5-12 (after P0, parallel with I1/A1) · **Effort:** 4-6 engineer-weeks
**Prereq:** TASK-P0 complete. Uses the existing restack machinery.

## Evidence (why upgrade)

- Curated lane is strong: FF++ 0.9121 (n=700), Celeb-DF 0.9482, DFD 0.8857, LOMO pass
  — but **wild video was never examined** and DF-Eval video exam is blocked.
- Text-to-video generators (GenVidBench, in-house): **kling 0.6035 / sora 0.6812 /
  opentora 0.7497** — the weakest real-world surface after audio.
- Temporal modeling is drift-statistics only; D-078 measured temporal features at
  **chance (0.5493)** for the 12-frame budget. API budget is 5 frames (`server.py:441`).
- The audio advisory is structurally unscoreable on training corpora: 0/60 FF++ and
  0/50 Celeb-DF samples contain audio streams (`benchmarks.json:168`).
- Context: every FF++-era detector collapses on 2024+ generators (DF40: SBI 0.514,
  Xception 0.546 avg AUC); DF26 (Sep 2026) reports SOTA near chance on 2026 t2v/i2v.
  Off-the-shelf best on DF-Eval video = 0.63 (GenConViT); fine-tuned = 0.82;
  best commercial = 0.79.

## Target architecture

| Slot | Component | Purpose | License note |
|---|---|---|---|
| Front-end | YuNet (HF `opencv/face_detection_yunet`, MIT, 232 KB) — verified 53.6 ms/frame Linux | replace untracked onnx; SCRFD-2.5G fallback only if small-face recall proves limiting (insightface models are NC) | MIT |
| Per-frame channel | **SBI** (EfficientNet-B4, 19M) on 4 frames | compression-robust face-forgery evidence (CDF 93.18, DFDCP 86.15 cross-dataset) | academic NC — commercial needs agreement or a self-trained SBI-recipe model |
| Temporal pilot (kill-gated) | **TALL++-style 4-frame thumbnail grid** | temporal evidence; corruption suite 98.0 avg AUC (best verified robustness) | DeepfakeBench weights CC BY-NC 4.0 (research only) |
| Generative channel | CLIP/ViT head fine-tuned on GenVidBench Tier-1 (14,114 clips in-house) + SAVGE-class public corpus if license allows | the actual t2v fix (DF-Eval: fine-tuning moved GenConViT 0.63→0.82; nothing else works) | own weights |
| Keep | dual CLIP heads, drift, scene, SyncNet, stacker | incumbent curated strength | — |
| Aggregation fix | mean-of-max-per-frame (replace max-only) + per-domain Youden thresholds | kills the one-bad-frame FPR inflation (your WaveRep 48% FPR class of failure) | — |

## Steps

1. **YuNet replacement** (P0 dependency): fetch from verified HF URL, sha-pin in
   models.lock, wire `fetch_face_models.py`.
2. **SBI integration**: mirror the GDrive checkpoint to the private HF repo with
   sha256 (GDrive links rot; license file copied alongside). Add `sbi_probe` channel
   behind the YuNet→crop→stacker seam (timm EfficientNet-B4; verify key-format under
   the pinned timm). Frame budget: 4 (not 12).
3. **TALL++ pilot behind a pre-registered gate**: hypothesis = 4-frame thumbnail grid
   adds ≥ +0.02 stacked AUC on a held-out split WITHOUT breaking B4 abstention-parity.
   Register the spec (D-number) BEFORE scoring. If it fails: shelve (kill-gate K2),
   incumbent lane unchanged — this exact class of feature measured chance before (D-078).
4. **Generative-video channel**: train on GenVidBench Tier-1 (in-house) with the
   existing CLIP service + a new head; evaluate on held-out kling/sora/opentora.
   Consider SAVGE (2.37M AIGV videos, Nov 2026) if licensing permits.
5. **Frame aggregation + calibration**: mean-of-max per frame; per-domain Youden
   thresholds as a new artifact type (`thresholds.json` beside the video stacker meta,
   fail-closed checked, surfaced in /api/health).
6. **Restack**: new feature row contract → `video_features.py FEATURE_NAMES` flip with
   sentinel gating (repo protocol) → `verify_dump_parity.py` re-based → retrain
   video stacker → full gate battery (FF++ ≥0.90, per-method bars, LOMO ≥0.70,
   CDF ≥0.75, DFD ≥0.85/≤0.35/≤0.25).
7. **First wild-video corpus run**: WildDeepfake test (HF copy, Apache-2.0) + DFDC
   sample, scope-labeled. This produces the first wild video number the project has.
8. **Latency check** against the corrected budget (master plan §9): YuNet×12 ≈ 0.25 s,
   SBI×4 + CLIP×4 ≈ 1.6 s, TALL++ 1 grid ≈ 0.3-0.7 s — target ≤4 s wall on 8 cores.
9. UI/health/e2e lockstep (instrument counts change) + D-numbers + benchmarks.json.

## Acceptance criteria + verification commands

```bash
# A1: new channels live and contracts hold
python deepfake/argus_detect.py tests/fixtures/wildface.mp4 --json \
  | jq '.feature_names'    # sbi block present; matches video_features.FEATURE_NAMES
jq -e '.feature_names == $REF' artifacts/<new>/stacker_meta.json   # CI assert
python deepfake/scripts/verify_dump_parity.py && echo PARITY-OK

# A2: curated gates — no regression (restack TOL table)
jq '.gates | map(select(.pass==false))' artifacts/<new>/gates.json   # empty or waived-with-WAIVER.md

# A3: wild first numbers recorded with scope
jq '.video.scope_wild' artifacts/benchmarks.json    # WildDeepfake + DFDC entries present

# A4: t2v staged target
jq '.video.scope_t2v' artifacts/benchmarks.json     # kling/sora/opentora ≥ 0.65 each

# A5: latency
python deepfake/scripts/perf_bench.py --modality video    # p50 ≤ 4 s/clip (8-core CPU)
```

## Staged targets (honest)
- **Stage 1:** curated held (FF++ ≥0.90, CDF ≥0.90) + SBI channel live + first
  WildDeepfake/DFDC numbers published (target CDF-class 0.72-0.80 on DFDC per
  published SBI range).
- **Stage 2 (t2v):** ≥0.65 per generator, 0.70 stretch. DF-Eval (if access): 0.75
  interim. **0.82 is stretch-only** — it exceeds every commercial system tested (0.79);
  never commit externally.
- TALL++ slot only if the pre-registered pilot passes.

## Risks
- SBI GDrive checkpoint fragility + NC license (commercial kill) — fallback: train
  own SBI-style model on public self-blended augmentation (recipe is public) or use a
  DeepfakeBench MIT-class detector.
- Temporal pilot failure is a REAL possibility (D-078 precedent) — that is why it is
  kill-gated, not scheduled as a deliverable.
- 2026 generators may keep everything near chance (DF26) — stage targets already
  assume humility.
