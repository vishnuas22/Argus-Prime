# Sealed spec — V1d-feasibility-v1 (new video-OOD signal class: ReStraV geometry)

**Status:** SEALED 2026-11-20. Pre-registered BEFORE any scoring. Any deviation
requires a new spec + explicit user approval.
**Owner:** user (V1d directed 2026-11-20).
**Dimension:** OOD-video (currently 1.2/10 — the only dimension that can reach 70).
**Receipts:** `artifacts/v1d_restrav/`
**Prior art:** `memory/restrav_probe_2026-11-20.md` (smoke + B4 addendum), D-107 (V1g FAIL B4).

## Hypothesis

ReStraV 21-D temporal-geometry features (DINOv2 ViT-S/14, 24 frames, 2 s centre
window; 7 consecutive-frame distances + 6 turning angles + 8 distribution stats)
carry **decorrelated** evidence versus the incumbent `p6vid_v4` on text-to-video
generators, such that a logistic-regression combination improves AUC over
`p6vid_v4` alone **without regressing** on curated face-swap exams.

## Why this candidate and not another (evidence)

- V1g smoke already measured **r=0.130** decorrelation and **Δ+0.125 AUC** on
  GenBuster++ T2V (B1/B2/B3 all passed). The only failure was **B4**, and that
  is now shown to be a **sampling artifact** (V1g took the alphabetically first
  100 files; a shuffled 250-file sample decodes 250/250).
- Every other video line is empirically closed: temporal statistics (D-099),
  D3 transfer (D-104, 16–44% FPR), WaveRep (D-100, 48% FPR), SPLIT (D-099).
  ReStraV is a **new signal class** (DINOv2 geometry), not another statistic.
- Assets are local: DINOv2 weights cached, no download, no disk pressure.

## Pre-registered bars (ALL must pass to advance)

Fitted on a **train split** only (D-035); evaluated on a **disjoint** split.

- **B1 (T2V gain):** combined 10-fold AUC on **GenBuster++ T2V** ≥ `p6vid_v4` AUC + 0.05.
- **B2 (no curated regression):** combined AUC on **FF++** and **CDF** each ≥
  `p6vid_v4` AUC − 0.02.
- **B3 (decorrelation):** |Pearson r(ReStraV best feature, p6vid_v4 p_fake)| ≤ 0.30
  on the T2V train split.
- **B4 (sample size):** ≥ **200 per class per domain**, measured **after** decode
  filtering (no silent attrition).
- **B5 (generalisation, NEW):** combined AUC on **held-out kling + sora** ≥
  `p6vid_v4` AUC on the same generators (no threshold transfer, AUC only).

## Procedure (sealed)

1. **Corpora (no download).**
   - Real: GenBuster++ `extracted/video/real` (1,000 files).
   - Fake T2V: GenVidBench `opentora` (13,800), `kling` (210), `sora` (51).
   - Curated regression guards: FF++ C23 real + `Deepfakes`; Celeb-DF-v2
     `YouTube-real` + `Celeb-synthesis`.
2. **Sampling.** `random.Random(20261120)`; for each domain shuffle then take
   `n=300`; run the decode filter; **keep the first 200 that decode per class**.
   Record every decode failure with its reason. Any domain that cannot reach
   200+200 is reported as a **failed B4**, not silently reduced.
3. **Splits.** 50/50 stratified **by clip** into fit / eval, disjoint, same seed.
   Fit the combiner on fit only. Report eval only.
4. **Scores.** `p6vid_v4` via `argus_detect.detect_video(n_frames=12)` — the
   deployed configuration, unchanged.
5. **Features.** ReStraV 21-D via the sealed extractor (DINOv2 ViT-S/14,
   24 frames, 2 s centre window, MPS). Feature order frozen as in
   `restrav_feats.FEATURE_NAMES`.
6. **Combiner.** `LogisticRegression` on `[ReStraV_best_feature, p6vid_v4]`.
   The ReStraV feature is selected on the **fit split only** (max |AUC−0.5|).
   10-fold stratified CV on the eval split; report mean ± std.
7. **Reporting.** `RESULT.md` with PASS/FAIL per bar, per-domain table, and the
   per-generator t2v table (kling/sora/opentora) so the result is comparable to
   the incumbent baseline (kling 0.6035 / sora 0.6812 / opentora 0.7497).

## Explicitly NOT allowed under this spec

- No training of a production ReStraV MLP, no wiring into `/api/detect`, no
  stacker retrain, no pointer/threshold change, no commit.
- No `benchmarks.json` promotion. Measured numbers only.
- No cherry-picking: seed and n are fixed above, before scoring.

## Kill conditions

- Any bar FAILs → line recorded CLOSED at 0 with the receipt. No re-litigation
  without a new pre-registered hypothesis.
- If `p6vid_v4` on GenVidBench is **above 0.85 AUC** (i.e. the incumbent is
  already strong on this exam), B1 becomes implausible and the spec is void —
  report that instead of forcing a comparison.

## Honest limits

- GenBuster reals are **stock footage**, GenVidBench fakes are **generated**;
  the real/fake axis is partly content. This is the same construct-validity
  problem recorded in D-104/D-106. AUC here is therefore an **upper bound** on
  deployed behaviour, and the spec says so rather than hiding it.
- Two of three generators (opentora, kling) are 2024-era; sora is 2024-03.
  None are 2026 frontier models. A pass would justify a **future** test on
  fresher generators, not a claim about them.
- Video face-swap (FF++/CDF) is a **different** problem; ReStraV is expected to
  add nothing there, and B2 exists to prove it does no harm.

## Approval

User directed V1d on 2026-11-20. This spec is the sealed record.