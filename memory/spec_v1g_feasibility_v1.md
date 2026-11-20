# Sealed spec — V1g-feasibility-v1

**Status:** SEALED (pre-registered before scoring). Any deviation requires a new spec + user approval.
**Owner:** user (approved 2026-11-20).
**Dimension:** OOD-video.
**Receipt dir:** `artifacts/v1g_restrav_feasibility/`.
**Memory receipt:** `memory/restrav_probe_2026-11-20.md` (prior smoke), this spec, and the final `RESULT.md`.

## Hypothesis
ReStraV 21-D geometry features (DINOv2 ViT-S/14, 24 frames, 2 s center window) provide **decorrelated** evidence vs the incumbent p6vid_v4 on T2V (GenBuster++) and CDF (face-swap/real-YouTube), such that a logistic-regression stacker on [ReStraV best feature, p6vid_v4] improves AUC over p6vid_v4 alone **without regressing** on either domain.

## Pre-registered bar (must pass ALL to advance)
- **B1 (T2V gain):** combined 10-fold AUC on GenBuster++ ≥ p6vid_v4 AUC + 0.05.
- **B2 (no CDF regression):** combined 10-fold AUC on CDF ≥ p6vid_v4 AUC − 0.02.
- **B3 (decorrelation):** |Pearson r(ReStraV best, p6vid_v4 pf)| ≤ 0.30 on GenBuster++ (proves the signal is not redundant).
- **B4 (sample size):** n ≥ 200 per class per domain (400 clips total minimum).

## What is allowed under this spec
- Read-only scoring: extract ReStraV 21-D features, score p6vid_v4, compute combined AUC.
- Fit a **diagnostic** logistic regression on the extracted features (no model promotion, no pointer change, no stacker retrain).
- Book the measured result to `memory/improvement_ledger.md` (Y = measured combined AUC delta, or 0 if bar fails).

## What is NOT allowed (requires separate spec + approval)
- Training a production ReStraV MLP.
- Wiring ReStraV into the serve path or `/api/detect`.
- Retraining the video stacker (p6vid_v4 → v5).
- Moving any pointer or threshold.
- Committing model weights or features to git.

## Procedure
1. Sample n=200 per class from GenBuster++ (100 real + 100 fake) and CDF (100 real + 100 fake). Use a fixed seed (42) for reproducibility; record paths in `RESULT.md`.
2. Extract ReStraV 21-D features (DINOv2 ViT-S/14, 24 frames, 2 s center window, MPS). Save to `artifacts/v1g_restrav_feasibility/features_{genbuster,cdf}.npz`.
3. Score p6vid_v4 (`detect_video`, 12 frames) on the same 400 clips. Save to `artifacts/v1g_restrav_feasibility/p6vid_scores.json`.
4. For each domain: compute ReStraV best single-feature AUC, p6vid_v4 AUC, combined LR AUC (10-fold stratified, report mean ± std), Pearson r.
5. Compare to bar B1–B4. Write `RESULT.md` with PASS/FAIL per bar and the measured numbers.
6. Book Y to `improvement_ledger.md`:
   - If all bars PASS: Y = measured combined AUC delta on GenBuster++ (the OOD-video dimension gain, conditional on a future restack spec).
   - If any bar FAILS: Y = 0, line closed at 0 with receipt, no re-litigation without a new hypothesis.

## Risks / kill conditions
- **Decode failures:** if >10% of clips fail to decode in either domain, the sample is biased; record and re-sample with a different seed.
- **p6vid_v4 on CDF is already 0.9482** (near ceiling) — B2 is a strict no-regression bar; even a small drop fails B2.
- **ReStraV on CDF is expected near-chance** (face-swap, not T2V) — B3 may fail on CDF even if B1 passes on GenBuster++. That is an acceptable outcome: the spec tests T2V primarily.
- **No cherry-picking:** the 200+200 sample is fixed by seed 42 before any scoring. Do not re-sample to improve results.

## Out of scope
- Video Reality Test (Sora-2/Veo-3.1) — separate acquisition + spec.
- Audio/image modalities.
- Any training, promotion, or pointer change.
- Commercial posture (U8) or license (U6) rulings.

## Approval
User approved the V1g feasibility wave (n=200, sealed spec) on 2026-11-20. This spec is the sealed record.