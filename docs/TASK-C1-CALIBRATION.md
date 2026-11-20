# TASK-C1 — Calibration and Court 2.0 (Fix the 41-71% FPR Collapse)

**Phase:** C1 · **Window:** weeks 3-5 (overlaps P0 tail) + per-deployment regenerations
**Effort:** 2-3 engineer-weeks initial · **Cheapest large win in the system — no new models required**

## Evidence (why this is a top-2 priority)

- **D-103 audit (repo's own)**: channels keep AUC 0.71-0.78 cross-domain, but fixed
  thresholds collapse: **FPR 41-71%**; fixed-threshold probes fail off-manifold 4-7x.
- Published 2026 confirmation ("When EER Hides Deployment Failure", arXiv 2606.21584):
  a SOTA model with EER 11.2% had **HTER 39.5% at the transferred threshold**; monotone
  transforms (z-norm, temperature) provably cannot fix EER (AUC-invariant) but DO move
  operating points; z-norm with a contaminated cohort made things WORSE (60.2% EER).
- ASVspoof 5 evaluation (arXiv 2601.03944): probability-like (softmax) scores collapse
  (actDCF 1.0 across wide threshold ranges); a logit transform or logistic-regression
  calibration recovers most of it (T27 near-oracle vs PAV).
- Current court outputs: `p_auth = sigmoid(3.0·fused)` (fusion.py:55) — exactly the
  probability-like output the literature warns about; conformal α per stacker is
  train-derived (good, D-035) but there is no domain-adaptive threshold layer.

## The five-layer stack (deploy 1-3 now; 4-5 per deployment)

1. **Output logits/LLRs, not softmax probabilities** — court + all channels.
2. **Court-level logistic-regression calibration** on held-out fused scores
   (one sklearn LR; turns the court score into a calibrated LLR so one global
   threshold means something).
3. **Class-pure cohort score packs** — per-domain (codec/source/language) Z/AS-norm
   with bona-fide-only cohorts. NEVER mixed cohorts (contamination collapse is
   published). Store cohorts as artifacts with train-only provenance (D-035).
4. **LTT / conformal threshold selection** — utility that takes any labeled probe set
   (a few hundred clips per domain) and returns a threshold with finite-sample
   FPR ≤ α guarantee (Learn-then-Test, arXiv 2110.01052; conformal risk control,
   2208.02814). Per-channel AND fused.
5. **Risk curves + abstention band** — a-DCF-style cost-vs-threshold curve in the API
   response (scope-labeled, D-037 law); keep the existing APS abstention as the
   safety net for fully-OOD inputs (the one thing no calibration can guarantee).

## Steps

1. Add `deepfake/bench/calibration.py`: `logit_transform`, `CourtCalibrator`
   (fit/predict LR on fused scores), `cohort_normalize(scores, cohort_stats)`,
   `ltt_threshold(scores, labels, alpha)` with the fixed-sequence testing procedure.
2. Fit the court calibrator on the existing train dumps (no new data collection).
3. Wire into `fusion.py`/`StackerJudge` output path behind a feature flag; API
   `/api/detect` gains `risk_curve` (array of (threshold, cost) points, scope-labeled)
   and keeps `abstained`.
4. Cohort packs: build from train-split reals per domain tag (codec via ffprobe,
   source tag where available). Class-pure by construction.
5. Evaluation harness extension: report HTER-at-transferred-threshold alongside AUC/EER
   for every exam (the 2606.21584 recommendation) — add to `run_benchmark.py` output
   and `benchmarks.json` schema (with scope labels).
6. Re-run the D-103 threshold-migration audit script with the new stack; produce the
   before/after table (this becomes the acceptance evidence).
7. Per-deployment procedure doc: how a new domain gets its LTT threshold from a
   few-hundred-clip labeled probe (this is operations, not code).

## Acceptance criteria + verification commands

```bash
# A1: calibration module + tests
pytest -m fast tests/test_calibration.py -q        # green

# A2: the D-103 re-audit — the money number
python deepfake/scripts/audit_threshold_migration.py --with-calibration
# Fixed-threshold FPR on held-out wild mix: ≤ 10%   (from 41-71%)

# A3: API surface
curl -s localhost:8420/api/detect -F file=@tests/fixtures/wild1.jpg \
  | jq '.risk_curve[0:3], .abstained'               # curve present + scope label

# A4: monotonicity sanity (LR calibration must not change ranking)
# AUC before == AUC after calibration on every exam set (monotone transform law)

# A5: HTER reporting live
jq '.image.exams[-1] | {auc, eer, hter_at_transferred_threshold}' artifacts/benchmarks.json
```

## What this stack honestly CANNOT fix (state it in LIMITATIONS.md)
- It never improves AUC (monotone transforms are AUC-invariant — mathematical fact).
- LTT guarantees hold only where calibration data is exchangeable with deployment;
  a fully novel generator gets NO guarantee — the abstention band is the only defense.
- Cohort packs need domain-matched bona-fide data; a brand-new platform (e.g., a new
  social codec) needs a new cohort before its scores are trustworthy.

## Risks
- Calibration fit on insufficiently diverse fused scores overfits — use LR (2 params),
  not isotonic/PAV, for small n; validate across at least 3 exam domains.
- The α monkey-patching at serve time (`server.py:199-210`) is not thread-safe —
  parameterize α per request BEFORE any fan-out work (S1 dependency).
