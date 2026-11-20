# Argus-Prime Agent Execution Pack — README

**Companion to:** `Argus-Prime_Upgrade_Master_Plan.docx` (v2.0, red-team corrected, 2026-11-20)
**Repo:** `github.com/vishnuas22/Argus-Prime` @ `b2cfae3` (audit baseline)
**Baseline score:** 42/100 independent audit. **Target:** 70-75 realistic after all phases.

## How to use this pack

Each `TASK-*.md` is self-contained: evidence, exact steps, acceptance criteria with
runnable verification commands, risks, rollback. Execute in the order below unless a
dependency is already satisfied. Never waive a gate silently — the repo's own law
(D-numbers, sealed specs, restack parity) is a feature, not an obstacle.

## Execution order and dependency graph

```
P0 EXISTENCE (weeks 1-3) ──┬──> I1 IMAGE (weeks 5-10)
                           ├──> V1 VIDEO (weeks 5-12)
                           ├──> A1 AUDIO (weeks 5-10)
                           └──> C1 CALIBRATION (weeks 3-5, overlaps P0 tail)
                                      |
        (I1, V1, A1 feature spaces FROZEN after their restacks)
                                      |
                                      v
                           S1 SCALABILITY (weeks 12-16, always LAST)
```

- **P0 is a hard prerequisite for everything.** Until artifacts are recovered from
  the dev Mac, hashed, hosted and bootstrapped, no end-to-end test can even run
  (`init_court()` crashes at `server.py:132`).
- **C1 lands before the modality waves finish**, but its per-deployment components
  (cohort packs, LTT thresholds) are regenerated after each restack.
- **S1 is last by design**: ONNX export and quantization change numerics; doing it
  before the feature spaces freeze means redoing parity gates per channel.

## Kill-gates (pre-registered — do not improvise)

| Gate | Condition | Action |
|---|---|---|
| K1 (week 2) | Deepfake-Eval-2024 registration not approved | All targets fall back to public-corpus definitions (see per-task "Stage 1") |
| K2 | TALL++ pilot fails its pre-registered gate | Shelve temporal channel; incumbent video lane unchanged |
| K3 | WaveSP-Net checkpoint unfetchable / non-permissive at download | Fallback: P3 in a Python 3.10 sidecar container, or self-train the 4.1M-param head (recipe published) |
| K4 | Any restack produces a FAIL | HOLD promotion; write a WAIVER.md or fix; never silently absorb |

## Effort (executor dry-run, honest)

P0: 3-4 wk · C1: 2-3 wk · I1/V1/A1: 4-6 wk each (parallel) · S1: 3-5 wk.
**Total 21-31 engineer-weeks.** 2 engineers ≈ 14-20 calendar weeks; 3 engineers ≈ 10-14.

## Non-negotiable process laws (from the repo's own decision log)

1. Train-only calibration (D-035): every threshold/alpha/coverage gate derived from
   train splits only. Exam numbers are never used for tuning.
2. Every channel change = sealed-spec amendment + D-number entry in `memory/decisions.md`
   + `memory/remaining_items.md` update.
3. Restack gates: AUC tolerances + B4 abstention-parity (`ab_new ≤ ab_old + 0.10`).
4. Fail-closed contracts: if a bundle asset is missing, refuse to construct (never
   degrade silently) — except documented fallback ladders.
5. No number without a scope label (D-037) — regenerate `artifacts/benchmarks.json`
   after every promotion.

## The 20 hardcoded-path files (P0b fix list, verified)

17 files with `/Users/Dataghost`: `deepfake/scripts/t1_stage0.sh`, `ffpp/watch_server.sh`,
`p47_chain.sh`, `build_inversion_aug.py`, `audit_neverseen_endpoint.py`,
`train_audio_w9_stacker.py`, `diag_dualdec_codec.py`, `build_audio_w9_rows.py`,
`train_audio_w9_wavlm.py`, `build_audio_w9_diet.py`, `diag_fused_itw.py`,
`smoke_fused_court.py`, `build_audio_w7_diet.py`, `exam_audio_2021_full.py`,
`extract_audio_w9.py`, `run_gate_battery_w9.py`, `tools/bin/exiftool`.
3 more with `/private/var/folders/.../opencode`: `perf_bench.py`, `prep_avlips_videos.py`,
`smoke_fused_court.py` (second occurrence).
