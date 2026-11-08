# Improvement Ledger — measured Y per change

**Law (user directive 2026-11-07):** every change records its MEASURED Y before it
counts as improvement. New work is planned as "do X → gain Y" with Y quantified from
documented evidence BEFORE compute is spent. No confirmed-Y path → SKIP (no compute,
no time). Re-measures that cannot move a court number are insurance at best — they
must be justified as anti-regression, never booked as improvement.

**Baseline: 63/100** (artifacts/audit_20261106/SOTA_SCORECARD.md, 2026-11-06).
Dimensions: in-domain 14.4/15 · OOD-image 8.5/10 · OOD-audio 5.0/10 · OOD-video 1.2/10
· abstention 11.3/15 · robustness 4.5/10 · data coverage 4.5/10 · system 11.7/15 ·
multimodal 2.0/5.

## Ledger

| Change | Date | Measured Y (exact) | Court impact | Status |
|---|---|---|---|---|
| Video restack v2→v4e (W4e, D-075) | 2026-11-03 | FF++ 0.7487→**0.9121 (+0.1634)**; per-method 0.9884-0.7917; LOMO 4/4; Celeb-DF 0.9482; DFD 0.8857 (real-elev 0.018, in-bar); dose-response lever A' (dfd_weight_mult 4.0) receipted | OOD-video in-domain leg +0.16 AUC | **DEPLOYED** (incumbent) |
| W-AV v5 lip-sync candidate (D-078) | 2026-11-05 | avlips external **0.6206 < 0.70 sealed bar** (miss 0.0794) → HOLD; 5/6 gates PASS | **0 deployed Y** (HOLD, no promotion); knowledge: lip-sync signal generalizes (offset_abs alone 0.7240) but stacker blend dilutes it | HOLD sealed; assets banked |
| W9 fused audio court p13_fused (D-079/D-080/D-083) | 2026-11-05/07 | **BATTERY MISS → HOLD**: D1 ITW 0.7377 (bar ≥0.95; p4 0.9153) · D2 FPR 0.1404 · D4 fake recall 0.083-0.267 (bar ≥0.84; p4 0.85) · D5 FPR 0.9873 · NR2 3/4 refs −0.11-−0.18 · NR4 no-abstain (0.9595); NR3 Kokoro **8/8 PASS** (incumbent 0/8); NR1-B 0.9225<0.98 (D-080). ROOT CAUSE receipted ×2: (1) sealed diet has **zero LA-spoof rows** (LA = bona-only parity anchor, spec §4) → B reads LA spoofs 0.0095-0.1605 REAL while p4 = 1.0000 on the same files; stacker lets B override p4; fused val 0.9639 was blind (zero LA fakes in val); (2) **bona slice 35% label-poisoned** — 904/2,580 "bona" rows are actual spoofs (parquet-verified; W9 builder sampled the mixed W7 label file unfiltered) | **0 deployed Y — promotion would have REGRESSED the court** (wild AUC −0.11-−0.18, codec recall 0.85→0.08-0.27); sealed bars caught it pre-promotion exactly as designed; p4 untouched throughout | HOLD — formal verdict on 2021 exam completion; **W9.1 diet-repair EXECUTING** (option b, both defects fixed, extraction in flight) |
| Ladder fix (audit F2) | 2026-11-07 | Detection Y = **0** (by design — ladder fires only on OOC/abstained rows). Honesty Y: **72** historical audio verdicts issued through the old ladder; rescored values overwhelmingly ≤0.15 (many 0.000 "confidently authentic" on wild/OOC audio); the measured-bias guard now refuses every real-ward certification (~60+ of 72); ~3-6 fake-ward (0.7492, 0.6611, 0.8496×2) still issue IF raw within the 0.15 confirmation cap. verify_ladder_fix.py 15/15 byte-exact (#608 0.0382, #610 0.5609; #606 image regression byte-equal) | closes the fake→"authenticated authentic" certification path; unifies API/UI (p_fake_adjudicated + banner); 0 AUC delta | **DEPLOYED** |
| D3 T2V channel (D-082) | 2026-11-07 | **MISS**: AUC 0.6959 (bar 0.85), catch 6/27 (bar 15; avlips 0/8), FPR 7.0%, per-generator AP inverted (sora 0.2216); τ=1.029278; 766 clips, 0 unavailable | **0** — nothing wired, deployed court byte-unchanged; banked: deterministic module + harness + content-motion-confound diagnosis | BANKED (not wired) |
| Audit/scorecard correction (2026-11-08) | 2026-11-08 | No court number moved; 3 stale audit claims corrected (D3 97.72 paper-reported; F2 fixed 2026-11-07; F5 drop-only/5MiB stale) | 0 (documentation honesty only) | DONE |

## Ceiling adjustment (honest)

The scorecard's recovery budget credited D3 +6.8 — **refuted on our corpora** (D-082
content-motion confound: the paper's benchmark is content-matched; our real arm is
static talking heads). Ladder/surface +4.0 — **partially banked** (defect closed;
the 84% audio over-abstention remains). Updated realistic recovery with public data
under the exam-only training scope: audio finetune ~+4.3 + W9 deployment ~+2-3 +
data acquisition +2.5 + AV joint +1.5 + robustness +2.0 + ladder remainder ~+2 ≈
**+14-15 → ceiling ~77-78/100**. The remaining ~22 is structural (no public corpus
for 2026-frontier generators; adversarial arms race; watermark layer compromised).
100/100 is not achievable by anyone with public data today.

## Skip list (no confirmed Y — do not spend compute)

- **D3 content-matched re-measure** — tests the paper's reachability, does not move
  the deployed court (our real arm would still hit the confound).
- **D3 motion-normalized variant** — research-grade, no documented prior, uncertain Y.
- **T2V training wave** — GenVidBench-class training data is exam-only (scope-blocked).
- **Standalone 2021 full-scoring** — folded into W9 STEP 6 as anti-regression
  insurance (subsampled per D-081 resolution: reals-complete + 50k seeded fakes,
  SE 0.00103 vs full 0.00071, both far inside the D6/NR2 tolerances). NOT booked as
  improvement; full scoring remains available on explicit user word.
- **AV-Deepfake1M++ (1.46TB) / DeepSpeak (44.7GB) pulls** — only feed a future
  video-AV diet wave (D-078: feasible ~2 days, payoff UNCERTAIN); deferred until the
  user prioritizes it over audio.

## Forward plan (X → Y)

| # | X (action) | Y (expected, sourced) | Cost |
|---|---|---|---|
| 1 | W9 STEP 6: gate battery + 2021 subsampled exam → sealed verdict | **MEASURED: HOLD** (D1/D2/D4/D5/NR2/NR4 miss; NR3 8/8; root cause = zero LA-spoof rows in the sealed diet — D-083 addendum). STEP 7 promotion moot on HOLD | done (2021 leg completing the formal receipt) |
| 2 | **W9.1 diet-repair (USER-AUTHORIZED, option b)**: extract LA-spoof 2,580 utts (same 2019-train parquet; absent from caches) + keep MLAAD + RADAR bona 6,000→8,580 + retrain B/stacker + full battery re-run. Construction receipts (D-083 addendum 3): v0 manifest had silently RESAMPLED 9,030 W9 rows (single RNG stream) and 96 spoof uids collided with the W9 cache → builder re-anchored on the sealed W9 manifest; final delta = 3,865 NEW uids (radar bona 2,588 + la bona 904 + MLAAD attrition retry 277 + la spoof top-up 96), 2,484/2,580 paid spoof extractions preserved. Delta extraction = data/caches/w9c (DONE: 3,579 new uids cached, 286 bad = 277 MLAAD + 9 radar bona, coverage 22,034/22,320 = 98.7%, parity 919/919); no bars touched | D4 recall 0.08→~0.85 (root cause removed — B sees the family it was blind to); D6/NR2-2021 → ~p4's 0.912; wild bars D1/D2/D5 UNCERTAIN (B's wild drift may persist) — bars measure | extraction ~0.5h MPS (3,865 utts, w9c) + rows minutes + retrain fast + battery ~1h (2021 leg re-scores from feature caches). **MEASURED 2026-11-07: D4 recall 0.8833/0.8833/0.9000/0.9000/0.8833 — all arms PASS (W9 0.083-0.267); D4 FPR 0.0050 in 3 MP3 arms (1/200, one file), 0.0000 FLAC/Opus — formal MISS by 1/200; B val AUC 0.9129 (NR1 miss); fused val AUC 0.9620; smoke PASS** |
| 3 | W-AUDIO wave (only if W9.1 wild bars still miss): wild reals in diet (ITW bona 19,963 on disk) + MLAAD v11 + codec augs; bars pre-registered BEFORE training | D2/D5 wild-real FPR repair; DF-Eval 0.5593→0.65-0.75 (uncertain — measured-only, escalate on miss) | training hours + exam re-runs |
| 4 | Image coverage wave (lowest priority — image is the most mature dimension) | moderate external-generator coverage gain | full restack cycle |

## Rules for future entries

1. Append one row per change with MEASURED numbers (receipts on disk, cited).
2. Expected-Y is written down before the work; measured-Y is booked after. Gaps
   escalate per D-046 (exact numbers, no waivers).
3. Anything with 0 measured court Y and no honesty/insurance justification gets
   moved to the skip list, not retried.
