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
**Current booked: 64.4/100** (P0c +0.8 bookkeeping; P2 FAIL adds 0; P8-fast +0.5 data coverage, 2026-11-13).
Conservative P0c floor: 63.5 if the stale-cap correction is disallowed.

## Ledger

| Change | Date | Measured Y (exact) | Court impact | Status |
|---|---|---|---|---|
| Video restack v2→v4e (W4e, D-075) | 2026-11-03 | FF++ 0.7487→**0.9121 (+0.1634)**; per-method 0.9884-0.7917; LOMO 4/4; Celeb-DF 0.9482; DFD 0.8857 (real-elev 0.018, in-bar); dose-response lever A' (dfd_weight_mult 4.0) receipted | OOD-video in-domain leg +0.16 AUC | **DEPLOYED** (incumbent) |
| W-AV v5 lip-sync candidate (D-078) | 2026-11-05 | avlips external **0.6206 < 0.70 sealed bar** (miss 0.0794) → HOLD; 5/6 gates PASS | **0 deployed Y** (HOLD, no promotion); knowledge: lip-sync signal generalizes (offset_abs alone 0.7240) but stacker blend dilutes it | HOLD sealed; assets banked |
| W9 fused audio court p13_fused (D-079/D-080/D-083) | 2026-11-05/07 | **BATTERY MISS → HOLD**: D1 ITW 0.7377 (bar ≥0.95; p4 0.9153) · D2 FPR 0.1404 · D4 fake recall 0.083-0.267 (bar ≥0.84; p4 0.85) · D5 FPR 0.9873 · NR2 3/4 refs −0.11-−0.18 · NR4 no-abstain (0.9595); NR3 Kokoro **8/8 PASS** (incumbent 0/8); NR1-B 0.9225<0.98 (D-080). ROOT CAUSE receipted ×2: (1) sealed diet has **zero LA-spoof rows** (LA = bona-only parity anchor, spec §4) → B reads LA spoofs 0.0095-0.1605 REAL while p4 = 1.0000 on the same files; stacker lets B override p4; fused val 0.9639 was blind (zero LA fakes in val); (2) **bona slice 35% label-poisoned** — 904/2,580 "bona" rows are actual spoofs (parquet-verified; W9 builder sampled the mixed W7 label file unfiltered) | **0 deployed Y — promotion would have REGRESSED the court** (wild AUC −0.11-−0.18, codec recall 0.85→0.08-0.27); sealed bars caught it pre-promotion exactly as designed; p4 untouched throughout | HOLD — formal verdict on 2021 exam completion; **W9.1 diet-repair EXECUTING** (option b, both defects fixed, extraction in flight) |
| Ladder fix (audit F2) | 2026-11-07 | Detection Y = **0** (by design — ladder fires only on OOC/abstained rows). Honesty Y: **72** historical audio verdicts issued through the old ladder; rescored values overwhelmingly ≤0.15 (many 0.000 "confidently authentic" on wild/OOC audio); the measured-bias guard now refuses every real-ward certification (~60+ of 72); ~3-6 fake-ward (0.7492, 0.6611, 0.8496×2) still issue IF raw within the 0.15 confirmation cap. verify_ladder_fix.py 15/15 byte-exact (#608 0.0382, #610 0.5609; #606 image regression byte-equal) | closes the fake→"authenticated authentic" certification path; unifies API/UI (p_fake_adjudicated + banner); 0 AUC delta | **DEPLOYED** |
| D3 T2V channel (D-082) | 2026-11-07 | **MISS**: AUC 0.6959 (bar 0.85), catch 6/27 (bar 15; avlips 0/8), FPR 7.0%, per-generator AP inverted (sora 0.2216); τ=1.029278; 766 clips, 0 unavailable | **0** — nothing wired, deployed court byte-unchanged; banked: deterministic module + harness + content-motion-confound diagnosis | BANKED (not wired) |
| Audit/scorecard correction (2026-11-08) | 2026-11-08 | No court number moved; 3 stale audit claims corrected (D3 97.72 paper-reported; F2 fixed 2026-11-07; F5 drop-only/5MiB stale) | 0 (documentation honesty only) | DONE |
| D3 content-matched T2V re-measure (D-084) | 2026-11-09 | **FAIL G4 / no wiring**: G1 0.926655 PASS (298 fakes; 2 genuine decode fails), G2 0.945461 PASS, G3 23/27 PASS, G4 **37/300 = 0.123333 FAIL** (bar ≤0.10; Wilson 0.0908–0.1654); τ=2.554618 (calib p10; calib FPR 30/300); content matching lifts prior 0.6959 AUC to 0.9267, confirming D-082's confound diagnosis, but real-guard cost is too high. Sentinel: arithmetic reproduced; **REJECT for pre-registration** because `d3_temporal.py` 8/16-frame policy and `fetch_genvideo_val.py` `real/` layout bypass were not in the sealed spec; fail valid as executed, not certifiable. Deployed hashes unchanged; no commit/wiring/training | **0** (no wiring; prior +6.8/+4–5 video-OOD projection closed) | CLOSED FAIL (no wiring); P1b SPLIT only as new pre-registered experiment |
| W9.1 formal closure (D-085) | 2026-11-10 | **HOLD, no promotion**: 7/11 sealed bars FAIL — D1 0.8858<.95; D2 0.3377>.30 and FPR .1667>.10; D4 1/200 in three MP3 arms vs sealed 0.000; D5 DF-Eval bona FPR .7139>.10; NR1 B val .9129<.98 / C pooled .5275<.70; NR2 all four exams; NR4 no abstain. D3/D6/NR3/REG PASS. D4 recall 0.883–0.900 repaired from 0.083–0.267. Receipt: `artifacts/gate_battery_w9_p13bfused/HOLD_CLOSURE.md` | **0**; p4 untouched; no fused serving | CLOSED HOLD; W-AUDIO successor |
| SPLIT Stage 0 motion gate (D-086) | 2026-11-11 | **PASS, marginal**: 2,000/2,000 GenBuster++ video clips (1,000 real / 1,000 fake), **AUC 0.70244** vs sealed PASS ≥0.70; report-only stratified bootstrap 95% CI **0.679236–0.725154**; motion-rebalanced AUC **0.736812** (99/100 bins; 1 class-imbalance bin skipped); motion-only AUC **0.495853** (negative control); 0 decode failures; all `[8,196,768]`, 24 fps, frame indices `[0,9,18,27,36,45,54,63]`. Recipe `3383d30f…3cd78`; scores SHA `fc1e6785…be3d8`; harness SHA `c4e254d6…5f6a94`; deployed p4/p6v11/p6vid_v4 unchanged (manual check; Stage 0 metrics lacks deployed-hash field) | **0 booked**; PASS authorizes a Stage 1 request only. Stage 1 conditional potential **+1.0–3.0 OOD-video** if all bars pass; public ceiling 73–76 | PASS; **Stage 1 later ran and closed FAIL S2 (D-087)**; no wiring/promotion |
| SPLIT Stage 1 content-matched GenVideo (D-087) | 2026-11-11 | **CLOSED FAIL S2**: 2,466/2,466 rows, zero unavailable/decode failures, all `[8,196,768]`; τ **6.120473727840235**; S4 FPR **0.08** (56/700) and S6 JPEG FPR **0.09** (63/700) PASS; **S2 GenVideo AUC 0.513919** vs ≥0.90, **S3 GenVidBench 0.546810** vs ≥0.85, **S5 5/27** vs ≥15, **S7 recall 0.076667/0.057604** vs ≥0.50. Recorded **9-stratum** deviation: `sora` empty (all 56 on-disk sora fakes are D3 SHA exclusions); bars unchanged; S2 is 9-generator, not the sealed 10-generator claim. Stage 0 0.70244 did not transfer; S8 motion-rebalanced 0.479688 report-only. 19.33 min clip latency, 3.72 GiB max RSS. Receipt: `artifacts/split_t2v_matched/stage1/RESULT.md`; scores SHA `f73b95ec…988a77`; metrics `2dcefc3c…ac517`; verdict `94fe36ca…df43a3` | **0 booked**; conditional +1.0–3.0 not realized; 63 remains | CLOSED FAIL S2; no Stage 2, no threshold/per-generator rescue |
| W-AUDIO A0z 1B DF-Eval (D-088) | 2026-11-11 | **AUC PASS / ECE FAIL**: 703/703 existing DF-Eval audio (395 real / 308 fake); 74 missing stems named (56 real / 18 fake), no substitution; frozen 4 s loader; class 0 = fake. **AUC 0.778066** (95% CI **0.742647–0.811393**) vs ≥0.650 PASS; **ECE 0.243034** vs ≤0.05 FAIL (max-confidence sensitivity **0.242634** also FAIL). AUC on logit difference equals p_fake; no class inversion. 345.5 s clip latency, 8.02 GiB max RSS. Guard arms, 13 s, 2B, and A0p not run. Receipt: `artifacts/wa_audio_a0z_1b/RESULT.md`; scores `fb6017e1…e45d0a8`; metrics `01d99c94…582e6b`; verdict `394ea6d8…e4716` | **0 booked**; A0z not certified; conditional +0.9–1.5 not realized | NOT CERTIFIED ECE FAIL; no A0p/A1 |

| P0c scorecard reconciliation | 2026-11-13 | **No detection Y**; receipt-proven **System 11.7→12.5 (+0.8)**: API/UI divergence fixed (`p_fake_adjudicated` + UI bridge), 5 MiB cap proven harness-only (product cap 100 MB), fresh ladder battery 23/23 (#608 0.0382, #610 0.5609, #606 0.8163); historical 63.1 snapshot preserved | **+0.8 bookkeeping**; court **63.9**; conservative floor **63.5** if stale-cap correction disallowed | **BOOKED** |
| P2 video OOD/scope gate | 2026-11-13 | **FAIL G3/G5**: G1 PASS (FF++ 2.71%, CDF 0.39%; issued AUC deltas +0.0009/−0.000008), G2 PASS (AVLips confirm 240/240 fake + 179/179 real, parity 0), G4 PASS (84/84 image-court), G6/G7/G8 PASS; **G3 face-route 48/105 = 45.71% vs ≥55%** (sora 76%, kling 52.5%, **opentora 8/40 = 20% vs ≥45%**); **G5 re-encoded 49/105 = 46.67% vs ≥50%** (diff 0.95pp, agreement 97.1%, image 100%); 0 feature/re-encode errors; stable re-encode → coverage fail, not metadata | **0 booked**; no wiring; projected +0.7–0.8 not realized | **CLOSED FAIL** |
| P8-fast data coverage (D-089) | 2026-11-13 | **PASS D1–D5**: DeepAction v1 rev `476932c3…`: **2,605 MP4s / 6.0592 GB / 7 folders / 100 classes**; license ledger CC BY 4.0 (AI) + Pexels (real) + arXiv:2412.00526; D3 **20/20 full decode** (resolutions 512×512–2160×4096; only 2/20 512²); D4 **70/70, 0 errors**, 52 `video_face` / 18 `image_court`, 14 abstain (20.0%), 56 p-scored; duplicate audit **2,594 unique contents** (5 extra `RunwayML/77 f–j` duplicate `27 a–e`; 11 duplicate files; 699/700 class-content sets). DF-Eval video draft updated, **not submitted** | **+0.5 data coverage** (dim 5 4.5→5.0); court **64.4**; no detection claim; +0.5 DF-Eval-video conditional not booked | **BOOKED** |

## Ceiling adjustment (honest)

The scorecard's recovery budget credited D3 +6.8 — **refuted on our corpora** (D-082
content-motion confound: the paper's benchmark is content-matched; our real arm is
static talking heads). Ladder/surface +4.0 — **partially banked** (defect closed;
the 84% audio over-abstention remains). Updated realistic recovery with public data
under the exam-only training scope: audio finetune ~+4.3 + W9 deployment ~+2-3 +
data acquisition +2.5 + AV joint +1.5 + robustness +2.0 + ladder remainder ~+2 ≈
**+14-15 → ceiling ~77-78/100** **before** the D3 content-matched result. D3 is now
**closed FAIL G4 (D-084)**, SPLIT is **closed FAIL S2 (D-087)**, and A0z 1B is
**AUC-pass/ECE-fail (D-088)**, so the measured court remains **63/100**. The
**73-76** figure is a theoretical public-data ceiling, not a current measured path;
77-78 remains conditional on a new pre-registered video-OOD channel clearing its
bars. The remaining ~22 is structural (no public corpus for 2026-frontier
generators; adversarial arms race; watermark layer compromised). 100/100 is not
achievable by anyone with public data today.

## Skip list (no confirmed Y — do not spend compute)

- **D3 content-matched re-measure** — **DONE 2026-11-09, FAIL G4 (D-084)**; content
  matching fixed the confound (AUC 0.6959→0.9267) but real FPR 12.33% breaks the
  ≤10% guard. No wiring. Do not rescue by post-hoc threshold tuning; any retry needs
  a new pre-registered split and spec.
- **D3 motion-normalized variant** — research-grade, no documented prior, uncertain Y.
- **SPLIT Stage 2 / post-hoc rescue** — **CLOSED FAIL S2 (D-087)**. S2 0.5139,
  S3 0.5468, S5 5/27, S7 recall 0.0767/0.0576; no threshold or per-generator
  rescue. The 9-stratum sora-empty deviation is recorded, not a silent pass.
- **A0z post-hoc calibration / unapproved A0p** — **ECE 0.2430 FAIL (D-088)**.
  No temperature scaling, threshold change, or training. A0p requires separate
  explicit training approval; A0z is not certified.
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
| 3 | **W-AUDIO A0z 1B DF-Eval (DONE 2026-11-11)**: 703/703 existing clips; AUC **0.778066** PASS, ECE **0.243034** FAIL; A0z not certified. Guard arms/13 s/2B/A0p not run | **0 booked**; conditional +0.9–1.5 not realized; no A0p/A1 | done 6.2 min CPU; any next audio spend requires fresh approval |
| 4 | **SPLIT Stage 1 (DONE 2026-11-11, CLOSED FAIL S2)**: 2,466/2,466 rows; S2 **0.513919**, S3 **0.546810**, S5 **5/27**, S7 **0.076667/0.057604**; recorded 9-stratum `sora`-empty deviation | **0 booked**; conditional +1.0–3.0 not realized; 63 remains | done 19.5 min CPU; no Stage 2 |
| 5 | Image coverage wave (lowest priority — image is the most mature dimension) | moderate external-generator coverage gain | full restack cycle |

## Rules for future entries

1. Append one row per change with MEASURED numbers (receipts on disk, cited).
2. Expected-Y is written down before the work; measured-Y is booked after. Gaps
   escalate per D-046 (exact numbers, no waivers).
3. Anything with 0 measured court Y and no honesty/insurance justification gets
   moved to the skip list, not retried.
