# Argus Prime — Remaining Items Inventory

**Last updated:** 2026-11-14
**Status:** execution update 2026-11-14: P0c booked; P2 closed FAIL; P8-fast
booked +0.5 data coverage; P4 oracle K1 FAIL and closed at 0; **P5 codec/
laundering kill-test GAP and closed at 0**; **P9 read-only C2PA validator PASS
and booked +0.4 system (0 detection)**. No training, wiring, promotion,
pointer change, or commit is authorized by this file.
**Baseline:** **63.1/100** weighted distance-to-frontier. This is not percent-complete.
**Current booked court:** **64.8/100** (P0c +0.8 bookkeeping; P2 0; P8-fast +0.5;
P4 0; P5 0; P9 +0.4 system).
**Current measured state:** D3 closed FAIL G4; W9.1 closed HOLD; SPLIT Stage 1
closed FAIL S2; A0z 1B DF-Eval AUC pass / ECE fail and **not certified**; P2
closed FAIL G3/G5; P4 closed FAIL K1 (oracle ceiling); **P5 closed GAP (audio
8 kHz and video CRF28; image q35/resize pass)**; P8-fast PASS D1–D5
(coverage only; no detection score); **P9 PASS B1-B15 (transparency-only;
0 detection)**.

## A. Immediate bookkeeping / reconciliation

| ID | Item | Status | Contribution | ETA | Notes |
|---|---|---|---|---|---|
| P0c | Formal scorecard rescore after Phase 0; ledger reconciliation | **Done 2026-11-13** | **+0.8 booked** (System 11.7→12.5); no detection claim | done | Receipt: `artifacts/scorecard_20261111/P0C_RECONCILIATION.md`; historical 63.1 preserved; conservative floor 63.5. |

## B. Score-bearing build candidates

The roadmap cumulative projection is a portfolio estimate, not a measured receipt.
A failed gate stops dependent rows.

| ID | Item | Status | Standalone contribution | ETA | Projected court (item-only, from 63.1) | Gate / dependency |
|---|---|---|---|---|---|---|
| P2 | Video OOD abstention/scope gate; abstain when video evidence is off-manifold; surface reason code | **CLOSED FAIL G3/G5 (2026-11-13)** | **0 booked**; G3 face-route 45.71% vs ≥55%; opentora 20% vs ≥45%; G5 re-encoded 46.67% vs ≥50% | done | 63.9 (P0c only); P2 adds 0 | Receipt: `artifacts/p2_video_scope/RESULT.md`; no wiring; retry needs a new pre-registered signal, not a lower bar. |
| P4 | Audio coverage-risk/calibration repair; the 84% gate masks raw FPR@0.5 **0.7949**; do not lower abstention blindly | **CLOSED FAIL K1 (2026-11-13)** | **0 booked**; oracle on existing p_fake+z cannot reach 30% coverage at FPR_c≤0.10 and R_all≥0.40 — no feasible policy at any coverage; best FPR_c at coverage≥30% is 0.333 (sealed grid) / 0.326 (0.005 sensitivity) | done | 64.4 (no change) | Receipt: `artifacts/p4_oracle/RESULT.md`; sealed spec + hashes; no calibration, wiring, or threshold change. Retry needs a better score/new signal, not a lower bar. |
| P5 | Robustness: H.264/CRF, JPEG, MP3/Opus, resampling; real codec diversity; measure retention | **CLOSED GAP (2026-11-14)** | **0 booked**; audio MP3/Opus PASS but **8 kHz FAIL R1/R2/R4** (AUC 0.9153→0.6708, coverage 0.937→0.252, raw FPR 0.093→0.940; all issued rows fake-ward); video **CRF28 FAIL R3 only** (AUC 0.9072→0.8580, TPR 0.729→0.561; R1 marginal); image q35/resize PASS (saturated spot-check) | done | 64.4 (no change) | Receipt: `artifacts/p5_robustness/RESULT.md` + `RESULT_ADDENDUM.md`; 3,840 rows, 0 errors/missing, hashes/recompute PASS. Codec-diverse retraining eligible only via separate sealed spec + explicit approval; no training/wiring. |
| P6 | AV sync as a separate `offset_abs` gate, not a stacker blend | **CLOSED KILL B3 (2026-11-14)** | **0 booked**; FP **106/358 = 29.6089%** vs ≤35 (Wilson 95% 0.251165–0.345343); TP 350/480 = 72.9167% PASS; AUC 0.724002 parity PASS; coverage 838/838 PASS; tau 2.5 frames; `>=` 107 FP / 375 TP report-only | done | 64.8 (no change) | Receipt: `artifacts/p6_av_gate/run/RESULT.md`; spec `p6-av-offset-gate-v1` sha `987846a7…`; sentinel APPROVE; no wiring/promotion/threshold change. Disclosed fps confound (RAVDESS 29.97 vs AVLips 25) and missing AVLips license. Retry needs target-domain calibration/normalization or a new signal, not a lower bar. |
| P7 | Image backbone audit + provider-real scalar calibration (DINOv3-class check); no per-generator heads | **CLOSED AT ZERO (2026-11-14)** | **0 booked**; 28.70 pp oracle loss at 10% FPR; LOCO FPR 0.1200/0.0300/0.1717; split 97.5th pct 0.1800; A/B1/B2/B3 false | done | 64.8 (no change) | Receipt: `artifacts/p7_image_audit/AUDIT_RECEIPT.md`; spec `p7-image-audit-v1` sha `a4f25c05…`; sentinel APPROVE; no wiring/promotion/threshold change. Parked: DINOv3 ViT-B/16 audit (license acceptance + new D-045-style spec + D-044-class contract re-derivation 224/CLS vs 256/avg) and a disjoint license-clean provider-real corpus. |
| P8 | Data coverage: DeepAction acquired; DF-Eval-2024 video gate request; license ledger | **P8-fast done 2026-11-13 (+0.5)**; P8a open; P8b deferred | **+0.5 booked**; +0.5 conditional on DF-Eval video, not booked | done + external gate | **64.8 current**; 65.3 if DF-Eval video lands | DeepAction 2,605 files / 2,594 unique; CC BY 4.0 + Pexels; DF-Eval video gated/eval-only; no training |
| P9 | System/transparency: C2PA read-only validator, security/ops checks, test/CI, model cards | **DONE 2026-11-14** | **+0.4 system booked**; 0 detection | done | **64.8 current** | Receipt: `artifacts/p9_system/RESULT.md`; spec `p9-c2pa-readonly-v1` sha `c6b551c7...`; 24 tests, 0 skips; fixture3 `well_formed_valid`/untrusted/hard-bound; tampered `invalid`/hard false; truncated `error`; trusted synthetic-only; no gate/wiring. |
| P3 | W-AUDIO A0/A1: A0z 1B DF-Eval done; AUC pass, ECE fail; not certified. Guard arms/13 s/2B/A0p/A1 open | Conditional; not certified | A0 pass +0.9–1.5 OOD-audio; A1 +1.5–2.5 | A0z 1B done; sub-arms separate | 63 now; 64.0–64.6 if A0; 64.6–65.6 if A1 | AUC pass alone is insufficient; ECE fail; A0p 1B linear probe ITW ECE 26.9% (Pascu 2024) makes the 0.05 bar unlikely; no training without explicit approval |

## C. Approval-gated sub-items (do not start without explicit user approval)

| ID | Sub-item | Status | Why it is not automatic |
|---|---|---|---|
| P3a | A0z guard arms: ASVspoof 2021 LA, ITW 600, codec pooled FPR, codec AUC drop | Not run | A0z ECE already fails; guards cannot repair DF-Eval calibration; separate approval required |
| P3b | A0z 13 s secondary | Not run | 4 s primary only; report-only secondary; approval required |
| P3c | A0z 2B checkpoint | Not run | 1B only; model-card AUC is lower for 2B; approval required |
| P3d | A0p frozen-feature `<2k` linear probe | Not run | Training approval required; A0z not certified; A0p is the pre-registered fallback for an AUC miss, not an ECE fail; 1B linear-probe ITW ECE 26.9% (Pascu 2024) makes ≤0.05 unlikely |
| P3e | A1 full audio candidate | Not run | Depends on a certified A0; full W9.1 bars; promotion approval |
| P6a | W-AV v5 / AV offset new spec | **CLOSED KILL (2026-11-14)** | Standalone `offset_abs` gate failed B3 (106/358 real FP); no blend, no wiring; any retry needs a new pre-registration. |
| P8a | DF-Eval-2024 video gate registration | Draft updated 2026-11-13; **not submitted** | User action / external approval; eval-only; institutional email or verifiable link |
| P8b | AV-Deepfake1M++ / DeepSpeak acquisition | Deferred | Gated/heavy; only feed a future video-AV diet wave; payoff uncertain |

## D. Open user decisions / engineering backlog (not score-bearing by themselves)

| ID | Item | Status | Notes |
|---|---|---|---|
| U1 | Audio ship-label decision | Blocked on user | ASVspoof-family banner only; no label change without ruling |
| U2 | `.aiff` dispatch | Open user item | Decoder/dispatch behavior; not a score move |
| U3 | CLI XLS-R reload performance backlog | Open | ~7.5 s warm reload per file; performance, not detection |
| U4 | Any model training / promotion / pointer movement | Not authorized | Requires explicit approval and pre-registered bars |
| U5 | Any commit of assistant changes | Not authorized | Leave uncommitted unless user asks |

## E. Closed / HOLD / do not restart

| Item | State | Reason |
|---|---|---|
| D3 content-matched T2V | CLOSED FAIL G4 | Real guard 0.1233 > 0.10; no wiring; retry needs new pre-registration |
| D3 motion-normalized variant | SKIP | Research-grade, no documented prior, confound already diagnosed |
| D3 wiring | CLOSED | P0 failed; +4–5 projection removed |
| SPLIT Stage 0 | Historical marginal PASS | 0.70244 vs 0.70; CI 0.679236–0.725154 |
| SPLIT Stage 1 | CLOSED FAIL S2 | S2 0.513919, S3 0.546810, S5 5/27, S7 0.076667/0.057604; 9-stratum sora-empty deviation; no Stage 2 |
| SPLIT Stage 2 / threshold rescue | SKIP | Near-chance aggregate; no post-hoc rescue |
| W9.1 | CLOSED HOLD | 7/11 sealed bars fail; no promotion; p4 remains |
| P5 codec/laundering kill-test | CLOSED GAP | Audio 8 kHz fails R1/R2/R4 (coverage 0.252; issued FPR 100%); video CRF28 fails R3 (TPR 0.729→0.561); image q35/resize pass; 0 booked; retraining only via separate sealed spec + approval |
| P9 read-only C2PA validator | DONE/BOOKED | +0.4 system; transparency-only; no gate/wiring; 24 tests, 0 skips |
| A0z 1B DF-Eval | NOT CERTIFIED | AUC 0.778066 PASS, ECE 0.243034 FAIL; 0 booked |
| A0z post-hoc calibration | SKIP | No temperature scaling/threshold change; would be a new experiment |
| HOLA reproduction / AV-Deepfake1M++ frozen featurization | SKIP | 8×A100, no public HOLA weights, 1.46 TB gated data; expected Y ≈ 0 |
| Per-generator image heads | SKIP | No head-to-head evidence; data alignment/backbone dominates |
| DeepSpeak v2 training corpus | SKIP | Gated/custom license/ND ambiguity |
| ASVspoof5 standalone pull | SKIP | 142.3 GB; no standalone confirmed Y |
| Full 2021 audio scoring | Insurance only | Subsample inside tolerance; not a score move |
| T2V training wave | SKIP for now | GenVidBench exam-only; no justified wave before a new measured hypothesis |
| AV diet diversification (SadTalker/VideoReTalking/MakeItTalk) | SKIP | W-AV v5 HOLD; uncertain payoff |
| Watermark detection as score feature | SKIP | Mark-to-frame attacks; transparency only |
| Frequency-invariance architecture | SKIP | 2026 controlled result: no marginal over plain EfficientNet-B0 |
| 2026-frontier red-team only | SKIP | No public corpus; unweighted probes are not a score |
| Chasing 100/100 | SKIP | ~22 points structural with public data |

## F. Research shortlist for “best next item”

1. **P2 — CLOSED FAIL G3/G5; 0 booked.** No production wiring; the original
   +0.7–0.8 expectation is not realized.
2. **P8-fast — DONE 2026-11-13; +0.5 data coverage booked.** Remaining P8 leg:
   DF-Eval-2024 video gate (user action; **not submitted**).
3. **P4 — CLOSED FAIL K1; 0 booked.** The oracle on existing `p_fake`+z
   cannot reach 30% coverage at FPR_c≤0.10 and R_all≥0.40; no feasible policy at
   any coverage (best FPR_c at coverage≥30% = 0.333). The 84% abstention is not
   the defect to “fix”; the score (AUC 0.5593) is. No calibration/wiring. Receipt:
   `artifacts/p4_oracle/RESULT.md`.
4. **P5 — CLOSED GAP; 0 booked.** Audio MP3/Opus pass, but 8 kHz resampling
   collapses coverage to 0.252 and issues 100% fake-ward; video CRF28 loses
   16.8 pp TPR at 0.5 (Face2Face/NeuralTextures); image q35/resize pass
   (saturated spot-check). Retraining only via a separate sealed spec + explicit
   approval. **P9 is DONE (+0.4 system booked; 0 detection)**, **P6 is
   CLOSED KILL B3 (0 booked; D-093)**, and **P7 is CLOSED AT ZERO
   (2026-11-14; D-094)**, so no no-wiring roadmap item remains; P3d is
   low-probability.
5. **P3d — A0p** (unlikely to fix ECE: 1B linear-probe ITW ECE 26.9%, Pascu 2024;
   training approval required; not next).
6. **P0c — scorecard reconciliation** (0–0.8 bookkeeping; not “next level”).

## G. Current score and honest ceiling

- Measured court: **64.8/100**.
- 73–76 is a theoretical public-data ceiling, not a current measured path.
- 77–78 requires a new pre-registered video-OOD channel clearing its bars.
- 100/100 is not achievable with public data today.

## Sources

- `artifacts/roadmap_20261109/ROADMAP.md`
- `memory/improvement_ledger.md` (D-084, D-085, D-087, D-088)
- `artifacts/prereg_audit_20261110.md`
- `artifacts/split_t2v_matched/stage1/RESULT.md`
- `artifacts/wa_audio_a0z_1b/RESULT.md`
- `memory/decisions.md` for open user decisions/backlog

## H. Research synthesis — 2026-11-11 (not executed)

**Decision state:** awaiting user confirmation. No training, wiring, promotion, pointer
change, or commit.

### Ranked next-item view

| Rank | Item | Expected court gain | ETA | Confidence | Why |
|---|---|---:|---|---|---|
| 1 | **P2 video OOD/scope gate** | **0 booked** | done | — | CLOSED FAIL G3/G5; no wiring; projected +0.7–0.8 not realized. |
| 2 | **P8-fast data coverage** | **+0.5 booked** | done 2026-11-13 | — | DeepAction acquired (2,605 files / 2,594 unique; CC BY 4.0 + Pexels); D4 70/70, 0 errors; DF-Eval video gate remains user action. |
| 3 | **P4 audio oracle coverage–risk test** | **0 booked** | done 2026-11-13 | — | **CLOSED FAIL K1.** Existing p_fake+z cannot reach 30% coverage at FPR_c≤0.10 and R_all≥0.40; no feasible policy at any coverage; best FPR_c at coverage≥30% = 0.333 (sealed) / 0.326 (0.005 sensitivity). The 84% gate is not the defect; AUC 0.5593 is. No calibration/wiring. Receipt: `artifacts/p4_oracle/RESULT.md`. |
| 4 | **P5 robustness** | **0 booked** | done 2026-11-14 | — | **CLOSED GAP.** Audio 8 kHz fails R1/R2/R4 (AUC 0.6708, coverage 0.252, raw FPR 0.940; all 151 issued rows p≥0.5, with issued FPR 68/68 and issued TPR 83/83); video CRF28 fails R3 only (TPR 0.729→0.561; Face2Face/NeuralTextures collapse); image q35/resize pass but baseline AUC is saturated. Retraining only via separate sealed spec + explicit approval; no training/wiring. Receipt: `artifacts/p5_robustness/RESULT.md` + `RESULT_ADDENDUM.md`. |
| 5 | **P3d A0p** | ~0 | 1 day + training approval | Low | A0z passed AUC; A0p is the pre-registered fallback for an AUC miss, not ECE. The same frozen XLS-R + logistic-regression recipe reports ITW ECE **26.9% (1B) / 16.1% (2B)** in Pascu et al. 2024; A0p is 1B, so 0.05 ECE is unlikely. |
| 6 | **P0c scorecard reconciliation** | 0–0.8 bookkeeping | 0.5 day | High | Books already-achieved credit; not a detection improvement. Do not call it “next level.” |

### Load-bearing evidence

- **P2 local defect:** `artifacts/audit_20261106/endpoint_summary_corrected.json` —
  `video:avlips_fake` 8 rows, **0 abstained**, mean `p_fake` 0.0109; `video:avlips_real`
  8 rows, 0 abstained. `artifacts/audit_20261106/SOTA_SCORECARD.md` F1: T2V+lip-sync
  catch 1/27, 0% abstention.
- **P2 shortcut risk:** Cakiroglu et al., *Where the Real Clips Came From: Corpus
  Identity in AI-Generated Video Detection*, arXiv:2606.31004 (v2, 3 Nov 2026):
  frame-type counts alone separate real/generated at **AUC 0.999** on GenVidBench;
  re-encoding every clip with one setting at fixed length still leaves **0.61–0.63**
  on three benchmarks; a fixed frame-type pattern removes the leak. Therefore P2 must
  include a **re-encoded T2V control**; metadata/scope features alone are not evidence.
- **P2 base rate:** the train-real-only coverage principle worked on audio (D-035:
  wild audio 2–3× off-manifold, 84% abstention) but failed twice on image (D-037
  CLIP z-norm; D-047 kNN/contradiction both NOT-AVAILABLE). Local base rate 1/3.
- **P4 local defect:** `artifacts/benchmarks.json` audio `scope_external_wild`:
  raw AUC 0.5593, **FPR@0.5 0.7949**, coverage gate abstention **0.839**, in-coverage
  residue AUC 0.6204. Lowering abstention without a better score is a product
  regression, not a score gain.
- **P4 method caution:** Salem et al., *Hierarchical Group-Conditional Conformal Risk
  Control for Selective Prediction in Language Models*, arXiv:2607.24562 (27 Jul 2026):
  standard CRC can violate per-group risk under mild group-composition shift; per-group
  thresholds/group-conditional control are required. This is LLM evidence, not direct
  audio-deepfake evidence; treat as method caution, not proof.
- **P3d calibration evidence:** Pascu et al., *Towards generalisable and calibrated
  audio deepfake detection with self-supervised representations*, arXiv:2309.05384v2
  (Interspeech 2024), Table 3/4: XLS-R 1B + logistic regression has ITW ECE **26.9%**;
  2B has **16.1%**. The bar is ≤5%. A0p is a 1B frozen-feature linear probe.
- **P8 access:** DeepAction v1 acquired 2026-11-13: **2,605 MP4s / 6.0592 GB /
  7 folders / 100 classes**; **2,594 unique contents** (5 extra `RunwayML/77 f–j`
  duplicate `27 a–e`; 11 duplicate files total); AI folders **CC BY 4.0**, real
  Pexels folder under the Pexels license; citation arXiv:2412.00526.
  Deepfake-Eval-2024 video: gated, CC-BY-SA-4.0, **evaluation only**; draft updated
  2026-11-13 but **not submitted** (institutional email or verifiable link
  required).
- **P5 evidence:** roadmap/Combei (arXiv:2506.09606, Interspeech 2025): data curation/
  pruning is the strong lever; codec augmentation was null-to-negative on wild audio.
  Measurement/operating-point changes are not a substitute for codec-diverse training.

### Recommendation

**P2 is closed FAIL (0 booked); P8-fast is done (+0.5 booked); P4 is closed
FAIL K1 (0 booked); P5 is closed GAP (0 booked); P9 is done and booked
(+0.4 system, 0 detection); P6 is closed KILL B3 (0 booked).** P5 measured a
real codec/laundering gap, but the gap is not an improvement receipt. **P7**
image backbone + provider-real calibration has since closed at 0 (D-094;
2026-11-14), so no no-wiring roadmap item remains; the next choice is a user
ruling: (a) draft a separate sealed codec-diverse retraining proposal (no
training until explicitly approved), or (b) the license-gated DINOv3 ViT-B/16
audit or a disjoint license-clean provider-real corpus. **P3d** is not next
because A0z passed AUC and the 1B linear-probe ECE evidence is poor. The
DF-Eval-2024 video gate is a user action; a grant can support a separate +0.5
but is not booked now.

**P7 closed at 0 (2026-11-14).** No no-wiring roadmap item remains. Reopen P7
only if: (a) the user accepts the DINOv3 License and approves a new ViT-B/16
D-045-style audit with the joint gate; or (b) a disjoint license-clean
provider-real corpus is acquired and a new spec is approved; or (c) a new frozen
backbone passes D-045 G1/G2. The audit licenses no AUC claim, no generalization,
and no court movement; court stays **64.8/100**.

**Execution has started; see the 2026-11-14 update below.**

## 2026-11-13 execution update

- **P0c — DONE (bookkeeping).** Receipt-proven **+0.8** System rescore
  (11.7→12.5/15); API/UI divergence fixed, 5 MiB cap proven harness-only,
  fresh ladder battery 15/15. Court after P0c: **63.9/100**; conservative floor
  **63.5** if the stale-cap correction is disallowed. Receipt:
  `artifacts/scorecard_20261111/P0C_RECONCILIATION.md`.
- **P2 — CLOSED FAIL G3/G5; 0 booked.** G1 PASS (FF++ 2.71%, CDF 0.39%),
  G2 PASS (AVLips confirm 240/240 fake + 179/179 real), G4 PASS (84/84
  image-court), G6/G7/G8 PASS. **G3 face-route 48/105 = 45.71% vs ≥55%**
  (sora 76%, kling 52.5%, **opentora 8/40 = 20% vs ≥45%**); **G5 re-encoded
  49/105 = 46.67% vs ≥50%** (difference 0.95pp, agreement 97.1%, image 100%).
  Zero feature/re-encode errors; the failure is coverage, not metadata. No
  production wiring. Receipt: `artifacts/p2_video_scope/RESULT.md`.
- **P8-fast — DONE; +0.5 data coverage booked.** DeepAction v1 acquired at
  revision `476932c3…`: 2,605 MP4s / 6.0592 GB / 7 folders / 100 classes; D3
  20/20 full decode; D4 70/70, 0 unhandled errors (52 `video_face` / 18
  `image_court`, 14 abstain = 20.0%); license ledger CC BY 4.0 + Pexels.
  Duplicate audit: 2,594
  unique contents (5 extra `RunwayML/77 f–j` duplicate `27 a–e`; 11 duplicate
  files). DF-Eval video draft updated but **not submitted**. Receipts:
  `artifacts/p8_data_coverage/RESULT.md`,
  `artifacts/scorecard_20261111/P8_RECONCILIATION.md`.
- **P4 — CLOSED FAIL K1; 0 booked.** Sealed oracle test on the deployed
  DF-Eval audio exam (703 clips, 395 real / 308 fake; AUC 0.5593, 95% CI
  0.5176–0.6022). No global policy on `(p_fake, z_norm)` reaches **FPR_c≤0.10**
  with **R_all≥0.40** at any coverage; K1 coverage is **0%**, not 30%. Best
  FPR_c at coverage≥30% is **0.333** on the sealed 0.01 grid and **0.326** on a
  post-seal 0.005 sensitivity grid — both far above 0.10. Deployed point:
  coverage 113/703 = 16.07%; FPR_c 35/58 = 0.6034; TPR_c 38/55 = 0.6909;
  **R_all 38/308 = 0.1234** (the old “TPR 0.6909” was conditional, not overall).
  No K2 cross-fit, no group oracle, no calibration/wiring/training. Receipt:
  `artifacts/p4_oracle/RESULT.md`; spec/script hashes in `seal.txt`.
- **Current measured court: 64.8/100** (P0c +0.8; P2 0; P8-fast +0.5; P4 0; P5 0;
  P9 +0.4 system).
  Historical 63.1 snapshot preserved. Any new video-OOD attempt needs a new
  pre-registered signal and explicit user approval; do not lower P2's bars.

## 2026-11-14 execution update

- **P5 — CLOSED GAP; 0 booked.** Sealed read-only codec/laundering kill-test:
  3,840 rows (audio ITW 600×4; video FF++ C23 420×2; image 100/100×3), **0
  errors, 0 missing**, all fixed-input/model hashes and baseline recomputes PASS
  (video max |Δ| 0.000050; audio orig re-score max |Δ| 0.000000). No media
  retained; 3,840 transform receipts. Receipts: `artifacts/p5_robustness/
  RESULT.md` and `RESULT_ADDENDUM.md`; spec/script hashes in `seal.txt`.
- **Audio:** MP3 128k PASS (AUC 0.9153→0.9164; coverage 0.938; FPR 0.093; TPR
  0.747→0.767); Opus 64k PASS (0.9161; coverage 0.940; FPR 0.093; TPR 0.743).
  **8 kHz resample FAIL R1/R2/R4:** AUC 0.6708 (Δ −0.2445, CI −0.2898..−0.2013);
  coverage 0.252 (Δ −0.685); raw FPR 0.940 (Δ +0.8467); raw TPR 0.973
  (Δ +0.2267). All 449 abstentions are OOC; all **68 covered reals** and **83
  covered fakes** score p≥0.5, so issued FPR/TPR are both 100%. The 8 kHz
  transform preserves duration/RMS; this is a fake-ward domain shift, not
  corruption.
- **Video:** **H.264 CRF28 FAIL R3 only:** AUC 0.9072→0.8580 (Δ −0.04917, CI
  −0.07027..−0.02777); coverage 1.0; FPR 0.100→0.050; TPR 0.729→0.561
  (Δ −0.1679). R1 point-passes by only 0.00083 and the CI includes −0.05.
  Per-method TPR: Deepfakes 0.957→0.929; Face2Face 0.643→0.300; FaceSwap
  0.914→0.857; NeuralTextures 0.400→0.157. The loss is an operating-point
  failure concentrated in the hard methods, not an aggregate ranking collapse.
- **Image:** JPEG q35 PASS (AUC 0.9992; coverage 0.725; raw FPR 0.040; TPR
  1.000); resize 50% PASS (0.9996; coverage 0.745; raw FPR 0.090; TPR 1.000).
  Issued FPR is 0/47 and 0/52 respectively. Baseline AUC is saturated at 1.0000
  on this 100/100 sample → spot-check only; the T11/T9 recapture 45% real-FPR
  gap remains separate and was not re-measured.
- **Disposition:** P5 adds **0**; court **64.4/100**. No training, wiring,
  promotion, pointer change, threshold change, or commit. A codec-diverse
  retraining proposal is eligible only via a separate sealed spec and explicit
  user approval. P6 closed KILL B3 (D-093); P7 closed at 0 (D-094; no
  no-wiring roadmap item remains); P3d remains low-probability.
- **P9 — PASS B1-B15; +0.4 system booked; 0 detection.** Sealed read-only C2PA
  validator (`p9-c2pa-readonly-v1`, sha `c6b551c7…`): 24 stdlib tests, 0 skips,
  0 failures/errors; fresh sentinel APPROVE. Fixture3 `well_formed_valid` /
  `trusted=false` / `hard_binding=true`; first-IDAT tamper `invalid` / hard
  false; truncated `error`; real no-manifest image `absent`; trusted/BMFF
  synthetic-only. System dimension 12.5→12.9; court **64.4→64.8**;
  transparency-only, no gate/wiring/promotion/pointer/threshold/commit.
  Receipt: `artifacts/p9_system/RESULT.md`.
- **P6 — CLOSED KILL B3; 0 booked.** Sealed read-only standalone
  `avs_syncnet_offset_abs` gate (`p6-av-offset-gate-v1`, spec sha
  `987846a7…`): calibration 53 AV-diet train reals, tau=q95=2.5 frames,
  strict `offset_abs > 2.5`; AVLips exam 838 (358 real / 480 fake), all
  in coverage, 838 unique `sha256_12`. AUC **0.724002** (parity
  0.7240 ± 0.0005 PASS); TP **350/480 = 72.9167%** PASS; coverage
  **838/838** PASS; FP **106/358 = 29.6089%** (Wilson 95%
  0.251165–0.345343) vs ≤35 → **B3 FAIL, exit 2 KILL**. `>=`
  sensitivity 107 FP / 375 TP, report-only. Two runs byte-identical;
  13 tests 0 failures/0 errors/0 skips; ruff clean; fresh sentinel
  APPROVE. Disclosed: RAVDESS 29.97 fps vs AVLips 25 fps (raw-frame
  threshold not normalized); `offset_abs` selected on this same exam
  (winner's curse); AVLips license missing (research/eval only).
  Court stays **64.8/100**; no wiring, promotion, pointer/threshold
  change, or commit. Receipt: `artifacts/p6_av_gate/run/RESULT.md`.
