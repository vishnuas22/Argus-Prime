# Argus Prime — Remaining Items Inventory

**Last updated:** 2026-11-18
**Open-item authority:** `memory/backlog.md` (2026-11-18). This file is the
provenance/archive for closed and historical items; do not delete it.
**Status:** 2026-11-17 inventory (A0z closed 0; SPLIT T=16 Stage 0 PASS, 0 booked;
**SPLIT LINE CLOSED by user ruling 2026-11-17 (D-099)** — Stage 1 T=16 diagnostic
`NO_TRANSFER_EVIDENCE`, spec never signed, MSR-VTT never downloaded; **WaveRep
fixed-threshold probe FAIL 2026-11-17 (D-100)** — 48% FPR on GenBuster++ reals,
~17.5 s/clip, OOD-video channel CLOSED at 0). P0c booked; P2 closed FAIL;
P8-fast booked +0.5 data coverage; P4 oracle K1 FAIL and closed at 0; P5 codec/
laundering kill-test GAP and closed at 0; P9 read-only C2PA validator PASS and
booked +0.4 system (0 detection); P6 AV offset gate KILL and closed at 0; P7
image/provider-real audit CLOSED AT ZERO. **No training, wiring, promotion,
pointer change, or commit is authorized by this file.**
**Baseline:** **63.1/100** weighted distance-to-frontier. This is not percent-complete.
**Current booked court:** **64.8/100** (P0c +0.8 bookkeeping; P2 0; P8-fast +0.5;
P4 0; P5 0; P6 0; P7 0; P9 +0.4 system).
**Current measured state:** D3 closed FAIL G4; W9.1 closed HOLD; SPLIT Stage 1
closed FAIL S2; **SPLIT Stage 0 T=16 PASS (AUC 0.710316; CI crosses 0.70; 0
booked)**; **SPLIT LINE CLOSED 2026-11-17 by user ruling (D-099)** — diagnostic
GenVidBench T=16 0.539859 (434/434, CI [0.504126, 0.574747]) vs two independent
high-fps negative arms; secondary GenVideo 0.728953 on 235/300; dedup 0.5388; Stage 1
spec parked unsigned, MSR-VTT download NOT performed, 0 booked; **WaveRep G1
fixed-threshold FAIL 2026-11-17 (D-100): 48/100 GenBuster++ 24 fps reals flagged,
~17.5 s/clip, full run early-stopped, 0 booked**; A0z 1B DF-Eval AUC
pass / ECE fail and **not certified**; P2 closed FAIL G3/G5; P4 closed FAIL K1
(oracle ceiling); **P5 closed GAP (audio 8 kHz and video CRF28; image q35/resize
pass)**; P8-fast PASS D1–D5 (coverage only; no detection score); **P9 PASS B1-B15
(transparency-only; 0 detection)**; **P6 CLOSED KILL B3**; **P7 CLOSED AT ZERO**.

## Consolidated remaining-items list (authoritative; updated 2026-11-17)

**DINOv3 pre-flight closed 2026-11-18 (D-103):** native
`facebook/dinov3-vitb16-pretrain-lvd1689m` weights downloaded and runnable
(327 MB; 224/16/768/12 layers/4 registers; `[B,201,768]`). Pre-flight on the 10
provider gates: random 5-fold probe AUC 0.953–0.963 / recall@10% FPR 86.8–89.3%,
but leave-one-generator-out + leave-one-real-component-out gives transferred
threshold FPR **41–71%** and oracle-ceiling recall **62–70.9%** (min 22–35%) at
~9% FPR. **No sealed P7b audit; P7b stays closed at 0.** Receipt
`artifacts/dinov3_feasibility_20261118/RESULT.md`.

### A. Ready or approved

| ID | Item | Status | Expected contribution (projected, not booked) | Expected outcome | Gate / dependency |
|---|---|---|---|---|---|
| **P7b** | DINOv3 ViT-B/16 frozen-backbone audit (P7 reopen) | **CLOSED at 0 after pre-flight 2026-11-18 (D-103)**; weights local; no sealed audit — LOCO FPR 41–71%, oracle recall 62–70.9% | **0 booked**; no sealed audit. The +0.5 OOD-image projection remains unearned. Low probability on the binding cell: only size-matched evidence is FGTS So-Fake-OOD +13.2 pp balanced accuracy (70.3 vs 57.1), and **no DINOv3 size has published real-image FPR**; D-045 closed 0/15; the 94.5 result is ViT-7B, not ViT-B/16. | Joint gate: provider-real **issued FPR ≤10%**, **fake recall ≥70%**, **coverage ≥30% both classes**, plus D-045 R4 G1/G2 strict non-regression (G1 mean hard-cell Δ ≥ +0.01; G2 Δ ≥ −0.01 every measurable cell). Gate needs a **1.44× trade-ratio improvement** over P7's oracle (1.68 pp FPR per pp recall vs 1.17:1). Five-arm contract derivation (CLS / mean-patch / CLS⊕mean-patch / top-K patch / layer-30), 256², provider-real FPR as primary endpoint. | Do not start without a user decision; sealed spec; audit-only registry/harness; fresh sentinel; no wiring/commit. |
| **P8a** | DF-Eval-2024 video gate registration | **REJECTED by repo authors 2026-11-18 (D-102)**; no route | **0 booked**; court stays **64.8**; data coverage **5.0** unchanged. No detection claim. | Access denial received; do not re-submit. | New access route supplied by the user; otherwise closed. |
| **P3** | Audio A0/A1 (A0z 1B not certified) | A0z AUC pass / ECE fail; A0p and A1 not run; training approval required | A0 **+0.9–1.5 OOD-audio**; A1 **+1.5–2.5**. But A0p 1B linear-probe ITW ECE ~26.9% (Pascu 2024) makes ≤0.05 ECE unlikely; A1 depends on certified A0. **Low expected value.** | A0p likely fails ECE → P3 stays 0. A1 only if A0 is certified and full W9.1 bars pass. Do not start by default. | Training approval; new calibration/score approach may be needed. |
| **P5** | Codec-diverse retraining | Measured GAP; eligible only via **separate sealed spec + explicit approval** | Roadmap **+1.5 robustness** if retention bars pass; court ~66.3. No confirmed Y; codec augmentation evidence null-to-negative on wild audio; video CRF28 loss concentrated in hard methods. | Retrain and measure retention; could regress. No booking until a receipt. | Sealed spec + explicit approval; no default start. |

### B. Requires a new pre-registered signal — no confirmed Y

| ID | Item | Status | Expected contribution | Expected outcome | Gate / dependency |
|---|---|---|---|---|---|
| **P6a** | New AV signal (standalone `offset_abs` gate failed) | **KILL B3**; no q99 rescue, no `>`→`>=`, no per-method threshold | Roadmap **+0.7 multimodal/video**, but the cheap signal failed. HOLA-class needs 8×A100 and 1.46 TB gated data. **Low probability.** | New target-domain calibration/fps normalization or a genuinely new signal; separate gate, never a blend. Expected 0 unless a new hypothesis is measured. | New spec + approval. |
| **P2** | Video OOD/scope retry | **FAIL G3/G5**; failure was coverage, not metadata | Roadmap **+1.0–1.5 abstention**, but no confirmed Y. | New pre-registered scope signal must pass face-route ≥55%, re-encoded ≥50%, and not suppress FF++/CDF in-domain. Do not lower bars. | New spec + approval. |
| **P4** | Audio abstention/calibration retry | **FAIL K1**; existing `p_fake`+z has no feasible policy | Roadmap **+0.7–1.0 abstention**, but needs a better score/new signal, not a lower bar. | New score or group-conditional method must reach 30% coverage at FPR_c≤0.10 and R_all≥0.40. Expected 0 otherwise. | New spec + approval. |
| **P1b** | New video-OOD channel | D3 **FAIL G4**; SPLIT **closed 2026-11-17 (D-099)**; WaveRep **FAIL-ACCURACY 2026-11-17 (D-100)** — 48% FPR at fixed threshold 0, ~17.5 s/clip; **no known channel** | **Only path to 77–78**; largest gap (OOD-video 1.2/10). Conditional +1.0–3.0 was not realized; D3’s +4–5 projection is removed. **Requires a genuinely new video-OOD signal** — do not re-litigate SPLIT/T=16, D3, or WaveRep’s fixed-threshold recipe; `sora` 0.8971 (n=51) would need its own pre-registration. | New pre-registered video-OOD channel must pass a real-FPR guard and fresh-generator bars. Research-grade; high upside, high failure probability. | New hypothesis + spec + approval. |
| **P7c** | Disjoint, license-clean provider-real corpus | Not acquired | Data coverage and a cleaner evaluation surface; does **not** fix P7 B3 (OOD-image is AUC-scored, so a monotone scalar cannot move the operating point). | New corpus enables a new pre-registered calibration/eval spec; no score by itself. | License + acquisition + new spec. |
| **P8b** | AV-Deepfake1M++ / DeepSpeak acquisition | Deferred; gated/heavy | Data coverage/multimodal only; feeds a future video-AV diet wave. **No score by itself.** | Acquisition if disk/license; 1.46 TB / 44.7 GB; multi-week. | User prioritization + disk plan. |
| **P8c** | LAV-DF / OpenFake / AIGIBench / other licensed corpora | Not scheduled | Data coverage only; no confirmed Y. | Evaluate under license; no training on eval-only data. | License audit + user priority. |

### C. User decisions / backlog (not score-bearing)

| ID | Item | Status | Notes |
|---|---|---|---|
| **U1** | Audio ship-label decision | Blocked on user | ASVspoof-family banner only; no label change without ruling. |
| **U2** | `.aiff` dispatch | Open | Decoder/dispatch behavior; not a score move. |
| **U3** | CLI XLS-R reload performance | Open | ~7.5 s warm reload per file; performance, not detection. |
| **U4** | Training / promotion / pointer movement | Not authorized | Requires explicit approval and pre-registered bars. |
| **U5** | Commit of assistant changes | Not authorized | Leave uncommitted unless user asks. |
| **P9-follow** | Security/ops, test/CI, model cards | Not scheduled | Insurance only; system dimension 12.9/15 is already near target; no confirmed court Y. |

### D. Closed at zero — do not re-litigate

- **SPLIT Stage 1 T=16** CLOSED by user ruling 2026-11-17 (D-099): diagnostic `NO_TRANSFER_EVIDENCE` — GenVidBench AUC **0.539859** (434/434, CI [0.504126, 0.574747]) against a high-fps real arm; dedup 0.5388, macro 0.6148, drop-opentora 0.5842, kling+opentora 0.4923; two independent negative arms agree (v1 3 fps 0.546810 / Stage 0 24 fps 0.539859); secondary GenVideo 0.728953 on 235/300 is selection- and fps-confounded. Stage 0 T=16 PASS (0.710316) banked as terminal, 0 booked. Stage 1 spec never signed; MSR-VTT never downloaded. Do not re-litigate without a new pre-registration; `sora` 0.8971 (n=51) is the only open thread.
- **WaveRep G1 (fixed threshold 0)** FAIL-ACCURACY + FAIL-DEPLOY 2026-11-17 (D-100): 48/100 GenBuster++ 24 fps reals flagged (**0.480**, CI 0.3846–0.5768) vs ≤0.10; GenVidBench recall 37/50 = 0.740 (kling 26/31, opentora 8/11, sora 3/8); GenVideo 29/50 = 0.580; AUC 0.7336/0.6128; frame-count agreement 8/16/32 = 0.755/0.790/0.860 <0.95 → 64 chosen; 64-frame CPU ~17.5 s/clip >15 s gate; early stop, full 1,734 not executed; 0 booked. No G4/H.264/threshold/GPU rescue under this spec.
- **P2** FAIL G3/G5: face-route 45.71% vs ≥55%; re-encoded 46.67% vs ≥50%; no wiring.
- **P4** FAIL K1: no policy at 30% coverage with FPR_c≤0.10 and R_all≥0.40; best FPR_c 0.333.
- **P5** GAP: audio 8 kHz AUC 0.6708, coverage 0.252, issued FPR 100%; video CRF28 TPR 0.729→0.561; image q35/resize pass (saturated spot-check).
- **P6** KILL B3: real FP 106/358 = 29.61% vs ≤35; AUC 0.724002; TP 350/480.
- **P7** CLOSED AT ZERO: 28.70 pp recall loss at 10% provider FPR; LOCO FPR 0.1200/0.1717; split 97.5th pct 0.1800.
- **D3** FAIL G4: AUC 0.9267 but real FPR 37/300 = 0.1233; no wiring.
- **SPLIT (v1, T=8)** FAIL S2: GenVideo AUC 0.513919; GenVidBench 0.546810; recall 0.0767/0.0576.
- **W9.1** HOLD: 7/11 sealed bars fail; D4 recall 0.883–0.900; FPR 0.005; NR1 0.9129.
- **A0z 1B** NOT CERTIFIED: AUC 0.778066 PASS; ECE 0.243034 FAIL.
- **P0c/P8-fast/P9** are booked: +0.8 system, +0.5 data, +0.4 system.

### E. Gap ranking and honest ceiling

1. **OOD-video — 4.8–5.8 points available.** Biggest gap, hardest, only route to 77–78; **no current channel** — D3 (G4 FPR), SPLIT (no transfer), and WaveRep (48% FPR) are all closed. A genuinely new pre-registered signal is required.
2. **OOD-audio — 2.5 points.** A0z passed AUC but failed ECE; A0p/A1 low-probability.
3. **Abstention — 1.7–2.2 points.** P2 and P4 failed on coverage; retries need new signals.
4. **Robustness — 1.5 points.** P5 measured a real gap; retraining unproven and approval-gated.
5. **Data coverage — 1.0 point.** P8a is the cheapest remaining +0.5; P8b/P8c are heavy.
6. **Multimodal — 0.7 point.** P6a failed the cheap signal; HOLA-class out of scope.
7. **In-domain — 0.6 point.** Saturated; not a practical lever.
8. **OOD-image — 0.5 point.** P7b is the last license-clean backbone question; small upside.
9. **System — ~0.** P9 already banked the available system credit.

**Ceiling:** 73–76 is the theoretical public-data ceiling; 77–78 requires a new
pre-registered video-OOD channel clearing its bars. The remaining ~22 points to
100 are structural; 100 is not achievable with public data today.

### F. 2026-11-15 research synthesis — best next item

Four scout reports landed (DINOv3, OOD-video, audio A0p/A1, codec). Ranked by
expected court points per hour of compute, with the blocking constraint named.

| Rank | Item | Cost | P(score) | Expected | Blocker |
|---|---|---|---|---|---|
| **1** | **A0z decision-quality re-read + owed guard arms** | **~1 h** (guards); re-read is minutes | 40–60% on guards | **+0.9–1.5 OOD-audio** | User ruling: is ECE≤0.05 or decision quality the certification target? |
| **2** | **SPLIT-faithful re-run** (T=16 @ 8 fps / 2 s) | **done** (0.56 h clip latency) | **PASS** | **0 booked**; AUC 0.710316 (CI crosses 0.70) | **Closed 2026-11-17 (D-099):** Stage 1 T=16 diagnostic `NO_TRANSFER_EVIDENCE`; no further SPLIT work |
| 3 | **P8a** DF-Eval video grant | free | — | **+0.5 data** → 65.3 | User submission only |
| 4 | Video CRF28 recalibration | 1–2 h | diagnostic only | 0 (needs a compression detector to deploy) | Defer behind #2 |
| 5 | DINOv3 B/16 audit | ~1 day | low on the binding cell | +0.5 max, wiring-gated | **Not confirmed to boost — see below** |
| 6 | P5 codec retraining | days, encoder-touching | low | unproven | Defer; not a bounded experiment |

**Update 2026-11-16 — A0z CLOSED at 0.** The guard wave returned
`A0Z_NOT_CERTIFIED_GUARD_FAIL`: K1-R PASS (point estimates, both CIs marginal),
G1/G2/G4/G5 PASS, **G3 FAIL 23/300** (bar 0–3). G5 max DF-Eval AUC drop
0.007439 (PASS ≤0.05) on 4,218/4,218 rows. 0 booked;
court stays **64.8**; OOD-audio stays **5.0**. Receipt:
`artifacts/wa_audio_a0z_guards/RESULT.md`; decision D-096. The next-item table is
otherwise unchanged; **#2 SPLIT is now CLOSED (D-099)**; #3 P8a is the live option.

**Update 2026-11-17 — SPLIT T=16 Stage 0 PASS, 0 booked.** The signed T=16 spec
(`artifacts/split_prereg/SPEC_T16_STAGE0.md`) ran on GenBuster++ 1,000/1,000 at
2.0 s / 8 fps / T=16: **AUC 0.710316** (95% bootstrap CI 0.686472–0.733444,
report-only; **CI crosses the 0.70 bar**; point estimate binds) →
`SPLIT_T16_STAGE0_PASS`. 2,000/2,000 rows, 0 unavailable, 0 errors; all 24 fps,
48 decoded, 16 unique indices. Motion-rebalanced 0.740819; motion-only
0.490672/0.509328 (chance); fps corr null. Receipt:
`artifacts/split_t2v_matched/stage0_t16/RESULT.md`; decision D-097. **0 booked;
court 64.8; OOD-video 1.2.** The PASS authorizes only a separate high-fps Stage 1
pre-registration; the v1 Stage 1 corpus cannot support T=16 (reals are 3 fps /
6 frames per 2 s).

**#1 — A0z already clears the court's own K1 constraint values, at full coverage.**
Verified directly from `artifacts/wa_audio_a0z_1b/scores.jsonl` (703 rows, AUC
0.778066): at the **untuned** 0.5 cut, **FP 31/395 = 0.0785** (≤0.10) and
**TP 138/308 = 0.4481** (≥0.40), with all 703 rows issued (**coverage 100%**,
3.3× the 30% K1 demanded). This is the same criterion that closed p4 at 0.5593
AUC and 16.07% coverage; it was never applied to A0z. It is **not** a threshold
sweep — 0.5 is the classifier's default cut and the receipt itself labels it
"not a gate and was not tuned."

Two honest caveats: (a) the numbers were already visible in the receipt, so a new
spec is a **confirmatory** re-read, not a discovery, and must say so; (b) A0z
still owes **four guard bars** that its own spec says run when DF-Eval AUC
≥0.610 — ASVspoof 2021 LA ≥0.900, ITW AUC ≥0.900, ITW bona FPR 0/300, codec
pooled FPR ≤1.5%. The 1B checkpoint is cached; 703 clips took 6.24 min, so the
guards are ~30–60 min. Also note A0z is **CC BY-NC-SA 4.0** (non-commercial).

**Bar-validity contradiction requiring a user ruling.** Audio ECE bar is ≤0.05
(`wa_audio_prereg/SPEC.md:111`); the image channel's own pre-registered T6 bar is
≤0.25 (`t6_gate_spec.md:33`, worst measured 0.2304). A0z's 0.243034 would **PASS
T6**. Three independent measurements put the frozen-SSL linear recipe's ECE floor
at 0.16–0.27 (Pascu 1B 26.9%, 2B 16.1%, our A0z 24.3%), so 0.05 is unreachable by
any source-trained member of the family and reachable only by a target-label fit
that DF-Eval's eval-only ToS forbids. The bar, not the model, is what fails.

**#2 — SPLIT receipt is not a faithful test of the published method.** Two
independent defects verified in-repo: (a) `deepfake/bench/split_temporal.py` is
frozen at `T=8`, `WINDOW_S=3.0` (2.67 fps) vs the paper's 16 frames @ 8 fps /
2 s; (b) `lsmi` alone scores **AUC 0.3141** on our 2,466 rows — inverted, because
it is a motion-magnitude statistic and our fakes are *smoother* than our reals.
The scout's `mean_ttr**8` sign hypothesis is **falsified**: only 9/2,466 rows have
negative TTR, all real. So the composite is near-chance by construction on this
corpus. A re-run at the published protocol is ~1 h and yields either a recovered
channel or a clean negative; the July-2026 bias audit (D3 0.4843 on
motion-balanced GenBuster++) says the negative is the likelier outcome, so
P(pass) is 20–30%, not the scout's 25–40%.

**Resolved 2026-11-17 (D-098/D-099):** Stage 0 T=16 PASSed (0.710316, 0 booked), then
the cheap cross-fps diagnostic returned **`NO_TRANSFER_EVIDENCE`** (GenVidBench T=16
0.539859, 434/434, CI [0.504126, 0.574747]). The user ruled the SPLIT line **CLOSED**;
Stage 1 T=16 was never run and MSR-VTT was never downloaded. This paragraph is kept as
the decision provenance, not as an open recommendation.

**Update 2026-11-17 — WaveRep probe CLOSED at 0.** The user approved a bounded,
inference-only WaveRep probe; the sealed calibration (100 GenBuster++ reals + 50
GenVidBench + 50 GenVideo fakes) measured **FPR 48/100 = 0.480** at the authors'
fixed threshold 0, with GenVidBench recall 37/50 = 0.740 and **~17.5 s/clip** at
64 frames. The pre-registered early stop fired (FPR >0.20); the full 1,734-clip run
was **not executed**. Verdict `WAVEREP_FIXED_THRESHOLD_FAIL`; OOD-video channel
closed at 0; court stays **64.8**. Receipt:
`artifacts/waverep_probe_20261117/RESULT.md`; decision D-100.

**#3 — P8a is closed: rejected by the repo authors (D-102).** The DF-Eval-2024
video gate books **0**; court stays **64.8**. No re-submission without a new
access route from the user. This removes the last certain +0.5.

**#4 — DINOv3 pre-flight is done and fails the binding cell (D-103).** Random
5-fold probe AUC 0.953–0.963 / recall@10% FPR 86.8–89.3%, but the honest
leave-one-generator-out + leave-one-real-component-out split gives transferred
threshold FPR **41–71%** and oracle-ceiling recall **62–70.9%** (min 22–35%) at
~9% FPR. **No sealed P7b audit; P7b closed at 0.** Receipt
`artifacts/dinov3_feasibility_20261118/RESULT.md`. A sealed audit now requires
explicit user override; the five-arm contract derivation is moot because the
failure is threshold transfer, not contract choice.

**Not the next item, for the record.** P5 retraining is deferred: the audio 8 kHz
arm is a *ranking* failure (AUC 0.9153→0.6708) and a 2026 audit proves no
monotone frozen-head correction can move EER, so "P5 retraining" silently means
encoder adaptation — the change class that failed four times (D-036/D-038/D-063/
D-064). The video CRF28 loss is mostly operating point (AUC 0.858, FPR *improves*
0.100→0.050, TPR collapses), but deploying a second threshold needs a compression
detector we do not have.

**Decision required from the user:** (1) audio — rule whether decision quality
(K1) or ECE≤0.05 is the certification target, and authorize the four owed guard
arms; (2) video — **SPLIT (D-099) and WaveRep (D-100) are closed; a video move now
requires a genuinely new pre-registered video-OOD signal**; (3) P8a — submit the
DF-Eval video request (free +0.5). No compute spend, threshold change, training,
wiring, or commit is authorized by this inventory.

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
| SPLIT Stage 0 | PASS at T=8 and T=16 | v1 T=8 0.70244 (CI 0.679236–0.725154); **T=16 0.710316 (CI 0.686472–0.733444 crosses 0.70)**; 0 booked |
| SPLIT Stage 1 T=16 | CLOSED — unrun (D-099) | Diagnostic `NO_TRANSFER_EVIDENCE`: GenVidBench T=16 **0.539859** (434/434, CI [0.504126, 0.574747]) vs 1,000 sealed 24 fps reals; dedup 0.5388, macro 0.6148; spec `SPEC_T16_STAGE1.md` PARKED unsigned; MSR-VTT never downloaded |
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
