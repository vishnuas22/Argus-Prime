# Argus Prime — Open-item backlog (authoritative list)

**As of:** 2026-11-18
**Court:** **64.8/100** measured and booked; **0 pending bookings**
**Status:** **STOP ruling (D-105)** — no further detection compute authorised. This
list is frozen as the record; items remain available only under a new sealed spec
plus explicit user approval.
**Next level:** 70.0 requires **+5.2 points**; no open item has a confirmed path.
**Hard rule:** this file is a list, not an authorization. No training, wiring,
promotion, download, pointer/threshold change, or commit without explicit user
approval. No confirmed-Y path → do not spend compute (ledger law,
`memory/improvement_ledger.md`).

This file supersedes the open-item tables in `memory/remaining_items.md` as of
2026-11-18. The older file remains the provenance/archive; do not delete it.

## 1. Scoreboard and arithmetic headroom

| Dimension | Score | Max | Headroom | Single-dimension route to 70? |
|---|---:|---:|---:|---|
| In-domain | 14.4 | 15 | +0.6 | no |
| OOD-image | 8.5 | 10 | +1.5 | no |
| OOD-audio | 5.0 | 10 | +5.0 | only at the full 5.0; not realistic |
| OOD-video | 1.2 | 10 | +8.8 | **yes, the only one** |
| Abstention | 11.3 | 15 | +3.7 | no |
| Robustness | 4.5 | 10 | +5.5 | only at the full 5.5; not realistic |
| Data coverage | 5.0 | 10 | +5.0 | no; data alone does not detect |
| System | 12.9 | 15 | +2.1 | no |
| Multimodal | 2.0 | 5 | +3.0 | no |
| **Total** | **64.8** | **100** | **+35.2** | |

The +35.2 is arithmetic headroom, not a plan. The honest public-data ceiling is
**73–76**; **77–78** requires a new video-OOD channel clearing its bars. No open
item has a confirmed +5.2.

**Arithmetic constraint (2026-11-18).** The Roadmap's public-data target for
OOD-video is **6.0–7.0/10**. From 1.2 that is **+4.8 to +5.8**. Therefore a
full video pass alone clears 70 only if it lands at the **top** of that range:
it needs **OOD-video ≥ 6.4/10**. A +4.0–5.0 outcome lands at **68.8–69.8**, i.e.
short. Any realistic route to 70 is **video plus a small second item**
(data +0.5 or system +0.4), or a video result at the very top of the target band.

**Reachability proof (2026-11-19).** Computed from the Roadmap's own dimension
targets:

| Scenario | Ceiling | Composition |
|---|---:|---|
| No video, no training | **65.4** | data +0.5, system +0.1 |
| Training authorised, no video | **69.0** | robustness +1.5, audio +1.5, multimodal +0.7, data +0.5 |
| **70.0** | — | **requires the video dimension** |

Non-video gaps and their gates: OOD-audio +2.5 (certification blocked), abstention
+2.2 (P2/P4 failed), robustness +1.5 (retraining only), data +1.0 (cheap, +0.5
realistic), multimodal +0.7 (AV diet HOLD), OOD-image +0.5 (P7/DINOv3 closed),
in-domain +0.6 (better model). **Even with training approval the realistic total
is 69.0. Video is the only dimension with enough weight to close the gap.**

## 2. Open items

### Tier 1 — the only single-dimension route to 70

| ID | Item | Dimension | Realistic gain | Cost | P(success) | Blocker / next action |
|---|---|---|---|---|---|---|
| **V1a** | ~~D3 real-collection transfer pre-flight~~ | OOD-video | 0 | done | — | **CLOSED FAIL 2026-11-18 (D-104).** α=6% on 200 fresh MSR-VTT reals → FPR **44.0% FF++**, 16.0% AVLips, 32.0% CelebDF-YouTube, 33.3% CelebDF-Celeb. Max catch at ≤10% FPR across all collections = **6/27**. Receipt `artifacts/d3_transfer_preflight/RESULT.md`. |
| **V1b** | ~~D3-v2 per-target-corpus calibration~~ | OOD-video | 0 | — | **0** | **CLOSED (D-104).** No threshold exists that both protects the court's real arms and catches ≥15/27. The 12.33% G4 miss was the mild form of a 16–44% transfer failure. |
| **V1c** | Fresh-generator probe (Video Reality Test) | OOD-video | diagnostic guard only | 4.3 GB + ~30 min | — | MIT, ungated, Sora-2 / Veo-3.1 / Wan2.2 / Kling. Fits the 68 GiB free disk. No published FPR, so it is a guard, not a score claim. |
| **V1d** | New video signal **class** | OOD-video | unknown; +5.2 only if a channel clears its bars | research | low | Temporal-statistics family now empirically closed (D3 pre-flight + SPLIT D-099). Any candidate must pre-register the **FF++/AVLips transfer test** before a full run. |
| **V1e** | OOD-video exam construct-validity re-spec | Scoring only | could change 1.2/10 materially | user ruling + sentinel | — | The exam compares **faceless T2V fakes vs FF++ face reals / AVLips talking heads**; content domain is collinear with the label. Re-specifying is a user scoring decision, not an engineering task. |
| **V1f** | **Content-matched video diagnostic** | OOD-video (diagnostic) | 0 booked by itself; tells us whether the 1.2 is a content confound | **~20 min, local only, read-only** | — | Score ~150 fresh MSR-VTT reals + ~150 fresh T2V fakes through the **deployed** `/api/detect` and compute AUC. No pointer change, no wiring, no download. This is the cheapest test of whether a legitimate re-spec could pass — and it is the *only* remaining experiment that could justify one. |
| **V2** | `sora`-specific follow-up | OOD-video | ~0 court; diagnostic only | hours | low | Only open SPLIT thread: AUC 0.8971, n=51, post-hoc, no real guard. Would need its own pre-registration. |
| **V3** | Re-request DF-Eval-2024 video | Data | +0.5 data only | free | **0** | **Rejected by repo authors (D-102).** Do not re-submit without a new route from the user. |
| **V4** | AIGVDBench selective acquisition | Data + OOD-video guard | +0.5–1.0 data; enables fresh-generator + FPR guard | 60–90 GB selective; 378 GB full | **blocked now** | **Disk: 68 GiB free** (926 GiB volume, 93% used). Selective download does not fit safely. Needs a disk plan or external volume first. |
| **V2** | `sora`-specific follow-up | OOD-video | ~0 court; diagnostic only | hours | low | Only open SPLIT thread: AUC 0.8971, n=51, post-hoc, no real guard. Would need its own pre-registration. |
| **V3** | Re-request DF-Eval-2024 video | Data | +0.5 data only | free | **0** | **Rejected by repo authors (D-102).** Do not re-submit without a new route from the user. |

### Tier 2 — audio (largest headroom after video)

| ID | Item | Dimension | Realistic gain | Cost | P(success) | Blocker / next action |
|---|---|---|---|---|---|---|
| **A1** | Certify A0z 1B (P3) | OOD-audio | +0.9–1.5 if A0 certified | guard wave already run; new model/calibration would be training-gated | low | A0z AUC 0.778 PASS, but **G3 ITW bona FPR 23/300 (7.7%)** vs 0–3, and **ECE 0.243** vs ≤0.05. Guards cannot repair DF-Eval calibration. |
| **A2** | A0p frozen-feature linear probe | OOD-audio | +0.9–1.5 only via A0 | ~1 day + training approval | low | Pre-registered fallback for an AUC miss, not an ECE/G3 miss; published ITW ECE ~26.9% (1B) makes ≤0.05 unlikely. |
| **A3** | A1 full audio candidate | OOD-audio | +1.5–2.5 | multi-day + training | very low | Depends on a certified A0; full W9.1 bars. Do not start. |
| **A4** | New audio checkpoint/calibration (research) | OOD-audio | unknown | unknown | unknown | Research question: is there a 2025–2026 open model with published ITW/DF-Eval FPR and ECE that clears the bars? No confirmed candidate yet. |

### Tier 3 — robustness and data (portfolio items, not a single +5.2)

| ID | Item | Dimension | Realistic gain | Cost | P(success) | Blocker / next action |
|---|---|---|---|---|---|---|
| **R1** | Codec-diverse retraining (P5) | Robustness | +1.5 if retention bars pass | days; encoder-touching | low | Separate sealed spec + explicit training approval. Audio 8 kHz is a ranking failure (AUC 0.9153→0.6708); video CRF28 is mostly operating point. No confirmed Y. |
| **R2** | Compression detector + second threshold (video CRF28) | Robustness | +0.5–1.0 at best | 1–2 days | low | Diagnostic only so far; deploying a second threshold needs a compression detector we do not have. |
| **D1** | Disjoint license-clean provider-real corpus (P7c) | Data coverage | +0.5–1.0 data; no detection Y | acquisition + license audit | medium on data, 0 on detection | Does not fix P7 B3 (OOD-image is AUC-scored). Enables a new calibration/eval spec. |
| **D2** | AV-Deepfake1M++ / DeepSpeak acquisition (P8b) | Data + multimodal | data only, feeds a future AV diet | 1.46 TB / 44.7 GB; multi-week | low | Gated/heavy; only feed a future video-AV diet wave. |
| **D3** | LAV-DF / OpenFake / AIGIBench / other licensed corpora (P8c) | Data coverage | +0.5–1.0 data; no detection Y | license audit + download | medium | No confirmed Y; eval-only data cannot train. |

### Tier 4 — abstention and multimodal (need a new signal, not a lower bar)

| ID | Item | Dimension | Realistic gain | Cost | P(success) | Blocker / next action |
|---|---|---|---|---|---|---|
| **AB1** | Video OOD/scope retry (P2) | Abstention | +1.0–1.5 | new signal + spec | low | Failed G3 face-route 45.71% vs ≥55%, G5 re-encoded 46.67% vs ≥50%. Retry needs a new pre-registered scope signal, not a lower bar. |
| **AB2** | Audio abstention retry (P4) | Abstention | +0.7–1.0 | new score + spec | low | Failed K1: existing `p_fake`+z has no policy at 30% coverage, FPR_c≤0.10, R_all≥0.40; best FPR_c 0.333. Needs a better score/new signal. |
| **MM1** | New AV signal (P6a) | Multimodal | +0.7 | research-grade | low | Cheap `offset_abs` gate killed B3 (106/358 real FP, 29.61%). HOLA-class needs 8×A100 + 1.46 TB gated data; out of scope. |
| **MM2** | AV diet diversification (SadTalker/VideoReTalking/MakeItTalk) | Multimodal | +0.7 | data + training | very low | W-AV v5 HOLD; uncertain payoff. Do not start. |

### Tier 5 — image, system, user decisions

| ID | Item | Dimension | Realistic gain | Cost | P(success) | Blocker / next action |
|---|---|---|---|---|---|---|
| **IM1** | DINOv3 / new image backbone (P7b) | OOD-image | 0; +0.5 max and wiring-gated | done | **closed** | **Closed at 0 after pre-flight (D-103):** transferred-threshold FPR 41–71%; oracle recall 62–70.9% (min 22–35%) at ~9% FPR. Override only. |
| **SY1** | P9 follow: security/ops, test/CI, model cards | System | ~0 | days | — | System is 12.9/15, already near target; insurance only, no confirmed court Y. |
| **U1** | Audio ship-label decision | — | 0 | minutes | — | Blocked on user: ASVspoof-family banner only; no label change without ruling. |
| **U2** | `.aiff` dispatch | — | 0 | small | — | Decoder/dispatch behavior; not a score move. |
| **U3** | CLI XLS-R reload performance | — | 0 | small | — | ~7.5 s warm reload per file; performance, not detection. |
| **U4** | Training / promotion / pointer movement | — | — | — | — | Not authorized; requires explicit approval and pre-registered bars. |
| **U5** | Commit of assistant changes | — | — | — | — | Not authorized; leave uncommitted unless user asks. |

## 3. Closed at zero — do not re-litigate

- **D3 content-matched T2V** — FAIL G4: real FPR 37/300 = 12.33% (bar ≤10%); AUC 0.9267. No wiring.
- **D3-v2 / D3 transfer** — **CLOSED FAIL (D-104, 2026-11-18):** per-collection transfer FPR 16–44%; max catch at ≤10% FPR across all real collections is 6/27. The ranking itself is corpus-bound (fakes vs FF++ reals AUC 0.6656). No re-threshold, no variant.
- **SPLIT v1 (T=8)** — FAIL S2: GenVideo AUC 0.513919; GenVidBench 0.546810.
- **SPLIT Stage 1 T=16** — CLOSED by user ruling (D-099): cross-fps diagnostic GenVidBench 0.539859 (434/434, CI [0.504126, 0.574747]); dedup 0.5388; spec never signed; MSR-VTT never downloaded.
- **WaveRep G1** — FAIL-ACCURACY + FAIL-DEPLOY (D-100): FPR 48/100 = 0.480; ~17.5 s/clip; early stop; full run not executed.
- **P2 video OOD/scope** — FAIL G3/G5: face-route 45.71% vs ≥55%; re-encoded 46.67% vs ≥50%.
- **P4 audio abstention** — FAIL K1: no policy at 30% coverage with FPR_c≤0.10 and R_all≥0.40; best FPR_c 0.333.
- **P5 codec/laundering (as run)** — GAP: audio 8 kHz AUC 0.6708, coverage 0.252, issued FPR 100%; video CRF28 TPR 0.729→0.561; image q35/resize pass (saturated).
- **P6 AV `offset_abs` gate** — KILL B3: real FP 106/358 = 29.61% vs ≤35; AUC 0.724002; recall 350/480.
- **P7 image backbone + provider-real calibration** — CLOSED AT ZERO (D-094): 28.70 pp recall loss at 10% provider FPR; LOCO FPR 0.1200/0.0300/0.1717.
- **P7b DINOv3** — CLOSED AT ZERO (D-103): honest threshold transfer FPR 41–71%; oracle recall 62–70.9% (min 22–35%).
- **P8a DF-Eval-2024 video** — REJECTED by repo authors (D-102); 0 booked.
- **A0z 1B (as run)** — NOT CERTIFIED: AUC 0.778066 PASS, ECE 0.243034 FAIL, G3 23/300 FAIL.
- **W9.1 fused audio** — HOLD: 7/11 sealed bars fail; p4 untouched.
- **HOLA / AV-Deepfake1M++ frozen featurization** — 8×A100, no public HOLA weights, 1.46 TB gated data; expected Y ≈ 0.
- **Per-generator image heads** — no head-to-head evidence; data alignment/backbone dominates.
- **ASVspoof5 standalone pull** — 142.3 GB; no standalone confirmed Y.
- **T2V training wave** — GenVidBench exam-only; no justified wave before a new measured hypothesis.
- **Watermark detection as score feature** — mark-to-frame attacks; transparency only.
- **Frequency-invariance architecture** — 2026 controlled result: no marginal over plain EfficientNet-B0.
- **Chasing 100/100** — ~22 points structural with public data.

## 4. Access and license constraints

- **DF-Eval-2024 video:** manual gate; request rejected by repo authors; no re-submission without a new route.
- **DF-Eval-2024 audio:** already granted; ToS is evaluation-only.
- **A0z 1B:** CC BY-NC-SA 4.0; court use is non-commercial (D-059).
- **ITW audio:** CC-BY-SA-4.0; 31,779 files / 37.9 h.
- **WaveRep:** official G1 weights used non-commercially/research-only; channel closed.
- **DINOv3:** native weights local; permissive with attribution, no non-commercial restriction found.
- **GenVidBench / GenVideo:** evaluation-only in our scope; do not train on them.

## 5. Decision pointers

- D-099 SPLIT line closed; D-100 WaveRep closed; D-101 P8a-only ruling; D-102 P8a rejected; D-103 DINOv3 pre-flight closed.
- Court and dimension scores: `memory/improvement_ledger.md`.
- Historical inventory and receipts: `memory/remaining_items.md`, `artifacts/`.

## 6. Research findings (2026-11-18, three scouts)

### 6.1 Video methods — the failure is the operating point, not the ranking

- **ReStraV:** 0.39% FP at τ=0.5 in-domain on 10k MSR-VTT reals, but **15.30 / 39.86 / 66.10% actual FPR** at nominal 0.1 / 1 / 5% on another real collection. Cleanest published proof that a low in-domain FPR does not transfer. (arXiv 2507.00583; github.com/ChristianInterno/ReStraV; SPLIT Table 4)
- **Corpus-identity audit:** thresholds set to flag 5% of a benchmark's reals flag **≥15%** of another real collection; choosing the threshold on **target-collection reals** restores ~5%. Frame-type counts alone separate real from generated at **AUC 0.999** on GenVidBench; a uniform re-encode drops ReStraV 0.85 → 0.60. (arXiv 2606.31004, 2026-06-30 / rev 2026-11-03)
- **D3's own threshold rule** is 85/90/95% of *source* real videos → nominal 5/10/15% FPR with no transfer test. Argus's 12.33% is that drift, measured. (ICCV 2025 supplemental; arXiv 2508.00701)
- **SPLIT** (ECCV 2026, MIT code) is the only training-free method reporting recall at fixed FPR with real-only calibration: **85.16 / 93.09 / 97.35%** at 0.1 / 1 / 5% FPR on GenVideo, **84.52%** at 0.1% on ViF-Bench. Cross-real transfer: 0.56 / 2.18 / 5.49% actual at 0.1 / 1 / 5% (MSR-VTT→ROVI direction 0.03 / 0.29 / 4.22). No independent replication; **never tested at a different frame rate** — Argus's cross-fps failure axis is untested by the authors. (arXiv 2607.02886)
- **TSC-Net** is the strongest trained 2026 result (1.731 M params, 4.68 ms/clip, 0.07% FP at τ=0.5 on 10k reals) but is in-domain on GenVideo-100K and training-gated. (arXiv 2610.35148)
- Dead names: "TempoToken" is a generator, "VideoGuard" is an adversarial protection method, "GenVideoDet" does not exist.

### 6.2 Video corpora — three license-clean ungated sets survive

- **AIGVDBench** — CC-BY-4.0, ungated, **378.39 GB**, 31 generators (11 closed-source test-only), 20k OpenVid-HD reals. Best fresh-generator exam + FPR guard; third-party transfer baseline exists (VidAudit). **Disk-blocked for now.**
- **GenBuster++** — MIT, 2.70 GB, ungated, already local (`data/raw/genbuster_plusplus`, 2.3 GB).
- **Video Reality Test** — MIT, 4.30 GB, ungated, only set with confirmed **Sora-2 / Veo-3.1**; 149 real ASMR + 149×13 fakes.
- **GenBuster-200K** — MIT, 113.66 GB, 100k real / 100k fake; generators are early-2025.
- Dead ends: OpenFake and AIGIBench have **no video**; Omni-Fake returns **HTTP 401**; DeepSpeak v2 / AV-Deepfake1M++ / FakeAVCeleb are all manual-gated. GenVideo-100K is 230.6 GB and **CC-BY-NC-SA-4.0** (stricter than the parent).

### 6.3 Audio — the bars are statistically unattainable, not a model gap

- **0/300 is a ~0.03% FPR bar**, not 1.2%: P(0 failures) = (1−p)³⁰⁰; a 1% system passes 4.9% of the time, 0.035% is needed for 90% reliability. The 95% upper bound on 0/300 is 0.99%, so the bar certifies almost nothing even when passed.
- **Best published ITW ECE is 16.1%** (xls-r-2b frozen probe; 26.9% for 1b) — 3× the ≤0.05 bar. No published ECE exists on DF-Eval audio at all.
- **No paper reports a 0/300-equivalent ITW FPR.** Published ITW EERs of 1.23–2.12% imply ~240–420 false positives across ITW's reals.
- **A0z's 23/300 is threshold misplacement**, not ranking failure: its published ITW EER is 1.35%, which implies ~4/300 at an EER-placed threshold. Local sweep: τ=0.999 gives ITW FPR 0.3% but DF-Eval TPR collapses to 18.5%.
- **DF-Eval ToS forbids training**, yet every headline 0.88–0.92 AUC fine-tunes on its train split. The honest zero-shot ceiling is ~0.65–0.78 AUC.
- Best license-plausible candidate **Spectra-AASIST** (XLS-R-300M + AASIST, ITW EER 1.461%) has no FPR/ECE and **contradictory licence provenance** (Apache-2.0 vs MIT vs possible MLAAD/CC-BY-NC lineage). Do not cite it without author confirmation and an independent re-score (AASIST3's card numbers differ from arena re-scores by ~10 EER points).
- **Recommendation: restructure the audio claim** (FPR bound + mandatory analyst review + selective risk–coverage), not shop for a model. That is a policy ruling for the user, not an engineering task.

### 6.4 Robustness — the ranking loss is provably unfixable without retraining

- **Monotone transforms cannot change AUC.** Formal Proposition 1 plus exact empirical confirmation: z-norm, temperature/shift, mean alignment, CORAL, AS-norm, BN adaptation all leave EER 11.18% unchanged on a frozen detector; four of them make HTER *worse* (39.49% baseline vs 45.02 / 46.65 / 43.69 / 60.70). (arXiv 2606.21584, 2026-06-19)
- The same paper shows the *operating point* is repairable: temperature/shift cut HTER 39.49 → **24.39%** on a frozen detector — then degenerated to **50.00%** under a different class prior. So a per-corpus threshold recalibration can fix coverage/FPR, but not ranking, and is prior-fragile.
- **Argus's audio 8 kHz failure is the operating-point class, not the ranking class**: AUC 0.9153 → 0.6708 (ranking survives above chance) but coverage 0.937 → 0.252 and FPR 0.940 among covered. The mechanism is published and class-asymmetric — low-pass strips the >4 kHz synthesis cue and the **bona-fide manifold moves toward fake** (SONAR arXiv 2511.21325; FP-NAC arXiv 2609.39651). Fixes in the literature are augmentation/retraining or a trained codec adapter (FP-NAC, MIT, 1.30M params) — none inference-only.
- **Restoration before the detector does not recover ranking**: JPEG artifact removal +0.01–0.035 AUC best case, **−0.146** worst case (DFF 2026); demoiréing leaves 8–11.6 AUC points of loss (NeurIPS 2025 D&B); super-resolution on inputs is a documented *attack* (−1.7 to −4.5 AUC, arXiv 2410.04205).
- **Video CRF28 has no published inference-only candidate at all.** No paper tests frame restoration before a frozen H.264-robust detector at CRF 28–40.
- Consequence: robustness can only move by **retraining on real codec variants** (+7.3 AUC points over synthetic JPEG augmentation, arXiv 2608.28685), which is unauthorised and caps the dimension at +1.5.
- **No open inference-only item exists here.** Any future attempt must pre-register a retraining spec plus retention bars, and cannot by itself reach 70.
