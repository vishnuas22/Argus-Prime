# 10-Scientist Research Report — Unexplored Detection Improvements

**Date:** 2026-11-22
**Method:** 10 hypothesis families, inline research (subagents unavailable in this session).
**Rule:** research only — **no execution performed**. Every candidate is grounded against the repo's closed decisions.
**Baseline:** court 64.8/100. Incumbent video p6vid_v4: FF++ 0.9121 · CDF 0.9482 · DFD 0.8857 · **GenBuster++ T2V 0.590**.

---

## Headline findings (ranked by expected value)

| # | Finding | Evidence | Why it's new |
|---|---|---|---|
| **1** | **MotionPhys: 0.872 AUC on GenBuster++** — the exact corpus where our incumbent scores **0.590** | arXiv 2608.20770 (Aug 2026) | Physical motion-consistency via optical-flow trajectories. **Not** temporal statistics (closed D-099) and **not** face-based (our incumbent fails non-face T2V) |
| **2** | **TriMoE / MoSTE: input-adaptive MoE fusion** | CVPRW 2026 (TriMoE), arXiv 2611.12443 (MoSTE, Nov 2026) | Validates the user's 11-channel instinct: **static fusion is the flaw**, not the channels. Our stacker is a *static* global fusion |
| **3** | **DiCoME: Dempster-Shafer evidential fusion** | ICML 2026 (PMLR v306), arXiv 2606.01885 | The repo already has a `conflict_metric` — this is the principled version. Handles "epistemic conflict" the stacker cannot |
| **5** | **WaveSP-Net: MIT, checkpoint on HF, 0.9426 DF-Eval** | ICASSP 2026, `xxuan-speech/WaveSP-Net` | The plan named it; **never fetched/tested**. 1.298% trainable params |
| **6** | **MoE for audio (Negroni 2025) + Mixture of Spectral Experts** | Interspeech 2026 | Audio MoE dynamically weights specialist detectors |

---

## Scientist 1 — Fusion architecture (the user's 11-channel insight)

**The insight is correct and the literature now proves it.** TriMoE (CVPRW 2026) states the problem exactly:

> "Spatial, spectral, and temporal domains each capture complementary forgery cues, but current multi-domain approaches **fuse them statically, with uniform weights failing to account for the variable informativeness of each domain across different videos**."

That is **precisely our architecture**: a HistGradientBoosting stacker is a *static* global fusion. It cannot know that `learned_ffpp_probe` is informative for this clip but useless for that one.

**Three concrete upgrades:**

| Idea | Mechanism | Cheap falsification |
|---|---|---|
| **Input-adaptive gating** | A small router `g(x)` → per-channel weights, replacing the global stacker | Fit a router on **existing** channel outputs; measure whether per-input weighting beats the global stacker on FF++/CDF dumps |
| **Rank-space fusion** | Combine per-channel **ranks** not raw probabilities — immune to the per-channel scale miscalibration the repo measured (D-103: 41–71% FPR transfer) | Re-fuse existing channel outputs by rank; no retraining |
| **Evidential / Dempster-Shafer** | Model belief + **uncertainty** per channel; fuse with DS combination | The repo already computes `conflict_metric`; DS is the principled version of what that reaches for |

**Key advantage:** all three can be tested on **existing artifacts** (`artifacts/restack_video_v4e/dump_ffpp/`, `artifacts/p6v11/` channel outputs). **No new inference, no training.**

**P(success):** moderate-high for *measurement*, moderate for *promotion* — a static stacker is hard to beat but a well-gated one has headroom.

---

## Scientist 2 — Physics-based T2V (the strongest lead)

**MotionPhys (arXiv 2608.20770, Aug 2026)** — the single most important finding.

- **Mechanism:** Shi-Tomasi + grid points → pyramidal Lucas-Kanade optical flow → per-trajectory **curvature, angular velocity, jerk** at 3 temporal scales (8/12/24 frames) → statistical aggregation → lightweight classifier.
- **Reported:** **GenBuster++ ACC 78.5% / AUC 0.872** (vs our 0.590). AIGVDBench cross-generator generalises.
- **Why it dodges every closed line:**
  - Not temporal *statistics* (D3/SPLIT/WaveRep measured inter-frame L2 in feature space — D-099). MotionPhys measures **differential geometry of trajectories** — curvature, angular velocity, jerk. Different mathematical object.
  - Not face-based — works on non-face content (the incumbent's T2V failure mode).
  - Not a heavy backbone — "lightweight and interpretable", no pretrained video model.
- **Caveat (critical):** arXiv 2607.00948 shows **motion-based detectors suffer dataset bias** — AI videos have *less* motion than real; rebalanced datasets degrade them. **MotionPhys's own paper says it beats appearance methods on GenBuster++, but the bias paper is a direct threat to any motion method.** This must be the first bar.

**Also strong:**
- **AIGVDet** (arXiv 2403.16638): optical-flow + RGB dual-branch, **Sora 0.921 AUC**.
- **Optical-flow residuals** (arXiv 2508.00397): Sora **0.995 AUC** — but suspiciously high, likely the same motion bias.
- **NSG-VD** (NeurIPS 2025): physics-driven Navier-Stokes features + MMD, **training-free**, +16% recall. `github.com/ZSHsh98/NSG-VD`.

**Falsification test (cheap):** run MotionPhys's trajectory features on our existing GenBuster++ 1,000+1,000. Bar: AUC ≥ 0.75 on GenBuster AND ≥ 0.60 on FF++ reals (the motion-bias control).

---

## Scientist 3 — Audio (the broken modality)

**WaveSP-Net is the clear lead and it is untested here.**

| Fact | Value |
|---|---|
| Venue | ICASSP 2026 |
| Licence | **MIT** (`github.com/xxuan-acoustics/WaveSP-Net`) |
| Checkpoint | **on HuggingFace** (`xxuan-speech/WaveSP-Net`) — the plan said "verify at fetch"; it exists |
| DF-Eval-2024 | **AUC 0.9426 / EER 10.58** |
| Trainable params | **1.298%** of total (frozen XLSR-300M + wavelet-domain prompts + Mamba backend) |
| Input | 4-second segments (matches our windowing) |

**Why it should fix our wild audio:** our lane is frozen XLS-R-300M + mean-pool + LogisticRegression. WaveSP-Net is **the same frozen backbone** but adds (a) learnable **wavelet-domain** prompt tuning, (b) **sparse** prompt regularisation, (b) a **Mamba** backend. The ablation is damning for naive approaches: removing sparsity costs **+35.5% EER**, fixed wavelets cost **+56.4% EER**.

**Caveats:** Python 3.8 required (repo says `conda create -n WaveSP-Net python=3.8`) — a sidecar container, same trap as P3's fairseq (D-129/V8). Mamba backend needs `mamba-ssm`.

**Also:** Negroni et al. 2025 — MoE for **audio** DF detection dynamically weighting specialist detectors. And Mixture of Spectral Experts (Interspeech 2026) on ASVspoof 2019/2021 + In-the-Wild.

**Falsification test:** fetch the HF checkpoint, run in a py3.8 sidecar on our existing DF-Eval audio subset (703 clips). Bar: AUC ≥ 0.85. **Zero training needed** — just inference.

---

## Scientist 4 — Calibration & abstention (unlocks score without detection gains)

The repo's abstention is 65–84% on wild. Two 2026 papers attack exactly this:

- **COF: Correlation-Optimized Fusion** (arXiv 2606.06666) — learns per-architecture uncertainty fusion weights by maximising `ρ(fused_uncertainty, prediction_error)`. Reports **in-domain UQ correlations of 0.41–0.47 collapsing to near-zero externally (mean degradation 90.7%)**.
- **DiCoME** (ICML 2026) — geometric view purification + evidential fusion, explicitly targets "semantic masking" where semantic features overwhelm subtle artifacts.

**Why this matters for us:** our abstention is a fixed conformal band. If we can *predict which inputs the court will get wrong*, we can abstain precisely instead of broadly. That converts abstention (11.3/15, +3.7 headroom) into a **coverage** win rather than a detection win.

**Falsification test:** on existing dumps, measure whether per-channel disagreement predicts error. If `ρ(disagreement, error) > 0.3` on held-out, the abstention gate can be tightened without losing precision.

---

## Scientist 5 — T2V via VLM reasoning (negative evidence)

ARGUS (arXiv 2608.06865) uses 4 MLLM forensic agents (texture/lighting/motion/physics) + a judge. But the reported numbers are **modest** (42–54% on OOD). And LLaVA-1.5 scores **exactly 50.00** on closed-source generators (DeCoF paper). **MLLM-as-detector is not a solution at our budget.** Deprioritise.

---

## Scientist 6 — Frequency / spectral (partially closed)

The image lane already has `spectral_probe` and `compression_history`. For **video**, 3D FFT / spatiotemporal spectra are unexplored here. But **AIGVDet's ablation** is informative: `S_optical` (0.950 avg AUC) beats `S_spatial` (0.949) — motion beats appearance. Supports Scientist 2.

---

## Scientist 7 — Data / benchmark (blocked but relevant)

- **GenVidBench** has 1,200 videos (Sora, Lumiere, Pandora, etc.) — already in-house (14,114 clips).
- **AV-Deepfake1M++** (1.46 TB) and **AIGVDBench** (378 GB) are disk-blocked (60 GiB free).
- **Deepfake-Eval-2024** gated to May 2027 (D-102 rejected).
- **D2 (AV-Deepfake1M++ / DeepSeek-V2.5)** — DeepSeek-V2.5 reports **98.9% on AV-Deepfake1M++**, but that's a 1.46 TB gated corpus.

---

## Scientist 8 — Multimodal / audio-visual

DF-MoE (arXiv 2608.23363) fuses 6 AV cues (head pose+gaze, segmentation, emotion, rPPG, lip-sync, AVFF) through sparse MoE. But our repo already **closed** MM1 (P6a offset_abs killed B3) and HOLA needs 8×A100. **Deprioritise unless the audio fix lands first.**

---

## Scientist 9 — Robustness / compression

The repo's R1/R2 (codec-diverse retraining, compression detector) remain untried but encoder-touching. **D-105 gated.** Not a near-term win.

---

## Scientist 10 — Meta: why single models fail (the honest synthesis)

The literature converges on one explanation, and it validates the user's instinct:

> Single detectors fail in real usage because they **rely on specific forensic cues** and **do not adapt well to shifts in how synthesis occurs** (DeepFakeBuster, Sci Reports 2026).
> "The need of the hour is to shift from standalone models to a group of models which specialise on their own" (TriMoE, CVPRW 2026).

Our 11-channel image court **already does this**. The video and audio lanes **do not** — they are single-model pipelines. **That is the structural gap.**

---

## Ranked recommendations (cheapest falsification first)

| Rank | Candidate | Cost | Bar | Why |
|---|---|---|---|---|---|
| **1** | **WaveSP-Net inference** | ~1 h (py3.8 sidecar, no training) | DF-Eval audio AUC ≥ 0.85 (from 0.5593) | MIT, checkpoint exists, same frozen backbone we use, 1.3% trainable |
| **2** | **MotionPhys features** | ~2 h (optical flow on existing clips) | GenBuster AUC ≥ 0.75 AND FF++ reals FPR sane | 0.872 published vs our 0.590; not temporal-stats, not face-based |
| **3** | **Rank-space / evidential re-fusion** | ~1 h (existing dumps, no inference) | beats global stacker on FF++/CDF | Directly exploits the 11-channel investment; no new models |
| **4** | **Input-adaptive gating (TriMoE-lite)** | ~3 h | beats global stacker + beats rank-fusion | The literature's #1 recommendation for our exact flaw |
| **5** | **COF uncertainty fusion** | ~2 h | `ρ(disagreement, error) > 0.3` held-out | Converts abstention into coverage; +3.7 headroom |

### The one that would change the project's trajectory

**MotionPhys + WaveSP-Net together** address the two dimensions where the project is weakest (OOD-video 1.2/10, OOD-audio 5.0/10) with **published, permissively-licensed, cheap** methods that **dodge every closed line** in the repo's decision log.

### Honest caveats

- **Motion-bias threat:** arXiv 2607.00948 shows motion detectors degrade on rebalanced datasets. MotionPhys must clear a motion-bias control before any promotion.
- **MotionPhys has no published code/checkpoint** (unlike WaveSP-Net). Reimplementation risk.
- **WaveSP-Net needs Python 3.8** (sidecar), Mamba (`mamba-ssm`).
- **No execution performed.** Every number above is from the papers, not measured here.

---

## What I did NOT re-propose (closed)

D3/SPLIT/WaveRep temporal stats (D-099/100/104) · ReStraV geometry (D-112/113) · V-JEPA2 frozen SSL (D-125) · GenD CLIP (D-127/128) · MLLM-as-detector (chance-level in this research) · offset_abs AV (D-088) · per-generator image heads · ASVspoof5 standalone · chasing 100/100.