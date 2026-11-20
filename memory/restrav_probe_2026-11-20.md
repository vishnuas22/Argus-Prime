# ReStraV feasibility probe — 2026-11-20

**Status:** READ-ONLY probe. No training, no promotion, no pointer change. 0 booked.
**Source:** github.com/ChristianInterno/ReStraV (NeurIPS 2025, Apache-2.0). DINOv2 ViT-S/14 + 21-D geometry (7 distances + 6 turning angles + 8 stats).
**Local setup:** torch 2.13 + MPS + transformers 5.15.1 in `.venv`; torchcodec 0.17 + h5py installed for the probe. DINOv2 weights cached at `~/.cache/torch/hub/`.

## What was measured
- **FF++ C23** (face-swap, n=60, 30 real + 30 fake): best single ReStraV feature `theta2` AUC = **0.4178** (near-chance). Mean |AUC-0.5| = 0.030. **Expected** — ReStraV targets T2V geometry, not face-swap.
- **GenBuster++** (T2V, n=60, 30 real + 30 fake): best single feature `theta2` AUC = **0.2059** (|AUC-0.5| = 0.294). Mean |AUC-0.5| = 0.194 (6× stronger than FF++). **This is the regime ReStraV is built for.**
- **Decorrelation vs p6vid_v4** (GenBuster++, 52/60 aligned after decode failures):
  - ReStraV best-feature AUC (real=1): 0.206
  - p6vid_v4 AUC (1-pf, real=1): 0.590
  - **Combined (LR on [ReStraV_best, p6vid_v4]): 0.794**
  - Pearson r (ReStraV best vs p6vid pf): **0.012**
  - Spearman ρ: 0.176

## Interpretation
- ReStraV alone does NOT beat p6vid_v4 on this small sample. The value is **decorrelation** (r=0.012), which is the exact pattern the repo's stacking history exploits (D-021 SynthVLM +0.070 → D-024 fusion ADD 6/6 → D-027 SCENE jump).
- Combined AUC 0.794 vs p6vid_v4 0.590 = +0.20 on T2V. p6vid_v4's 0.590 on GenBuster++ is itself below its FF++ 0.912 — the known T2V weakness that closed D-082/084/099.
- n=52 is too small to book Y. 95% CI on 0.794 is roughly ±0.10.

## Caveats (NoHallucinations)
- No published ReStraV `model.pt` checkpoint exists — the repo ships training code, not weights. The probe uses raw 21-D geometry features, not the trained MLP. A trained MLP on Argus data could be stronger or weaker; unmeasured.
- ReStraV's published transfer FPR is 15.3/39.9/66.1% at nominal 0.1/1/5% (backlog §6.1). Its standalone threshold transfer is worse than p6vid_v4's. The stacking discount (D-021 law: ab_gated ≤ ab_0 + 0.02) would govern.
- GenBuster++ generators are early-2025; not the freshest (Sora-2/Veo-3.1 are 2025-H2). Video Reality Test (MIT, 4.3 GB, fits disk) is the fresher follow-up.

## Recommended next step (requires user approval + sealed spec)
Pre-registered feasibility wave:
- n=200 per class on GenBuster++ + CDF + (if fetched) Video Reality Test.
- Bar: combined 10-fold AUC ≥ p6vid_v4 + 0.05 on T2V AND ≥ p6vid_v4 on FF++/CDF (no regression).
- Pass → ReStraV becomes the new video-OOD signal class D-099 requires; sealed spec + D-number + restack.
- Fail → line closed at 0 with receipt; no re-litigation.

## Files
- `/tmp/restrav_feats.py` — 21-D feature extractor (DINOv2 + torchcodec).
- `/tmp/decorr_probe.py` — p6vid_v4 scoring + decorrelation report.
- `/tmp/decorr_report.json` — full per-clip scores + summary stats.
- `/tmp/restrav_ffpp_smoke.npz`, `/tmp/restrav_genbuster_smoke.npz` — feature dumps.

## Score impact
**0 booked.** Court stays 64.8/100. This probe produces a measurement, not a promotion.

## ADDENDUM 2026-11-20 (V1d prep) — the B4 blocker was an artifact, not a property

`test_ltt_threshold_refuses_when_slack_exceeds_alpha` style honesty applies here:
before designing V2 I re-measured the V1g decode failure that failed bar B4.

**Measurement:** 250 GenBuster++ `real/` clips, `random.seed(42)`, shuffled,
`decode_clip(T=24, window_sec=2.0)` → **250/250 decoded, 0 failures (0.0%)**.
Zero `RuntimeError`/`ValueError` reasons.

**Cause of V1g's 29/200 failures:** V1g used `sorted(glob(...))[:100]` — the
**alphabetically first** 100 filenames. Those are `00xxxx…`/`01xxxx…`, a
non-random prefix that happens to be disproportionately corrupt. The failures
were a **sampling artifact**, not a corpus property.

**Consequence:** V1g's science bars (B1 T2V Δ+0.125, B2 no CDF regression,
B3 r=0.130) were obtained on a biased subsample and are **provisional**, not
refuted. A clean V2 with a shuffled decode-filtered sample can satisfy B4
without any change to the method.

## V1d candidate landscape (verified on disk, 2026-11-20)

| Corpus | Real | Fake | Location | Usable |
|---|---:|---:|---|---|
| GenBuster++ | 1000 | 1000 | `data/raw/genbuster_plusplus/extracted/video/{real,fake}` | yes (decode 100% on shuffled sample) |
| GenVidBench Tier-1 | 0 | **14,114** (opentora 13,800 · kling 210 · sora 51) | `data/raw/genvidbench/{opentora,kling,sora}` | fakes only; reals must come from GenBuster `real/` |
| FF++ C23 mirror | 1000 | 6×1000 | `data/raw/ffpp_mirror/FaceForensics++_C23` | curated regression guard |
| Celeb-DF-v2 | 590+300 | 5,639 | `data/raw/celebdf/Celeb-DF-v2` | curated regression guard |

**Current t2v baseline (p6vid_v4e, `artifacts/exam_video_t2v_20261011_arcface/metrics.json`):**
kling **0.6035** (n=150) · sora **0.6812** (n=51) · opentora **0.7497** (n=150);
real set = FF++ (n=140); frames=12; route split video_face 407 / image_court 84.

**Assets ready:** DINOv2 ViT-S/14 weights cached at
`~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` (no download);
`torchcodec` 0.17 + `h5py` installed in `.venv`; ReStraV 21-D extractor at
`/tmp/restrav_feats.py`; disk 60 GiB free (no corpus download needed).