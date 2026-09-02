# P0 FINDINGS — First Quantitative Truth About Argus_Master

Date: 2026-08-24 · Corpus: 150 COCO-reals vs 150 SD1.5-fakes (matched COCO-style content) · 5 laundering rungs · n=300/rung

## Headline

**The pipeline as-built performs at chance against its target threat on matched-content synthetic images.**
Fused clean AUC **0.507** (coin flip; 95% CI ≈ ±0.06). Best single module: perturbation probe at **0.590**.

## Module-by-module (clean rung)

| Module | AUC | Verdict |
|---|---|---|
| metadata | 0.500 | structurally blind here (both classes lack EXIF on this corpus) |
| compression_history | 0.493 | blind (SD PNGs have uniform single-generation history; reals likewise) |
| spectral_probe | 0.441 | **below chance** — inverted/absent signal for SD1.5@512 |
| real_distribution_probe (DINOv2 kNN) | 0.469 | chance — one-class usage wastes foundation features |
| perturbation_probe (RIGID-style) | 0.590 | weak but real signal; best in panel |
| FUSED | 0.507 | no signal to fuse |

## What WORKS (important — not everything failed)

1. **Triage machinery is functional**: evidence-capacity mix degrades correctly under laundering
   (clean: 170 HIGH / 130 MODERATE → q50_r05: 49 HIGH / 251 MODERATE). The r_m(d)
   reliability-conditioning idea is implemented and behaves as designed.
2. **The court is honest**: abstention rate 1.0 at α=0.10. It never commits because module
   probabilities carry no signal — fail-safe behavior confirmed.
3. **Runtime is production-viable**: ~776 ms/image full panel on M1 Max CPU (triage 43ms +
   five modules), DINOv2 embedding dominating.

## Root-cause analysis (why the panel is blind)

- **Semantic embeddings ≠ artifact detectors.** DINOv2 kNN measures content-similarity to a
  real reference. SD-generated everyday scenes are semantically indistinguishable from COCO —
  distances overlap completely. Published SOTA (UnivFD et al.) gets ≥0.9 AUC from the SAME
  class of features only with SUPERVISED heads. One-class usage discards the signal.
- **Handcrafted thresholds tuned for GAN-era signatures** (spectral peaks, HF deficit) don't
  fire on diffusion upsamplers at these settings — and slightly anti-correlate (0.441).
- **Fusion cannot manufacture signal** — voting over five near-uniform streams stays uniform.

## Corpus caveats (honesty section)

- n=300/rung → AUC CI ±0.06; conclusions about "weak" robust, exact values noisy.
- ONE generator (SD1.5, 512px, 20 steps, COCA-style matched prompts — deliberately hostile
  to shortcut learning). Findings describe this regime, not all regimes.
- EER column recomputed via ROC interpolation after discovering naive-sort EER degenerates
  on tied scores (see memory/lessons.md).

## Implications for the unified plan (evidence-ranked priorities)

1. **P0 vindicated**: benchmark-first saved weeks of building on an unmeasured base.
2. **Promote P5 (supervised IMAGE-AIGC probe) to next-build position** — CLIP linear probe
   (UnivFD recipe) trained on GenImage + our SD corpus, plugged into the SAME harness.
   Expected step-change to ≥0.85 AUC if literature transfers.
3. Keep triage/court shell untouched — proven functional.
4. Fusion upgrade (P6) requires calibrated per-module likelihoods first; current
   logistic-squash saturates and destroys ranking information near extremes.
5. Spectral probe needs recalibration against diffusion-era envelope before being trusted,
   or demotion to optional.

## Reproduce

```bash
PATH="$PWD/tools/bin:$PATH" ./.venv/bin/python deepfake/bench/run_benchmark.py --limit 150 --out artifacts/p0_baseline
# or recompute metrics only:
./.venv/bin/python deepfake/bench/recompute.py artifacts/p0_baseline
```
