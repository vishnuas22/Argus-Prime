# CROSS-GENERATOR EVALUATION — Detector or Fingerprint Checker?

Date: 2026-08-24 · Probe: CLIP ViT-L/14 linear (P5), trained ONLY on COCO-reals vs SD1.5-fakes.
Unseen-generator fakes generated locally (40/generator, COCO-style prompts, disjoint seeds).
Reals: COCO val2017 slice (150). Clean rung, no laundering. n=190/rung → CI ≈ ±0.07 on xgen rows.

## Per-generator results

| Generator (architecture) | Seen in training? | learned AUC | learned EER | legacy best | FUSED |
|---|---|---|---|---|---|
| SD 1.5 (UNet, ref) | YES | 0.999 | 1.3% | 0.590 perturb | 0.970 |
| **SDXL 1.0** (scaled UNet, new VAE) | no | **0.998** | 2.7% | 0.548 realness | 0.974 |
| **SANA 1600M** (DiT + DC-AE autoencoder) | no | **0.873** | 19.3% | 0.593 spectral | 0.734 |

## Verdict: a real detector — with a mapped frontier

1. **Not an SD-fingerprint checker.** SDXL shares no checkpoints/VAE/scheduler with SD1.5,
   yet transfer is near-perfect (0.998). The probe reads artifact-level statistics of the
   diffusion-photograph manifold, not pipeline-specific noise signatures.
2. **The frontier is architectural distance.** SANA's DC-AE compression + pure-DiT stack
   costs ~13 AUC points. Generalization degrades gracefully, not catastrophically.
3. **The legacy panel is blind everywhere** (all ≤0.59 across all three generators),
   confirming P0's diagnosis was about the panel design, not one corpus quirk.
4. **Fusion dilution reconfirmed**: fused < probe on every unseen generator because four
   blind modules outvote one expert. P6 trained-stacker upgrade is now evidence-mandatory.

## Consequences for the roadmap

- Multi-generator training data is the highest-leverage robustness investment: adding even
  ONE non-UNet family (SANA-class) to training should recover most of the SANA gap.
  → Full-P5 training mix: {SD1.5 ✓, SDXL ✓, SANA ✓} × COCO-reals, then re-eval all three.
- PixArt-α and Stable-Cascade repos returned HTTP 401 during this phase (auth-restricted /
  deprecated-out) — documented as unavailable supply lines; revisit via GenImage mirror later.
- FACE head (P3) proceeds next per plan; its eval inherits this cross-family harness pattern.

## Reproduce

```bash
./.venv/bin/python deepfake/scripts/gen_cross_fakes.py xgen_sana xgen_sdxl
for g in sana sdxl sd15_ref; do
  FAKE=data/corpora/xgen_$g; [ $g = sd15_ref ] && FAKE=data/corpora/eval_fake
  PATH="$PWD/tools/bin:$PATH" ./.venv/bin/python deepfake/bench/run_benchmark.py \
    --limit 150 --no-ladder --out artifacts/xgen_$g --fake $FAKE --with-learned artifacts/p5/probe.joblib
done
```
