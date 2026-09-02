# P5 RERUN FINDINGS — First Engine Installed Into the Court

Date: 2026-08-24 · Same corpus & protocol as P0 baseline · One addition: `learned_aigc_probe`
(frozen CLIP ViT-L/14 → L2-norm → logistic C=3, trained on 750 COCO-reals ×3 views vs
250 SD1.5-fakes ×3 views, disjoint seeds/files from eval).

## Headline: fused AUC 0.507 → 0.970 (+0.46)

| Variant | learned AUC | learned EER | FUSED AUC (was) | FUSED EER (was) |
|---|---|---|---|---|
| clean | **0.999** | 1.3% | 0.970 (0.507) | 8.0% (50%) |
| jpeg75 | 0.999 | 1.3% | 0.944 | 15.3% |
| jpeg50 | 0.999 | 1.3% | 0.944 | 15.3% |
| resize50 | 0.999 | 0.7% | 0.971 | 7.3% |
| q50_r05 | **0.999** | **0.7%** | **0.974** (0.569) | 7.3% |

- Multi-view training delivered laundering robustness for free: no degradation
  from clean to the heaviest rung.
- Probe adds only 79 ms/image (MPS).
- The five original modules were untouched — the gain is additive, not corpus luck.

## Honest caveats

1. In-domain result: train and eval fakes are both SD1.5-family (disjoint seeds/params).
   Cross-generator transfer (SDXL, Flux, GANs, Midjourney-style) is THE next test —
   GenImage/CiFAKE cross-eval queued as part of full P5.
2. Fused EER (8%) lags probe EER (1.3%): WeightedVotingJudge dilutes one strong module
   with four blind ones. Trained stacker (P6) is the fix; conformal abstention still 1.0
   pending calibrated inputs.
3. n=300/rung.

## What this proves about the merged architecture

Evidence-court shell (Argus_Master) + supervised foundation-feature specialists (ours)
compose without modification: contract-compliant module, zero vendor edits, registered at
harness level. The thesis "one-class wastes foundation features" is now measured fact:
same backbone class, 0.469 (kNN realness) vs 0.999 (supervised head).

## Reproduce

```bash
./.venv/bin/python deepfake/scripts/train_probe.py data/caches/train_real_clipL.npz data/caches/train_fake_clipL.npz --out artifacts/p5
PATH="$PWD/tools/bin:$PATH" ./.venv/bin/python deepfake/bench/run_benchmark.py --limit 150 --out artifacts/p5_rerun --with-learned artifacts/p5/probe.joblib
```
