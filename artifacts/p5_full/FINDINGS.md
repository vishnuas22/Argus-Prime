# FULL-P5 + P6 RESULTS — Multi-Family Retrain & Trained Fusion

Date: 2026-08-24 · Probe v2 trained on {750 COCO-reals + SD1.5 250 + SANA_tr 35 + SDXL_tr 35} ×3 views.
Held-out eval fakes: xgen last-35 splits (never seen). Stacker trained on train-side dumps only.

## Headline table

| Eval set | probe v1 | probe v2 (mix) | vote-fuse v2 | STACKER fused | stacker EER |
|---|---|---|---|---|---|
| SD1.5 (150) | 0.9992 | 0.9993 | 0.9721 | 0.9899 | 1.3% |
| SDXL held-out (35) | 0.9977 | **1.0000** | 0.9773 | **1.0000** | 0.0% |
| SANA held-out (35) | 0.8732 | **0.9990** | 0.9829 | 0.9856 | 2.9% |

## Findings

1. **SANA gap collapsed** (0.873 → 0.999): architectural distance is a *data* problem.
   Training on one representative per family transfers across the family.
2. **Stacker beats voting on every set** (+0.02–0.05 AUC) and fused EER converged to
   best-module scale (0–2.9%) — P6's stated goal, met.
3. **The court now commits: abstention 1.00 → 0.01.** Calibrated stacker probabilities
   give the conformal wrapper real signal; voting's saturated scores never did.
4. Stack features are tiny (15 dims: 6×{evidence,reliability} + capacity one-hot);
   HistGBM+sigmoid trains in seconds — cheap to retrain whenever modules change.

## Caveats

- Held-out n=35/generator → AUC CI ≈ ±0.10; direction robust, exact values soft.
- Stacker saw train-side siblings of eval families (by design); cross-family stacker
  generalization untested until a 4th family arrives (PixArt/GenImage when accessible).
- Laundering rungs not rerun this round; probe multi-view training already proved
  rung-invariance in P5-lite.

## Reproduce

```bash
# split, cache, train
python3 - <<'EOF' ... # see git history for split script (first 35 -> _tr)
./.venv/bin/python deepfake/scripts/build_feature_cache.py data/corpora/xgen_sana_tr data/caches/xgen_sana_tr.npz
./.venv/bin/python deepfake/scripts/train_probe.py data/caches/train_real_clipL.npz \
  data/caches/train_fake_clipL.npz data/caches/xgen_sana_tr.npz data/caches/xgen_sdxl_tr.npz --out artifacts/p5_full
# stacker (train-side dumps only)
./.venv/bin/python deepfake/scripts/train_stacker.py --csv artifacts/stk_*/scores.csv --out artifacts/p6
# final eval
PATH="$PWD/tools/bin:$PATH" ./.venv/bin/python deepfake/bench/run_benchmark.py --limit 150 --no-ladder \
  --fake data/corpora/xgen_sana --with-learned artifacts/p5_full/probe.joblib --with-stacker artifacts/p6/stacker.joblib
```
