# Argus Deepfake Benchmark (P0)

Scores the vendored `integrations/Argus_Master` evidence pipeline on a labeled
corpus with laundering variants, producing the system's first quantitative
report card.

## Pipeline order

```bash
./.venv/bin/python -m pip install -r deepfake/requirements.txt   # once

bash deepfake/scripts/fetch_coco.sh          # COCO val2017 -> /tmp/val2017
python deepfake/scripts/prepare_corpus.py    # ref/calib -> /tmp/argus_corpus, eval slice -> data/corpora/eval_real

# Argus calibration data (needs reference + calib slices above)
cd integrations/Argus_Master/backend && python scripts/build_envelope.py && python scripts/build_reference.py

python deepfake/scripts/generate_fakes.py    # SD1.5-on-MPS fake corpus -> data/corpora/eval_fake

python deepfake/bench/run_benchmark.py       # scores.csv + metrics.json + REPORT.md -> artifacts/p0_baseline
```

## Corpus

| Split | Source | N | Use |
|---|---|---|---|
| reference | COCO val2017 [0:2500] | 2500 | DINOv2 kNN reference + spectral envelope |
| calib | COCO val2017 [2500:2900] | 400 | kNN/perturb calibration |
| eval_real | COCO val2017 [2900:3400] | 500 | scoring (real class) |
| eval_fake | SD 1.5 local generation | 300 | scoring (fake class), COCO-style prompts |

Deterministic slices are disjoint; regeneration is reproducible.

## Reading the report card

`artifacts/p0_baseline/REPORT.md` — per-module + fused AUC/EER across five
laundering rungs (clean, JPEG75, JPEG50, resize0.5, JPEG50+resize0.5),
calibration (Brier), abstention rate, trust/risk means, runtime ms/image.
