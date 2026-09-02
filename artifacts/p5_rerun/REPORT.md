# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 150 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.441 | 0.469 | 0.590 | 0.999 | 0.970 |
| clean | EER | 0.500 | 1.000 | 0.532 | 0.540 | 0.417 | 0.013 | 0.080 |
| jpeg75 | AUC | 0.500 | 0.441 | 0.454 | 0.469 | 0.590 | 0.999 | 0.944 |
| jpeg75 | EER | 0.500 | 0.820 | 0.525 | 0.540 | 0.417 | 0.013 | 0.153 |
| jpeg50 | AUC | 0.500 | 0.441 | 0.454 | 0.469 | 0.590 | 0.999 | 0.944 |
| jpeg50 | EER | 0.500 | 0.820 | 0.525 | 0.540 | 0.417 | 0.013 | 0.153 |
| resize50 | AUC | 0.500 | 0.500 | 0.510 | 0.470 | 0.604 | 0.999 | 0.971 |
| resize50 | EER | 0.500 | 0.500 | 0.495 | 0.537 | 0.400 | 0.007 | 0.073 |
| q50_r05 | AUC | 0.500 | 0.523 | 0.510 | 0.470 | 0.604 | 0.999 | 0.974 |
| q50_r05 | EER | 0.500 | 0.370 | 0.495 | 0.537 | 0.400 | 0.007 | 0.073 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1499`
- Abstention rate: `1.0`
- Mean trust score: `0.6738`
- Mean risk score: `0.0012`
- Evidence-capacity mix: {'HIGH': 170, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 46.3 |
| metadata | 129.8 |
| compression_history | 124.8 |
| spectral_probe | 106.0 |
| real_distribution_probe | 167.4 |
| perturbation_probe | 243.9 |
| learned_aigc_probe | 79.4 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
