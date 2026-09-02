# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 150 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | FUSED |
|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.441 | 0.469 | 0.590 | 0.507 |
| clean | EER | 0.500 | 1.000 | 0.532 | 0.540 | 0.417 | 0.507 |
| jpeg50 | AUC | 0.500 | 0.441 | 0.454 | 0.469 | 0.590 | 0.465 |
| jpeg50 | EER | 0.500 | 0.820 | 0.525 | 0.540 | 0.417 | 0.527 |
| jpeg75 | AUC | 0.500 | 0.441 | 0.454 | 0.469 | 0.590 | 0.465 |
| jpeg75 | EER | 0.500 | 0.820 | 0.525 | 0.540 | 0.417 | 0.527 |
| q50_r05 | AUC | 0.500 | 0.523 | 0.510 | 0.470 | 0.604 | 0.569 |
| q50_r05 | EER | 0.500 | 0.370 | 0.495 | 0.537 | 0.400 | 0.460 |
| resize50 | AUC | 0.500 | 0.500 | 0.510 | 0.470 | 0.604 | 0.542 |
| resize50 | EER | 0.500 | 0.500 | 0.495 | 0.537 | 0.400 | 0.487 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.2564`
- Abstention rate: `1.0`
- Mean trust score: `0.672`
- Mean risk score: `0.0012`
- Evidence-capacity mix: {'HIGH': 170, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 42.7 |
| metadata | 103.1 |
| compression_history | 118.0 |
| spectral_probe | 104.9 |
| real_distribution_probe | 165.3 |
| perturbation_probe | 241.5 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
