# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 150 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.441 | 0.469 | 0.590 | 0.999 | 0.990 |
| clean | EER | 0.500 | 1.000 | 0.532 | 0.540 | 0.417 | 0.020 | 0.013 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.0122`
- Abstention rate: `0.0067`
- Mean trust score: `0.7731`
- Mean risk score: `0.0012`
- Evidence-capacity mix: {'HIGH': 170, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 51.1 |
| metadata | 132.2 |
| compression_history | 127.8 |
| spectral_probe | 105.7 |
| real_distribution_probe | 170.9 |
| perturbation_probe | 251.2 |
| learned_aigc_probe | 79.4 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
