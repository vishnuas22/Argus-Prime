# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 40 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.481 | 0.548 | 0.415 | 0.998 | 0.974 |
| clean | EER | 0.500 | 1.000 | 0.511 | 0.525 | 0.560 | 0.027 | 0.053 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1496`
- Abstention rate: `1.0`
- Mean trust score: `0.6363`
- Mean risk score: `0.0019`
- Evidence-capacity mix: {'HIGH': 60, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 45.7 |
| metadata | 131.6 |
| compression_history | 131.4 |
| spectral_probe | 109.8 |
| real_distribution_probe | 178.1 |
| perturbation_probe | 251.8 |
| learned_aigc_probe | 89.3 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
