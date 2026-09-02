# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.542 | 0.591 | 0.422 | 0.999 | 0.983 |
| clean | EER | 0.500 | 1.000 | 0.381 | 0.465 | 0.571 | 0.029 | 0.057 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1459`
- Abstention rate: `1.0`
- Mean trust score: `0.633`
- Mean risk score: `0.0019`
- Evidence-capacity mix: {'HIGH': 55, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 42.2 |
| metadata | 125.8 |
| compression_history | 132.0 |
| spectral_probe | 102.9 |
| real_distribution_probe | 165.4 |
| perturbation_probe | 231.8 |
| learned_aigc_probe | 89.1 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
