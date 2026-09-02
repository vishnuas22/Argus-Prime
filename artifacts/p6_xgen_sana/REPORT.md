# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.542 | 0.591 | 0.422 | 0.999 | 0.986 |
| clean | EER | 0.500 | 1.000 | 0.381 | 0.465 | 0.571 | 0.029 | 0.029 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.0074`
- Abstention rate: `0.0054`
- Mean trust score: `0.7325`
- Mean risk score: `0.0019`
- Evidence-capacity mix: {'HIGH': 55, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 55.9 |
| metadata | 139.1 |
| compression_history | 155.1 |
| spectral_probe | 114.5 |
| real_distribution_probe | 190.6 |
| perturbation_probe | 271.2 |
| learned_aigc_probe | 100.2 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
