# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.429 | 0.578 | 0.493 | 1.000 | 0.977 |
| clean | EER | 0.500 | 1.000 | 0.547 | 0.486 | 0.486 | 0.000 | 0.073 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1465`
- Abstention rate: `1.0`
- Mean trust score: `0.6355`
- Mean risk score: `0.0019`
- Evidence-capacity mix: {'HIGH': 55, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 40.6 |
| metadata | 128.5 |
| compression_history | 119.2 |
| spectral_probe | 105.0 |
| real_distribution_probe | 169.0 |
| perturbation_probe | 240.4 |
| learned_aigc_probe | 87.1 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
