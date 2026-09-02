# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 150 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.441 | 0.469 | 0.590 | 0.999 | 0.972 |
| clean | EER | 0.500 | 1.000 | 0.532 | 0.540 | 0.417 | 0.020 | 0.093 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1484`
- Abstention rate: `1.0`
- Mean trust score: `0.6738`
- Mean risk score: `0.0012`
- Evidence-capacity mix: {'HIGH': 170, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 44.6 |
| metadata | 126.9 |
| compression_history | 127.9 |
| spectral_probe | 106.5 |
| real_distribution_probe | 168.3 |
| perturbation_probe | 244.5 |
| learned_aigc_probe | 80.4 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
