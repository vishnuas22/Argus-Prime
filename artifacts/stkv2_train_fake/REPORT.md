# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 50 real (COCO val2017 slice) vs 50 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | learned_face_probe | FUSED |
|---|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.490 | 0.449 | 0.603 | 1.000 | 0.933 | 0.995 |
| clean | EER | 0.500 | 0.500 | 0.505 | 0.540 | 0.400 | 0.000 | 0.125 | 0.020 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1292`
- Abstention rate: `1.0`
- Mean trust score: `0.6616`
- Mean risk score: `0.0`
- Evidence-capacity mix: {'MODERATE': 43, 'HIGH': 57}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 44.3 |
| metadata | 125.8 |
| compression_history | 122.7 |
| spectral_probe | 104.4 |
| real_distribution_probe | 181.2 |
| perturbation_probe | 234.6 |
| learned_aigc_probe | 113.4 |
| learned_face_probe | 31.4 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
