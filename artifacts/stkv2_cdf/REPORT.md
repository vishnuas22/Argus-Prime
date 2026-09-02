# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 150 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | learned_face_probe | FUSED |
|---|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.480 | 0.629 | 0.468 | 0.430 | 1.000 | 0.997 |
| clean | EER | 0.500 | 0.500 | 0.521 | 0.420 | 0.520 | 0.580 | 0.007 | 0.013 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1714`
- Abstention rate: `1.0`
- Mean trust score: `0.6701`
- Mean risk score: `0.18`
- Evidence-capacity mix: {'MODERATE': 158, 'HIGH': 142}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 9.6 |
| metadata | 119.8 |
| compression_history | 64.5 |
| spectral_probe | 102.7 |
| real_distribution_probe | 160.9 |
| perturbation_probe | 234.4 |
| learned_aigc_probe | 75.3 |
| learned_face_probe | 54.5 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
