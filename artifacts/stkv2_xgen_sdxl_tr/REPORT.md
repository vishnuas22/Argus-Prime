# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 50 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | learned_face_probe | FUSED |
|---|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.503 | 0.533 | 0.426 | 1.000 | 0.726 | 0.979 |
| clean | EER | 0.500 | 0.500 | 0.499 | 0.540 | 0.600 | 0.000 | 0.250 | 0.086 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.139`
- Abstention rate: `1.0`
- Mean trust score: `0.6522`
- Mean risk score: `0.0`
- Evidence-capacity mix: {'MODERATE': 43, 'HIGH': 42}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 45.7 |
| metadata | 130.2 |
| compression_history | 128.1 |
| spectral_probe | 107.3 |
| real_distribution_probe | 193.8 |
| perturbation_probe | 248.4 |
| learned_aigc_probe | 124.6 |
| learned_face_probe | 30.6 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
