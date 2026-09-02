# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 50 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | learned_face_probe | FUSED |
|---|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.623 | 0.550 | 0.458 | 1.000 | 0.806 | 0.982 |
| clean | EER | 0.500 | 0.500 | 0.099 | 0.482 | 0.540 | 0.000 | 0.333 | 0.040 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1342`
- Abstention rate: `1.0`
- Mean trust score: `0.6508`
- Mean risk score: `0.0`
- Evidence-capacity mix: {'MODERATE': 43, 'HIGH': 42}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 46.5 |
| metadata | 125.8 |
| compression_history | 152.7 |
| spectral_probe | 104.4 |
| real_distribution_probe | 194.2 |
| perturbation_probe | 242.7 |
| learned_aigc_probe | 125.3 |
| learned_face_probe | 38.9 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
