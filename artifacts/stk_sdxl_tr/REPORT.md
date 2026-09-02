# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 60 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.506 | 0.516 | 0.403 | 1.000 | 0.975 |
| clean | EER | 0.500 | 0.500 | 0.497 | 0.571 | 0.600 | 0.000 | 0.050 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1513`
- Abstention rate: `1.0`
- Mean trust score: `0.6633`
- Mean risk score: `0.0`
- Evidence-capacity mix: {'MODERATE': 50, 'HIGH': 45}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 44.0 |
| metadata | 126.8 |
| compression_history | 120.3 |
| spectral_probe | 104.3 |
| real_distribution_probe | 186.7 |
| perturbation_probe | 245.6 |
| learned_aigc_probe | 110.8 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
