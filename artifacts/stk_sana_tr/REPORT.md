# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 60 real (COCO val2017 slice) vs 35 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.624 | 0.529 | 0.437 | 1.000 | 0.986 |
| clean | EER | 0.500 | 0.500 | 0.099 | 0.495 | 0.568 | 0.000 | 0.033 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1474`
- Abstention rate: `1.0`
- Mean trust score: `0.6613`
- Mean risk score: `0.0`
- Evidence-capacity mix: {'MODERATE': 50, 'HIGH': 45}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 46.9 |
| metadata | 126.1 |
| compression_history | 151.9 |
| spectral_probe | 103.6 |
| real_distribution_probe | 182.6 |
| perturbation_probe | 230.8 |
| learned_aigc_probe | 116.5 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
