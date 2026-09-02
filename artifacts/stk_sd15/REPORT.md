# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 60 real (COCO val2017 slice) vs 60 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.500 | 0.484 | 0.420 | 0.581 | 1.000 | 0.985 |
| clean | EER | 0.500 | 0.500 | 0.508 | 0.567 | 0.417 | 0.000 | 0.050 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1448`
- Abstention rate: `1.0`
- Mean trust score: `0.6765`
- Mean risk score: `0.0`
- Evidence-capacity mix: {'MODERATE': 50, 'HIGH': 70}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 45.7 |
| metadata | 129.7 |
| compression_history | 126.8 |
| spectral_probe | 104.8 |
| real_distribution_probe | 181.9 |
| perturbation_probe | 242.9 |
| learned_aigc_probe | 103.8 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
