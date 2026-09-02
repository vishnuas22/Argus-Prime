# ARGUS_MASTER — P0 Baseline Report Card

Corpus: 150 real (COCO val2017 slice) vs 40 fake (SD 1.5 local). Labels: 1=fake. Module discriminant = −evidence_score. Fused = 1−P(camera_original).

## Detection performance by laundering rung

| Variant | Metric | metadata | compression | spectral | realness | perturbation | learned | FUSED |
|---|---|---|---|---|---|---|---|---|
| clean | AUC | 0.500 | 0.493 | 0.593 | 0.577 | 0.464 | 0.873 | 0.734 |
| clean | EER | 0.500 | 1.000 | 0.244 | 0.448 | 0.513 | 0.193 | 0.307 |

## Fusion behavior (clean rung)

- Brier(P(real)): `0.1817`
- Abstention rate: `1.0`
- Mean trust score: `0.6351`
- Mean risk score: `0.0019`
- Evidence-capacity mix: {'HIGH': 60, 'MODERATE': 130}

## Runtime (ms/image, clean rung)

| Stage | ms |
|---|---|
| triage | 44.9 |
| metadata | 130.7 |
| compression_history | 141.6 |
| spectral_probe | 106.7 |
| real_distribution_probe | 176.7 |
| perturbation_probe | 242.4 |
| learned_aigc_probe | 91.8 |

## Reading guide

- AUC 0.5 = coin flip; higher is better. EER lower is better.
- The gap between `clean` and `q50_r05` AUCs measures laundering fragility.
- Modules unavailable (missing envelope/reference data) show as —; build Argus data files before trusting those rows.
