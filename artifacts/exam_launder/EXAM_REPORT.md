# Full-court laundering/screenshot robustness exam

Court: p6v8 · 166 images x 5 views

## Pivot

| corpus | view | AUC | FPR | TPR | abstain | n |
|---|---|---|---|---|---|---|
| eval_real | pristine | nan | 0.000 | 0.000 | 0.11 | 45 |
| train_fake | pristine | nan | 0.000 | 1.000 | 0.00 | 45 |
| cdf_flat_fake | pristine | nan | 0.000 | 1.000 | 0.12 | 25 |
| sg2_probe | pristine | nan | 0.000 | 1.000 | 0.18 | 11 |
| flux_train | pristine | nan | 0.000 | 1.000 | 0.30 | 20 |
| mj_train | pristine | nan | 0.000 | 1.000 | 0.25 | 20 |
| eval_real | jpeg_q50 | nan | 0.000 | 0.000 | 0.18 | 45 |
| train_fake | jpeg_q50 | nan | 0.000 | 0.933 | 0.87 | 45 |
| cdf_flat_fake | jpeg_q50 | nan | 0.000 | 1.000 | 0.16 | 25 |
| sg2_probe | jpeg_q50 | nan | 0.000 | 1.000 | 1.00 | 11 |
| flux_train | jpeg_q50 | nan | 0.000 | 0.900 | 0.95 | 20 |
| mj_train | jpeg_q50 | nan | 0.000 | 1.000 | 0.80 | 20 |
| eval_real | whatsapp | nan | 0.000 | 0.000 | 0.16 | 45 |
| train_fake | whatsapp | nan | 0.000 | 0.933 | 0.60 | 45 |
| cdf_flat_fake | whatsapp | nan | 0.000 | 0.960 | 0.28 | 25 |
| sg2_probe | whatsapp | nan | 0.000 | 1.000 | 1.00 | 11 |
| flux_train | whatsapp | nan | 0.000 | 0.750 | 1.00 | 20 |
| mj_train | whatsapp | nan | 0.000 | 1.000 | 0.95 | 20 |
| eval_real | screenshot | nan | 0.022 | 0.000 | 0.56 | 45 |
| train_fake | screenshot | nan | 0.000 | 0.978 | 0.18 | 45 |
| cdf_flat_fake | screenshot | nan | 0.000 | 1.000 | 0.24 | 25 |
| sg2_probe | screenshot | nan | 0.000 | 0.818 | 0.55 | 11 |
| flux_train | screenshot | nan | 0.000 | 0.900 | 0.50 | 20 |
| mj_train | screenshot | nan | 0.000 | 0.950 | 0.30 | 20 |
| eval_real | screenshot_jpg | nan | 0.044 | 0.000 | 0.67 | 45 |
| train_fake | screenshot_jpg | nan | 0.000 | 0.978 | 0.91 | 45 |
| cdf_flat_fake | screenshot_jpg | nan | 0.000 | 1.000 | 1.00 | 25 |
| sg2_probe | screenshot_jpg | nan | 0.000 | 0.909 | 1.00 | 11 |
| flux_train | screenshot_jpg | nan | 0.000 | 1.000 | 0.55 | 20 |
| mj_train | screenshot_jpg | nan | 0.000 | 1.000 | 0.95 | 20 |

## Verdict: ALL GATES PASS
