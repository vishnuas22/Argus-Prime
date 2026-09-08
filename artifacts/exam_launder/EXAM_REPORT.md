# Full-court laundering/screenshot robustness exam

Court: p6v7 · 166 images x 5 views

## Pivot

| corpus | view | AUC | FPR | TPR | abstain | n |
|---|---|---|---|---|---|---|
| eval_real | pristine | nan | 0.000 | 0.000 | 0.11 | 45 |
| train_fake | pristine | nan | 0.000 | 1.000 | 0.00 | 45 |
| cdf_flat_fake | pristine | nan | 0.000 | 1.000 | 0.04 | 25 |
| sg2_probe | pristine | nan | 0.000 | 1.000 | 0.45 | 11 |
| flux_train | pristine | nan | 0.000 | 1.000 | 0.25 | 20 |
| mj_train | pristine | nan | 0.000 | 1.000 | 0.05 | 20 |
| eval_real | jpeg_q50 | nan | 0.022 | 0.000 | 0.16 | 45 |
| train_fake | jpeg_q50 | nan | 0.000 | 0.889 | 0.78 | 45 |
| cdf_flat_fake | jpeg_q50 | nan | 0.000 | 0.960 | 0.16 | 25 |
| sg2_probe | jpeg_q50 | nan | 0.000 | 0.909 | 1.00 | 11 |
| flux_train | jpeg_q50 | nan | 0.000 | 0.900 | 0.90 | 20 |
| mj_train | jpeg_q50 | nan | 0.000 | 1.000 | 0.70 | 20 |
| eval_real | whatsapp | nan | 0.000 | 0.000 | 0.13 | 45 |
| train_fake | whatsapp | nan | 0.000 | 0.933 | 0.33 | 45 |
| cdf_flat_fake | whatsapp | nan | 0.000 | 0.920 | 0.24 | 25 |
| sg2_probe | whatsapp | nan | 0.000 | 0.909 | 1.00 | 11 |
| flux_train | whatsapp | nan | 0.000 | 0.750 | 0.95 | 20 |
| mj_train | whatsapp | nan | 0.000 | 1.000 | 0.90 | 20 |
| eval_real | screenshot | nan | 0.111 | 0.000 | 0.36 | 45 |
| train_fake | screenshot | nan | 0.000 | 0.956 | 0.04 | 45 |
| cdf_flat_fake | screenshot | nan | 0.000 | 0.880 | 1.00 | 25 |
| sg2_probe | screenshot | nan | 0.000 | 0.727 | 0.36 | 11 |
| flux_train | screenshot | nan | 0.000 | 1.000 | 0.00 | 20 |
| mj_train | screenshot | nan | 0.000 | 1.000 | 0.15 | 20 |
| eval_real | screenshot_jpg | nan | 0.178 | 0.000 | 0.44 | 45 |
| train_fake | screenshot_jpg | nan | 0.000 | 0.911 | 0.89 | 45 |
| cdf_flat_fake | screenshot_jpg | nan | 0.000 | 0.920 | 1.00 | 25 |
| sg2_probe | screenshot_jpg | nan | 0.000 | 0.636 | 0.91 | 11 |
| flux_train | screenshot_jpg | nan | 0.000 | 1.000 | 0.40 | 20 |
| mj_train | screenshot_jpg | nan | 0.000 | 1.000 | 0.90 | 20 |

## Verdict: GATES FAIL: G1:screenshot, G1:screenshot_jpg, G2:screenshot_jpg
