# Lessons — mistakes made once, never twice

Format: `- YYYY-MM-DD [tag] mistake → fix/prevention`

- 2026-08-24 [deployment] A file whose write was rejected once (sentinel.md during plan mode) was dropped from the redeploy batch → after any failed operation, re-verify its completion in the final pass; verification caught it.
- 2026-08-24 [testing] Always smoke-test (n≤10) before long benchmark/training runs → caught 3 harness bugs in 30s instead of after a 19-min silent run.
- 2026-08-24 [metrics] Naive sorted-array EER degenerates on tied/constant scores (produced EER=0.01 at AUC=0.50) → compute EER from sklearn roc_curve with linear interpolation between the crossing points.
- 2026-08-24 [vendoring] Third-party scripts hardcode absolute paths (/app, /tmp) → read builder scripts fully before running; plan path overrides or wrappers up front, not mid-run.
- 2026-08-24 [tooling] Two tool calls in ONE message targeting the same file can race (write executed after edit, destroying the edit and the file's prior content) → same-file mutations must go in SEPARATE sequential messages; after any batch touching one file, re-read it.
- 2026-08-24 [mps] Feature extraction belongs on MPS (15.4/s) vs CPU (2.5/s); schedule GPU jobs so generation and extraction don't contend.
- 2026-08-24 [tooling] SECOND path-targeting failure: wrote FINDINGS content to memory/patterns.md (intended artifacts/xgen/FINDINGS.md), silently destroying accumulated conventions → before every Write, re-read the filePath aloud; batched multi-file writes are the danger zone. Reconstruction cost ~30 min.
- 2026-08-24 [hf] Deprecated HF repos vanish behind auth walls without notice (stable-cascade-decoder, PixArt-XL-2 returned 401 mid-phase) → curl-check repo accessibility BEFORE planning a pipeline around it.
- 2026-08-24 [zsh] Quoted-pair for-loops do NOT word-split in zsh ("a b" stays one word) — burned 3 separate loops → never use `for x in "p1 p2"; do set -- $x`; write explicit commands or use ${=var}.
- 2026-08-24 [ops] Hard shell timeouts kill the whole process group (nohup included) → keep long polls under the limit and spawn long jobs in their own short call.
- 2026-08-24 [models] Crop pipelines and backbones must agree on input size (256px crop fed to 224px CLIP = silent contract-crash loop) → assert/embed resize at the module boundary, never assume upstream dimensions.
- 2026-09-02 [ffpp] FF++ joint-training interferes: even 4k balanced FF++ views drop the WildDeepfake exam 0.906→0.790, and full-dose (86k, 65% of train) sinks it to 0.693 — while FF++ in-domain never exceeds 0.868 AUC. Destructive interference in CLIP space, NOT the SANA mix-law (D-012 does not transfer). Before adding any domain to a shared linear probe, run a dose-response sweep; domain count ≠ domain quality.
