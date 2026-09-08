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

## [scoreboard] Stale hardcoded stacker default nearly caused a false regression verdict (2026-09-07)
- scoreboard.py had STACKER hardcoded to p6v3; a 'regression run' silently replayed ancient cached exam CSVs (mj 0.7651 = exact p6v3-era number was the tell).
- Antidote: no consumer may hardcode an artifact tag — import the wired STACKER from argus_detect (fixed). Before believing any regression, confirm WHICH stacker actually ran (tag in work dir path) and whether scores.csv was fresh (mtime) or cached.
- Corollary: exact reproduction of a historical score by a 'new' run is a cache-replay signature, not a coincidence.

## [ui/api] Hardcoded artifact tags rot in every consumer, not just benchmark scripts (2026-09-07)
- After promoting p6v7, the API server (server.py) and the web UI still hardcoded p6v5 in 4 places — the deployed court silently served the old 9-channel stacker for hours. Same disease as the scoreboard bug, different organ.
- Antidote: single source of truth = argus_detect.STACKER; server health/detect now derive the tag from it and the UI prints what the server reports (v.stacker). When promoting a new stacker, grep ALL consumers for the old tag, not just the bench scripts.

## [vlm] Structured-output parsing must tolerate markdown decoration (2026-09-07)
- Qwen2.5-VL emits "**SCENE:** ..." (markdown bold) regardless of the format instructions; naive startswith("SCENE:") matched nothing and fields came back empty while the raw narrative was perfect.
- Antidote: strip '*', '**' and normalize whitespace before tag matching; always log/keep the raw narrative so parsing failures are visible, not silent.

## [inversion] A signal-destroying view poisons a probe — test the probe ON the view before adding it to the diet (2026-09-08)
- Adding screenshot-canvas views to the SD-VAE inversion probe's training collapsed 5-fold CV to 0.43±0.22: the 1280×800 canvas (≈70% flat background + chrome) dominates the round-trip and drives real/fake content to identical feature profiles (per-view AUC 0.46-0.47) — the probe learns view-type noise instead of class signal. The same views are fine for CLIP probes (CV 1.0).
- Antidote: before laundering a probe's training set with a new view family, measure per-view AUC on the fresh cache. If the view destroys the signal, keep the probe content-only and let the STACKER discount the channel on that domain via dump rows. Also: overwriting a deployed probe joblib in place is risky — retrain from the prior npz to restore (inversion v2 restored: CV 0.7284).
