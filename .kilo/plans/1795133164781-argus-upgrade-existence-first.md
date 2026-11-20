# Argus-Prime Upgrade — Existence-First Execution Plan

**Status:** Implementation-ready plan (planning agent — no source edits made).
**Baseline:** repo @ `4a71e59` (GLM audit baseline `b2cfae3`, 2 commits earlier).
**Companion to:** `docs/Argus-Prime_Upgrade_Master_Plan.docx` (GLM 5.3 MAX swarm, v2.0) + `docs/TASK-*.md`.
**Decision locked by user:** **Existence-first** — commit P0→C1→S1 + LICENSE/CI; downgrade I1/V1/A1 to pre-registered research bets requiring a new-hypothesis gate + explicit approval before any training.

> Execution note: this plan requires source edits and mutating commands. Switch to an implementation-capable agent to execute it. Do not start any training/wiring/promotion/commit without the gates below.

---

## 1. Verdict on the GLM plan (evidence-based)

The GLM plan's **research is sound** (every load-bearing external number I re-verified holds) and its **P0/C1/S1 + license work is correct and high-value**. It is **not executable as written** for the three modality waves, which re-tread experiments the repo already closed.

### What I verified (holds)
- Fresh-clone crash: `deepfake/api/server.py:132` unconditional `joblib.load(STACKER)`; `artifacts/p6v11/` + `artifacts/p6vid_v4/` are **untracked** (only `p3_full`, `p3_ffpp`, `p4/audio_probe`, `benchmarks.json` are tracked). CONFIRMED.
- `benchmarks.json` matches every current-state figure (image val 0.9877; audio DF-Eval 0.5593 / FPR 0.7949 / abstention 0.839 / residue 0.6204; video FF++ 0.9121 / CDF 0.9482 / DFD 0.8857 / t2v 0.6035–0.7497).
- Threshold-transfer 41–71% FPR is real repo data (D-103 DINOv3 pre-flight).
- WaveSP-Net: MIT, arXiv 2510.05305, ICASSP 2026, AUC 0.9426 / EER 10.58, ~1.3% trainable. CONFIRMED.
- DF-Eval-2024 commercial ceilings image 0.90 / audio 0.93 / video 0.79 AUC; off-shelf 0.56/0.58/0.63 → fine-tuned 0.73/0.92/0.82; NPR 0.73 (Table 4) vs 0.55 (Table S9) conflict is REAL. CONFIRMED.
- Tuned linear probe 81% on DF-Eval (arXiv 2509.04150, CLIP ViT-b32/ConvNeXt-base). CONFIRMED.

### Why the modality waves are downgraded (prior-art gate)
| GLM wave | Repo prior art (closed) | Verdict |
|---|---|---|
| **A1** audio fine-tune + fuse | D-036/038/063/065 diets failed; **W9.1 (D-085) HOLD 7/11 bars**; fused p13/p13b HOLD. Only **frozen SSL** moved wild audio (A0z 1B 0.5593→**0.778**, D-088). | Re-tread. Fine-tune = empirically worst direction. |
| **V1** temporal + genVideo channel | Temporal line **CLOSED** (D-099; D-082/084/087/097-100 all failed). GenVidBench **training is scope-blocked** (D-041 exam-only). Wild-video FPR target never met (WaveRep 48%, D3 44%, V1f 36%). SBI slot already filled by D-070/D-075 SCENE family. | 2/3 closed; 1/3 scope-blocked. |
| **I1** tuned CLIP-L + Detectra | Backbone-swap closed twice (D-045 0/15; D-103 LOCO 41–71% FPR). Tuned CLIP-L = incumbent recipe (D-017/018). | Only **SAFE** channel is genuinely new. |

### Score reconciliation (42 vs 64.8)
The GLM "42/100" is an **external audit rubric**; the repo's own ledger is **64.8/100** (`memory/improvement_ledger.md`, weighted distance-to-frontier, `artifacts/roadmap_20261109/ROADMAP.md`). **Adopt 64.8 as the single plan-of-record baseline**; cite the 42 audit only as a secondary external data point. Never mix. Both rubrics agree ~70–78 is achievable and ~22 points are structural.

---

## 2. Locked decisions
- **D-P1 (posture):** Existence-first. P0→C1→S1 + license are committed. I1/V1/A1 become research-gated bets.
- **D-P2 (score):** Single baseline = repo ledger 64.8. Every task books **measured** Y into `memory/improvement_ledger.md` (no expected-Y booking).
- **D-P3 (vendor pristine):** No edit under `integrations/Argus_Master/` (D-009). All new logic is harness-level under `deepfake/`.
- **D-P4 (research posture default):** Commercial vs research posture (GLM §10) remains an **owner decision**; default = research for evaluation, commercial-safe roster flagged. NC weights (SBI/TALL++/AntiDeepfake/GenConViT) are named in `WEIGHTS-NOTICES.md` and gated.
- **D-P5 (process law):** Train-only calibration (D-035); sealed spec + D-number before any channel change; restack parity + B4 abstention-parity; fail-closed contracts; scope label on every number (D-037).

---

## 3. Swarm execution structure
Mirror the repo's own BLUEPRINT (Planner–Executor–Critic; mutation confined to `@builder`; destructive bash ask-gated).

```
P0 EXISTENCE (single critical path, no fan-out) ──┬──> C1 CALIBRATION
                                                   ├──> LICENSE + CI/DOCS
                                                   └──> (I1/V1/A1 research bets — ONLY if a new-hypothesis gate passes)
                                                          │
                              (feature spaces frozen after C1 + any approved bet)
                                                          v
                                                   S1 SERVING (always last)
```

- **Per step:** `architect` writes a sealed spec (steps/files/verify/risks/rollback) → `builder` implements (only component that mutates) → `sentinel` (fresh instance, sees spec+diff only) returns APPROVE/REJECT/ESCALATE. Max two reject cycles → escalate to user.
- **Read-only scouts** verify external facts / fetch-and-load checkpoints (never mutate).
- **Memory out:** after each task, append to `memory/patterns.md` / `lessons.md` / `decisions.md` and book measured Y to `improvement_ledger.md`.
- **Never waive a gate silently** — a FAIL → HOLD + `WAIVER.md` + user sign-off (repo law).

---

## 4. Phase plan (ordered; no time estimates — dependencies + gates only)

### P0 — Existence (blocks everything)
1. **Artifact recovery + hosting.** On the dev Mac, export `artifacts/p6v11/`, `artifacts/p6vid_v4/` (+ `drift_calibration.json`, `av_calibration.json`), `artifacts/aigc_shot_v2/`, `p7_mj/`, `inversion_probe/`, `provenance_probe/`, `tools/models/{yunet, sface, arcface_w600k_r50}.onnx`, `integrations/syncnet_python/data/syncnet_v2.model`, `integrations/Argus_Master/backend/data/*` (spectral_envelope.npz, reference_embeddings.npy, perturb_calib.npz). Upload to a **private HF repo**; generate `models.lock` (sha256 + bytes + library pins: torch/sklearn/timm/transformers/opencv/onnxruntime versions). Write `deepfake/bootstrap.py` (`snapshot_download(revision=pinned)` → verify sha → lay out at `ARGUS_ARTIFACTS_DIR`, default `./artifacts`; `--verify` mode).
2. **De-hardcode + config seam.** Introduce `ARGUS_ARTIFACTS_DIR`; refactor `argus_detect.py:28,43,44`, `video_score.py:35-42`, `video_channels.py:41`. Replace **20** hardcoded paths (**18** `/Users/Dataghost` + **3** `/private/var/folders` − 1 overlap = 20; the GLM list omits `deepfake/scripts/diag_video_content_matched.py`). exiftool via `shutil.which("exiftool")` (drop the `tools/bin/exiftool` Mac shim — it also execs an untracked `tools/Image-ExifTool-13.10/`); ffmpeg via `shutil.which` + `imageio-ffmpeg` fallback; never silently set `audio_present=0`.
3. **Fail-closed with bootstrap instruction.** Guard `init_court` `server.py:132` like the meta block at `:122-123` (missing bundle ⇒ SystemExit naming the file + `python deepfake/bootstrap.py`).
4. **YuNet + face models.** Fetch YuNet (MIT, `huggingface.co/opencv/face_detection_yunet`, sha256 `8f2383e4dd3cfbb4…`), sface, arcface; sha-pin in `models.lock`; wire `fetch_face_models.py`. Confirm `video_face_track.py:75-79` no longer raises on a fresh clone.
5. **requirements + CI guard + markers.** Add `matplotlib` (imported by `viz.py`, which 4/5 core modules import — its absence kills 4 channels), `pyexiftool`, `mlx-vlm`; move `mflux`, `diffusers` to a `[training]` extra (mflux is training-only `gen_modern_corpus.py`; note it is **not** strictly macOS-only — it has a Linux `mlx[cuda13]` path — so justify by "unused on serve path", not "macOS-only"). **Create `pytest.ini` with `fast`/`slow` markers** (none exist today; `pytest -m fast` selects nothing). CI grep guard must be **scoped to git-tracked files** (`git ls-files | xargs rg '/Users/|/private/var/folders'`) — a repo-wide grep hits 178+ in `artifacts/`+`logs/`. Fix the CI stacker-meta assert to accept `2n+6` and to tolerate `p6vid_v4` having **no `module_ids` key** (currently `KeyError`).
6. **LICENSE + docs.** Top-level `LICENSE` (owner: MIT or Apache-2.0) + `WEIGHTS-NOTICES.md` (ArcFace buffalo NC, Wav2Lip research, SBI/TALL++/AntiDeepfake/GenConViT NC) + root `README.md` (quickstart, scope table, `LIMITATIONS.md`).
7. **Two-tier CI.** `fast` (fixtures, every PR, tokenless) / `slow` (real models, nightly, HF_TOKEN + cache keyed on `models.lock`). Commit tiny fixtures (synthetic PNG/WAV/MP4 + dummy 5-feature stacker + 232 KB YuNet). Triage `test_laundering_monotonicity` (RED today: `tests/regression_set/` missing) — fix or waive in writing.

**P0 acceptance:** fresh Linux clone → `pip install -r requirements.txt` → `python deepfake/bootstrap.py --verify` = ALL-SHAS-OK → API boots → CLI on all 3 modalities with fixtures → `pytest -m fast` green → `git ls-files | xargs rg '/Users/|/private/var/folders'` = 0 hits.

### C1 — Calibration (harness-level, no vendored edits)
1. `deepfake/bench/calibration.py`: `logit_transform`, `CourtCalibrator` (LR on fused scores), `cohort_normalize`, `ltt_threshold(scores, labels, alpha)`.
2. **`CalibratedStackerJudge(StackerJudge)`** in `deepfake/bench/` — calibrate **inside `decide()`** so the conformal set / abstention / trust derive from calibrated probs. Inject via the existing `fuse(judge=...)` seam. **Do NOT edit `integrations/Argus_Master/backend/fusion.py`** (D-009). Note: the deployed stacker already emits `predict_proba` (`stacker_judge.py:74`); the `sigmoid(3.0·fused)` squash (`fusion.py:51`) is only the non-stacker `WeightedVotingJudge` path — layer-1 "logits not softmax" applies there; the real win is layer-2 court LR.
3. **Fix two latent seams:** recovery R2 uses the vendored default α=0.10 and ignores the bound α (`recovery.py:297-321`) → route R2 through the calibrated judge; `core.conformal.alpha` misreports the constant (`fusion.py:226`) → report the bound value.
4. Class-pure cohort packs (train-only provenance, D-035); LTT threshold utility; a-DCF risk curves in `/api/detect` (scope-labeled, D-037); HTER-at-transferred-threshold in `benchmarks.json`.
5. α parameterization: prefer a `ContextVar` shim (anyio.to_thread copies context) over the global monkeypatch (`server.py:198-210`). Decide explicitly whether α reaches audio (currently takes none) and the video_face route (currently ignores it).

**C1 acceptance:** `pytest -m fast tests/test_calibration.py` green; re-run D-103 threshold-migration audit → fixed-threshold FPR on held-out wild mix ≤10% (from 41–71%); AUC unchanged on every exam (monotone-law assert); `risk_curve` present in `/api/detect`; `benchmarks.json` carries `hter_at_transferred_threshold`.

### LICENSE / CI / docs
Fold P0 steps 5–6; run parallel to the P0 tail. Gate: `LICENSE` + `WEIGHTS-NOTICES.md` + `README.md` exist; `fast` CI green tokenless.

### Research-gated modality bets (only if a new-hypothesis gate passes; architect spec + user approval required)
- **I1 (image):** only **SAFE** is genuinely new (noise/wavelet expert, Apache-2.0, 60 ms). Measure on **provider-real issued-FPR** (D-054 flux2 20/25), not generator-attribution. Do NOT re-run tuned-CLIP-L or backbone swaps without a new hypothesis (D-045/D-103 closed).
- **V1 (video):** temporal CLOSED (D-099); GenVidBench **exam-only** (D-041) — do not train on it. Cheapest true info gain = **first wild-video corpus run** (WildDeepfake video + DFDC, scope-labeled). TALL++ only behind a pre-registered kill-gate (K2). A genuinely-new video-OOD signal is required (D-099).
- **A1 (audio):** fine-tune ≈ W9.1 HOLD (D-085). Only **frozen-SSL** moved the number (A0z-class, D-088). Gate = frozen-SSL + **owner ruling on the ECE bar** (Pascu 2024: 1B frozen-head ECE ~26.9%; ≤0.05 likely unreachable → adopt a decision-quality target). AntiDeepfake is CC BY-NC-SA.

### S1 — Serving (always last; feature spaces frozen)
1. **Full shared-state audit before removing `CapacityLimiter(1)` (`server.py:96`).** Close, in order:
   - a. Parameterize α via ContextVar shim (closes C-1 global rebind).
   - b. **Per-request embedder** — thread embedder through `observe_video`/`_embed`; **stop calling `set_embedder`** (`argus_detect.py:241-243`) (closes C-3).
   - c. **Per-thread YuNet detector + ArcFace `cv2.dnn` net** — `_DET`/`_ARC_NET` are stateful (`setInputSize`/`setInput`); the GLM S1 misses these (C-9/C-10). Do not share them across threads.
   - d. **Locks span the FULL lazy load** — set `loaded=True` after load (not before) and fix the fail-open-on-exception (`argus_detect.py:199-200`) (closes C-4/5/6/7/8/14).
   - e. **Vendored caches** (`registry._cache` C-11, `realness/spectral/perturbation` C-12, `DinoService._cache` C-13) — mitigate harness-side (build modules once + reuse) or accept+document; D-009 forbids editing them.
   - f. Fix `sys.path` growth (C-15), `torch.set_grad_enabled` in `audio_fusion` (C-16), `torch.set_num_threads/seed` (C-17). Decide α scope for audio/video (C-18).
2. **ONNX export + parity gates** per frozen channel. Note: the repo has **no `onnxruntime`** — "ONNX sessions" = OpenCV DNN nets; use `cv2.dnn` (and ORT only where a channel is ported). Gate = `max|p_onnx − p_torch| < 1e-3` on fixtures + repo dump-parity law.
3. **INT8** for wav2vec2-class (EER delta ≤0.3% vs fp32); ViT only with PTQ4ViT-class method.
4. **Perf receipts:** `perf_bench.py` p50/p95 per modality into `benchmarks.json` (scope-labeled); nightly trend.

**S1 acceptance:** `pytest -m slow tests/test_concurrency_races.py` (8-way α isolation) green; image p50 ≤1.5 s / video p50 ≤4.0 s / audio p50 ≤2.0 s (8-core); ONNX parity <1e-3; INT8 EER delta ≤0.3%.

---

## 5. Corrections to the GLM `docs/TASK-*.md` (apply before executing them)
- **TASK-P0:** paths are **18+3−1=20** (add `diag_video_content_matched.py`); CI grep guard scoped to tracked files; `matplotlib` (not just pyexiftool) is a serve-path gap; mflux is training-only, not "macOS-only"; create pytest markers; CI `2n+3`→`2n+6` + handle missing `module_ids`.
- **TASK-C1:** calibrate via a harness-level `CalibratedStackerJudge` (never edit `fusion.py`); note the stacker already outputs `predict_proba`; fix recovery-R2 α + `conformal.alpha` misreport.
- **TASK-S1:** "ONNX sessions" = OpenCV DNN nets (no onnxruntime); add the stateful YuNet `_DET` + ArcFace net (C-9/C-10) and vendored-cache handling (C-11/12/13) to the thread-safety scope.
- **TASK-I1/V1/A1:** add a prior-art gate citing D-085 / D-099 / D-041 / D-045 / D-103; require a new-hypothesis spec + user approval before any training.

## 6. Risks & kill-gates
- **Dev-Mac artifacts unrecoverable** (catastrophic single point) → P0 week-1 recovery + sha lock; record corpora availability in a D-number.
- **DF-Eval-2024 gated** (registration denied, D-102) → public-corpus fallback (OpenFake/CommFor/ITW/MLAAD); week-2 kill-gate.
- **Restack parity FAIL** → HOLD + waiver, never silent absorb.
- **S1 thread-safety incomplete** → do NOT raise `CapacityLimiter(1)` until steps a–f pass the race test.
- **Calibrated judge changes ranking** → assert AUC invariance (monotone law).
- **License contamination** → research posture default; NC rosters gated + named.

## 7. Out of scope
- Commercial image parity (0.90 AUC) — no open model has it.
- Committing to 0.82 video as a promise (stretch only; exceeds commercial 0.79).
- Sub-second CPU latency (needs GPU).
- Adversarial-robustness guarantees; conformal guarantees on unseen generators.
- Any training/wiring/promotion/commit without explicit approval.

## 8. Open questions (owner)
1. Commercial vs research posture (GLM §10) — default research.
2. Audio certification target: ECE ≤0.05 (likely unreachable) vs decision-quality (K1) — needs a ruling.
3. Priority of the research-gated bets vs stopping after P0+C1+S1.