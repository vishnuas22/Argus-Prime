# Argus-Prime

Evidence-first deepfake detection across **image**, **video**, and **audio**, fused
through a calibrated court with explicit abstention.

**Current state: 64.8 / 100** (weighted distance-to-frontier scorecard, measured).
**Target: 70+** — reachable, but not with the paths already closed; see
`memory/backlog_2026-11-20.md`.

> **Honesty first.** This is a research-grade system with measured weak spots.
> Read [Scope and limits](#scope-and-limits) before quoting any number.

---

## Quickstart

> **Artifact bundle.** The runtime weights (stackers, face-detection ONNX,
> probes) are **not** in git. `deepfake/bootstrap.py` fetches them from the
> pinned private bundle and verifies every sha256 against `models.lock`.
> A fresh clone needs `HF_TOKEN` (or `hf auth login`) with access to
> `thedataghost22/Argus_prime`.

```bash
# 1. install (serve path only)
pip install -r deepfake/requirements.txt
#    training/generation extras (mflux, diffusers, ...) live in:
pip install -r deepfake/requirements-training.txt

# 2. fetch + verify the artifact bundle (content-addressed)
hf auth login                                 # if not already authenticated
python deepfake/bootstrap.py                  # download pinned revision + verify
python deepfake/bootstrap.py --verify         # check only -> ALL-SHAS-OK

# 3. run the API
python -m uvicorn deepfake/api.server:app --port 8420
#    -> http://127.0.0.1:8420  (UI + REST)

# 4. or the CLI
python deepfake/argus_detect.py path/to/image.jpg --json
python deepfake/argus_detect.py path/to/clip.mp4  --frames 12 --json
python deepfake/argus_detect.py path/to/audio.wav --json
```

### Configuration

| Env var | Purpose | Default |
|---|---|---|
| `ARGUS_ROOT` | Repository root | derived from `__file__` |
| `ARGUS_ARTIFACTS_DIR` | Runtime artifact bundle location | `<repo>/artifacts` |
| `ARGUS_ARTIFACTS_REPO` | HF repo to fetch from | `models.lock` `bundle.repo` |
| `ARGUS_ARTIFACTS_REVISION` | Pinned HF revision | `models.lock` `bundle.revision` |
| `ARGUS_TMP` | Scratch directory | system temp |
| `ARGUS_CALIBRATION` | Enable the C1 calibration layer (`1`=on) | off |
| `ARGUS_THREADSAFE_ALPHA` | Use the ContextVar α shim | off |

Every artifact pointer resolves through `ARGUS_ARTIFACTS_DIR`; nothing is
hardcoded to a machine path.

### Note on the video path

`tools/models/arcface_w600k_r50.onnx` (insightface, **non-commercial**) is
**excluded** from the HF bundle. Fetch it separately:

```bash
python deepfake/scripts/fetch_face_models.py
```

The image path is fully commercial-safe without it.

### Checking your environment

```bash
pytest -m fast -q        # fixtures only, no models, no corpora
# -> 48 passed

python deepfake/bootstrap.py --verify   # -> ALL-SHAS-OK (17 artifacts, ~218 MB)
```

---

## Architecture

Three modality lanes feed a shared court.

```
image  ──► 11 instruments ──┐
video  ──►  5 instruments ──┼──► court (stacker + conformal abstention) ──► verdict
audio  ──►  3 instruments ──┘
```

**Image (11):** metadata rules, ELA compression history, FFT spectral probe,
DINOv2 kNN realness, DINOv2 perturbation drops, CLIP AIGC probe, CLIP face probe,
CLIP FF++ probe, Midjourney probe, SD-VAE inversion probe, C2PA provenance.

**Video (5):** YuNet face detection → ArcFace identity clustering → dual CLIP
face heads (generalist + FF++ specialist) + identity drift + scene features,
fused by a 32-feature HistGradientBoosting stacker.

**Audio (3):** frozen wav2vec2-XLS-R-300m mean-pooled over 4 s windows → logistic
probe with a train-derived coverage gate.

**Court:** reliability gate → harness-level `StackerJudge` → conformal prediction
set → APS-style abstention with a recovery ladder. The judge is pluggable
(`fuse(judge=...)`), which is how the calibration layer and future channels attach
without editing vendored code.

---

## Verified performance (scope-labelled — do not quote without the scope)

Source: `artifacts/benchmarks.json` (`as_of` 2026-11-04). Regenerate after any
promotion.

| Scope | Image | Video | Audio |
|---|---|---|---|
| **In-domain** | stacker val AUC **0.9877** | FF++ test n=700 AUC **0.9121** | ASVspoof 2019 LA EER **3.5%** |
| **Near-domain** | screenshot-launder AUC 0.9943–0.9975 | LOMO 4/4 pass · DFD 0.8857 | 2021 LA AUC 0.912 / EER 17.2% |
| **External generators** | 0.67–1.00 across 13 domains | t2v: kling 0.6035 · sora 0.6812 · opentora 0.7497 | — |
| **Wild** | WildDeepfake: issued FPR 0.0% but **abstention 65–73%** | **never measured on a wild-video corpus** | **AUC 0.5593, FPR 79.5% @0.5, abstention 83.9%** |

**Read the last row carefully.** Wild performance is the honest weak spot, and
large abstention is doing real work there.

---

## Scope and limits

### What this system does well
- Strong **curated** benchmarks: FF++ 0.9121, Celeb-DF 0.9482, DFD 0.8857.
- **Abstention-first** posture: it says "I don't know" rather than guessing.
- Every number in the repo carries a **scope label** (D-037 law).
- Reproducible, portable serve path: no machine-local paths, cross-platform.

### What it does not do (measured, not opinion)
- **Wild audio is broken** — AUC 0.5593 is worse than useless; the coverage gate
  abstains on 83.9% of clips to hide it.
- **Wild video has never been measured** on a wild-video corpus.
- **Modern generative video (Sora/Kling era)** scores 0.60–0.75; every FF++-era
  detector degrades on 2024+ generators.
- **Commercial image parity (0.90 AUC)** — no open model has demonstrated it.
- **Sub-second CPU latency** for the full pipeline — needs GPU.
- **Adversarial robustness** — out of scope, no claims.
- **Conformal guarantees on unseen generators** — mathematically impossible; the
  abstention band is the only defence.

### What was measured and closed (do not re-propose without new evidence)
`C1 calibration` (hurts; the court is AUC-limited) · `ReStraV geometry signal`
(T2V gain is real but unroutable label-free) · `D3 / SPLIT / WaveRep` (temporal
statistics) · `P7 / P7b DINOv3` · `A0z 1B` · `W9.1 audio` · `HOLA`.

Full reasoning in `memory/decisions.md` (D-001 … D-113).

---

## Development

```bash
# fast tier: fixtures only, no models, tokenless (every PR)
pytest -m fast -q

# slow tier: real models/corpora (nightly, needs the artifact bundle)
ARGUS_THREADSAFE_ALPHA=1 pytest -m slow tests/test_concurrency_races.py -q

# CI runs the fast tier + path guard + registry/meta contracts
```

### Process laws (from the repo's own decision log)
1. **Train-only calibration** (D-035) — thresholds derived from train splits only.
2. **Sealed spec + D-number** before any channel change.
3. **Restack parity gates** — AUC tolerances + abstention parity (`ab_new ≤ ab_old + 0.10`).
4. **Fail-closed contracts** — a missing bundle refuses to construct, never degrades silently.
5. **No number without a scope label** (D-037).

---

## Documentation

| Document | Contents |
|---|---|
| `memory/backlog_2026-11-20.md` | Consolidated to-do, ranked |
| `memory/decisions.md` | Full decision log (D-001 … D-113) |
| `memory/improvement_ledger.md` | Measured Y per change |
| `memory/lessons.md` | Past mistakes and their antidotes |
| `memory/patterns.md` | Repo conventions |
| `docs/Argus-Prime_Upgrade_Master_Plan.docx` | External upgrade plan (GLM 5.3 MAX swarm) |
| `WEIGHTS-NOTICES.md` | Third-party licences — **read before commercial use** |
| `LICENSE` | MIT (code and trained probes) |

---

## Licence

**MIT** for the source code and trained probe/stacker artifacts.

Third-party weights retain their own licences — see `WEIGHTS-NOTICES.md`.
**Notably non-commercial:** ArcFace (insightface), Wav2Lip, SBI, TALL++,
AntiDeepfake, GenConViT. **These block a commercial video deployment.**