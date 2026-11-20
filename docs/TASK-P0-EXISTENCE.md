# TASK-P0 — Make the Product Exist (Reproducibility Foundation)

**Phase:** P0 · **Window:** weeks 1-3 · **Effort:** 3-4 engineer-weeks · **Blocks:** everything
**Expected score movement:** reproducibility 3→12/15, testing/CI 4→8/10 (audit rubric)

## Evidence (why this task exists)

1. `init_court()` crashes on any fresh clone: `server.py:132` does an unconditional
   `joblib.load(STACKER)` for `artifacts/p6v11/stacker.joblib` — file is untracked
   (`.gitignore:8 /artifacts/`). Verified traceback captured in audit.
2. Video route raises `FileNotFoundError: YuNet model missing` (`video_face_track.py:75-79`)
   — `tools/models/` is entirely untracked.
3. 20 files hardcode `/Users/Dataghost` or `/private/var/folders` paths (list in README).
4. CI (`.github/workflows/ci.yml`) runs ZERO tests: AST parse + 2 asserts. It also
   asserts a stale stacker-meta formula (`2n+3` — deployed metas are `2n+6`).
5. `requirements.txt:21` pins `mflux==0.22.0` (MLX/macOS-only; verified `import mlx.core`
   fails on Linux: `libmlx.so` missing). `pyexiftool` and `matplotlib` are imported at
   runtime but absent from requirements. `tools/ffmpeg` is an x86-mac evermeet binary;
   `tools/bin/exiftool` shim execs a Mac-only perl path.
6. No LICENSE, no root README.
7. Artifact pointers are constants, not config: `STACKER/VIDEO_STACKER/AUDIO_PROBE`
   (`argus_detect.py:28,43,44`), `FACE_PROBES` (`video_score.py:35-42`).

## Goal

A fresh Linux clone boots the API, runs the CLI on all three modalities with committed
tiny fixtures, CI runs a fast test suite green, and every artifact is content-addressed.

## Steps

### P0a — Artifact recovery (WEEK 1, before anything else)
1. On the dev Mac, export the deployed artifact set: `artifacts/p6v11/` (stacker.joblib,
   stacker_meta.json, WAIVER.md), `artifacts/p6vid_v4/` (+ drift_calibration.json,
   av_calibration.json), `artifacts/p4/audio_probe.joblib` (already tracked — verify sha),
   `artifacts/aigc_shot_v2/probe.joblib`, `p7_mj`, `inversion_probe`, `provenance_probe`,
   `tools/models/{yunet.onnx, face_recognition_sface_2021dec.onnx, arcface_w600k_r50.onnx}`,
   `integrations/syncnet_python/data/syncnet_v2.model`,
   `integrations/Argus_Master/backend/data/` (spectral_envelope.npz, reference_embeddings.npy,
   knn_calib_distances.npy, reference_meta.json, perturb_calib.npz).
2. Create a private HuggingFace repo (free tier: 100 GB private storage; Git LFS free
   quota is only 10 GiB + 10 GiB/mo bandwidth — use HF). Upload with `huggingface_hub`.
3. Generate `models.lock`: sha256 + byte size + library pins
   (torch, scikit-learn, timm, transformers, opencv, onnxruntime versions from the dev
   Mac env — joblib/HistGBM pickles are version-sensitive).
4. Write `deepfake/bootstrap.py`: `snapshot_download(repo, revision=<pinned>)`,
   verify every sha256, lay files out at `ARGUS_ARTIFACTS_DIR` (default `./artifacts`).
   Add `--verify` mode that only checks.

### P0b — De-hardcode + config seam
5. Introduce `ARGUS_ARTIFACTS_DIR` env (default: repo-relative `artifacts/`); refactor
   `argus_detect.py:28,43,44`, `video_score.py:35-42`, `video_channels.py:41` to resolve
   through it. Keep fail-closed contracts (missing bundle ⇒ SystemExit with the exact
   missing file), but `init_court()` must now fail with a bootstrap instruction, not a
   bare FileNotFoundError.
6. Replace all 20 hardcoded paths (README list): `WS = Path(os.environ.get("ARGUS_ROOT",
   Path(__file__).resolve().parents[2]))` pattern for scripts; exiftool via
   `shutil.which("exiftool")` with `tools/bin` fallback removed; ffmpeg via
   `shutil.which` with `imageio-ffmpeg` fallback; log a visible decision when absent
   (never silently set `audio_present=0`).
7. Add CI guard: `rg -n "/Users/|/private/var/folders" --glob '!memory/**'` must exit 0 hits.

### P0c — Licensing + docs
8. Add top-level LICENSE (owner decision: MIT or Apache-2.0 for code) + `WEIGHTS-NOTICES.md`
   listing every vendored/downloadable weight and its license (insightface ArcFace and
   Wav2Lip are research-only TODAY — flag them; see master plan §10).
9. Root README: what it is, quickstart (bootstrap → CLI → API), scope table from
   `benchmarks.json`, link to `LIMITATIONS.md`.

### P0d — Two-tier CI
10. `requirements.txt`: move `mflux`, `diffusers` to `[training]` extra; add `pyexiftool`,
    `matplotlib`; pin the serve-path set to the `models.lock` versions.
11. Commit tiny fixtures (<10 MB total): 1 synthetic PNG/WAV/MP4, dummy 5-feature
    stacker.joblib + meta matching a `fast` fixture contract, the 232 KB YuNet ONNX
    (MIT — verified URL `huggingface.co/opencv/face_detection_yunet`, sha256 starts
    `8f2383e4dd3cfbb4`).
12. pytest markers: `fast` (fixtures, every PR, tokenless) / `slow` (real models,
    nightly, HF_TOKEN secret + `actions/cache` keyed on models.lock sha).
13. Fix the CI stacker-meta assert to accept both `2n+3` and `2n+6` (or import the
    constant from `train_stacker.py`).
14. Triage `test_laundering_monotonicity` (currently RED even on the runnable subset) —
    fix or document the waiver before it becomes a CI gate.

### P0e — Cross-platform serve path
15. Device abstraction: `mps` → `cuda` → `cpu` selection helper (currently macOS-MPS or
    cpu only, `argus_detect.py:440`).
16. Verify in a clean Ubuntu container: `pip install -r requirements.txt` succeeds;
    `python deepfake/bootstrap.py --verify` green; API boots; metadata module not
    `unavailable`.

## Acceptance criteria + verification commands

```bash
# A1: fresh clone boots end-to-end (Linux container)
git clone <repo> && cd Argus-Prime && pip install -r requirements.txt
python deepfake/bootstrap.py --verify && echo BOOTSTRAP-OK
uvicorn deepfake.api.server:app --port 8420 & sleep 8
curl -s localhost:8420/api/health | jq '.stacker, .modalities.video.artifact'
# expect: p6v11 pointer reported, no crash

# A2: CLI on all 3 modalities with fixtures
python deepfake/argus_detect.py tests/fixtures/tiny.png --json | jq '.p_fake_fused, .abstained'
python deepfake/argus_detect.py tests/fixtures/tiny.wav --json | jq '.p_fake_audio_mean'
python deepfake/argus_detect.py tests/fixtures/tiny.mp4 --json | jq '.video_score'

# A3: fast CI green, tokenless, <10 min
pytest -m fast -q          # 0 failed
rg -n "/Users/|/private/var/folders" --glob '!memory/**' ; echo "hits=$?"   # expect hits=1 (no matches)

# A4: nightly workflow exists and passed once with real models
# A5: models.lock sha verification
python deepfake/bootstrap.py --verify | tail -1   # ALL-SHAS-OK
```

## Risks / rollback
- **Dev-Mac artifacts unrecoverable** = catastrophic single point of failure. If any
  bundle is lost: probes can be retrained from recipes in `deepfake/scripts/train_*`,
  but stackers need the dump chain (`data/` corpora) — check what corpora still exist
  on the Mac during P0a and record in a D-number.
- All changes are additive/config — rollback = revert commits; artifact pointers keep
  their fail-closed behavior.

## Deliverables
`bootstrap.py`, `models.lock`, private HF repo, 20 path fixes + CI guard, LICENSE +
WEIGHTS-NOTICES + README, `requirements[-training].txt`, fixtures, two CI workflows,
D-number entries for every promotion, regenerated `benchmarks.json`.
