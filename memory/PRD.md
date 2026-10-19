# Argus Prime - PRD

## Project Overview
Multi-Modal Deepfake Detection & Forensic Platform — an evidence court that
scores images, audio and video for synthesis/manipulation and renders an
abstention-aware verdict with per-module evidence.

## Repository
- Source: https://github.com/vishnuas22/Argus-Prime.git
- Branch: main

## System (as it EXISTS — revised 2026-10-09)
- **API**: single-file FastAPI server `deepfake/api/server.py`, port **8420**
  — `POST /api/detect`, `GET /api/health`, `GET /api/history`,
  `POST /api/explain`, static UI at `/`.
- **UI**: single-file web UI `web/index.html` (no Next.js, no React build,
  no WebSocket layer).
- **Persistence**: SQLite `artifacts/verdicts.db` (verdict log).
  No PostgreSQL, no Redis — those were deleted 2026-09-02.
- **ML**: PyTorch (MPS), Transformers, scikit-learn; deps listed in
  `deepfake/requirements.txt`.

## Detection branches (trust labels)
- **Image court** (primary): 11 evidence channels + HistGBM stacker
  `artifacts/p6v11/stacker.joblib`, wired through `argus_detect.STACKER`
  (never hardcoded in consumers), 28-dim feature space with conflict-scalars
  and non-destructive conflict gate (D-040), per-candidate train-only
  conformal recalibration (P4.7 B3, D-051); robust to screenshots and
  messenger laundering (D-032) and to modern generators (D-026/D-027).
  Promoted 2026-10-18 under user waiver (`artifacts/p6v11/WAIVER.md`):
  HOLD verdict taken fail-closed (AUC 7/13, B4 parity 8/13), accepted for the
  flux2 coverage gain +0.1725 (0.7712→0.9436); family-cost residuals
  (mj/flux/shards AUC −0.010…−0.039; core-domain abstention aigc 0.265 /
  cdf 0.54 / ffpp 0.635) are on the waiver and absorbed by the D-053
  abstention-first UI default.
- **Audio**: frozen XLS-R voice-clone probe `artifacts/p4` + D-035 coverage
  gate. **UNTRUSTED for in-the-wild audio** (Deepfake-Eval-2024 AUC ≈0.56,
  D-033); off-manifold input → the branch abstains instead of voting.
  r8/r9 wild-diet retrains are recorded negative results (D-036/D-038).
- **Video**: identity-aware dual-probe scorer (CLIP users / p6vid_v2, D-044)
  — sampled-frame face heads (generalist + FF++ specialist) over ArcFace
  identity tracks + temporal-dispersion channel + XLS-R audio track. FF++
  Gate A 0.9016 (F2F waived at 0.876 vs 0.88), Celeb-DF 0.9411. **UNTRUSTED
  for in-the-wild video** (D-043/D-044 residuals; GenVidBench exam-only).

## Operating model
- Channel intake: dump → train → gate → verdict via
  `deepfake/scripts/restack.py`; promotion requires an explicit
  `--promote-to <tag>` (never a default, never auto-wired).
- Regression suite: `deepfake/scripts/scoreboard.py` vs
  `artifacts/golden.json` — missing golden fails closed, stale score caches
  re-run against the current stacker mtime.
- Benchmark claims live in `artifacts/benchmarks.json` with explicit scope
  and trust labels; memory/{patterns,lessons,decisions}.md hold the
  institutional record.
