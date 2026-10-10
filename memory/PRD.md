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
  `artifacts/p6v9/stacker.joblib`, wired through `argus_detect.STACKER`
  (never hardcoded in consumers), conformal abstention; robust to screenshots
  and messenger laundering (D-032) and to modern generators (D-026/D-027).
- **Audio**: frozen XLS-R voice-clone probe `artifacts/p4` + D-035 coverage
  gate. **UNTRUSTED for in-the-wild audio** (Deepfake-Eval-2024 AUC ≈0.56,
  D-033); off-manifold input → the branch abstains instead of voting.
- **Video**: thin scorer — sampled-frame face head + temporal dispersion +
  XLS-R audio track. **UNTRUSTED** for in-the-wild video (n=39 FF++ exam only,
  no video stacker).

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
