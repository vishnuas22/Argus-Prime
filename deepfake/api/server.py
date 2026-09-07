"""Argus API — FastAPI adapter serving the court to the web UI.

  POST /api/detect   multipart file -> verdict + per-module evidence
  GET  /api/history  recent verdicts (SQLite)
  GET  /api/health   court + artifact status
Static UI served from web/ at /.

Run: ./.venv/bin/python -m uvicorn deepfake.api.server:app --port 8420
"""
import json
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "integrations" / "Argus_Master" / "backend"))

import os
os.environ["PATH"] = f"{WS/'tools'/'bin'}:{os.environ['PATH']}"

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DB = WS / "artifacts" / "verdicts.db"
STATIC = WS / "web"

app = FastAPI(title="Argus Court", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_state = {}


def init_court():
    """Preload stacker + channel modules once (meta-driven, matches CLI)."""
    import joblib
    from argus_detect import build_modules
    from triage import DegradationEstimator
    mids = None
    meta = WS / "artifacts" / "p6v5" / "stacker_meta.json"
    if meta.exists():
        mids = json.loads(meta.read_text()).get("module_ids")
    modules, ids = build_modules()
    _state["modules"] = modules
    _state["ids"] = ids
    _state["clf"] = joblib.load(WS / "artifacts" / "p6v5" / "stacker.joblib")
    _state["mids"] = mids or ids
    _state["estimator"] = DegradationEstimator()
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS verdicts (
        id INTEGER PRIMARY KEY, ts REAL, filename TEXT, kind TEXT,
        p_fake REAL, abstained INTEGER, trust REAL, risk REAL,
        alpha REAL, modules_json TEXT)""")
    con.commit()
    con.close()


def run_court(pil, alpha=None):
    """Full court pass; returns verdict dict with per-module detail."""
    import functools
    import hashlib
    import io
    import fusion
    from base import ImageContext
    from fusion import fuse
    from stacker_judge import StackerJudge
    if alpha is not None:
        fusion.conformal_set = functools.partial(fusion.conformal_set, alpha=alpha)
    modules = _state["modules"]
    tmp = Path(tempfile.mkdtemp(prefix="argus_api_"))
    art = Path(tempfile.mkdtemp(prefix="argus_art_"))
    vid = hashlib.sha256(str(pil.size).encode()).hexdigest()[:12]
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    raw = buf.getvalue()
    src = tmp / f"{vid}.png"
    src.write_bytes(raw)
    ctx = ImageContext(pil=pil.copy(), raw_bytes=raw,
                       sha256=hashlib.sha256(raw).hexdigest()[:16], fmt="png",
                       verdict_id=vid, artifact_dir=art, src_path=src)
    d = _state["estimator"].estimate(ctx)
    outputs = [m.run(ctx, d) for m in modules]
    judge = StackerJudge(_state["clf"], d.evidence_capacity, _state["mids"])
    core, *_ = fuse(outputs, d, judge=judge)
    detail = []
    for mo in outputs:
        if getattr(mo, "unavailable_reason", None):
            continue
        claim = ""
        if mo.artifacts:
            claim = mo.artifacts[0].checkable_claim or mo.artifacts[0].description
        detail.append({"id": mo.module_id, "evidence": mo.evidence_score,
                       "reliability": mo.reliability_score,
                       "direction": mo.verdict_direction,
                       "confidence": mo.confidence_score, "claim": claim})
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(art, ignore_errors=True)
    return {"p_fake_fused": round(1.0 - core.probabilities["camera_original"], 4),
            "abstained": bool(core.abstained), "trust": core.trust_score,
            "risk": core.risk_score, "hypotheses": core.hypothesis_set,
            "modules": detail}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@app.on_event("startup")
def _startup():
    init_court()


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/api/health")
def health():
    return {"status": "ok", "channels": _state.get("ids", []),
            "stacker": "artifacts/p6v5", "n_modules": len(_state.get("modules", []))}


@app.get("/api/history")
def history(limit: int = 30):
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM verdicts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    return [dict(r) | {"modules": json.loads(r["modules_json"])} for r in rows]


def log_verdict(filename, kind, v, alpha):
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO verdicts (ts, filename, kind, p_fake, abstained, trust, risk, alpha, modules_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), filename, kind, v["p_fake_fused"], int(v["abstained"]),
                 v["trust"], v["risk"], alpha, json.dumps(v["modules"])))
    con.commit()
    con.close()


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), alpha: float = None):
    ext = Path(file.filename or "").suffix.lower()
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    tmp = Path(tempfile.mkdtemp(prefix="argus_up_")) / f"f{ext}"
    tmp.write_bytes(data)
    t0 = time.time()
    try:
        if ext in IMAGE_EXTS:
            from PIL import Image
            from argus_detect import detect_image
            pil = Image.open(tmp)
            pil.load()
            v = run_court(pil.convert("RGB"), alpha)
            kind = "image"
        elif ext in AUDIO_EXTS:
            from argus_detect import detect_audio
            a = detect_audio(tmp)
            kind = "audio"
            v = {"p_fake_fused": a["p_fake_audio_mean"], "abstained": False,
                 "trust": 0.7, "risk": 0.0, "modules": [],
                 "audio": {k: v2 for k, v2 in a.items() if k != "per_segment"}}
        elif ext in VIDEO_EXTS:
            from argus_detect import detect_video
            v = detect_video(tmp, 5)
            kind = "video"
            v.setdefault("abstained", False)
            v.setdefault("modules", [])
        else:
            raise HTTPException(400, f"unsupported type {ext}")
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    v.update({"kind": kind, "filename": file.filename,
              "elapsed_s": round(time.time() - t0, 2), "alpha": alpha})
    try:
        log_verdict(file.filename, kind, v, alpha)
    except Exception:
        pass
    return v

