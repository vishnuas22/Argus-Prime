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
sys.path.insert(0, str(WS / "deepfake" / "api"))
sys.path.insert(0, str(WS / "integrations" / "Argus_Master" / "backend"))

import os
os.environ["PATH"] = f"{WS/'tools'/'bin'}:{os.environ['PATH']}"

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

DB = WS / "artifacts" / "verdicts.db"
STATIC = WS / "web"

app = FastAPI(title="Argus Court", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_state = {}


def init_court():
    """Preload stacker + channel modules once (meta-driven, matches CLI).

    The stacker path comes from argus_detect.STACKER — the wired court —
    never hardcoded (same lesson as the scoreboard p6v3 cache incident)."""
    import joblib
    from argus_detect import build_modules, STACKER
    from triage import DegradationEstimator
    mids = None
    feature_names = None
    conflict_p99 = None
    meta = STACKER.parent / "stacker_meta.json"
    if meta.exists():
        meta_d = json.loads(meta.read_text())
        mids = meta_d.get("module_ids")
        feature_names = meta_d.get("feature_names")
        conflict_p99 = meta_d.get("conflict_p99")
    modules, ids = build_modules()
    _state["modules"] = modules
    _state["ids"] = ids
    _state["clf"] = joblib.load(STACKER)
    _state["mids"] = mids or ids
    _state["feature_names"] = feature_names
    _state["conflict_p99"] = conflict_p99
    _state["estimator"] = DegradationEstimator()
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS verdicts (
        id INTEGER PRIMARY KEY, ts REAL, filename TEXT, kind TEXT,
        p_fake REAL, abstained INTEGER, trust REAL, risk REAL,
        alpha REAL, modules_json TEXT)""")
    con.commit()
    con.close()


def run_court(pil, alpha=None, original_raw: bytes | None = None):
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
                       verdict_id=vid, artifact_dir=art, src_path=src,
                       original_raw=original_raw)
    d = _state["estimator"].estimate(ctx)
    outputs = [m.run(ctx, d) for m in modules]
    judge = StackerJudge(_state["clf"], d.evidence_capacity, _state["mids"],
                         feature_names=_state.get("feature_names"),
                         conflict_p99=_state.get("conflict_p99"))
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
            "out_of_conflict": bool(getattr(judge, "out_of_conflict", False)),
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
    from argus_detect import STACKER
    out = {"status": "ok", "channels": _state.get("ids", []),
           "stacker": "artifacts/" + STACKER.parent.name,
           "n_modules": len(_state.get("modules", []))}
    # Scope-labeled benchmark snapshot (D-037 arc): every number carries its
    # measurement scope — operators must never quote a number without it.
    bm_path = WS / "artifacts" / "benchmarks.json"
    if bm_path.exists():
        try:
            out["benchmarks"] = json.loads(bm_path.read_text())
        except Exception as e:
            out["benchmarks_error"] = str(e)[:120]
    return out


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
async def detect(file: UploadFile = File(...), alpha: float = Form(None)):
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
            v = run_court(pil.convert("RGB"), alpha, original_raw=data)
            kind = "image"
        elif ext in AUDIO_EXTS:
            from argus_detect import detect_audio
            a = detect_audio(tmp)
            kind = "audio"
            # D-035: coverage gate — off-manifold audio abstains, never votes
            v = {"p_fake_fused": a["p_fake_audio_mean"],
                 "abstained": bool(a.get("out_of_coverage", False)),
                 "trust": 0.7 if not a.get("out_of_coverage") else 0.2,
                 "risk": 0.9 if a.get("out_of_coverage") else 0.0,
                 "modules": [],
                 "audio": {k: v2 for k, v2 in a.items() if k != "per_segment"}}
        elif ext in VIDEO_EXTS:
            from argus_detect import detect_video
            # frame budget 5 for the API path by design (decision #4)
            v = detect_video(tmp, 5)
            kind = "video"
            # honesty fields (Phase 5): trust/risk/route come from detect_video —
            # never hardcoded here (D-035 lesson: hardcoded trust is silent
            # overconfidence)
            v.setdefault("route", "video_face")
            v.setdefault("confidence_degraded", False)
            v.setdefault("out_of_conflict", False)
            v.setdefault("abstained", False)
            v.setdefault("stacker", None)
            v.setdefault("modules", [])
        else:
            raise HTTPException(400, f"unsupported type {ext}")
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)
    from argus_detect import STACKER
    v.update({"kind": kind, "filename": file.filename,
              "elapsed_s": round(time.time() - t0, 2), "alpha": alpha})
    if kind == "video":
        v.setdefault("stacker", None)     # the VIDEO stacker, not the image court's
    else:
        v["stacker"] = STACKER.parent.name
    v.setdefault("out_of_conflict", False)
    try:
        log_verdict(file.filename, kind, v, alpha)
    except Exception as e:
        print(f"[log_verdict] failed: {e}")
    return v


@app.post("/api/explain")
async def explain(file: UploadFile = File(...)):
    """Layer-1 explanation engine: neutral VLM narration of an image (D-031).

    Not evidence — the court never sees this. Human-readable observations
    attached to an already-rendered verdict. Cached by image hash."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, "explain supports images only")
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty file")
    tmp = Path(tempfile.mkdtemp(prefix="argus_ex_")) / f"f{ext}"
    tmp.write_bytes(data)
    try:
        from PIL import Image
        from vlm_explain import explain_image
        pil = Image.open(tmp)
        pil.load()
        return explain_image(pil.convert("RGB"), save_path=tmp)
    except Exception as e:
        raise HTTPException(500, f"explain failed: {e}")
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

