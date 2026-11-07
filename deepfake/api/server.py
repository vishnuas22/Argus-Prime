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
from contextlib import asynccontextmanager
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "api"))
sys.path.insert(0, str(WS / "integrations" / "Argus_Master" / "backend"))

import os
os.environ["PATH"] = f"{WS/'tools'/'bin'}:{os.environ['PATH']}"

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from urllib.parse import urlparse
import anyio

DB = WS / "artifacts" / "verdicts.db"
STATIC = WS / "web"
MAX_UPLOAD = 100 * 1024 * 1024  # user ruling 2026-10-19: 413 above 100 MB (courts on faxed slabs; blocks zip-bombs/multi-GB bodies)

@asynccontextmanager
async def _lifespan(_app):
    # hop11: @app.on_event("startup") -> lifespan (audit P4 cosmetic; the
    # forward reference to init_court resolves at startup, after its def)
    init_court()
    yield


app = FastAPI(title="Argus Court", version="1.0", lifespan=_lifespan)
# CORS scoped to local origins by default (user ruling 2026-10-19 — "*" let any web
# page in the operator's browser POST + read verdicts cross-origin). Allow-list is
# env-overridable for legitimate cross-origin client integrations:
#   ARGUS_CORS_ORIGINS="https://client.example" uvicorn deepfake.api.server:app
_CORS = os.environ.get("ARGUS_CORS_ORIGINS",
                       "http://127.0.0.1:8420,http://localhost:8420").split(",")
app.add_middleware(CORSMiddleware, allow_origins=_CORS, allow_methods=["*"],
                   allow_headers=["*"])

# F8 (hop7): cross-origin mutation reject — the court accepts mutators only from its
# own origin. POST lanes are gated; GET lanes (health/history) stay open.
# No Origin header => allow (curl, lane batteries and pinned probe suites all lack it).
# Origin "null" (file:// sandbox) or a netloc mismatch with Host => 403 JSON.
_MUTATION_PATHS = ("/api/detect", "/api/explain")

@app.middleware("http")
async def _origin_gate(request, call_next):
    if request.method == "POST" and request.url.path in _MUTATION_PATHS:
        origin = request.headers.get("origin")
        if origin is not None:
            o = urlparse(origin.strip())
            same = o.scheme == "http" and \
                   o.netloc.lower() == request.headers.get("host", "").lower()
            if not same:
                return JSONResponse({"detail": "cross-origin POST rejected"},
                                    status_code=403)
    return await call_next(request)

# F7 (hop7): vendored three.js/gsap + fonts serve same-origin (zero CDN call-outs).
app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")

_state = {}

# Audit F2: the video lane's α bracket mutates a module global
# (argus_detect.USER_ALPHA); two concurrent video requests could interleave
# the bracket and cross-bind α. One lock serialises bracket + adjudication
# (ordering, never silence).
# hop11/FP-3: _VIDEO_ALPHA_LOCK retired — the α bracket it serialised is gone
# (form α is now a parameter threaded through detect_video; audit F2 closed at
# the root rather than guarded).

# hop9/FP-1: the court lanes are CPU-bound and ran ON the event loop (measured
# hop9: /api/health degraded 0.010s -> 11.8s during a k=8 image ramp — every
# endpoint frozen for the full adjudication). Court calls move to worker
# threads; CapacityLimiter(1) preserves the exact one-at-a-time serialization
# the loop used to provide, so all hop9-measured thread-safety properties hold
# while health/history/UI stay live. /api/explain gets its OWN gate: the VLM
# loads lazily (7s+ cold) and must never block a court adjudication.
_COURT_GATE = anyio.CapacityLimiter(1)
_EXPLAIN_GATE = anyio.CapacityLimiter(1)


def init_court():
    """Preload stacker + channel modules once (meta-driven, matches CLI).

    The stacker path comes from argus_detect.STACKER — the wired court —
    never hardcoded (same lesson as the scoreboard p6v3 cache incident)."""
    import joblib
    from argus_detect import build_modules, STACKER
    from triage import DegradationEstimator

    # Audit F13: sweep stale request tempdirs (>1 h = crash debris) at
    # startup — single-shot here, so nothing in-flight can be swept.
    troot = Path(tempfile.gettempdir())
    now_ts = time.time()
    for d in troot.glob("argus_*"):
        try:
            if d.is_dir() and now_ts - d.stat().st_mtime > 3600:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass
    mids = None
    feature_names = None
    conflict_p99 = None
    meta = STACKER.parent / "stacker_meta.json"
    if meta.exists():
        meta_d = json.loads(meta.read_text())
        mids = meta_d.get("module_ids")
        feature_names = meta_d.get("feature_names")
        conflict_p99 = meta_d.get("conflict_p99")
        _state["conformal_alpha"] = meta_d.get("conformal_alpha")  # B3
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
        alpha REAL, modules_json TEXT, recovery_json TEXT)""")
    try:                          # Phase 3 (§5.3): additive column on old DBs
        con.execute("ALTER TABLE verdicts ADD COLUMN recovery_json TEXT")
    except sqlite3.OperationalError:
        pass
    # Audit F5: WAL + busy timeout — concurrent adjudications must never
    # surface "database is locked" as a silently-lost ledger row (D-046).
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
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
    try:
        d = _state["estimator"].estimate(ctx)
        outputs = [m.run(ctx, d) for m in modules]
        judge = StackerJudge(_state["clf"], d.evidence_capacity, _state["mids"],
                             feature_names=_state.get("feature_names"),
                             conflict_p99=_state.get("conflict_p99"))
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
        # B3/D-039 + audit F3: BOTH α bindings under ONE save/restore umbrella —
        # the operator form α wins; else the stacker meta α binds for THIS fuse
        # call; module state is restored before returning either way (no
        # operator-α pin, no partial-chain growth, no cross-request residue)
        bind_alpha = None if alpha is not None else _state.get("conformal_alpha")
        prev_conformal = None
        if alpha is not None:
            prev_conformal = fusion.conformal_set
            fusion.conformal_set = functools.partial(prev_conformal, alpha=alpha)
        elif bind_alpha is not None:
            prev_conformal = fusion.conformal_set
            fusion.conformal_set = functools.partial(prev_conformal, alpha=bind_alpha)
        try:
            core, *_ = fuse(outputs, d, judge=judge)
        finally:
            if prev_conformal is not None:
                fusion.conformal_set = prev_conformal
    finally:
        # Audit F4: tempdirs are swept even when a court exception aborts the
        # pass (previously only the straight-through path cleaned up)
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(art, ignore_errors=True)
    from argus_detect import STACKER
    from fusion import conflict_metric, gate
    survivors, _unavailable = gate(outputs, d)
    verdict = {"p_fake_fused": round(1.0 - core.probabilities["camera_original"], 4),
               "abstained": bool(core.abstained), "trust": core.trust_score,
               "risk": core.risk_score, "hypotheses": core.hypothesis_set,
               "out_of_conflict": bool(getattr(judge, "out_of_conflict", False)),
               "modules": detail}
    # Phase 3 recovery ladder (§5.3): flagged layer; extras already in scope.
    # Top-level verdict fields are NEVER mutated (C-MECHB) — recovery rides
    # alongside as verdict["recovery"]; needs_review stays a flagged HTTP 200.
    extras = {"outputs": outputs, "d": d, "survivors": survivors, "core": core,
              "judge": judge, "k": conflict_metric(survivors), "pil": pil,
              "modules": modules, "estimator": _state["estimator"],
              "original_raw": original_raw,
              "stacker_clfs": {STACKER.parent.name: {
                  "clf": _state["clf"], "module_ids": _state["mids"],
                  "feature_names": _state.get("feature_names"),
                  "conflict_p99": _state.get("conflict_p99"),
                  # B3: R1 rescores at the court's operating point unless a
                  # form α was requested for this call (D-039)
                  "alpha": (None if alpha is not None
                            else _state.get("conformal_alpha"))}}}
    import recovery as recovery_mod
    verdict["recovery"] = recovery_mod.run("image", verdict, extras)
    return verdict


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/backdrop.jpg")
def backdrop():
    p = STATIC / "backdrop_opt.jpg"
    if not p.exists():
        p = STATIC / "backdrop.jpg"
    if p.exists():
        return FileResponse(p, media_type="image/jpeg")
    raise HTTPException(404, "Backdrop not found")


@app.get("/api/health")
def health():
    from argus_detect import STACKER
    out = {"status": "ok", "channels": _state.get("ids", []),
           "stacker": "artifacts/" + STACKER.parent.name,
           "n_modules": len(_state.get("modules", []))}
    # B3 transparency: the stacker's own conformal alpha (what the court binds
    # when a request supplies no explicit alpha)
    out["conformal_alpha"] = _state.get("conformal_alpha")
    # hop11/FP-6a: ledger row counter — log_verdict failure is SILENT by design
    # (detect L~386 try/except), so external row accounting is the only
    # row-loss detector; surface it where operators already look.
    try:
        _hcon = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=2)
        out["ledger_rows"] = _hcon.execute(
            "SELECT count(*) FROM verdicts").fetchone()[0]
        _hcon.close()
    except Exception as e:
        out["ledger_rows_error"] = str(e)[:80]
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
    # hop9/FP-2: clamp 1..200 (audit P4: the raw value hit SQLite LIMIT, where a
    # negative limit returns ALL rows — unbounded dump on one request)
    limit = max(1, min(limit, 200))
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM verdicts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    con.close()
    out = []
    for r in rows:
        d = dict(r) | {"modules": json.loads(r["modules_json"])}
        try:                       # Phase 3 (§5.3): merge recovery back
            d["recovery"] = json.loads(r["recovery_json"]) if r["recovery_json"] else None
        except (KeyError, TypeError, json.JSONDecodeError):
            d["recovery"] = None
        out.append(d)
    return out


def log_verdict(filename, kind, v, alpha):
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO verdicts (ts, filename, kind, p_fake, abstained, trust, risk, alpha, modules_json, recovery_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (time.time(), filename, kind, v["p_fake_fused"], int(v["abstained"]),
                 v["trust"], v["risk"], alpha, json.dumps(v["modules"]),
                 json.dumps(v.get("recovery") or {})))
    con.commit()
    con.close()


@app.post("/api/detect")
async def detect(request: Request, file: UploadFile = File(...), alpha: float = Form(None)):
    # Audit F6: refuse oversized BODIES before multipart buffering — a GB-scale
    # POST would spool entirely to disk before the parsed part-size check could
    # act. Slack covers multipart framing overhead.
    _cl = request.headers.get("content-length", "")
    if _cl.isdigit() and int(_cl) > MAX_UPLOAD + (1 << 20):
        raise HTTPException(413, "file too large (100 MB court cap)")
    ext = Path(file.filename or "").suffix.lower()
    if getattr(file, "size", None) and file.size > MAX_UPLOAD:
        raise HTTPException(413, "file too large (100 MB court cap)")
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
            try:
                pil = Image.open(tmp)
                pil.load()
            except Exception as e:
                # uploaded bytes are not a decodable image (truncated/corrupt):
                # 400 with clean body — never a raw 500 (release hardening 2026-10-19)
                raise HTTPException(400, f"unparseable image ({type(e).__name__})")
            v = await anyio.to_thread.run_sync(
                lambda: run_court(pil.convert("RGB"), alpha, original_raw=data),
                limiter=_COURT_GATE)
            kind = "image"
        elif ext in AUDIO_EXTS:
            # W3 wiring (ruled W2-(ii) accept-list): undecodable containers fail
            # CLOSED with the measured reason (W2 decoder table 2026-10-19:
            # ogg-vorbis not decodable by this libsndfile build) — 415 with
            # actionable copy, not a generic 400.
            try:
                import soundfile as _sf
                _sf.info(tmp)
            except Exception as e:
                raise HTTPException(415, (
                    "AUDIO FORMAT NOT DECODABLE BY THIS COURT — re-encode to "
                    "wav, flac, mp3, or ogg (libsndfile cannot open this "
                    f"container; measured undecodable: aac/m4a; {type(e).__name__})"))
            from argus_detect import detect_audio
            a = await anyio.to_thread.run_sync(lambda: detect_audio(tmp),
                                               limiter=_COURT_GATE)
            kind = "audio"
            # D-035: coverage gate — off-manifold audio abstains, never votes.
            # W3 wiring: a sub-2s score is a padding artifact (W2 padprobe) —
            # the court does not vote on it either; the D-035 posture applies.
            # The Phase-3 recovery layer (attached by detect_audio) carries the
            # operative numbers when triggered (§5.3, C-MECHB).
            short = bool(a.get("short_upload"))
            ooc = bool(a.get("out_of_coverage")) or short
            v = {"p_fake_fused": a["p_fake_audio_mean"],
                 "abstained": ooc,
                 "trust": 0.7 if not ooc else 0.2,
                 "risk": 0.9 if ooc else 0.0,
                 "modules": [],
                 "audio": {k: v2 for k, v2 in a.items() if k != "per_segment"},
                 "recovery": a.get("recovery")}
            # W3 wiring: the ruled accept-list flags lossy sources; the reason
            # rides to the UI's existing Degraded meta cell.
            v["confidence_degraded"] = bool(a.get("confidence_degraded"))
            if a.get("degraded_reason"):
                v["degraded_reason"] = a["degraded_reason"]
        elif ext in VIDEO_EXTS:
            from argus_detect import detect_video
            # hop11/FP-3: form α threads through detect_video -> _route_image_court
            # -> _image_stacker_clfs, so the D-039 "form α wins on every sub-route"
            # law now holds by PARAMETER instead of by mutating the module global.
            # The former _VIDEO_ALPHA_LOCK bracket is retired (audit F2 closed at
            # the root). Frame budget 5 for the API path by decision #4; the call
            # runs on a worker thread (hop9/FP-1) — the loop stays live.
            v = await anyio.to_thread.run_sync(
                lambda: detect_video(tmp, 5, alpha=alpha), limiter=_COURT_GATE)
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
    # Audit F2 (2026-11-06): unify surfaces. The UI dials the recovery-
    # adjudicated p (web/index.html reads rec.verdict.p_fake_fused) while the
    # API previously exposed only the raw p — divergent numbers on the same
    # case with no explicit bridge. Both numbers are now first-class on the
    # API response; the UI notes already print "original → adjudicated".
    _rec = v.get("recovery") or {}
    _rvp = (_rec.get("verdict") or {}).get("p_fake_fused") \
        if _rec.get("outcome") == "recovered" else None
    v["p_fake_adjudicated"] = _rvp if isinstance(_rvp, (int, float)) else None
    v["adjudicated"] = isinstance(_rvp, (int, float))
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
async def explain(request: Request, file: UploadFile = File(...)):
    """Layer-1 explanation engine: neutral VLM narration of an image (D-031).

    Not evidence — the court never sees this. Human-readable observations
    attached to an already-rendered verdict. Cached by image hash."""
    # Audit F6: same early body-cap precheck as /api/detect.
    _cl = request.headers.get("content-length", "")
    if _cl.isdigit() and int(_cl) > MAX_UPLOAD + (1 << 20):
        raise HTTPException(413, "file too large (100 MB cap)")
    ext = Path(file.filename or "").suffix.lower()
    if ext not in IMAGE_EXTS:
        raise HTTPException(400, "explain supports images only")
    if getattr(file, "size", None) and file.size > MAX_UPLOAD:
        raise HTTPException(413, "file too large (100 MB cap)")
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
        return await anyio.to_thread.run_sync(
            lambda: explain_image(pil.convert("RGB"), save_path=tmp),
            limiter=_EXPLAIN_GATE)
    except Exception as e:
        # 500 stays, but the detail is server-log only (user ruling 2026-10-19:
        # raw exception text never leaks to clients); court /api/detect paths
        # keep full detail by design — only this analyst layer is sanitized.
        print(f"[explain] {type(e).__name__}: {e}", file=sys.stderr)
        raise HTTPException(500, "analyst report unavailable (server log holds the detail)")
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

