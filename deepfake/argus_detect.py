"""argus-detect: unified multi-sense deepfake verdict CLI.

Routes any media file through the right senses:
  image  -> Argus_Master evidence court + learned AIGC probe + face head (+ stacker fusion)
  video  -> sampled-frame face head + temporal dispersion + XLS-R audio (ffmpeg)
  audio  -> frozen XLS-R voice-clone probe

Usage:
  .venv/bin/python deepfake/argus_detect.py <file> [--frames 5] [--json]
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "integrations" / "Argus_Master" / "backend"))

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

STACKER = WS / "artifacts" / "p6v10" / "stacker.joblib"
# Video stacker (Phase 5). Loaded ONLY if the file exists — promotion is the
# file being present (restack_video.py --stage verdict + --promote). Until
# then detect_video serves the documented naive fallback with stacker=null.
VIDEO_STACKER = WS / "artifacts" / "p6vid_v1" / "stacker.joblib"
AUDIO_PROBE = WS / "artifacts" / "p4" / "audio_probe.joblib"


def build_modules():
    """Module set derived from the deployed stacker's meta (feature-space
    contract); falls back to every registered channel when no stacker exists."""
    from registry import get_modules
    from channels import CORE_IDS, OPTIONAL_CHANNELS, build as build_channels
    import json
    ids = None
    if STACKER.exists():
        meta_p = STACKER.with_name("stacker_meta.json")
        if meta_p.exists():
            ids = json.loads(meta_p.read_text()).get("module_ids")
    if ids is None:
        modules, ids = [], CORE_IDS + list(build_channels("all")[1])
        core = {m.module_id: m for m in get_modules(force_reload=True)}
        modules = [core[i] for i in CORE_IDS if i in core]
        opt, opt_ids = build_channels(",".join(i for i in ids if i in OPTIONAL_CHANNELS))
        modules += opt
        ids = CORE_IDS + opt_ids
        return modules, ids
    core = {m.module_id: m for m in get_modules(force_reload=True)}
    modules = [core[i] for i in ids if i in core]
    opt_ids = [i for i in ids if i in OPTIONAL_CHANNELS]
    opt, _ = build_channels(",".join(opt_ids))
    modules += opt
    return modules, ids


def detect_image(pil, stacker_clfs=None, original_raw: bytes | None = None):
    from base import ImageContext
    from triage import DegradationEstimator
    from fusion import fuse
    from run_benchmark import run_image
    modules, mids = build_modules()
    estimator = DegradationEstimator()
    tmp = Path(tempfile.mkdtemp(prefix="argus_detect_"))
    art = Path(tempfile.mkdtemp(prefix="argus_art_"))
    vid = hashlib.sha256(str(pil.size).encode()).hexdigest()[:12]
    row = run_image(modules, estimator, pil, None, art, vid, tmp, stacker_clfs,
                    original_raw=original_raw)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(art, ignore_errors=True)
    module_scores = {m: row.get(m) for m in mids if row.get(m) is not None}
    return {
        "p_fake_fused": round(1.0 - row["p_real"], 4),
        "abstained": row["abstained"],
        "trust": row["trust"],
        "risk": row["risk"],
        "out_of_conflict": bool(row.get("out_of_conflict", False)),
        "module_evidence": {k: round(-v, 4) for k, v in module_scores.items()},
    }


def _image_stacker_clfs() -> dict:
    """The image court's stacker bundle — ONE construction for CLI image calls
    and for the no-face video route, so a faceless video sees the same court
    the image CLI shows (not a stacker-less degraded variant)."""
    if not STACKER.exists():
        return {}
    import joblib
    meta_p = STACKER.with_name("stacker_meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return {STACKER.parent.name: {
        "clf": joblib.load(STACKER),
        "module_ids": meta.get("module_ids"),
        "feature_names": meta.get("feature_names"),
        "conflict_p99": meta.get("conflict_p99")}}


def _route_image_court(path: Path, k: int = 4) -> dict:
    """No-face route: whole-frame image court on k evenly-sampled frames.

    Rationale: whole-frame AIGC detection is the honest answer for faceless
    video (T2V, scenes, screen recordings). Confidence is DEGRADED because a
    single-frame court on a video frame carries no temporal evidence; trust
    and risk are hard-capped (<=0.4 / >=0.6). Median over >=2 scored frames,
    else the route abstains outright."""
    import cv2
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(total - 1, 1), k, dtype=int)
    stacker_clfs = _image_stacker_clfs()
    per_frame = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:                       # frame-level isolation: one bad frame ≠ route failure
            per_frame.append(detect_image(Image.fromarray(rgb), stacker_clfs))
        except Exception as e:
            print(f"[image_court] frame {fi} failed: {e}", file=sys.stderr)
    cap.release()
    scored = [p for p in per_frame
              if p.get("p_fake_fused") is not None and not p.get("abstained")]
    if len(scored) >= 2:
        med = round(float(np.median([p["p_fake_fused"] for p in scored])), 4)
        trust = min(0.4, float(np.median([p["trust"] for p in scored])))
        risk = max(0.6, float(np.median([p["risk"] for p in scored])))
        abstained = False
    else:
        med, trust, risk, abstained = None, 0.3, 0.8, True
    return {"route": "image_court", "confidence_degraded": True,
            "frames_court": len(per_frame), "frames_scored": len(scored),
            "per_frame": [{"p_fake_fused": p.get("p_fake_fused"),
                           "abstained": p.get("abstained")} for p in per_frame],
            "p_fake_fused": med, "p_fake_video_combined": None,
            "abstained": abstained, "trust": trust, "risk": risk}


_VID_STACKER_CACHE = {"loaded": False, "clf": None, "embedder": None}


def _load_video_stacker():
    """Load the promoted video stacker and pin its embedder — fail closed.

    Serve-contract asserts (D-040 skew class), same spirit as the image side
    (stacker_judge.py:20 refuses a stacker without its module_ids):
      - stacker_meta.json must exist and its feature_names must equal
        video_features.FEATURE_NAMES — otherwise the deployed model would be
        scored in a feature space it was never trained in;
      - drift_calibration.json must exist beside the stacker — without it
        drift_spikes silently zeros while the trained model expects calibrated
        spike counts.
    The embedder used at dump/train time MUST be the serve-time embedder —
    identity clustering and drift features are embedder-dependent; the choice
    is recorded in stacker_meta.json."""
    if not _VID_STACKER_CACHE["loaded"]:
        _VID_STACKER_CACHE["loaded"] = True
        if VIDEO_STACKER.exists():
            import joblib
            meta_p = VIDEO_STACKER.with_name("stacker_meta.json")
            if not meta_p.exists():
                raise SystemExit(
                    f"[video_stacker] {VIDEO_STACKER} exists but {meta_p} is "
                    f"missing — feature-space contract unverifiable; refusing "
                    f"to serve (fail closed)")
            meta = json.loads(meta_p.read_text())
            from video_features import FEATURE_NAMES
            if meta.get("feature_names") != FEATURE_NAMES:
                raise SystemExit(
                    f"[video_stacker] stacker_meta.feature_names != "
                    f"FEATURE_NAMES — the deployed video stacker was trained in "
                    f"a different feature space (D-040 skew class); refusing to "
                    f"serve (fail closed)")
            cal_p = VIDEO_STACKER.parent / "drift_calibration.json"
            if not cal_p.exists():
                raise SystemExit(
                    f"[video_stacker] {VIDEO_STACKER} exists but "
                    f"{cal_p} is missing — drift_spikes would silently zero "
                    f"while the trained model expects calibrated spikes "
                    f"(D-040); refusing to serve (fail closed)")
            _VID_STACKER_CACHE["clf"] = joblib.load(VIDEO_STACKER)
            try:
                emb = meta.get("embedder")
                if emb in ("sface", "arcface"):
                    from video_face_track import EMBEDDER_NAME, set_embedder
                    if emb != EMBEDDER_NAME:
                        set_embedder(emb)
                    _VID_STACKER_CACHE["embedder"] = emb
            except Exception as e:
                print(f"[video_stacker] embedder meta read failed: {e}",
                      file=sys.stderr)
    return _VID_STACKER_CACHE["clf"]


def detect_video(path: Path, n_frames: int = 12) -> dict:
    """Identity-aware video route with an honest no-face fallback.

    route=video_face : visual aggregation + identity drift + audio; verdict
        comes from the promoted video stacker when present, else from the
        documented naive fallback (combined()).
    route=image_court: zero faces in the sampled frames -> whole-frame image
        court on k=4 frames, median, confidence degraded (spec item 5.2).
    """
    sys.path.insert(0, str(WS / "deepfake" / "scripts"))
    from video_score import audio_channel, combined, visual_channel_bundle
    # Load the stacker FIRST: it pins the embedder recorded at train time
    # (D-042), so face embedding/clustering below runs with the same embedder
    # the stacker was trained on — otherwise the first video of a process
    # would be featurized with the default (sface) and scored by an
    # arcface-trained model.
    clf = _load_video_stacker()
    vis, tracks, _per_track, _obs = visual_channel_bundle(path, n_frames)
    if vis is None:
        court = _route_image_court(path)
        court.update({
            "p_fake_video_combined": court.get("p_fake_fused"),
            "stacker": None, "frame_budget": n_frames,
            "out_of_conflict": False,
            "identities": [], "n_identities": 0, "n_voting_identities": 0,
            "video_score": None, "visual": None,
            "drift": {"evidence": None, "p95": None, "spikes": None,
                      "unavailable_reason": "no_face_route"},
            "audio": {}, "audio_out_of_coverage": False,
            "confidence_degraded": True,
        })
        return court

    from video_channels import IdentityDriftProbe, VideoContext, drift_features
    from video_features import compute_row, row_to_X

    cal_p = VIDEO_STACKER.parent / "drift_calibration.json"
    spike = (float(json.loads(cal_p.read_text())["spike_thresh"])
             if cal_p.exists() else None)
    # no track_scores: IdentityDriftProbe.assess never reads ctx.track_scores —
    # scoring every track here (2 CLIP forwards + seeks each) was pure serve
    # latency with zero effect on the verdict.
    ctx = VideoContext(path=path, frames_sampled=n_frames, tracks=tracks)
    out = IdentityDriftProbe(calib_path=cal_p if cal_p.exists() else None).run(ctx)
    feats = drift_features(tracks, spike_thresh=spike)
    drift = {"evidence": None if out.unavailable_reason else out.evidence_score,
             "p95": feats["drift_p95"], "spikes": feats["drift_spikes"],
             "unavailable_reason": out.unavailable_reason,
             "drift_mean": feats["drift_mean"],
             "n_drift_tracks": feats["n_drift_tracks"]}

    aud = audio_channel(path)
    if clf is not None:
        row = compute_row(vis, feats, aud)
        p = round(float(clf.predict_proba(row_to_X(row).reshape(1, -1))[0, 1]), 4)
        stacker_tag = VIDEO_STACKER.parent.name
    else:
        p = combined(vis, aud)          # documented naive fallback (pre-promotion)
        stacker_tag = None
    scored = p is not None
    return {
        "route": "video_face",
        "p_fake_video_combined": p,
        "p_fake_fused": p,
        "stacker": stacker_tag,
        "frame_budget": n_frames,
        "abstained": not scored,
        # ADVISORY flag only — never overwrites p (mechanism-b, memory/patterns:33)
        "out_of_conflict": bool(vis.get("identity_conflict")),
        # trust/risk are INTENTIONAL CONSTANTS on the video_face route: the
        # video branch has no calibrated confidence model yet, so unlike the
        # image court (whose fusion layer derives trust/risk from evidence) a
        # fixed operating point is the honest ceiling — 0.5/0.4 when scored,
        # 0.3/0.8 on abstention (2026-10-11; documented decision, not drift).
        "trust": 0.5 if scored else 0.3,
        "risk": 0.4 if scored else 0.8,
        "identities": vis["identities"],
        "n_identities": vis["n_identities"],
        "n_voting_identities": vis["n_voting_identities"],
        "video_score": vis.get("video_score"),
        "largest_face_mean": vis.get("largest_face_mean"),
        "drift": drift,
        "audio": {k: v for k, v in (aud or {}).items() if k != "per_segment"},
        "audio_out_of_coverage": bool((aud or {}).get("out_of_coverage", False)),
        "confidence_degraded": False,
        "visual": vis,
    }


_AUDIO_MODEL = None


def _get_audio_model(device):
    """Process-level cache so batch video scoring loads XLS-R once."""
    global _AUDIO_MODEL
    if _AUDIO_MODEL is None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval()
        _AUDIO_MODEL = (fe, model)
    return _AUDIO_MODEL


def detect_audio(path: Path):
    import torch
    import soundfile as sf
    import joblib

    SR = 16000
    SEG = SR * 4
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        g = np.arange(len(wav)) * (SR / sr)
        wav = np.interp(g, np.arange(len(wav)), wav)
    segs = [wav[i:i + SEG] for i in range(0, max(1, len(wav) - SEG + 1), SEG)][:8] or \
           [np.pad(wav[:SEG], (0, max(0, SEG - len(wav))))]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    fe, model = _get_audio_model(device)
    bundle = joblib.load(AUDIO_PROBE)

    probs = []
    znorms = []
    with torch.no_grad():
        for seg in segs:
            if len(seg) < SEG:
                seg = np.pad(seg, (0, SEG - len(seg)))
            inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            h = model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
            x = ((h - bundle["mu"]) / bundle["sd"]).reshape(1, -1)
            probs.append(float(bundle["clf"].predict_proba(x)[0, 1]))
            znorms.append(float(np.linalg.norm(x)))
    del model
    # Coverage gate (D-035): embeddings far outside the training manifold mean
    # the probe is extrapolating — the lab-TTS failure mode measured on
    # Deepfake-Eval-2024 (AUC 0.559, FPR 0.795 with 0% abstention). Off-manifold
    # input => the court abstains instead of voting.
    zmean = float(np.mean(znorms))
    cov_thresh = bundle.get("coverage_threshold")
    ooc = bool(cov_thresh and zmean > cov_thresh)
    return {
        "p_fake_audio_mean": round(float(np.mean(probs)), 4),
        "segments_analyzed": len(probs),
        "per_segment": [round(p, 4) for p in probs],
        "znorm_mean": round(zmean, 1),
        "coverage_threshold": cov_thresh,
        "out_of_coverage": ooc,
        "abstained": ooc,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", default=None, type=float, metavar="A",
                    help="abstention policy: conformal alpha (vendored default 0.10 "
                         "commits at top-prob>=0.90; e.g. 0.3 commits at >=0.70 with "
                         "AUC unchanged on modern exams — see memory D-028)")
    ap.add_argument("file")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    assert path.exists(), f"no such file: {path}"
    ext = path.suffix.lower()
    if args.alpha is not None:
        import functools
        import fusion
        fusion.conformal_set = functools.partial(fusion.conformal_set, alpha=args.alpha)
    stacker_clfs = {}
    if STACKER.exists():
        import joblib
        meta_p = STACKER.with_name("stacker_meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        stacker_clfs[STACKER.parent.name] = {
            "clf": joblib.load(STACKER),
            "module_ids": meta.get("module_ids"),
            "feature_names": meta.get("feature_names"),
            "conflict_p99": meta.get("conflict_p99")}

    result = {"file": path.name}
    if ext in IMAGE_EXTS:
        from PIL import Image
        result.update(kind="image", **detect_image(Image.open(path), stacker_clfs,
                                                   original_raw=path.read_bytes()))
    elif ext in VIDEO_EXTS:
        result.update(kind="video", **detect_video(path, args.frames))
    elif ext in AUDIO_EXTS:
        result.update(kind="audio", **detect_audio(path))
    else:
        print(f"unsupported type: {ext}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nARGUS DETECT — {path.name}")
        print("=" * 46)
        kind = result["kind"]
        if kind == "image":
            pf = result["p_fake_fused"]
            verdict = "SYNTHETIC/MANIPULATED" if pf > 0.5 else "AUTHENTIC-LEANING"
            if result["abstained"]:
                verdict += " (court abstains: low confidence)"
            if result.get("out_of_conflict"):
                verdict += " (court abstains: out of conflict — panel unresolved)"
            print(f"fused P(fake): {pf:.3f}  ->  {verdict}")
            print(f"trust {result['trust']:.2f} | risk {result['risk']:.2f}")
            print("evidence (P-fake per module):")
            for k, v in sorted(result["module_evidence"].items(), key=lambda kv: -abs(kv[1] - .5)):
                print(f"  {k:26s} {v:.3f}")
        elif kind == "video":
            print(f"video fused P(fake): {result.get('p_fake_fused')}  "
                  f"-> route {result.get('route')} "
                  f"({'DEGRADED ' if result.get('confidence_degraded') else ''}"
                  f"stacker {result.get('stacker') or 'none'})")
            v = result.get("visual") or {}
            if result.get("route") == "video_face":
                print(f"  video_score {result.get('video_score')} | identities "
                      f"{result.get('n_identities')} (voting "
                      f"{result.get('n_voting_identities')}) | largest_face_mean "
                      f"{result.get('largest_face_mean')} | conflict "
                      f"{result.get('out_of_conflict')}")
                d = result.get("drift") or {}
                print(f"  drift p95 {d.get('p95')} spikes {d.get('spikes')} "
                      f"({d.get('unavailable_reason') or 'ok'})")
            else:
                print(f"  no face in sampled frames -> image court on "
                      f"{result.get('frames_court')} frames "
                      f"(scored {result.get('frames_scored')})")
            a = result.get("audio", {})
            if isinstance(a, dict) and "p_fake_audio_mean" in a:
                print(f"  audio voice-clone P(fake) {a['p_fake_audio_mean']}")
        else:
            pf = result["p_fake_audio_mean"]
            print(f"voice-clone P(fake): {pf:.3f} over {result['segments_analyzed']} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
