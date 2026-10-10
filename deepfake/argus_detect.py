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


def detect_video(path: Path, n_frames: int):
    sys.path.insert(0, str(WS / "deepfake" / "scripts"))
    from video_score import visual_channel, audio_channel, combined
    vis = visual_channel(path, n_frames)
    aud = audio_channel(path)
    if vis is None:
        return {"error": "no scoreable faces in sampled frames",
                "p_fake_fused": None,
                "abstained": True, "trust": 0.3, "risk": 0.8}
    comb = combined(vis, aud)
    return {
        "p_fake_video_combined": comb,
        "p_fake_fused": comb,
        "abstained": comb is None,
        "trust": 0.3 if comb is None else 0.5,
        "risk": 0.8 if comb is None else 0.4,
        "visual": vis,
        "audio": {k: v for k, v in (aud or {}).items() if k != "per_segment"},
        "audio_out_of_coverage": bool((aud or {}).get("out_of_coverage", False)),
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
            v = result.get("visual", {})
            print(f"video combined P(fake): {result.get('p_fake_video_combined')}")
            print(f"  visual mean {v.get('mean')} | temporal std {v.get('std')} | "
                  f"frames {v.get('frames_scored')}")
            a = result.get("audio", {})
            if isinstance(a, dict) and "p_fake_audio_mean" in a:
                print(f"  audio voice-clone P(fake) {a['p_fake_audio_mean']}")
        else:
            pf = result["p_fake_audio_mean"]
            print(f"voice-clone P(fake): {pf:.3f} over {result['segments_analyzed']} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
