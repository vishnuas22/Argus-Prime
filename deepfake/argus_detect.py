"""argus-detect: unified multi-sense deepfake verdict CLI.

Routes any media file through the right senses:
  image  -> Argus_Master evidence court + learned AIGC probe + face head (+ stacker fusion)
  video  -> N sampled frames through the image path (audio track requires ffmpeg)
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

PROBE_AIGC = WS / "artifacts" / "p5_full" / "probe.joblib"
PROBE_FACE = WS / "artifacts" / "p3_full" / "face_probe.joblib"
STACKER = WS / "artifacts" / "p6v2" / "stacker.joblib"
AUDIO_PROBE = WS / "artifacts" / "p4" / "audio_probe.joblib"


def build_modules():
    from registry import get_modules
    from learned_aigc_probe import LearnedAIGCProbe
    from learned_face_probe import LearnedFaceProbe
    modules = list(get_modules(force_reload=True))
    ids = ["metadata", "compression_history", "spectral_probe",
           "real_distribution_probe", "perturbation_probe"]
    if PROBE_AIGC.exists():
        modules.append(LearnedAIGCProbe(str(PROBE_AIGC)))
        ids.append("learned_aigc_probe")
    if PROBE_FACE.exists():
        modules.append(LearnedFaceProbe(str(PROBE_FACE)))
        ids.append("learned_face_probe")
    return modules, ids


def detect_image(pil, stacker_clf=None):
    from base import ImageContext
    from triage import DegradationEstimator
    from fusion import fuse
    from run_benchmark import run_image
    modules, _ = build_modules()
    estimator = DegradationEstimator()
    tmp = Path(tempfile.mkdtemp(prefix="argus_detect_"))
    art = Path(tempfile.mkdtemp(prefix="argus_art_"))
    vid = hashlib.sha256(str(pil.size).encode()).hexdigest()[:12]
    row = run_image(modules, estimator, pil, None, art, vid, tmp, stacker_clf)
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(art, ignore_errors=True)
    module_scores = {m: row.get(m) for m in
                     ("metadata", "compression_history", "spectral_probe",
                      "real_distribution_probe", "perturbation_probe",
                      "learned_aigc_probe", "learned_face_probe")
                     if row.get(m) is not None}
    return {
        "p_fake_fused": round(1.0 - row["p_real"], 4),
        "abstained": row["abstained"],
        "trust": row["trust"],
        "risk": row["risk"],
        "module_evidence": {k: round(-v, 4) for k, v in module_scores.items()},
    }


def detect_video(path: Path, n_frames: int):
    sys.path.insert(0, str(WS / "deepfake" / "scripts"))
    from video_score import visual_channel, audio_channel, combined
    vis = visual_channel(path, n_frames)
    aud = audio_channel(path)
    if vis is None:
        return {"error": "no scoreable faces in sampled frames"}
    return {
        "p_fake_video_combined": combined(vis, aud),
        "visual": vis,
        "audio": {k: v for k, v in (aud or {}).items() if k != "per_segment"},
    }


def detect_audio(path: Path):
    import torch
    import soundfile as sf
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
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
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval()
    bundle = joblib.load(AUDIO_PROBE)

    probs = []
    with torch.no_grad():
        for seg in segs:
            if len(seg) < SEG:
                seg = np.pad(seg, (0, SEG - len(seg)))
            inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            h = model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
            x = ((h - bundle["mu"]) / bundle["sd"]).reshape(1, -1)
            probs.append(float(bundle["clf"].predict_proba(x)[0, 1]))
    del model
    return {
        "p_fake_audio_mean": round(float(np.mean(probs)), 4),
        "segments_analyzed": len(probs),
        "per_segment": [round(p, 4) for p in probs],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    path = Path(args.file)
    assert path.exists(), f"no such file: {path}"
    ext = path.suffix.lower()
    stacker_clf = None
    if STACKER.exists():
        import joblib
        stacker_clf = joblib.load(STACKER)

    result = {"file": path.name}
    if ext in IMAGE_EXTS:
        from PIL import Image
        result.update(kind="image", **detect_image(Image.open(path), stacker_clf))
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
