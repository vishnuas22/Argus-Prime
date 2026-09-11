"""Video deepfake scoring: three channels.

  VISUAL   - face crops per sampled frame through the multi-domain face head
  TEMPORAL - dispersion of frame scores (fakes wobble more than reals)
  AUDIO    - extracted soundtrack through the XLS-R voice-clone probe

Usage:
  .venv/bin/python deepfake/scripts/video_score.py <video> [--frames 10] [--json]
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
FFMPEG = WS / "tools" / "ffmpeg"
FACE_PROBE = WS / "artifacts" / "p3_full" / "face_probe.joblib"
AUDIO_PROBE = WS / "artifacts" / "p4" / "audio_probe.joblib"


class _FaceScorer:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            import joblib
            from learned_face_probe import _ClipService
            clf = joblib.load(FACE_PROBE)
            model, torch, device = _ClipService.get()
            cls._inst = (clf, model, torch, device)
        return cls._inst


def frame_pfake(pil_crop):
    from learned_aigc_probe import _MEAN, _STD
    clf, model, torch, device = _FaceScorer.get()
    if hasattr(pil_crop, "resize"):
        import cv2
        arr_rgb = cv2.resize(np.asarray(pil_crop), (224, 224), interpolation=cv2.INTER_AREA)
        arr = arr_rgb.astype(np.float32) / 255.0
    else:
        arr = np.asarray(pil_crop, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        f = model(x).squeeze(0).float().cpu().numpy()
    f = f / (np.linalg.norm(f) + 1e-8)
    return float(clf.predict_proba(f.reshape(1, -1))[0, 1])


def visual_channel(path: Path, n_frames: int):
    from learned_face_probe import crop_largest_face
    from PIL import Image
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(total - 1, 1), n_frames, dtype=int)
    scores = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = crop_largest_face(Image.fromarray(rgb))
        if face is None:
            continue
        scores.append(frame_pfake(face))
    cap.release()
    if not scores:
        return None
    a = np.array(scores)
    return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "frac_gt_half": round(float((a > 0.5).mean()), 3), "frames_scored": len(scores)}


def audio_channel(path: Path):
    if not FFMPEG.exists():
        return {"error": "ffmpeg missing"}
    tmp = Path(tempfile.mkdtemp(prefix="argus_av_")) / "a.wav"
    r = subprocess.run([str(FFMPEG), "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                        str(tmp)], capture_output=True, timeout=300)
    if not tmp.exists() or tmp.stat().st_size < 1000:
        return {"note": "no extractable audio track"}
    sys.path.insert(0, str(WS / "deepfake"))
    from argus_detect import detect_audio
    out = detect_audio(tmp)
    import shutil
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return out


def combined(visual, audio):
    parts = []
    if visual:
        parts.append(visual["mean"])
        parts.append(min(0.99, visual["frac_gt_half"] + visual["std"]))
    # D-035: out-of-coverage audio (off-manifold) must not vote in the fusion
    if isinstance(audio, dict) and "p_fake_audio_mean" in audio \
            and not audio.get("out_of_coverage", False):
        parts.append(audio["p_fake_audio_mean"])
    if not parts:
        return None
    return round(float(np.mean(parts)), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=10)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    path = Path(args.video)
    assert path.exists(), path

    vis = visual_channel(path, args.frames)
    aud = None if args.no_audio else audio_channel(path)
    result = {
        "file": path.name,
        "visual": vis,
        "temporal_consistency": ("stable" if vis and vis["std"] < 0.08 else
                                 "unstable" if vis else None),
        "audio": aud,
        "combined_naive": combined(vis, aud),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nARGUS VIDEO — {path.name}")
        print("=" * 44)
        v = vis or {}
        print(f"VISUAL : mean P(fake) {v.get('mean')} | frac>0.5 {v.get('frac_gt_half')} "
              f"| {v.get('frames_scored', 0)} frames")
        print(f"TEMPORAL: {result['temporal_consistency']} (std={v.get('std')})")
        a = aud or {}
        if "p_fake_audio_mean" in a:
            print(f"AUDIO  : voice-clone P(fake) {a['p_fake_audio_mean']}")
        elif "note" in a:
            print(f"AUDIO  : {a['note']}")
        elif "error" in a:
            print(f"AUDIO  : {a['error']}")
        print(f"COMBINED (naive): P(fake) {result['combined_naive']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
