"""Video deepfake scoring: identity-aware visual + advisory audio.

  VISUAL   - ALL faces on ALL sampled frames, clustered into identity tracks
             (SFace); per-identity robust stats; verdict = max over voting
             identities of per-identity top-k-mean (the MINTIME multi-identity
             fix — a manipulated secondary face no longer hides behind the
             largest face). Legacy largest-face stats stay in-row for A/B.
  TEMPORAL - identity-drift module lives in video_channels.py (item 2); the
             old `std < 0.08` pseudo-temporal is retired as unsupported.
  AUDIO    - extracted soundtrack through the XLS-R voice-clone probe (D-035
             coverage gate; advisory only).

Usage:
  .venv/bin/python deepfake/scripts/video_score.py <video> [--frames 12] [--json]
"""
import argparse
import json
import math
import shutil
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


def _ffmpeg_bin() -> Path | None:
    """Prefer the pinned tools/ffmpeg; fall back to a system ffmpeg.

    tools/ffmpeg is an x86 evermeet build — on arm64 hosts it fails with
    Errno 86 ("Bad CPU type"), so a bare .exists() check is not enough."""
    for cand in (FFMPEG, Path(shutil.which("ffmpeg") or "/nonexistent")):
        if not cand.exists():
            continue
        try:
            if subprocess.run([str(cand), "-version"], capture_output=True,
                              timeout=20).returncode == 0:
                return cand
        except OSError:
            continue
    return None


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


def _robust(scores: np.ndarray) -> dict:
    return {"mean": round(float(scores.mean()), 4),
            "std": round(float(scores.std()), 4),
            "min": round(float(scores.min()), 4),
            "max": round(float(scores.max()), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "frac_gt_half": round(float((scores > 0.5).mean()), 3)}


def _identity_stats(scores: np.ndarray, track, rank: int) -> dict:
    """Per-identity robust stats + top-k-mean (k = max(3, ceil(0.4*n)), clipped)."""
    n = len(scores)
    k = min(n, max(3, math.ceil(0.4 * n)))
    topk = float(np.sort(scores)[-k:].mean())
    d = _robust(scores)
    d.update(id=rank, frames=n, topk_mean=round(topk, 4),
             obs_only=bool(track.obs_only))
    return d


def visual_channel(path: Path, n_frames: int = 12):
    """Identity-aware visual channel. Returns None only when ZERO faces are
    detected in the sampled frames (the no-face route upstream)."""
    vis, _tracks, _tscores, _obs = visual_channel_bundle(path, n_frames)
    return vis


def visual_channel_bundle(path: Path, n_frames: int = 12):
    """visual_channel + the observation bundle the drift module needs.

    Returns (visual dict | None, tracks, per-track CLIP scores, obs).
    Single decode pass; every face's CLIP score is shared between the
    identity stats and the legacy largest-face stats."""
    from video_face_track import (FRAME_BUDGET_DEFAULT, cluster_identities,
                                  observe_video)
    obs, stats = observe_video(path, n_frames or FRAME_BUDGET_DEFAULT,
                               with_scores=True)
    if not obs:
        return None, [], {}, []
    scores = np.asarray(stats["scores"], dtype=np.float32)
    tracks = cluster_identities(obs)

    track_of = {id(x): t.track_id for t in tracks for x in t.obs}
    per_track = {t.track_id: [] for t in tracks}
    for o, s in zip(obs, scores):
        per_track[track_of[id(o)]].append(s)
    identity_rows = [_identity_stats(np.asarray(per_track[t.track_id], np.float32),
                                     t, t.track_id) for t in tracks]
    identity_rows.sort(key=lambda r: (-r["frames"], r["id"]))

    voting = [r for r in identity_rows if not r["obs_only"]]
    video_score = (round(max(r["topk_mean"] for r in voting), 4) if voting else None)

    big = scores[np.asarray(stats["largest_idx"], dtype=int)] \
        if stats["largest_idx"] else np.zeros(0, dtype=np.float32)
    all_stats = _robust(scores)
    legacy = _robust(big) if len(big) else {}

    conflict_votes = [r for r in voting if r["frames"] >= 3]
    identity_conflict = bool(
        len(conflict_votes) >= 2 and
        (max(r["mean"] for r in conflict_votes) -
         min(r["mean"] for r in conflict_votes)) > 0.5)

    from video_channels import drift_features, DRIFT_CALIB
    import json as _json
    spike = None
    if DRIFT_CALIB.exists():
        spike = float(_json.loads(DRIFT_CALIB.read_text())["spike_thresh"])

    visual = {
        "identities": identity_rows,
        "n_identities": len(identity_rows),
        "n_voting_identities": len(voting),
        "frames_scored": int(len({o.frame_idx for o in obs})),
        "faces_detected": len(obs),
        "video_score": video_score,
        **all_stats,
        "largest_face_mean": legacy.get("mean", 0.0),
        "largest_face_std": legacy.get("std", 0.0),
        "largest_face_frames": len(big),
        "identity_conflict": identity_conflict,
        "_drift": drift_features(tracks, spike_thresh=spike),
    }
    return visual, tracks, per_track, obs


def audio_channel(path: Path):
    ff = _ffmpeg_bin()
    if ff is None:
        return {"error": "ffmpeg missing"}
    tmp = Path(tempfile.mkdtemp(prefix="argus_av_")) / "a.wav"
    r = subprocess.run([str(ff), "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
                        str(tmp)], capture_output=True, timeout=300)
    if not tmp.exists() or tmp.stat().st_size < 1000:
        shutil.rmtree(tmp.parent, ignore_errors=True)
        return {"note": "no extractable audio track"}
    sys.path.insert(0, str(WS / "deepfake"))
    from argus_detect import detect_audio
    out = detect_audio(tmp)
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
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    path = Path(args.video)
    assert path.exists(), path

    vis, tracks, _tscores, _obs = visual_channel_bundle(path, args.frames)
    aud = None if args.no_audio else audio_channel(path)
    result = {
        "file": path.name,
        "visual": vis,
        "audio": aud,
        "combined_naive": combined(vis, aud),
        "drift": (vis or {}).get("_drift"),
    }
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"\nARGUS VIDEO — {path.name}")
        print("=" * 44)
        v = vis or {}
        print(f"VISUAL : video_score {v.get('video_score')} | identities "
              f"{v.get('n_identities')} (voting {v.get('n_voting_identities')}) | "
              f"faces {v.get('faces_detected')}")
        print(f"LEGACY : largest_face_mean {v.get('largest_face_mean')} | "
              f"all-face mean {v.get('mean')} | frac>0.5 {v.get('frac_gt_half')} | "
              f"{v.get('frames_scored', 0)} frames")
        print(f"IDENT  : conflict {v.get('identity_conflict')} "
              f"(advisory flag only)")
        d = result.get("drift") or {}
        print(f"DRIFT  : p95 {d.get('drift_p95')} | spikes {d.get('drift_spikes')} | "
              f"{d.get('unavailable_reason') or 'ok'}")
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
