"""Generate models.lock — sha256 + size + library pins for every deployed artifact.

P0a deliverable. Run on the dev Mac where the artifacts live:

    .venv/bin/python deepfake/scripts/make_models_lock.py

Writes models.lock at the repo root. The lock is the content-addressed manifest
that bootstrap.py verifies; it is the thing that makes the deployed court
reproducible on a fresh clone.
"""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
LOCK = WS / "models.lock"

# Every file the DEPLOYED court loads. Verified against:
#   - deepfake/argus_detect.py (STACKER, VIDEO_STACKER, AUDIO_PROBE)
#   - deepfake/bench/channels.py OPTIONAL_CHANNELS (learned probes)
#   - deepfake/bench/video_channels.py (drift calibration)
#   - deepfake/bench/video_score.py (face probes)
#   - deepfake/bench/video_face_track.py (yunet, arcface, sface)
# av_calibration.json is deliberately absent: it is required only by AV-capable
# (38-feature) stackers, and the deployed p6vid_v4 is 32-feature.
ARTIFACTS = [
    "artifacts/p6v11/stacker.joblib",
    "artifacts/p6v11/stacker_meta.json",
    "artifacts/p6vid_v4/stacker.joblib",
    "artifacts/p6vid_v4/stacker_meta.json",
    "artifacts/p6vid_v4/drift_calibration.json",
    "artifacts/p6vid_v2/drift_calibration.json",
    "artifacts/aigc_shot_v2/probe.joblib",
    "artifacts/p3_full/face_probe.joblib",
    "artifacts/p3_ffpp/ffpp_only_probe.joblib",
    "artifacts/p7_mj/probe.joblib",
    "artifacts/inversion_probe/inversion_probe.joblib",
    "artifacts/provenance_probe/config.joblib",
    "artifacts/p4/audio_probe.joblib",
    "artifacts/benchmarks.json",
    "tools/models/yunet.onnx",
    "tools/models/arcface_w600k_r50.onnx",
    "tools/models/face_recognition_sface_2021dec.onnx",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main() -> int:
    missing = [a for a in ARTIFACTS if not (WS / a).exists()]
    if missing:
        print("MISSING artifacts (cannot build lock):", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        return 1

    libs = {}
    for name in ("torch", "torchvision", "numpy", "scikit-learn", "scipy",
                 "transformers", "onnxruntime", "opencv-python",
                 "opencv-python-headless", "joblib", "timm", "pillow"):
        v = pkg_version(name)
        if v:
            libs[name] = v

    lock = {
        "schema": "argus.models-lock.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_on": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
        },
        "library_pins": libs,
        "artifacts": {},
    }
    total = 0
    for rel in ARTIFACTS:
        p = WS / rel
        size = p.stat().st_size
        total += size
        lock["artifacts"][rel] = {"bytes": size, "sha256": sha256_of(p)}
        print(f"  {size:>10}  {rel}")

    lock["total_bytes"] = total
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"\nwrote {LOCK}  ({len(ARTIFACTS)} artifacts, {total/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())