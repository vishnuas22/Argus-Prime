"""Inject coverage_threshold into an audio probe bundle (train per-segment z-norm p99.5).

train_audio_probe.py dumps {clf, mu, sd} only; the serve path (argus_detect.detect_audio)
reads bundle["coverage_threshold"] for the D-035 coverage gate. This post-process computes
the threshold from a training cache exactly as p4's was derived (per-segment z-norm
distribution, p99.5) and re-dumps the bundle with it, without touching clf/mu/sd.

Verify mode (default when the bundle already carries a threshold): computes and compares,
writes nothing — use --force to actually overwrite.

Usage: .venv/bin/python deepfake/scripts/inject_audio_coverage.py \
    --bundle artifacts/p10_audio/audio_probe.joblib \
    --train data/caches/asv_stageA_train.npz [--force]
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing coverage_threshold (default: verify only)")
    args = ap.parse_args()

    b = joblib.load(args.bundle)
    z = np.load(args.train)
    keys = list(z.files)
    X = np.stack([z[k] for k in keys]).astype(np.float32)
    x = (X - b["mu"]) / b["sd"]                    # same standardization as the serve path
    zn = np.linalg.norm(x, axis=1)                 # per-segment z-norm = serve statistic
    thr = float(np.percentile(zn, 99.5))
    stats = {
        "znorm_median": round(float(np.median(zn)), 1),
        "znorm_p99": round(float(np.percentile(zn, 99)), 1),
        "znorm_p99_5": round(thr, 1),
        "n_vectors": int(len(zn)),
    }
    print(json.dumps(stats, indent=2))

    existing = b.get("coverage_threshold")
    if existing is not None and not args.force:
        print(f"VERIFY-ONLY: bundle has coverage_threshold={existing}, computed p99.5={thr:.1f} "
              f"(delta={thr - existing:+.1f}); wrote nothing")
        return 0

    if existing is not None:
        print(f"NOTE: overwriting existing coverage_threshold={existing} with {thr:.1f}")
    b["coverage_threshold"] = thr
    joblib.dump(b, args.bundle)
    meta_p = Path(args.bundle).parent / "train_meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    meta["coverage"] = {**stats,
                        "coverage_threshold": round(thr, 1),
                        "basis": "train per-segment z-norm p99.5 (p4-derivation parity)"}
    meta_p.write_text(json.dumps(meta, indent=2))
    print(f"injected coverage_threshold={thr:.1f} -> {args.bundle}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
