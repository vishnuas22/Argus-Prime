"""V1d stage 2a — resumable ReStraV 21-D feature extraction.

Appends one JSON line per clip to artifacts/v1d_restrav/features.jsonl.
Re-running skips clips already present, so an interrupted run resumes in place.

Usage:
    .venv/bin/python deepfake/scripts/v1d_features.py [--device mps]
"""
import argparse
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

OUT = WS / "artifacts" / "v1d_restrav"
MANIFEST = OUT / "manifest.json"
FEAT = OUT / "features.jsonl"


def unique_clips(manifest):
    """Dedupe: the GenBuster real pool is shared across 4 domains."""
    seen, order = set(), []
    for name, d in manifest["domains"].items():
        for cls in ("real", "fake"):
            for p in d.get(cls, []):
                if p not in seen:
                    seen.add(p)
                    order.append(p)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from restrav_feats import (decode_clip, preprocess, dinov2,
                               features_from_Z, FEATURE_NAMES)
    import numpy as np
    import torch

    manifest = json.load(open(MANIFEST))
    clips = unique_clips(manifest)
    print(f"unique clips: {len(clips)}")

    done = set()
    if FEAT.exists():
        for line in open(FEAT):
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                continue
        print(f"resuming: {len(done)} already extracted")
    todo = [c for c in clips if c not in done]
    print(f"to do: {len(todo)}")

    if not todo:
        print("nothing to do")
        return

    device = torch.device(args.device)
    model = dinov2().to(device)
    print(f"DINOv2 on {device}")

    t0 = time.time()
    with open(FEAT, "a") as fh:
        for i, p in enumerate(todo):
            rec = {"path": p}
            try:
                clip = decode_clip(p, T=24, window_sec=2.0)
                batch = preprocess(clip).to(device)
                with torch.no_grad():
                    f = model.forward_features(batch)
                    cls = f["x_norm_clstoken"].unsqueeze(1)
                    patches = f["x_norm_patchtokens"]
                    Z = torch.cat([cls, patches], dim=1).flatten(1)
                    Z = Z.view(1, clip.shape[0], -1)
                feat = features_from_Z(Z)[0].cpu().numpy()
                rec["features"] = [round(float(x), 6) for x in feat]
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:160]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(todo) - i - 1) / rate
                print(f"  [{i+1}/{len(todo)}] {el/60:.1f}m elapsed, ETA {eta/60:.1f}m",
                      flush=True)
    print(f"features done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()