"""SBI-style pseudo-fake generator: self-blended faces with feathered masks.

From each REAL face crop, produce a pseudo-fake: affine-warped, color-jittered
self-paste composited through a feathered elliptical mask (blend-boundary cue).
Usage: .venv/bin/python deepfake/scripts/sbi_pseudo.py --in DIR --out DIR [--limit N]
"""
import argparse
import io
import sys
from pathlib import Path

import cv2
import numpy as np


def blend_fake(face_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = face_bgr.shape[:2]
    angle = rng.uniform(-8, 8)
    scale = rng.uniform(0.90, 1.06)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += rng.uniform(-0.04, 0.04, size=2) * w
    warped = cv2.warpAffine(face_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    a = rng.uniform(0.9, 1.1)
    b = rng.uniform(-12, 12)
    warped = np.clip(warped.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

    mask = np.zeros((h, w), np.float32)
    cx, cy = w / 2 + rng.uniform(-0.08, 0.08) * w, h / 2 + rng.uniform(-0.08, 0.08) * h
    ax, ay = rng.uniform(0.28, 0.42) * w, rng.uniform(0.34, 0.48) * h
    cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), rng.uniform(0, 180),
                0, 360, 1.0, -1)
    k = int(rng.integers(11, 31)) * 2 + 1
    mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]

    out = face_bgr.astype(np.float32) * (1 - mask) + warped.astype(np.float32) * mask
    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    src, dst = Path(args.indir), Path(args.outdir)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.jpg"))
    if args.limit:
        files = files[:args.limit]
    rng = np.random.default_rng(args.seed)
    n = 0
    for p in files:
        outp = dst / f"sbi_{p.stem}.jpg"
        if outp.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        fake = blend_fake(img, rng)
        q = int(rng.integers(85, 96))
        ok, buf = cv2.imencode(".jpg", fake, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            outp.write_bytes(buf.tobytes())
            n += 1
    print(f"sbi fakes written: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
