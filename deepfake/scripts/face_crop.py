"""Face detection + cropping via OpenCV YuNet. Produces 256px crops + QA contact sheet.
Usage: .venv/bin/python deepfake/scripts/face_crop.py --in DIR --out DIR [--limit N]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "tools" / "models" / "yunet.onnx"
EXTS = {".jpg", ".jpeg", ".png"}


def largest_face(detector, img_bgr):
    h, w = img_bgr.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    f = max(faces, key=lambda x: x[2] * x[3])
    x, y, fw, fh = [int(v) for v in f[:4]]
    pad = int(0.30 * max(fw, fh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
    return img_bgr[y0:y1, x0:x1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.indir)
    dst = Path(args.outdir)
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in EXTS)
    if args.limit:
        files = files[:args.limit]
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), score_threshold=0.55)
    ok, miss = 0, 0
    crops = []
    for i, p in enumerate(files):
        outp = dst / f"{p.stem}.jpg"
        if outp.exists():
            ok += 1
            continue
        img = cv2.imread(str(p))
        if img is None:
            miss += 1
            continue
        if min(img.shape[:2]) < 200:
            f = max(1.0, 260 / min(img.shape[:2]))
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        face = largest_face(det, img)
        if face is None:
            miss += 1
            continue
        side = min(face.shape[:2])
        face = cv2.resize(face, (256, 256)) if abs(face.shape[0] - face.shape[1]) < 4 else cv2.resize(
            _center_square(face), (256, 256))
        cv2.imwrite(str(outp), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if len(crops) < 36:
            crops.append(face)
        ok += 1
        if (i + 1) % 500 == 0:
            print(f"[{i+1}/{len(files)}] ok={ok} noface={miss}", flush=True)
    if crops:
        sheet = np.hstack([np.vstack(crops[i::6][:6]) for i in range(6)])
        cv2.imwrite(str(dst.parent / "qa_contact_sheet.jpg"), sheet)
    print(f"done: {ok} crops, {miss} without detectable face")
    return 0


def _center_square(img):
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return img[y0:y0 + s, x0:x0 + s]


if __name__ == "__main__":
    sys.exit(main())
