"""Extract face crops from videos: uniform frame sampling -> YuNet -> 256px crops.

Output layout mirrors input: <out>/<subdir>/<videoname>/frame_XXXX.jpg
Usage:
  .venv/bin/python deepfake/scripts/face_crop_video.py \
      --in data/raw/celebdf/Celeb-DF-v2/Celeb-synthesis --out data/corpora/cdf_faces/synthesis \
      --frames-per-video 8 [--max-videos N] [--stride-step S]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parents[2] / "tools" / "models" / "yunet.onnx"


def largest_face(det, img):
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return None
    x, y, fw, fh = [int(v) for v in max(faces, key=lambda t: t[2] * t[3])[:4]]
    pad = int(0.30 * max(fw, fh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    return img[y0:y1, x0:x1]


def process_video(det, vid: Path, out_dir: Path, n_frames: int) -> int:
    cap = cv2.VideoCapture(str(vid))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, max(total - 1, 0), n_frames, dtype=int) if total else range(n_frames)
    got, last_idx = 0, -1
    for k, fi in enumerate(idxs):
        if total == 0:
            cap.release()
            cap = cv2.VideoCapture(str(vid))
        if fi != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            last_idx = fi
        ok, frame = cap.read()
        if not ok:
            continue
        face = largest_face(det, frame)
        if face is None:
            continue
        side = min(face.shape[:2])
        ys, xs = (face.shape[0] - side) // 2, (face.shape[1] - side) // 2
        crop = cv2.resize(face[ys:ys + side, xs:xs + side], (256, 256))
        cv2.imwrite(str(out_dir / f"frame_{k:03d}.jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        got += 1
    cap.release()
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True)
    ap.add_argument("--out", dest="outdir", required=True)
    ap.add_argument("--frames-per-video", type=int, default=8)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.indir)
    vids = sorted(src.rglob("*.mp4"))
    if args.max_videos:
        vids = vids[:args.max_videos]
    print(f"{len(vids)} videos", flush=True)
    det = cv2.FaceDetectorYN.create(str(MODEL), "", (320, 320), score_threshold=0.55)
    done, empty = 0, 0
    for i, v in enumerate(vids):
        rel = v.relative_to(src).with_suffix("")
        out_dir = Path(args.outdir) / rel
        marker = out_dir / ".done"
        if marker.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        got = process_video(det, v, out_dir, args.frames_per_video)
        if got == 0:
            empty += 1
            out_dir.rmdir()
            continue
        marker.write_text("ok")
        done += 1
        if done % 25 == 0:
            print(f"[{i+1}/{len(vids)}] extracted={done} no-face={empty}", flush=True)
    print(f"done: {done} videos processed, {empty} without usable faces")
    return 0


if __name__ == "__main__":
    sys.exit(main())
