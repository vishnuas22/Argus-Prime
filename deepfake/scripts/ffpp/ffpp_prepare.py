"""FF++ post-download prep: mirror dir -> face crops per split -> manifest.

Run AFTER the HF mirror lands (already-extracted, NOT a zip):
  data/raw/ffpp_mirror/FaceForensics++_C23/real/<id>.mp4            (1000 youtube ids)
  data/raw/ffpp_mirror/FaceForensics++_C23/fake/<Method>/<a>_<b>.mp4 (500 pairs x 2 directions)

Produces data/corpora/ffpp_faces/{train,val,test}/{real,Deepfakes,Face2Face,FaceSwap,NeuralTextures}/<video>/frame_*.jpg
Fake <a>_<b>.mp4 is assigned to the split of its pair (either direction).
Output dir = full stem (a_b) so both pair directions survive as separate samples.

Usage: .venv/bin/python deepfake/scripts/ffpp/ffpp_prepare.py [--limit N] [--all-methods]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from face_crop_video import process_video  # noqa: E402
import cv2  # noqa: E402

MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
OUT = WS / "data" / "corpora" / "ffpp_faces"
SPLITS = WS / "data" / "raw" / "ffpp"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
ALL_METHODS = METHODS + ["FaceShifter", "DeepFakeDetection"]
FRAMES = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke-test cap per (split,label)")
    ap.add_argument("--all-methods", action="store_true")
    args = ap.parse_args()
    methods = ALL_METHODS if args.all_methods else METHODS
    if not MIRROR.exists():
        print("mirror missing:", MIRROR)
        return 1
    det = cv2.FaceDetectorYN.create(str(WS / "tools" / "models" / "yunet.onnx"),
                                    "", (320, 320), score_threshold=0.55)

    # split maps: id -> split (real), (a,b) either direction -> split (fakes)
    id_split, pair_split = {}, {}
    for s in ("train", "val", "test"):
        for a, b in json.load(open(SPLITS / f"{s}.json")):
            pair_split[(a, b)] = pair_split[(b, a)] = s
            id_split[a] = id_split[b] = s

    def extract_subset(jobs, out_split, label_dir):
        count = skipped = 0
        for vid_path in jobs:
            vid_id = vid_path.stem
            out_video = OUT / out_split / label_dir / vid_id
            if (out_video / ".done").exists():
                skipped += 1
                continue
            out_video.mkdir(parents=True, exist_ok=True)
            got = process_video(det, vid_path, out_video, FRAMES)
            if got:
                (out_video / ".done").write_text("ok")
                count += 1
            else:
                shutil.rmtree(out_video, ignore_errors=True)
            if args.limit and count >= args.limit:
                break
        print(f"{out_split}/{label_dir}: {count} new videos ({skipped} already done)", flush=True)

    # reals: real/<id>.mp4 -> split by id
    for s in ("train", "val", "test"):
        vids = [MIRROR / "real" / f"{i}.mp4" for i in sorted(
            k for k, v in id_split.items() if v == s) if (MIRROR / "real" / f"{i}.mp4").exists()]
        extract_subset(vids, s, "real")

    # fakes: fake/<Method>/<a>_<b>.mp4 -> split by pair (either direction)
    for method in methods:
        fdir = MIRROR / "fake" / method
        if not fdir.exists():
            print(f"method dir missing: {fdir}")
            continue
        by_split = {s: [] for s in ("train", "val", "test")}
        for v in sorted(fdir.glob("*.mp4")):
            parts = v.stem.split("_")
            if len(parts) < 2:
                continue
            s = pair_split.get((parts[0], parts[1]))
            if s:
                by_split[s].append(v)
        for s in ("train", "val", "test"):
            extract_subset(by_split[s], s, method)
    print("prep complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

