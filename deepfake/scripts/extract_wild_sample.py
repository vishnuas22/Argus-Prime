"""Sample a balanced eval set from WildDeepfake test shards.

Extracts up to --per-class images each from real_test and fake_test tars.
Faces are pre-cropped by the dataset authors; we only filter tiny images.
Usage: .venv/bin/python deepfake/scripts/extract_wild_sample.py [--per-class 2000]
"""
import argparse
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw" / "wilddeepfake" / "deepfake_in_the_wild"
OUT = Path(__file__).resolve().parents[2] / "data" / "corpora" / "wild_sample"


def harvest(split_dir: Path, out_dir: Path, per_class: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    have = len(list(out_dir.glob("*.jpg")))
    if have >= per_class:
        print(f"{out_dir.name}: already {have}")
        return 0
    need = per_class - have
    n = 0
    tars = sorted(split_dir.glob("*.tar.gz")) + sorted(split_dir.glob("*.tar"))
    for tar in tars:
        if n >= need:
            break
        try:
            try:
                tf = tarfile.open(tar, "r:gz")
            except tarfile.ReadError:
                tf = tarfile.open(tar, "r:")
            with tf:
                for m in tf:
                    if not m.isfile() or not m.name.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue
                    dst_name = f"{tar.stem.split('.')[0]}_{Path(m.name).stem}_{n:05d}.jpg"
                    dst = out_dir / dst_name
                    if dst.exists():
                        continue
                    src = tf.extractfile(m)
                    data = src.read()
                    dst.write_bytes(data)
                    n += 1
                    if n >= need:
                        break
        except (tarfile.TarError, EOFError) as e:
            print(f"bad shard {tar.name}: {e}", flush=True)
            continue
    print(f"{split_dir.name}: wrote {n}, total {len(list(out_dir.glob('*.jpg')))}")
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=2000)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    args = ap.parse_args()
    out_sub = OUT.parent / ("wild_train" if args.split == "train" else "wild_sample")
    harvest(ROOT / f"real_{args.split}", out_sub / "real", args.per_class)
    harvest(ROOT / f"fake_{args.split}", out_sub / "fake", args.per_class)
    return 0


if __name__ == "__main__":
    sys.exit(main())
