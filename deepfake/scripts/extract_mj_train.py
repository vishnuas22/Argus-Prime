"""Extract extra training images from the already-downloaded modern-exam shards.

Skips the first --offset indices (reserved for the exam sets) to keep
training and exam data disjoint. Output: data/corpora/mj_train/{fake,real}/
Usage: .venv/bin/python deepfake/scripts/extract_mj_train.py --n 500 --offset 150
"""
import argparse
import io
from pathlib import Path

import pyarrow.parquet as pq

WS = Path(__file__).resolve().parents[2]
RAW = WS / "data" / "raw" / "modern_exam"
OUT = WS / "data" / "corpora" / "mj_train"

SHARDS = {
    "fake": RAW / "fake_mj.parquet",
    "real": RAW / "real_ext.parquet",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=150)
    args = ap.parse_args()

    from PIL import Image
    for label, shard in SHARDS.items():
        out_dir = OUT / label
        out_dir.mkdir(parents=True, exist_ok=True)
        have = len(list(out_dir.glob("*.jpg")))
        if have >= args.n:
            print(f"{label}: already {have}, skipped")
            continue
        table = pq.read_table(shard)
        col = table["image"] if "image" in table.column_names else table["img"]
        got, skipped, i = 0, 0, args.offset
        while got < args.n and i < len(col):
            rec = col[i].as_py()
            i += 1
            data = rec["bytes"] if isinstance(rec, dict) else rec
            if not data:
                continue
            try:
                im = Image.open(io.BytesIO(data))
                im.load()
                if min(im.size) < 64:
                    skipped += 1
                    continue
                im.convert("RGB").save(out_dir / f"{label}_{got:04d}.jpg", quality=95)
                got += 1
            except Exception:
                skipped += 1
        print(f"{label}: {got} new ({skipped} skipped, scanned to row {i})", flush=True)
    return 0


if __name__ == "__main__":
    main()
