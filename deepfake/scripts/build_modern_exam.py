"""Build the external modern-generator exam corpus (D-026 exam set).

Samples flat image dirs from three ungated BitMind HF shards:
  real_ext   bitmind/bm-real                                (external reals)
  fake_mj    bitmind/GenImage_MidJourney                    (Midjourney v5, commercial)
  fake_flux  bitmind/white_paper_holdout_3___FLUX.1-dev     (FLUX.1-dev holdout)

Output: data/corpora/modern_exam/{real_ext,fake_mj,fake_flux}/NNN.jpg
Usage: .venv/bin/python deepfake/scripts/build_modern_exam.py [--per-class 150]
"""
import argparse
import io
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import requests

WS = Path(__file__).resolve().parents[2]
OUT = WS / "data" / "corpora" / "modern_exam"

SHARDS = {
    "real_ext": "https://huggingface.co/datasets/bitmind/bm-real/resolve/main/data/train-00000-of-00045.parquet",
    "fake_mj": "https://huggingface.co/datasets/bitmind/GenImage_MidJourney/resolve/main/data/train-00000-of-00402.parquet",
    "fake_flux": "https://huggingface.co/datasets/bitmind/white_paper_holdout_3___FLUX.1-dev/resolve/main/data/train-00000-of-00004.parquet",
}


def image_column(table):
    for col in ("image", "img"):
        if col in table.column_names:
            return table.column(col)
    raise SystemExit(f"no image column; have {table.column_names}")


def extract(col, out_dir, n):
    out_dir.mkdir(parents=True, exist_ok=True)
    got, skipped = 0, 0
    for i in range(len(col)):
        if got >= n:
            break
        rec = col[i].as_py()
        data = rec["bytes"] if isinstance(rec, dict) else rec
        if not data:
            skipped += 1
            continue
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            im.load()
            if min(im.size) < 64:
                skipped += 1
                continue
            im.convert("RGB").save(out_dir / f"{got:04d}.jpg", quality=95)
            got += 1
        except Exception as e:
            skipped += 1
            if skipped % 50 == 0:
                print(f"  skip={skipped} last_err={e}", flush=True)
    print(f"{out_dir.name}: {got} images ({skipped} skipped)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=150)
    args = ap.parse_args()

    for name, url in SHARDS.items():
        out_dir = OUT / name
        have = len(list(out_dir.glob("*.jpg"))) if out_dir.exists() else 0
        if have >= args.per_class:
            print(f"{name}: already {have}, skipped")
            continue
        shard = WS / "data" / "raw" / "modern_exam" / f"{name}.parquet"
        shard.parent.mkdir(parents=True, exist_ok=True)
        if not shard.exists() or shard.stat().st_size < 10_000_000:
            print(f"{name}: downloading shard...", flush=True)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(shard.with_suffix(".part"), "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            shutil.move(shard.with_suffix(".part"), shard)
        table = pq.read_table(shard)
        extract(image_column(table), out_dir, args.per_class)
    print("modern exam corpus ready:", OUT)
    return 0


if __name__ == "__main__":
    main()
