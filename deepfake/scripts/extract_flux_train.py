"""Extract FLUX.1-dev training fakes from the ash12321 10k shard (offset 0).

Disjoint from the FLUX exam by construction (different source dataset:
exam = bitmind white_paper_holdout_3, train = ash12321/flux-1-dev-generated-10k).
Output: data/corpora/flux_train/fake/
Usage: .venv/bin/python deepfake/scripts/extract_flux_train.py [--n 300]
"""
import argparse
import io
import shutil
from pathlib import Path

import requests

WS = Path(__file__).resolve().parents[2]
URL = ("https://huggingface.co/datasets/ash12321/flux-1-dev-generated-10k/"
       "resolve/main/data/train-00000-of-00007.parquet")
OUT = WS / "data" / "corpora" / "flux_train" / "fake"
SHARD = WS / "data" / "raw" / "flux_train.parquet"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from PIL import Image

    have = len(list(OUT.glob("*.jpg"))) if OUT.exists() else 0
    if have >= args.n:
        print(f"flux_train/fake: already {have}")
        return 0
    if not SHARD.exists() or SHARD.stat().st_size < 10_000_000:
        SHARD.parent.mkdir(parents=True, exist_ok=True)
        print("downloading flux shard (~430MB)...", flush=True)
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(SHARD.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        shutil.move(SHARD.with_suffix(".part"), SHARD)
    OUT.mkdir(parents=True, exist_ok=True)
    col = pq.read_table(SHARD)["image"]
    got, i = 0, 0
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
                continue
            im.convert("RGB").save(OUT / f"flux_{got:04d}.jpg", quality=95)
            got += 1
        except Exception:
            continue
    print(f"flux_train/fake: {got} images", flush=True)
    return 0


if __name__ == "__main__":
    main()
