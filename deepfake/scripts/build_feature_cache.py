"""Extract frozen CLIP ViT-L/14 penultimate features into an NPZ cache.

Views per image: clean + jpeg75 + resize75 (robustness without extra corpus).
MPS-accelerated when available. Resumable via cache file.
Usage: .venv/bin/python deepfake/scripts/build_feature_cache.py <dir> <out.npz> [--limit N]
"""
import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
MODEL_NAME = "vit_large_patch14_clip_224.openai"
EXTS = {".jpg", ".jpeg", ".png"}


def to_tensor(pil: Image.Image) -> "torch.Tensor":
    arr = np.asarray(pil.convert("RGB").resize((224, 224), Image.LANCZOS), dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr.transpose(2, 0, 1))


def views(pil: Image.Image):
    yield "clean", pil
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "JPEG", quality=75)
    yield "jpeg75", Image.open(buf)
    w, h = pil.size
    yield "resize75", pil.resize((max(64, int(w * .75)), max(64, int(h * .75))), Image.BILINEAR)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("outnpz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default="clean,jpeg75,resize75")
    args = ap.parse_args()
    VIEWS = [v.strip() for v in args.views.split(",") if v.strip()]

    out = Path(args.outnpz)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if out.exists():
        z = np.load(out)
        cache = {k: z[k] for k in z.files}
    root = Path(args.indir)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS and p.is_file())
    if args.limit:
        files = files[:args.limit]
    def key(f, v):
        return f"{f.relative_to(root)}:{v}"

    todo = [(f, v) for f in files for v in VIEWS if key(f, v) not in cache]
    print(f"{len(files)} files, {len(todo)} embeddings to compute")

    device = __import__("os").environ.get("ARGUS_DEVICE") or (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device}")
    import timm
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(device).eval()
    t0, done = time.time(), 0
    with torch.no_grad():
        for f, v in todo:
            try:
                x = to_tensor(Image.open(f)).unsqueeze(0).to(device)
                feat = model(x).squeeze(0).float().cpu().numpy()
                cache[key(f, v)] = feat.astype(np.float16)
            except Exception as e:
                print(f"skip {f.name}:{v}: {e}", flush=True)
            done += 1
            if done % 200 == 0:
                np.savez_compressed(out, **cache)
                rate = done / (time.time() - t0)
                print(f"[{done}/{len(todo)}] {rate:.2f}/s", flush=True)
    np.savez_compressed(out, **cache)
    print(f"cache saved: {len(cache)} vectors -> {out} ({(time.time()-t0)/60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
