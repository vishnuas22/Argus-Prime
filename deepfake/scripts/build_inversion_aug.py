"""Build laundering-augmented views of inversion-training corpora (Phase 2).

For every source image create:
  _q50.jpg  : JPEG re-save quality 50
  _q30.jpg  : JPEG q30 + 0.6x Lanczos downscale (screenshot-style laundering)

Reals get augmented views (teach: compression => good VAE reconstruction != fake);
fakes get symmetric views (so compression isn't a real-only tell).
"""
import sys
from pathlib import Path
from PIL import Image

WS = Path("/Users/Dataghost/Argus_Prime")
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

JOBS = [
    # (src, dst, limit)  limit=None => all
    ("data/corpora/eval_real", "data/corpora/inv_aug/real", None),
    ("data/corpora/train_fake", "data/corpora/inv_aug/fake_sd15", None),
    ("data/corpora/xgen_sdxl", "data/corpora/inv_aug/fake_sdxl", None),
    ("data/corpora/xgen_sana", "data/corpora/inv_aug/fake_sana", None),
]

# ffpp train-split faces: one frame per video, capped per class.
# fakes = sample across all manipulation-method dirs proportionally.
def ffpp_sample(split_cls, limit):
    base = WS / "data/corpora/ffpp_faces/train"
    if split_cls == "real":
        roots = [base / "real"]
    else:
        roots = [d for d in base.iterdir() if d.is_dir() and d.name != "real"]
    per_root = max(1, limit // len(roots))
    out = []
    for root in roots:
        n = 0
        for vid in sorted(root.iterdir()):
            frames = sorted(vid.glob("frame_*.jpg"))
            if frames:
                out.append(frames[0])
                n += 1
            if n >= per_root or len(out) >= limit:
                break
    return out[:limit]

def augment(pil, dst_stem):
    pil.convert("RGB").save(f"{dst_stem}_q50.jpg", quality=50)
    w, h = pil.size
    small = pil.convert("RGB").resize((int(w * 0.6), int(h * 0.6)), Image.LANCZOS)
    small = small.resize((w, h), Image.LANCZOS)  # back up: laundering-then-restore
    small.save(f"{dst_stem}_q30.jpg", quality=30)
    # screenshot family (D-032): canvas PNG + recompressed screenshot JPEG
    _shot_augment(pil, dst_stem)


CHROME_H = 60


def _shot_canvas(pil):
    cw, ch = 1280, 800
    avail = ch - CHROME_H
    img = pil.convert("RGB")
    scale = min(cw / img.width, avail / img.height, 1.0)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    from PIL import ImageDraw
    canvas = Image.new("RGB", (cw, ch), (240, 240, 245))
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, cw, CHROME_H], fill=(52, 101, 164))
    d.rectangle([90, 16, cw - 90, CHROME_H - 16], fill=(255, 255, 255))
    canvas.paste(img, ((cw - img.width) // 2, CHROME_H + (avail - img.height) // 2))
    return canvas


def _shot_augment(pil, dst_stem):
    canvas = _shot_canvas(pil)
    canvas.save(f"{dst_stem}_shot.png")
    canvas.save(f"{dst_stem}_shot_q60.jpg", quality=60)


def shot_only(pil, dst_stem):
    """Generate ONLY the screenshot views (existing q50/q30 stay cached on disk)."""
    _shot_augment(pil, dst_stem)

def main():
    total = 0
    for src, dst, limit in JOBS:
        dst = WS / dst
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in (WS / src).rglob("*") if p.suffix.lower() in EXTS)
        if limit:
            files = files[:limit]
        for i, p in enumerate(files):
            out = dst / f"{p.stem}_{p.suffix[1:]}"
            if out.exists():
                continue
            try:
                augment(Image.open(p), str(dst / p.stem))
                total += 2
            except Exception as e:
                print(f"skip {p.name}: {e}", flush=True)
            if (i + 1) % 200 == 0:
                print(f"[{dst.name}] {i+1}/{len(files)}", flush=True)
        print(f"[{dst.name}] done", flush=True)

    # ffpp faces: copy originals (already c23-compressed) + one q50 view
    for cls, limit in (("real", 250), ("fake", 250)):
        dst = WS / f"data/corpora/inv_aug/ffpp_{cls}"
        dst.mkdir(parents=True, exist_ok=True)
        for j, f in enumerate(ffpp_sample(cls, limit)):
            out = dst / f"{f.parent.parent.name}_{f.parent.name}_{f.name}"
            if not out.exists():
                out.write_bytes(f.read_bytes())
            q50 = dst / f"{f.parent.parent.name}_{f.parent.name}_{f.stem}_q50.jpg"
            if not q50.exists():
                try:
                    Image.open(f).convert("RGB").save(q50, quality=50)
                except Exception as e:
                    print(f"skip {f.name}: {e}", flush=True)
            total += 2
            if (j + 1) % 100 == 0:
                print(f"[ffpp_{cls}] {j+1}/{limit}", flush=True)
        print(f"[ffpp_{cls}] done", flush=True)
    print(f"AUG_DONE total_new={total}", flush=True)

if __name__ == "__main__":
    main()
