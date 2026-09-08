"""Materialize screenshot-view dump corpus for the stacker diet (D-032 r6).

data/corpora/shot_aug/real  <- screenshot_jpg views of eval_real (real photos as users actually upload them)
data/corpora/shot_aug/fake  <- screenshot_jpg views of train_fake + cdf_flat_fake
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exam_launder import view_screenshot_jpg
from PIL import Image

WS = Path(__file__).resolve().parents[2]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

JOBS = [
    ("data/corpora/eval_real", "data/corpora/shot_aug/real", 250),
    ("data/corpora/train_fake", "data/corpora/shot_aug/fake", 130),
    ("data/corpora/cdf_flat_fake", "data/corpora/shot_aug/fake", 70),
]


def main():
    for src, dst, limit in JOBS:
        d = WS / dst
        d.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in (WS / src).rglob("*") if p.suffix.lower() in EXTS)[:limit]
        n = 0
        for p in files:
            out = d / f"{p.stem}_shot.jpg"
            if out.exists():
                continue
            try:
                pil = Image.open(p); pil.load()
                vpil, _ = view_screenshot_jpg(pil.convert("RGB"))
                vpil.save(out, "JPEG", quality=92)
                n += 1
            except Exception as e:
                print(f"skip {p.name}: {e}", flush=True)
        print(f"{dst}: {n} files", flush=True)


if __name__ == "__main__":
    main()
