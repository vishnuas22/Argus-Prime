"""Generate ONLY screenshot views for inversion-aug corpora (D-032 r6).

Source images mirror build_inversion_aug.JOBS; writes <stem>_shot.png and
<stem>_shot_q60.jpg next to the existing _q50/_q30 views. Skips existing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_inversion_aug import JOBS, EXTS, shot_only, ffpp_sample
from PIL import Image

WS = Path(__file__).resolve().parents[2]


def main():
    total = 0
    for src, dst, limit in JOBS:
        dst_dir = WS / dst
        dst_dir.mkdir(parents=True, exist_ok=True)
        if "ffpp" in src:
            files = ffpp_sample("real" if "real" in src else "fake", limit or 500)
        else:
            files = sorted(p for p in (WS / src).rglob("*") if p.suffix.lower() in EXTS)
            if limit:
                files = files[:limit]
        for p in files:
            stem = dst_dir / p.stem
            if (dst_dir / f"{p.stem}_shot.png").exists():
                continue
            try:
                pil = Image.open(p); pil.load()
                shot_only(pil.convert("RGB"), str(stem))
                total += 1
            except Exception as e:
                print(f"skip {p.name}: {e}", flush=True)
            if total % 200 == 0 and total:
                print(f"{total} sources processed", flush=True)
    print(f"done: screenshot views generated for {total} new sources", flush=True)


if __name__ == "__main__":
    main()
