"""Extract frozen-backbone features into an NPZ cache.

Backbone registry (single source of truth, shared with serve):
deepfake/bench/backbone_embed.py — model id, input size, mean/std, dim, pooling.

Views per image: clean + jpeg75 + resize75 by default (the 3-view CLIP-baseline
set); pass --views for the 5-view AIGC diet (clean,jpeg75,resize75,shot,shot_q60)
or clean-only for wild. MPS-accelerated when available. Resumable via cache file
(plus a .meta.json sidecar carrying backbone identity — a resumed npz can never
silently mix two backbones' keys, D-044).

Usage: .venv/bin/python deepfake/scripts/build_feature_cache.py <dir> <out.npz> \
           [--backbone clip|dv2b|dv2l|siglip|dv3] [--crop-face] [--limit N]

Output naming (caller-passed, enforced): data/caches/<name>.<bbtag>.npz where
bbtag is the backbone tag; an existing un-suffixed <name>.npz is the legacy CLIP
baseline and stays untouched. fp16 storage, NO L2 at storage — normalization
happens at train/serve consumption (lessons 2026-10-12).
"""
import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from backbone_embed import BACKBONES, embed, entry, get_embedder, resolve_revision  # noqa: E402

BBTAGS = ("clip", "dv2b", "dv2l", "siglip", "dv3")
EXTS = {".jpg", ".jpeg", ".png"}


def _shot_canvas(pil: Image.Image) -> Image.Image:
    """Browser-window screenshot simulation: 1280x800 canvas, chrome bar, center fit."""
    cw, ch, chrome = 1280, 800, 60
    avail = ch - chrome
    img = pil.convert("RGB")
    scale = min(cw / img.width, avail / img.height, 1.0)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (cw, ch), (240, 240, 245))
    from PIL import ImageDraw
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, cw, chrome], fill=(52, 101, 164))
    d.rectangle([90, 16, cw - 90, chrome - 16], fill=(255, 255, 255))
    canvas.paste(img, ((cw - img.width) // 2, chrome + (avail - img.height) // 2))
    return canvas


def views(pil: Image.Image):
    yield "clean", pil
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, "JPEG", quality=75)
    yield "jpeg75", Image.open(buf)
    w, h = pil.size
    yield "resize75", pil.resize((max(64, int(w * .75)), max(64, int(h * .75))), Image.BILINEAR)
    # screenshot family (D-032): canvas PNG + recompressed screenshot JPEG
    shot = _shot_canvas(pil)
    yield "shot", shot
    buf2 = io.BytesIO()
    shot.save(buf2, "JPEG", quality=60)
    yield "shot_q60", Image.open(buf2)


def _tagged_name(out: Path, backbone: str) -> str | None:
    """Validate <name>.<bbtag>.npz naming; returns the tag (None = legacy un-suffixed).

    A suffixed output must match the requested backbone, and an un-suffixed
    output is only legal for the CLIP baseline (existing un-suffixed caches ARE
    the CLIP baseline and are never rebuilt).
    """
    if out.suffix != ".npz":
        raise SystemExit(f"output must end in .npz: {out}")
    stem = out.stem
    parts = stem.split(".")
    if len(parts) > 1 and parts[-1] in BBTAGS:
        tag = parts[-1]
        if tag != backbone:
            raise SystemExit(
                f"output name says backbone '{tag}' but --backbone {backbone} "
                f"was requested — refusing to mix backbones in one npz (D-044)")
        return tag
    if backbone != "clip":
        raise SystemExit(
            f"--backbone {backbone} requires a '<name>.{backbone}.npz' output "
            f"path, got {out.name}")
    return None


def _file_list_sha(files) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(str(f).encode())
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("outnpz")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--views", default="clean,jpeg75,resize75",
                    help="comma list; 3-view CLIP-baseline set by default "
                         "(clean,jpeg75,resize75); pass "
                         "clean,jpeg75,resize75,shot,shot_q60 for the AIGC diet")
    ap.add_argument("--backbone", default="clip", choices=list(BBTAGS))
    ap.add_argument("--crop-face", action="store_true",
                    help="route each image through YuNet largest-face crop "
                         "(crop_largest_face) before resize — required for "
                         "face-row exam caches so they measure the serve path")
    args = ap.parse_args()
    VIEWS = [v.strip() for v in args.views.split(",") if v.strip()]
    if not VIEWS:
        raise SystemExit("--views produced an empty view list")

    cfg = entry(args.backbone)          # dv3 raises SystemExit here (license)
    out = Path(args.outnpz)
    tag = _tagged_name(out, args.backbone)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta_p = out.with_suffix(".meta.json") if tag else None
    if meta_p is not None and meta_p.exists():
        prev = json.loads(meta_p.read_text())
        if prev.get("backbone_id") != args.backbone:
            raise SystemExit(
                f"identity guard: {meta_p} says backbone_id="
                f"{prev.get('backbone_id')!r} but --backbone "
                f"{args.backbone!r} was requested — a resumed npz must not "
                f"mix backbones (D-044). Delete the npz+sidecar to rebuild.")
        if bool(prev.get("crop_face")) != bool(args.crop_face):
            raise SystemExit(
                f"identity guard: {meta_p} was built with crop_face="
                f"{prev.get('crop_face')} but this run requests "
                f"crop_face={args.crop_face} — face and full-image caches "
                f"cannot share one npz (D-044).")
    elif meta_p is not None and out.exists():
        raise SystemExit(
            f"identity guard: {out} exists but its sidecar {meta_p.name} is "
            f"missing — backbone identity unverifiable (D-044). Restore the "
            f"sidecar or delete the npz to rebuild.")

    recipe = "build_feature_cache.py" + f" --backbone {args.backbone}" + \
        f" --views {','.join(VIEWS)}" + \
        (" --crop-face" if args.crop_face else "") + \
        (f" --limit {args.limit}" if args.limit else "") + \
        f" {args.indir} {out.name}"

    cache = {}
    if out.exists():
        z = np.load(out)
        cache = {k: z[k] for k in z.files}
    root = Path(args.indir)
    all_files = sorted(p for p in root.rglob("*")
                       if p.suffix.lower() in EXTS and p.is_file())
    files = all_files[:args.limit] if args.limit else list(all_files)
    if not files:
        raise SystemExit(f"no {sorted(EXTS)} files under {root}")

    def key(f, v):
        return f"{f.relative_to(root)}:{v}"

    todo = [(f, v) for f in files for v in VIEWS if key(f, v) not in cache]
    n_no_face = 0
    crops = {}
    if args.crop_face and todo:
        from learned_face_probe import crop_largest_face
        size = int(cfg["input_size"])
        for f in sorted({f for f, _ in todo}):
            try:
                pil = Image.open(f)
                pil.load()
                arr = crop_largest_face(pil)
            except Exception as e:
                print(f"skip unreadable {f.name}: {e}", flush=True)
                n_no_face += 1
                continue
            if arr is None:
                n_no_face += 1
                continue
            crops[f] = Image.fromarray(arr).resize((size, size), Image.LANCZOS)
        todo = [(f, v) for f, v in todo if f in crops]

    n_done_files = len({f for f, _ in todo})
    print(f"{len(files)} files ({n_done_files} with pending work), "
          f"{len(todo)} embeddings to compute"
          + (f" | no-face skipped: {n_no_face}" if args.crop_face else ""),
          flush=True)

    get_embedder(args.backbone)         # load weights first (populates HF cache)
    revision = resolve_revision(args.backbone)

    if meta_p is not None:
        meta_p.write_text(json.dumps({
            "status": "building",
            "backbone_id": args.backbone,
            "model_repo": cfg.get("repo"),
            "model_id": cfg["timm_or_hf_id"],
            "revision": revision,
            "input_size": int(cfg["input_size"]),
            "mean": [float(x) for x in cfg["mean"]],
            "std": [float(x) for x in cfg["std"]],
            "dim": int(cfg["dim"]),
            "pooling": entry(args.backbone)["pooling"],
            "views": VIEWS,
            "crop_face": bool(args.crop_face),
            "source_dir": str(root),
            "n_files": len(all_files),
            "file_list_sha256": _file_list_sha(all_files),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "recipe": recipe,
            "n_keys": len(cache),
            "n_no_face_skipped": n_no_face,
        }, indent=2) + "\n")

    device = __import__("os").environ.get("ARGUS_DEVICE") or (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} | backbone: {args.backbone} "
          f"({cfg['timm_or_hf_id']} @ {cfg['input_size']}px, dim={cfg['dim']})",
          flush=True)

    t0, done = time.time(), 0
    with torch.no_grad():
        for f, v in todo:
            try:
                pil = crops.get(f)
                if pil is None:
                    pil = Image.open(f)
                feat = embed(pil, args.backbone)
                if feat.shape[0] != int(cfg["dim"]):
                    raise SystemExit(
                        f"backbone {args.backbone} returned dim "
                        f"{feat.shape[0]}, registry says {cfg['dim']}")
                cache[key(f, v)] = np.asarray(feat, dtype=np.float16)
            except Exception as e:
                print(f"skip {f.name}:{v}: {e}", flush=True)
            done += 1
            if done % 200 == 0:
                np.savez_compressed(out, **cache)
                rate = done / (time.time() - t0)
                print(f"[{done}/{len(todo)}] {rate:.2f}/s", flush=True)
    np.savez_compressed(out, **cache)
    if not cache:
        raise SystemExit(f"no embeddings produced for {out} — refusing to "
                         f"leave an empty cache behind")

    views_in_cache = sorted({k.rsplit(":", 1)[1] for k in cache if ":" in k})
    if meta_p is not None:
        prev = json.loads(meta_p.read_text())
        prev.update({
            "status": "complete",
            "revision": revision,
            "n_keys": len(cache),
            "views_in_cache": views_in_cache,
            "n_no_face_skipped": n_no_face,
            "n_files_embedded": len({k.rsplit(":", 1)[0] for k in cache
                                     if ":" in k}),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        meta_p.write_text(json.dumps(prev, indent=2) + "\n")
        print(f"sidecar: {meta_p}", flush=True)
    if views_in_cache != VIEWS:
        print(f"WARNING: requested views {VIEWS} != views in cache "
              f"{views_in_cache}", flush=True)
    print(f"cache saved: {len(cache)} vectors -> {out} "
          f"({(time.time()-t0)/60:.1f} min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
