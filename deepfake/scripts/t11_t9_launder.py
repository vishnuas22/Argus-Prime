#!/usr/bin/env python3
"""T11/T9 laundering rungs — regen (img2img) + Chimera-class screen recapture + claim ladder.

Pre-registered spec: artifacts/generator_intake/t11_t9_launder_spec.md (2026-10-19).
Reuses exam_launder machinery (views, B3-alpha-aware court loader, score) unchanged.
Provenance evidence per row is measured with a dedicated single-probe pass (the claim
ladder), because exam_launder.score returns only (p_fake_fused, abstained).

Modes:
  --craft   CPU: build laundered specimens (jpeg_q50, whatsapp, screenshot,
            screenshot_jpg, recapture) under artifacts/exam_launder2/<name>/
  --regen   GPU, idle court only: SDXL img2img regeneration at 0.3/0.5/0.7
  --score   court pass over all specimens -> results.json
  --report  EXAM_REPORT.md from results.json

D-046: any bar miss escalates; this tool measures, it does not fix.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

import exam_launder as EL  # noqa: E402
from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402

OUT = WS / "artifacts" / "exam_launder2"
FIX = WS / "artifacts" / "mbeta" / "fixtures"
REGEN_STRENGTHS = (0.3, 0.5, 0.7)

# ---------------- new views ----------------


def view_recapture(pil):
    """Chimera-class phone-photo-of-screen approximation (programmatic, deterministic)."""
    img = pil.convert("RGB")
    w, h = img.size
    smol = img.resize((max(1, int(w * 0.88)), max(1, int(h * 0.88))), Image.LANCZOS)
    img = smol.resize((w, h), Image.LANCZOS)
    a = np.asarray(img).astype(np.float32) / 255.0
    enc = np.power(a, 1.0 / 2.2)
    dec = np.power(enc, 2.2)
    bleed = np.asarray(
        Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6))
    ).astype(np.float32) / 255.0
    mixed = 0.7 * dec + 0.3 * bleed
    rng = np.random.default_rng(42)
    mixed = np.clip(mixed + rng.normal(0.0, 1.5 / 255.0, mixed.shape), 0.0, 1.0)
    out = Image.fromarray((mixed * 255).astype(np.uint8))
    out = out.filter(ImageFilter.GaussianBlur(0.6))
    out = ImageEnhance.Brightness(out).enhance(0.98)
    return EL._recode(out, "JPEG", quality=85), "jpg"


def _dest(name, view_name, fmt):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{view_name}.{fmt}"


# ---------------- modes ----------------


def src_images():
    imgs = []
    for f in sorted(FIX.glob("fixture*.*")):
        imgs.append((f.stem, Image.open(f).convert("RGB")))
    for corpus_name in ("eval_fake", "eval_real"):
        for p in EL.corpus(corpus_name, 20):
            imgs.append((f"{corpus_name}_{p.stem}", Image.open(p).convert("RGB")))
    return imgs


def craft():
    n = 0
    for name, pil in src_images():
        for view_name, view in [
            ("jpeg_q50", EL.view_jpeg_q50),
            ("whatsapp", EL.view_whatsapp),
            ("screenshot", EL.view_screenshot),
            ("screenshot_jpg", EL.view_screenshot_jpg),
            ("recapture", view_recapture),
        ]:
            art, fmt = view(pil)
            p = _dest(name, view_name, fmt)
            p.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            art.save(buf, "JPEG" if fmt == "jpg" else "PNG")
            p.write_bytes(buf.getvalue())
            n += 1
    print(f"[craft] {n} specimens written under {OUT}")


def regen():
    """SDXL img2img on eval_fake + fixture3 (attack and claim routes)."""
    from diffusers import StableDiffusionXLImg2ImgPipeline
    import torch

    dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype)
    pipe.to("mps")
    gen = torch.Generator("mps").manual_seed(42) if torch.backends.mps.is_available() else None
    n = 0
    for name, pil in src_images():
        if not (name.startswith("eval_fake") or name.startswith("fixture3")):
            continue
        w, h = pil.size
        if max(w, h) > 1024:
            pil = pil.resize((1024, int(1024 * h / w)), Image.LANCZOS)
        for st in REGEN_STRENGTHS:
            out = pipe(prompt="high quality photo", image=pil, strength=st,
                       num_inference_steps=24, generator=gen).images[0]
            p = _dest(name, f"regen_{st:.1f}", "png")
            out.save(p)
            n += 1
            print(f"[regen] {name} strength={st}", flush=True)
    print(f"[regen] {n} regen specimens done", flush=True)


def _ctx_for(pil, path):
    """ImageContext replica of exam_launder.score's construction (raw bytes from disk)."""
    from base import ImageContext
    import hashlib, tempfile
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()[:16]
    fmt = "jpg" if path.suffix.lower() in (".jpg", ".jpeg") else "png"
    tmp = Path(tempfile.mkdtemp(prefix="t1x_tmp_"))
    src = tmp / f"{path.stem}.{fmt}"
    src.write_bytes(raw)
    adir = Path(tempfile.mkdtemp(prefix="t1x_art_")) / path.stem
    adir.mkdir(parents=True, exist_ok=True)
    return ImageContext(pil=pil.copy(), raw_bytes=raw, sha256=sha,
                        fmt="jpeg" if fmt == "jpg" else "png",
                        verdict_id=path.stem, artifact_dir=adir, src_path=src)


def provenance_evidence(pil, path):
    from channels import build_channels
    from triage import DegradationEstimator
    entry, = build_channels("provenance_probe")
    ctx = _ctx_for(pil, path)
    d = DegradationEstimator().estimate(ctx)
    out = entry.run(ctx, d)
    return float(getattr(out, "evidence", 0.0))


def score():
    modules, estimator, stacker_clfs = EL.load_court()
    res = {}
    for child in sorted(OUT.iterdir()):
        if not child.is_dir():
            continue
        rel = {}
        for p in sorted(child.glob("*")):
            if p.suffix.lower() not in EL.EXTS:
                continue
            pil = Image.open(p).convert("RGB")
            pf, ab = EL.score(modules, estimator, stacker_clfs, pil, p.suffix[1:], p.stem)
            prov = provenance_evidence(pil, p)
            rel[p.name] = {"p_fake_fused": round(float(pf), 4), "abstained": bool(ab),
                           "provenance": round(prov, 4)}
            print(f"[score] {child.name}/{p.name} pf={pf:.3f} abst={ab} prov={prov:.3f}",
                  flush=True)
        res[child.name] = rel
    json.dump(res, open(OUT / "results.json", "w"), indent=1)
    print(f"[score] -> {OUT/'results.json'}")


def report():
    res = json.load(open(OUT / "results.json"))
    lines = ["# T11/T9 laundering rungs — results",
             "(spec: artifacts/generator_intake/t11_t9_launder_spec.md, pre-registered 2026-10-19)", ""]
    for spec in sorted(res):
        lines.append(f"## {spec}")
        for fn, row in sorted(res[spec].items()):
            lines.append(f"- {fn} → p_fake={row['p_fake_fused']:.3f} "
                         f"abstained={row['abstained']} provenance={row['provenance']:.3f}")
        lines.append("")
    (OUT / "EXAM_REPORT.md").write_text("\n".join(lines))
    print(f"[report] -> {OUT/'EXAM_REPORT.md'}")


ap = argparse.ArgumentParser()
ap.add_argument("--craft", action="store_true")
ap.add_argument("--regen", action="store_true")
ap.add_argument("--score", action="store_true")
ap.add_argument("--report", action="store_true")
a = ap.parse_args()
if a.craft:
    craft()
if a.regen:
    regen()
if a.score:
    score()
if a.report:
    report()
