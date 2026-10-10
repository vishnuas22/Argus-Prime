"""Full-court laundering/screenshot robustness exam (Priority 2).

Question: does the DEPLOYED p6v7 court survive the actual user input
distribution — screenshots, messenger recompression, resize laundering —
on BOTH sides (fake AUC retention AND real FPR: a real screenshot of a
real photo must NOT read as fake)?

Views per source image:
  pristine       as-is (baseline)
  jpeg_q50       single JPEG re-save q50
  whatsapp       0.8x resize -> JPEG q70 -> restore -> JPEG q40 (messenger double compression)
  screenshot     pasted onto 1280x800 canvas w/ browser chrome bar, saved PNG (screen capture sim)
  screenshot_jpg screenshot canvas then JPEG q60 (messenger screenshot recompression)

Acceptance gates (defined BEFORE running):
  G1  pooled AUC (eval_real vs SD1.5+CDF fakes) >= pristine_pooled_AUC - 0.05 on every laundered view
  G2  fused FPR@0.5 on laundered/screenshot REALS <= 0.15 (user-facing false accusations)
  G3  report-only: sg2/flux/mj per-view AUC tracked (SG2 known-weak post-processing; no gate)

Usage: ./.venv/bin/python deepfake/scripts/exam_launder.py [--out DIR] [--force]
Outputs: artifacts/exam_launder_<stacker>_<YYYYMMDD>/launder_exam.csv + pivot to stdout + EXAM_REPORT.md
"""
import argparse
import csv
import io
import json
import sys
import tempfile
import time
import shutil
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

from PIL import Image, ImageDraw

N_SAMPLE = {"eval_real": 45, "train_fake": 45, "cdf_flat_fake": 25,
            "sg2_probe": 11, "flux_train": 20, "mj_train": 20}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def corpus(name, limit):
    root = WS / "data" / "corpora" / name
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)[:limit]


# ---------------- laundering transforms ----------------

def _recode(pil, fmt, **kw):
    buf = io.BytesIO()
    pil.save(buf, fmt, **kw)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def view_pristine(pil):
    return pil, "png"


def view_jpeg_q50(pil):
    return _recode(pil, "JPEG", quality=50), "jpg"


def view_whatsapp(pil):
    w, h = pil.size
    small = pil.resize((int(w * 0.8), int(h * 0.8)), Image.LANCZOS)
    mid = _recode(small, "JPEG", quality=70).resize((w, h), Image.LANCZOS)
    return _recode(mid, "JPEG", quality=40), "jpg"


CHROME_H = 60


def _canvas(pil):
    """Paste image into a browser-window-like 1280x800 canvas (center fit)."""
    cw, ch = 1280, 800
    avail = ch - CHROME_H
    img = pil.convert("RGB")
    scale = min(cw / img.width, avail / img.height, 1.0)
    img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGB", (cw, ch), (240, 240, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, cw, CHROME_H], fill=(52, 101, 164))
    for i, col in enumerate(((237, 106, 94), (245, 191, 79), (97, 197, 84))):
        draw.ellipse([12 + i * 26, CHROME_H // 2 - 7, 26 + i * 26, CHROME_H // 2 + 7], fill=col)
    draw.rectangle([90, 16, cw - 90, CHROME_H - 16], fill=(255, 255, 255))
    canvas.paste(img, ((cw - img.width) // 2, CHROME_H + (avail - img.height) // 2))
    return canvas


def view_screenshot(pil):
    return _canvas(pil), "png"


def view_screenshot_jpg(pil):
    return _recode(_canvas(pil), "JPEG", quality=60), "jpg"


VIEWS = [("pristine", view_pristine), ("jpeg_q50", view_jpeg_q50),
         ("whatsapp", view_whatsapp), ("screenshot", view_screenshot),
         ("screenshot_jpg", view_screenshot_jpg)]


# ---------------- court harness (wired stacker, modules built once) ----------------

def load_court():
    import joblib
    from argus_detect import STACKER, build_modules
    from triage import DegradationEstimator
    meta = STACKER.parent / "stacker_meta.json"
    meta_d = json.loads(meta.read_text()) if meta.exists() else {}
    stacker_clfs = {STACKER.parent.name: {
        "clf": joblib.load(STACKER),
        "module_ids": meta_d.get("module_ids"),
        "feature_names": meta_d.get("feature_names"),
        "conflict_p99": meta_d.get("conflict_p99")}}
    modules, ids = build_modules()
    return modules, DegradationEstimator(), stacker_clfs


def score(modules, estimator, stacker_clfs, pil, fmt, vid):
    from run_benchmark import run_image
    from base import ImageContext
    from stacker_judge import StackerJudge
    from fusion import fuse
    import hashlib
    tmp = Path(tempfile.mkdtemp(prefix="exam_l_"))
    art = Path(tempfile.mkdtemp(prefix="exam_a_"))
    try:
        if fmt == "jpg":
            buf = io.BytesIO(); pil.save(buf, "JPEG", quality=92)
        else:
            buf = io.BytesIO(); pil.save(buf, "PNG")
        raw = buf.getvalue()
        sha = hashlib.sha256(raw).hexdigest()[:16]
        adir = art / vid
        adir.mkdir(parents=True, exist_ok=True)
        src = tmp / f"{vid}.{fmt}"
        src.write_bytes(raw)
        ctx = ImageContext(pil=pil.copy(), raw_bytes=raw, sha256=sha,
                           fmt="jpeg" if fmt == "jpg" else "png",
                           verdict_id=vid, artifact_dir=adir, src_path=src)
        d = estimator.estimate(ctx)
        outputs = [m.run(ctx, d) for m in modules]
        tag = list(stacker_clfs.keys())[0]
        rec = stacker_clfs[tag]
        judge = StackerJudge(rec["clf"], d.evidence_capacity, rec["module_ids"],
                             feature_names=rec.get("feature_names"),
                             conflict_p99=rec.get("conflict_p99"))
        core, *_ = fuse(outputs, d, judge=judge)
        return 1.0 - core.probabilities.get("camera_original"), bool(core.abstained)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(art, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output dir (default artifacts/exam_launder_<stacker>_<YYYYMMDD>)")
    ap.add_argument("--force", action="store_true",
                    help="write into an output dir that already contains results")
    args = ap.parse_args()

    from argus_detect import STACKER
    out = Path(args.out) if args.out else WS / "artifacts" / (
        f"exam_launder_{STACKER.parent.name}_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force to overwrite",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    from sklearn.metrics import roc_auc_score
    modules, estimator, stacker_clfs = load_court()
    tag = list(stacker_clfs.keys())[0]
    print(f"court wired: stacker={tag} n_modules={len(modules)}", flush=True)

    corpora = {name: corpus(name, lim) for name, lim in N_SAMPLE.items()}
    n_runs = sum(len(v) for v in corpora.values()) * len(VIEWS)
    print(f"sampled {sum(len(v) for v in corpora.values())} images x {len(VIEWS)} views = {n_runs} court runs", flush=True)

    rows, done, t0 = [], 0, time.time()
    for name, files in corpora.items():
        label = 0 if "real" in name else 1
        for p in files:
            try:
                pil = Image.open(p); pil.load(); pil = pil.convert("RGB")
            except Exception as e:
                print(f"ERR open {p}: {e}", flush=True); continue
            for vname, vfn in VIEWS:
                try:
                    vpil, fmt = vfn(pil)
                    pf, abst = score(modules, estimator, stacker_clfs, vpil, fmt, f"{vname}_{p.stem[:20]}")
                except Exception as e:
                    print(f"ERR {p.name}/{vname}: {e}", flush=True)
                    pf, abst = None, False
                rows.append({"corpus": name, "label": label, "file": p.name,
                             "view": vname, "p_fake_fused": pf, "abstained": abst})
                done += 1
                if done % 25 == 0:
                    el = time.time() - t0
                    print(f"{done}/{n_runs} ({el:.0f}s, {el/done:.2f}s/run)", flush=True)

    csvp = out / "launder_exam.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"saved {csvp}", flush=True)

    print("\n=== PIVOT (per corpus x view): AUC | FPR@0.5 | TPR@0.5 | abstain% ===", flush=True)
    pivot = {}
    for vname, _ in VIEWS:
        for cname in N_SAMPLE:
            sub = [r for r in rows if r["corpus"] == cname and r["view"] == vname
                   and r["p_fake_fused"] is not None]
            if not sub:
                continue
            y = [r["label"] for r in sub]
            s = [r["p_fake_fused"] for r in sub]
            auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
            reals = [r for r in sub if r["label"] == 0]
            fakes = [r for r in sub if r["label"] == 1]
            fpr = sum(r["p_fake_fused"] >= 0.5 for r in reals) / max(1, len(reals))
            tpr = sum(r["p_fake_fused"] >= 0.5 for r in fakes) / max(1, len(fakes))
            ab = sum(r["abstained"] for r in sub) / len(sub)
            pivot[(cname, vname)] = (auc, fpr, tpr, ab, len(sub))
            print(f"{cname:16s} {vname:14s} AUC={auc:.4f} FPR={fpr:.3f} TPR={tpr:.3f} abst={ab:.2f} n={len(sub)}", flush=True)

    print("\n=== GATES ===", flush=True)
    gate_fail = []
    pooled = {v: ([], []) for v, _ in VIEWS}
    for (cname, vname), cell in pivot.items():
        if cname in ("eval_real", "train_fake", "cdf_flat_fake"):
            sub = [r for r in rows if r["corpus"] == cname and r["view"] == vname
                   and r["p_fake_fused"] is not None]
            pooled[vname][0].extend(r["label"] for r in sub)
            pooled[vname][1].extend(r["p_fake_fused"] for r in sub)
    pauc = {v: roc_auc_score(y, s) for v, (y, s) in pooled.items() if len(set(y)) > 1}
    base_auc = pauc.get("pristine")
    for vname, auc in pauc.items():
        if vname == "pristine":
            continue
        ok = auc >= base_auc - 0.05
        print(f"G1 {vname}: pooled AUC {auc:.4f} vs pristine {base_auc:.4f} (>= -0.05): {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            gate_fail.append(f"G1:{vname}")
    for vname, _ in VIEWS:
        if vname == "pristine":
            continue
        cell = pivot.get(("eval_real", vname))
        if cell:
            ok = cell[1] <= 0.15
            print(f"G2 {vname}: real FPR {cell[1]:.3f} (<=0.15): {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                gate_fail.append(f"G2:{vname}")

    verdict = "ALL GATES PASS" if not gate_fail else f"GATES FAIL: {', '.join(gate_fail)}"
    print(verdict, flush=True)

    report = ["# Full-court laundering/screenshot robustness exam", "",
              f"Court: {tag} · {sum(len(v) for v in corpora.values())} images x {len(VIEWS)} views", "",
              "## Pivot", "",
              "| corpus | view | AUC | FPR | TPR | abstain | n |", "|---|---|---|---|---|---|---|"]
    for (cname, vname), (auc, fpr, tpr, ab, n) in pivot.items():
        report.append(f"| {cname} | {vname} | {auc:.4f} | {fpr:.3f} | {tpr:.3f} | {ab:.2f} | {n} |")
    report += ["", f"## Verdict: {verdict}", ""]
    (out / "EXAM_REPORT.md").write_text("\n".join(report))
    return 0 if not gate_fail else 1


if __name__ == "__main__":
    sys.exit(main())