"""P0 baseline benchmark for the vendored Argus_Master pipeline.

Runs triage -> all registered evidence modules -> fusion over a labeled corpus,
optionally through a laundering ladder. Emits scores CSV, metrics JSON, and a
markdown report card.

Usage:
  .venv/bin/python deepfake/bench/run_benchmark.py \
      --real data/corpora/eval_real --fake data/corpora/eval_fake \
      --out artifacts/p0_baseline [--limit 200] [--no-ladder]
"""
import argparse
import csv
import hashlib
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _r4(x):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, 4)

WS = Path(__file__).resolve().parents[2]
BACKEND = WS / "integrations" / "Argus_Master" / "backend"
sys.path.insert(0, str(BACKEND))

from registry import get_modules  # noqa: E402
from base import ImageContext  # noqa: E402
from triage import DegradationEstimator  # noqa: E402
from fusion import fuse  # noqa: E402
from schemas import HYPOTHESES  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from metrics import auc, eer, brier, abstention_rate, mean_ms  # noqa: E402
from report import render_report  # noqa: E402

MODULE_IDS = ["metadata", "compression_history", "spectral_probe",
              "real_distribution_probe", "perturbation_probe"]

LADDER = {
    "clean": None,
    "jpeg75": {"q": 75},
    "jpeg50": {"q": 50},
    "resize50": {"scale": 0.5},
    "q50_r05": {"q": 50, "scale": 0.5},
}


def apply_variant(pil: Image.Image, spec) -> Image.Image:
    if spec is None:
        return pil
    img = pil.convert("RGB")
    if "scale" in spec:
        s = spec["scale"]
        img = img.resize((max(64, int(img.width * s)), max(64, int(img.height * s))), Image.BILINEAR)
    return img


def encode(pil: Image.Image, spec) -> bytes:
    buf = io.BytesIO()
    if spec is None or "q" not in spec:
        pil.save(buf, "PNG")
    else:
        pil.save(buf, "JPEG", quality=spec["q"])
    return buf.getvalue()


def collect(paths: list, limit: int):
    exts = {".jpg", ".jpeg", ".png"}
    out = []
    for d in paths:
        out += sorted([p for p in Path(d).iterdir() if p.suffix.lower() in exts])
    return out[:limit]


def run_image(modules, estimator, pil: Image.Image, spec, artifact_root: Path, vid: str, tmp_root: Path, stacker_clf=None):
    img = apply_variant(pil, spec)
    raw = encode(img, spec)
    sha = hashlib.sha256(raw).hexdigest()[:16]
    adir = artifact_root / vid
    adir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if raw[:3] == b"\xff\xd8\xff" else "png"
    src_path = tmp_root / f"{vid}.{ext}"
    src_path.write_bytes(raw)
    ctx = ImageContext(pil=img.copy(), raw_bytes=raw, sha256=sha,
                       fmt="jpeg" if ext == "jpg" else "png",
                       verdict_id=vid, artifact_dir=adir, src_path=src_path)
    t0 = time.perf_counter()
    d = estimator.estimate(ctx)
    triage_ms = (time.perf_counter() - t0) * 1000
    row = {"sha": sha, "triage_ms": round(triage_ms, 1),
           "capacity": d.evidence_capacity}
    outputs = []
    for m in modules:
        mo = m.run(ctx, d)
        outputs.append(mo)
        row[mo.module_id] = mo.evidence_score if mo.unavailable_reason is None else None
        row[f"{mo.module_id}_ms"] = mo.compute_ms
        row[f"{mo.module_id}_rel"] = mo.reliability_score
    if stacker_clf is not None:
        from stacker_judge import StackerJudge
        core, ranking, contradictions, unavailable, k = fuse(outputs, d, judge=StackerJudge(stacker_clf, d.evidence_capacity))
    else:
        core, ranking, contradictions, unavailable, k = fuse(outputs, d)
    row["p_real"] = core.probabilities.get("camera_original")
    row["abstained"] = bool(core.abstained)
    row["set_size"] = len(core.hypothesis_set)
    row["trust"] = core.trust_score
    row["risk"] = core.risk_score
    row["conflict"] = k
    shutil.rmtree(adir, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(WS / "data/corpora/eval_real"))
    ap.add_argument("--fake", default=str(WS / "data/corpora/eval_fake"))
    ap.add_argument("--out", default=str(WS / "artifacts/p0_baseline"))
    ap.add_argument("--limit", type=int, default=150, help="per class per variant")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--with-learned", default=None, metavar="PROBE_JOBLIB",
                    help="append learned_aigc_probe module using trained probe")
    ap.add_argument("--with-stacker", default=None, metavar="STACKER_JOBLIB",
                    help="replace voting fusion with trained P6 stacker")
    ap.add_argument("--with-face", default=None, metavar="FACE_PROBE_JOBLIB",
                    help="append learned_face_probe module using trained face probe")
    args = ap.parse_args()

    module_ids = list(MODULE_IDS)
    modules = get_modules(force_reload=True)
    stacker_clf = None
    if args.with_stacker:
        import joblib
        stacker_clf = joblib.load(args.with_stacker)
    if args.with_learned:
        from learned_aigc_probe import LearnedAIGCProbe
        modules.append(LearnedAIGCProbe(args.with_learned))
        module_ids.append("learned_aigc_probe")
    if getattr(args, "with_face", None):
        from learned_face_probe import LearnedFaceProbe
        modules.append(LearnedFaceProbe(args.with_face))
        module_ids.append("learned_face_probe")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    artifact_root = Path(tempfile.mkdtemp(prefix="argus_bench_art_"))
    tmp_root = Path(tempfile.mkdtemp(prefix="argus_bench_src_"))

    real_paths = collect([args.real], args.limit)
    fake_paths = collect([args.fake], args.limit)
    assert real_paths and fake_paths, "corpus missing: run prepare_corpus.py + generate_fakes.py"
    variants = {"clean": LADDER["clean"]} if args.no_ladder else LADDER

    estimator = DegradationEstimator()
    print(f"modules: {[m.module_id for m in modules]}", flush=True)

    rows = []
    total = len(real_paths) + len(fake_paths)
    n_done = 0
    t_start = time.time()
    for label, paths in ((0, real_paths), (1, fake_paths)):
        for p in paths:
            try:
                pil = Image.open(p)
                pil.load()
            except Exception as e:
                print(f"skip unreadable {p.name}: {e}", flush=True)
                continue
            for vname, spec in variants.items():
                vid = f"{p.stem}_{vname}"
                try:
                    row = run_image(modules, estimator, pil, spec, artifact_root, vid, tmp_root, stacker_clf)
                except Exception as e:
                    print(f"FAIL {vid}: {e}", flush=True)
                    continue
                row.update({"image": p.name, "label": label, "variant": vname})
                rows.append(row)
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t_start
                print(f"[{n_done}/{total}] {el:.0f}s elapsed, ~{el/n_done:.1f}s/img-set", flush=True)

    shutil.rmtree(artifact_root, ignore_errors=True)
    shutil.rmtree(tmp_root, ignore_errors=True)
    csv_path = out / "scores.csv"
    fields = sorted({k for r in rows for k in r.keys()},
                    key=lambda k: (k not in ("image", "label", "variant"), k))
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    results = compute_results(rows, list(variants.keys()), module_ids)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "REPORT.md").write_text(render_report(results, len(real_paths), len(fake_paths), module_ids))
    print(json.dumps({v: {k: r[k] for k in ("fused_auc", "fused_eer")}
                      for k, r in results[v].items()} if False else
                     {v: results[v]["fusion"] for v in results}, indent=2))
    print(f"\nreport: {out/'REPORT.md'}")


def compute_results(rows, variant_names, module_ids):
    module_cols = module_ids
    results = {}
    for vname in variant_names:
        sub = [r for r in rows if r["variant"] == vname and r["label"] in (0, 1)]
        labels = np.array([r["label"] for r in sub])
        block = {}
        for c in module_cols:
            scores = np.array([-r[c] if r[c] is not None else np.nan for r in sub])
            mask = ~np.isnan(scores)
            s, l = scores[mask], labels[mask]
            block[c] = {
                "auc": _r4(auc(s, l)),
                "eer": _r4(eer(s, l)),
                "n": int(mask.sum()),
            }
        preal = np.array([r["p_real"] if r["p_real"] is not None else np.nan for r in sub])
        m = ~np.isnan(preal)
        block["fusion"] = {
            "auc": _r4(auc(-preal[m], labels[m])),
            "eer": _r4(eer(-preal[m], labels[m])),
            "n": int(m.sum()),
            "brier_p_real": _r4(brier(preal[m], labels[m])),
            "abstention_rate": _r4(abstention_rate([r["abstained"] for r in sub])),
            "mean_trust": _r4(np.mean([r["trust"] for r in sub])),
            "mean_risk": _r4(np.mean([r["risk"] for r in sub])),
        }
        block["runtime_ms"] = {
            "triage": round(mean_ms([r["triage_ms"] for r in sub]), 1),
            **{c: round(mean_ms([r[f"{c}_ms"] for r in sub]), 1) for c in module_cols},
        }
        cap_counts = {}
        for r in sub:
            cap_counts[r["capacity"]] = cap_counts.get(r["capacity"], 0) + 1
        block["evidence_capacity_mix"] = cap_counts
        results[vname] = block
    return results


if __name__ == "__main__":
    main()
