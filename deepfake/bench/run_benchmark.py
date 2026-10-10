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


def run_image(modules, estimator, pil: Image.Image, spec, artifact_root: Path, vid: str, tmp_root: Path, stacker_clfs=None, original_raw: bytes | None = None):
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
                       verdict_id=vid, artifact_dir=adir, src_path=src_path,
                       original_raw=original_raw)
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
    from fusion import gate, build_contradictions
    survivors, _ = gate(outputs, d)
    contras = build_contradictions(survivors)
    row["n_survivors"] = len(survivors)
    row["contradiction_count"] = len(contras)
    row["contradiction_mass"] = round(sum(c.conflict_contribution for c in contras), 4)
    from stacker_judge import StackerJudge
    tags = list((stacker_clfs or {}).keys())
    if not tags:
        core, ranking, contradictions, unavailable, k = fuse(outputs, d)
        row["p_real"] = core.probabilities.get("camera_original")
        row["abstained"] = bool(core.abstained)
        row["trust"] = core.trust_score
        row["risk"] = core.risk_score
        row["conflict"] = k
    for tag in tags:
        rec = stacker_clfs[tag]
        judge = StackerJudge(rec["clf"], d.evidence_capacity, module_ids=rec["module_ids"],
                             feature_names=rec.get("feature_names"),
                             conflict_p99=rec.get("conflict_p99"))
        core, ranking, contradictions, unavailable, k = fuse(outputs, d, judge=judge)
        row[f"p_real_{tag}"] = core.probabilities.get("camera_original")
        row[f"abstained_{tag}"] = bool(core.abstained)
        row[f"trust_{tag}"] = core.trust_score
        row[f"risk_{tag}"] = core.risk_score
        row[f"conflict_{tag}"] = k
        row[f"out_of_conflict_{tag}"] = bool(getattr(judge, "out_of_conflict", False))
        if len(tags) == 1:  # legacy single-stacker column names
            row["p_real"] = row[f"p_real_{tag}"]
            row["abstained"] = row[f"abstained_{tag}"]
            row["trust"] = row[f"trust_{tag}"]
            row["risk"] = row[f"risk_{tag}"]
            row["conflict"] = k
            row["out_of_conflict"] = row[f"out_of_conflict_{tag}"]
    shutil.rmtree(adir, ignore_errors=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(WS / "data/corpora/eval_real"))
    ap.add_argument("--fake", default=str(WS / "data/corpora/eval_fake"))
    ap.add_argument("--out", default=str(WS / "artifacts/p0_baseline"))
    ap.add_argument("--limit", type=int, default=150, help="per class per variant")
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--alpha", default=None, type=float, metavar="A",
                    help="abstention policy: conformal alpha (default = vendored 0.10; "
                         "higher A commits more, e.g. 0.3 commits at top-prob>=0.70)")
    ap.add_argument("--channels", default=None, metavar="SPEC",
                    help="optional learned channels: 'core' | 'all' | comma ids; "
                         "overrides legacy --with-learned/--with-face")
    ap.add_argument("--with-learned", default=None, metavar="PROBE_JOBLIB",
                    help="append learned_aigc_probe module using trained probe")
    ap.add_argument("--with-stacker", default=None, action="append", metavar="STACKER_JOBLIB",
                    help="replace voting fusion with trained stacker; repeatable "
                         "for A/B gating (tag = parent dir name)")
    ap.add_argument("--with-face", default=None, metavar="FACE_PROBE_JOBLIB",
                    help="append learned_face_probe module using trained face probe")
    args = ap.parse_args()

    if args.alpha is not None:
        # harness-level abstention policy: rebind the conformal stub's alpha
        # without touching vendored code (D-009). fuse() resolves conformal_set
        # in the fusion module namespace at call time.
        import functools
        import fusion
        fusion.conformal_set = functools.partial(fusion.conformal_set, alpha=args.alpha)

    modules = get_modules(force_reload=True)
    from channels import CORE_IDS, build as build_channels
    learned_modules, learned_ids = [], []
    if args.channels:
        learned_modules, learned_ids = build_channels(args.channels)
    else:
        if args.with_learned:
            from learned_aigc_probe import LearnedAIGCProbe
            learned_modules.append(LearnedAIGCProbe(args.with_learned))
            learned_ids.append("learned_aigc_probe")
        if getattr(args, "with_face", None):
            from learned_face_probe import LearnedFaceProbe
            learned_modules.append(LearnedFaceProbe(args.with_face))
            learned_ids.append("learned_face_probe")
    modules += learned_modules
    module_ids = CORE_IDS + learned_ids

    import joblib
    stacker_clfs = {}
    for sp in (args.with_stacker or []):
        spath = Path(sp)
        tag = spath.parent.name or spath.stem
        meta_p = spath.with_name("stacker_meta.json")
        meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        stacker_clfs[tag] = {"clf": joblib.load(spath),
                             "module_ids": meta.get("module_ids"),
                             "feature_names": meta.get("feature_names"),
                             "conflict_p99": meta.get("conflict_p99")}

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
                file_raw = p.read_bytes()
            except Exception as e:
                print(f"skip unreadable {p.name}: {e}", flush=True)
                continue
            for vname, spec in variants.items():
                vid = f"{p.stem}_{vname}"
                try:
                    row = run_image(modules, estimator, pil, spec, artifact_root, vid, tmp_root, stacker_clfs, original_raw=file_raw)
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

    results = compute_results(rows, list(variants.keys()), module_ids,
                              list(stacker_clfs.keys()))
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "REPORT.md").write_text(render_report(results, len(real_paths), len(fake_paths), module_ids))
    fusion_out = {}
    for v, block in results.items():
        fusion_out[v] = {k: block[k] for k in block if k.startswith("fusion")}
    print(json.dumps(fusion_out, indent=2))
    print(f"\nreport: {out/'REPORT.md'}")


def compute_results(rows, variant_names, module_ids, stacker_tags=None):
    module_cols = module_ids
    tags = stacker_tags or []
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

        def fusion_block(key_preal, key_abst, key_trust, key_risk):
            preal = np.array([r[key_preal] if r.get(key_preal) is not None else np.nan
                              for r in sub])
            m = ~np.isnan(preal)
            return {
                "auc": _r4(auc(-preal[m], labels[m])),
                "eer": _r4(eer(-preal[m], labels[m])),
                "n": int(m.sum()),
                "brier_p_real": _r4(brier(preal[m], labels[m])),
                "abstention_rate": _r4(abstention_rate([r[key_abst] for r in sub])),
                "mean_trust": _r4(np.mean([r[key_trust] for r in sub])),
                "mean_risk": _r4(np.mean([r[key_risk] for r in sub])),
            }

        if tags:
            for tag in tags:
                blk = fusion_block(f"p_real_{tag}", f"abstained_{tag}",
                                   f"trust_{tag}", f"risk_{tag}")
                block[f"fusion_{tag}"] = blk
                if tag == tags[0]:  # primary tag keeps the legacy key
                    block["fusion"] = blk
        else:
            block["fusion"] = fusion_block("p_real", "abstained", "trust", "risk")
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
