"""D3 T2V exam — PRE-REGISTERED D-046 evaluation of the second-order temporal channel
(audit 2026-11-06 fix; SOTA_SCORECARD solution #1; architect spec 2026-11-07).

D-046 BAR (fixed BEFORE measurement — copy verbatim, not negotiable at verdict time):
  Eval set: fakes = Sora all 51 + Kling all 263 + OpenSora seeded 120 (seed 42)
    from data/raw/genvidbench/{sora,kling,opentora}/
    reals = FF++ test-split reals + Celeb-DF test reals seeded 160 (seed 42).
  PASS requires ALL of:
    (1) channel standalone AUC >= 0.85 on that eval (score = -std; low std = fake)
    (2) audit fixture catch >= 15/27 ISSUED at the calibrated threshold
    (3) real-side FPR <= 10% at that threshold.
  On PASS: wire as an ADVISORY INSTRUMENT (never a stacker feature — GenVidBench
    is exam-only per user scope, so the stacker could never learn this signal).
  On MISS: escalate with exact numbers; do NOT wire anything.

  .venv/bin/python deepfake/scripts/exam_d3_t2v.py --smoke        # pipeline check
  .venv/bin/python deepfake/scripts/exam_d3_t2v.py --calibrate    # tau = p10 of real sigma
  .venv/bin/python deepfake/scripts/exam_d3_t2v.py                # full eval + verdict
  .venv/bin/python deepfake/scripts/exam_d3_t2v.py --fixtures     # fixture arm only
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
for p in ("deepfake", "deepfake/bench", "deepfake/scripts"):
    if str(WS / p) not in sys.path:
        sys.path.insert(0, str(WS / p))

GENVID = WS / "data" / "raw" / "genvidbench"
MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
SPLITS = WS / "data" / "raw" / "ffpp"
CDF = WS / "data" / "raw" / "celebdf" / "Celeb-DF-v2"
MANIFEST = WS / "artifacts" / "audit_20261106" / "endpoint_manifest.csv"
RESULTS = WS / "artifacts" / "audit_20261106" / "endpoint_results.csv"
CALIB = WS / "artifacts" / "d3_t2v" / "calibration.json"
SEED = 42


def list_clips(d: Path) -> list:
    """Case-insensitive + junk-excluded: kling ships `.MP4`; macOS leaves `._*`."""
    return sorted(q for q in d.iterdir() if q.is_file()
                  and q.suffix.lower() == ".mp4" and not q.name.startswith("._"))


def ffpp_ids(split: str) -> list:
    ids = {i for p in json.load(open(SPLITS / f"{split}.json")) for i in p}
    return sorted(ids)


def ffpp_real_jobs(split: str, n: int) -> list:
    vids = [MIRROR / "real" / f"{i}.mp4" for i in ffpp_ids(split)]
    vids = [v for v in vids if v.exists()]
    if len(vids) > n:
        vids = random.Random(SEED).sample(vids, n)
    return [{"arm": "calib" if split == "train" else "eval_real",
             "generator": f"ffpp_{split}", "label": 0, "path": v,
             "filename": v.name} for v in vids]


def cdf_real_jobs(n: int) -> list:
    vids = []
    for line in (CDF / "List_of_testing_videos.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2 and int(parts[0]) == 1 and (CDF / parts[1]).exists():
            vids.append(CDF / parts[1])
    vids = sorted(set(vids))
    if len(vids) > n:
        vids = random.Random(SEED).sample(vids, n)
    return [{"arm": "eval_real", "generator": "celebdf_test", "label": 0,
             "path": v, "filename": v.name} for v in vids]


def fake_jobs(sora_n: int, kling_n: int, opentora_n: int) -> list:
    rows = []
    for gen, n in (("sora", sora_n), ("kling", kling_n), ("opentora", opentora_n)):
        clips = list_clips(GENVID / gen)
        if n and len(clips) > n:
            clips = random.Random(SEED).sample(clips, n)
        rows += [{"arm": "eval_fake", "generator": gen, "label": 1,
                  "path": v, "filename": v.name} for v in clips]
    return rows


def fixture_jobs() -> list:
    """Audit battery's fake video fixtures; `issued` = the court did NOT abstain."""
    man = [r for r in csv.DictReader(open(MANIFEST))
           if r["kind"] == "video" and int(r["label"]) == 1]
    abstained = set()
    for r in csv.DictReader(open(RESULTS)):
        if (r["kind"] == "video" and int(r["label"]) == 1
                and r["abstained"].strip().lower() in ("true", "1")):
            abstained.add(r["path"])
    rows = []
    for r in man:
        p = Path(r["path"])
        if not p.exists():
            print(f"WARN: fixture missing on disk: {p}", file=sys.stderr)
            continue
        rows.append({"arm": "fixture", "generator": r["family"], "label": 1,
                     "path": p, "filename": p.name,
                     "issued": r["path"] not in abstained})
    return rows


def score_all(rows: list, scores_path: Path, encoder: str) -> list:
    """Resume by (arm, generator, filename) — dir-qualified keys only."""
    from d3_temporal import d3_std
    done: dict = {}
    if scores_path.exists():
        for line in scores_path.read_text().splitlines():
            if line.strip():
                _r = json.loads(line)
                done[(_r["arm"], _r["generator"], _r["filename"])] = _r
    out = []
    t0 = time.time()
    with open(scores_path, "a") as f:
        for i, row in enumerate(rows):
            key = (row["arm"], row["generator"], row["filename"])
            if key in done:
                out.append(done[key])
                continue
            res = d3_std(row["path"], encoder)
            rec = {**row, **res,
                   "path": str(row["path"])}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            out.append(rec)
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                el = time.time() - t0
                print(f"[{i+1}/{len(rows)}] {row['generator']}/{row['filename']} "
                      f"std={res['std']} ({el / max(1, i + 1):.2f}s/clip)",
                      flush=True)
    return out


def recipe_hash(encoder: str) -> str:
    import d3_temporal as dt
    payload = json.dumps({
        "encoder": encoder,
        "fps_target": dt.FPS_TARGET, "window_s": dt.WINDOW_S,
        "max_frames": dt.MAX_FRAMES, "min_frames": dt.MIN_FRAMES,
        "jpeg_q": dt.JPEG_Q, "crop_frac": dt.CROP_FRAC, "resize": dt.RESIZE,
        "mean": [float(x) for x in dt.IMAGENET_MEAN],
        "std": [float(x) for x in dt.IMAGENET_STD],
        "polarity": "low_std_fake",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def metrics_from(rows: list, tau: float, encoder: str) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    scored = [r for r in rows if r.get("std") is not None]
    unavail = [r for r in rows if r.get("std") is None]
    ef = [r for r in scored if r["arm"] == "eval_fake"]
    er = [r for r in scored if r["arm"] == "eval_real"]
    fx = [r for r in scored if r["arm"] == "fixture"]
    m = {"encoder": encoder, "tau": tau, "recipe_hash": recipe_hash(encoder),
         "n_eval_fake": len(ef), "n_eval_real": len(er), "n_fixture": len(fx),
         "n_unavailable": len(unavail),
         "unavailable_by_generator": {}}
    for r in unavail:
        m["unavailable_by_generator"][r["generator"]] = \
            m["unavailable_by_generator"].get(r["generator"], 0) + 1
    if ef and er:
        y = [1] * len(ef) + [0] * len(er)
        s = [-r["std"] for r in ef] + [-r["std"] for r in er]
        m["auc_eval"] = round(float(roc_auc_score(y, s)), 4)
        m["per_generator_ap"] = {}
        for g in sorted({r["generator"] for r in ef}):
            gf = [r for r in ef if r["generator"] == g]
            m["per_generator_ap"][g] = round(float(average_precision_score(
                [1] * len(gf) + [0] * len(er),
                [-r["std"] for r in gf] + [-r["std"] for r in er])), 4)
        m["real_std_p10"] = round(float(np.percentile([r["std"] for r in er], 10)), 6)
        m["real_std_p50"] = round(float(np.percentile([r["std"] for r in er], 50)), 6)
        m["fpr_real_eval"] = round(float(np.mean([r["std"] < tau for r in er])), 4)
        m["fpr_real_ffpp"] = round(float(np.mean([r["std"] < tau for r in er
                                                  if r["generator"] == "ffpp_test"])), 4)
        m["fpr_real_celebdf"] = round(float(np.mean([r["std"] < tau for r in er
                                                     if r["generator"] == "celebdf_test"])), 4)
    if fx:
        issued = [r for r in fx if r.get("issued")]
        m["n_fixture_issued"] = len(issued)
        m["catch_issued"] = int(sum(r["std"] < tau for r in issued))
        m["catch_all_fixtures"] = int(sum(r["std"] < tau for r in fx))
        m["catch_by_family"] = {}
        for fam in sorted({r["generator"] for r in fx}):
            sub = [r for r in issued if r["generator"] == fam]
            m["catch_by_family"][fam] = {
                "n_issued": len(sub),
                "catch": int(sum(r["std"] < tau for r in sub))}
    return m


def guard(out: Path, force: bool) -> int:
    if out.exists() and any(out.iterdir()) and not force:
        print(f"ERROR: output dir {out} already contains results — pass --force",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--encoder", default="xclip-b16",
                    choices=["xclip-b16", "clip-b16"])
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    enc = args.encoder

    if args.smoke:
        out = Path(args.out) if args.out else WS / "artifacts/d3_t2v/smoke"
        if guard(out, args.force):
            return 1
        rows = (fake_jobs(2, 2, 2) + ffpp_real_jobs("test", 3) + cdf_real_jobs(3))
        scored = score_all(rows, out / "scores.jsonl", enc)
        for r in scored:
            print(f"  {r['generator']}/{r['filename']} std={r['std']} "
                  f"n={r['n_frames']} ({r['unavailable_reason']})")
        print(f"SMOKE OK — {len(scored)} rows -> {out / 'scores.jsonl'}")
        return 0

    if args.calibrate:
        out = Path(args.out) if args.out else WS / "artifacts/d3_t2v/calibration"
        if guard(out, args.force):
            return 1
        rows = ffpp_real_jobs("train", args.n_calib)
        print(f"CALIBRATION: {len(rows)} FF++ TRAIN-split reals (seed {SEED})")
        scored = score_all(rows, out / "scores.jsonl", enc)
        stds = [r["std"] for r in scored if r["std"] is not None]
        if len(stds) < args.n_calib * 0.9:
            print(f"ERROR: only {len(stds)}/{len(rows)} calibrated — fail closed",
                  file=sys.stderr)
            return 1
        tau = round(float(np.percentile(stds, 10)), 6)
        cal = {"tau": tau, "encoder": enc, "n": len(stds),
               "real_std_p10": tau,
               "real_std_p50": round(float(np.percentile(stds, 50)), 6),
               "real_std_p90": round(float(np.percentile(stds, 90)), 6),
               "fit_on": "ffpp train-split real videos only",
               "recipe_hash": recipe_hash(enc),
               "date": time.strftime("%Y-%m-%d")}
        CALIB.parent.mkdir(parents=True, exist_ok=True)
        CALIB.write_text(json.dumps(cal, indent=2))
        print(f"CALIBRATED: tau={tau} (p10 of {len(stds)} real sigmas) -> {CALIB}")
        return 0

    if not CALIB.exists():
        print(f"ERROR: {CALIB} missing — run --calibrate first", file=sys.stderr)
        return 1
    tau = json.loads(CALIB.read_text())["tau"]

    if args.fixtures:
        out = Path(args.out) if args.out else WS / "artifacts/d3_t2v/fixtures"
        if guard(out, args.force):
            return 1
        rows = fixture_jobs()
        scored = score_all(rows, out / "scores.jsonl", enc)
        m = metrics_from(scored, tau, enc)
        (out / "metrics.json").write_text(json.dumps(m, indent=2))
        print(json.dumps({k: m[k] for k in m if k.startswith("catch") or
                          k.startswith("n_fixture")}, indent=2))
        return 0

    # ---- full eval + D-046 verdict ----
    out = Path(args.out) if args.out else WS / "artifacts/d3_t2v/eval"
    if guard(out, args.force):
        return 1
    rows = (fake_jobs(0, 0, 120) + ffpp_real_jobs("test", 140) + cdf_real_jobs(160)
            + fixture_jobs())
    print(f"FULL EVAL: {len(rows)} clips (fakes=sora 51 + kling 263 + opentora 120; "
          f"reals=ffpp_test + celebdf 160; + audit fixtures)", flush=True)
    scored = score_all(rows, out / "scores.jsonl", enc)
    m = metrics_from(scored, tau, enc)
    m["d046"] = {
        "auc_ok": m.get("auc_eval", 0) >= 0.85,
        "catch_ok": m.get("catch_issued", 0) >= 15,
        "fpr_ok": m.get("fpr_real_eval", 1.0) <= 0.10,
        "bar": "AUC>=0.85 AND catch>=15/27 AND real FPR<=10%",
    }
    m["d046"]["pass"] = all(m["d046"][k] for k in ("auc_ok", "catch_ok", "fpr_ok"))
    m["wall_min"] = None
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    print("\n==== D-046 VERDICT ====")
    print(f"  AUC (eval, {m.get('n_eval_fake', 0)}f/{m.get('n_eval_real', 0)}r): "
          f"{m.get('auc_eval')}  [bar >= 0.85]")
    print(f"  per-generator AP: {json.dumps(m.get('per_generator_ap', {}))}")
    print(f"  FPR@tau reals: {m.get('fpr_real_eval')} "
          f"(ffpp {m.get('fpr_real_ffpp')}, cdf {m.get('fpr_real_celebdf')})  [bar <= 0.10]")
    print(f"  fixture catch: {m.get('catch_issued')}/{m.get('n_fixture_issued')} issued "
          f"({m.get('catch_all_fixtures')}/{m.get('n_fixture')} all)  [bar >= 15]")
    print(f"  by family: {json.dumps(m.get('catch_by_family', {}))}")
    print(f"  unavailable: {json.dumps(m.get('unavailable_by_generator', {}))}")
    print(f"  VERDICT: {'PASS — wire as advisory instrument (STEP 7)' if m['d046']['pass'] else 'MISS — escalate with exact numbers; do NOT wire'}")
    return 0 if m["d046"]["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
