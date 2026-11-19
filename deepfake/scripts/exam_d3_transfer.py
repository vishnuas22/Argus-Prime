"""D3 real-collection transfer pre-flight — executes SEALED SPEC
artifacts/d3_transfer_preflight/SPEC.md (2026-11-18).

Measurement only: no wiring, no court files, no threshold/pointer change.
Bars are fixed in the spec and must not be edited after scores exist.

  .venv/bin/python deepfake/scripts/exam_d3_transfer.py --prepare
  .venv/bin/python deepfake/scripts/exam_d3_transfer.py --score
  .venv/bin/python deepfake/scripts/exam_d3_transfer.py --verdict
  .venv/bin/python deepfake/scripts/exam_d3_transfer.py            # all three
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

from exam_d3_t2v import score_all, list_clips  # noqa: E402

OUT = WS / "artifacts" / "d3_transfer_preflight"
D3_SCORES = WS / "artifacts" / "d3_t2v_matched" / "scores.jsonl"
SEED = 20261118
ALPHAS = (0.05, 0.06, 0.07, 0.10)
PRIMARY_ALPHA = 0.06
T1_PASS, T1_HOLD = 0.10, 0.15
T2_PASS, T2_HOLD = 0.85, 0.95
T3_PASS, T3_HOLD = 15, 12
N_FIXTURE_ISSUED = 27

# arm -> (source dir, n, generator label)
ARMS = {
    "calib_msrvtt": (WS / "data/raw/genvideo_val/real_msrvtt", 200, "msrvtt"),
    "ffpp_original": (WS / "data/raw/ffpp_mirror/FaceForensics++_C23/real", 150, "ffpp"),
    "avlips_real": (WS / "data/raw/avlips/videos/0_real", 150, "avlips"),
    "celebdf_youtube": (WS / "data/raw/celebdf/Celeb-DF-v2/YouTube-real", 150, "celebdf_yt"),
    "celebdf_celeb": (WS / "data/raw/celebdf/Celeb-DF-v2/Celeb-real", 150, "celebdf_celeb"),
}


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _used_msrvtt() -> set:
    if not D3_SCORES.exists():
        return set()
    used = set()
    for line in D3_SCORES.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("generator") == "msrvtt":
            used.add(r["filename"])
    return used


def prepare() -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    used = _used_msrvtt()
    rows = []
    for arm, (src, n, gen) in ARMS.items():
        clips = list_clips(src)
        if arm == "calib_msrvtt":
            clips = [c for c in clips if c.name not in used]
        if len(clips) < n:
            raise SystemExit(f"{arm}: need {n}, found {len(clips)}")
        picked = sorted(random.Random(SEED).sample(clips, n))
        rows.extend({"arm": arm, "generator": gen, "label": 0,
                     "path": p, "filename": p.name} for p in picked)
    with open(OUT / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "generator", "label", "path", "filename"])
        w.writeheader()
        w.writerows(rows)
    manifest_sha = _sha256_bytes((OUT / "manifest.csv").read_bytes())
    spec_sha = _sha256_file(OUT / "SPEC.md")
    (OUT / "input_hashes.txt").write_text(
        f"spec_sha256 {spec_sha}\n"
        f"manifest_sha256 {manifest_sha}\n"
        f"n_rows {len(rows)}\n"
        f"seed {SEED}\n"
        f"selection excluded_msrvtt {len(used)}\n"
    )
    (OUT / "seal.txt").write_text(
        f"sealed_at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
        f"spec_sha256 {spec_sha}\n"
        f"manifest_sha256 {manifest_sha}\n"
        f"recipe_hash 1ba8154ca03c4732\n"
        f"primary_alpha {PRIMARY_ALPHA}\n"
    )
    print(f"prepared {len(rows)} rows; spec {spec_sha[:12]} manifest {manifest_sha[:12]}")
    return rows


def load_rows() -> list:
    with open(OUT / "manifest.csv") as f:
        return [{"arm": r["arm"], "generator": r["generator"],
                 "label": int(r["label"]), "path": Path(r["path"]),
                 "filename": r["filename"]} for r in csv.DictReader(f)]


def _cp_interval(k: int, n: int, level: float = 0.95) -> tuple:
    from scipy.stats import beta
    a = 1 - level
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return round(lo, 4), round(hi, 4)


def verdict() -> dict:
    rows = [json.loads(l) for l in (OUT / "scores.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("std") is not None]
    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["std"])
    calib = np.array(by_arm["calib_msrvtt"])

    thrs = {a: float(np.quantile(calib, a, method="lower")) for a in ALPHAS}
    thr = thrs[PRIMARY_ALPHA]

    per_arm = {}
    for arm, (src, n, gen) in ARMS.items():
        if arm == "calib_msrvtt":
            continue
        s = np.array(by_arm[arm])
        fprs = {f"a{a}": round(float((s < thrs[a]).mean()), 4) for a in ALPHAS}
        k = int((s < thr).sum())
        lo, hi = _cp_interval(k, len(s))
        per_arm[arm] = {
            "n": int(len(s)),
            "fpr": fprs,
            f"fpr_primary": round(k / len(s), 4),
            "fpr_primary_ci95": [lo, hi],
            "median_std": round(float(np.median(s)), 4),
        }

    # T2 — can the score alone tell calib reals from this collection's reals?
    from sklearn.metrics import roc_auc_score
    for arm in per_arm:
        s = np.array(by_arm[arm])
        y = [0] * len(calib) + [1] * len(s)
        per_arm[arm]["auc_vs_calib"] = round(
            float(roc_auc_score(y, list(-calib) + list(-s))), 4)

    # T3 — fixture catch, reusing already-scored audit fixtures
    fixture = [json.loads(l) for l in D3_SCORES.read_text().splitlines() if l.strip()]
    fixture = [r for r in fixture if r.get("arm") == "fixture" and r.get("std") is not None]
    issued = [r for r in fixture if r.get("issued", True)]
    catch = int(sum(r["std"] < thr for r in issued))
    catch_all = int(sum(r["std"] < thr for r in fixture))
    n_issued = len(issued)

    t1_max = max(v["fpr_primary"] for v in per_arm.values())
    t2_max = max(v["auc_vs_calib"] for v in per_arm.values())

    t1 = "PASS" if t1_max <= T1_PASS else ("HOLD" if t1_max <= T1_HOLD else "FAIL")
    t2 = "PASS" if t2_max <= T2_PASS else ("HOLD" if t2_max <= T2_HOLD else "FAIL")
    t3 = "PASS" if catch >= T3_PASS else ("HOLD" if catch >= T3_HOLD else "FAIL")
    bars = {"T1": t1, "T2": t2, "T3": t3}
    verdict_s = "FAIL" if "FAIL" in bars.values() else (
        "HOLD" if "HOLD" in bars.values() else "PASS")

    m = {
        "exam": "D3 real-collection transfer pre-flight",
        "date": time.strftime("%Y-%m-%d"),
        "recipe_hash": "1ba8154ca03c4732",
        "primary_alpha": PRIMARY_ALPHA,
        "thresholds": {f"a{a}": round(v, 6) for a, v in thrs.items()},
        "per_arm": per_arm,
        "fixture": {"catch": catch, "n_issued": n_issued,
                    "catch_all_32": catch_all, "n": len(fixture),
                    "ci95": _cp_interval(catch, n_issued)},
        "bars": {
            "T1": {"value": round(t1_max, 4), "verdict": t1,
                   "rule": f"<= {T1_PASS} PASS, <= {T1_HOLD} HOLD, else FAIL"},
            "T2": {"value": round(t2_max, 4), "verdict": t2,
                   "rule": f"<= {T2_PASS} PASS, <= {T2_HOLD} HOLD, else FAIL"},
            "T3": {"value": catch, "verdict": t3,
                   "rule": f">= {T3_PASS} PASS, >= {T3_HOLD} HOLD, else FAIL"},
        },
        "verdict": verdict_s,
    }
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    (OUT / "verdict.json").write_text(json.dumps(
        {"verdict": verdict_s, **{k: v["verdict"] for k, v in m["bars"].items()},
         "detail": m["bars"]}, indent=2) + "\n")
    print(json.dumps({"verdict": verdict_s, "T1": t1_max, "T2": t2_max, "T3": catch}, indent=2))
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args()
    all_steps = not (a.prepare or a.score or a.verdict)
    if a.prepare or all_steps:
        prepare()
    if a.score or all_steps:
        rows = load_rows()
        score_all(rows, OUT / "scores.jsonl", "xclip-b16")
    if a.verdict or all_steps:
        verdict()
    return 0


if __name__ == "__main__":
    sys.exit(main())
