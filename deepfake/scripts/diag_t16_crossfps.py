#!/usr/bin/env python3
"""DIAG_T16_CROSSFPS — report-only T=16 cross-frame-budget diagnostic.

Binding spec: artifacts/split_prereg/DIAG_T16_CROSSFPS.md
Sealed scoring module: deepfake/bench/split_t16_temporal.py (read-only).

Not a Stage 1 result. 0 booked. Never imported by serving code.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "deepfake" / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
import split_t16_temporal as st  # noqa: E402

DEFAULT_OUT = ROOT / "artifacts" / "split_t2v_matched" / "diag_t16_crossfps"
SPEC = ROOT / "artifacts" / "split_prereg" / "DIAG_T16_CROSSFPS.md"
V1_SPLITS = ROOT / "artifacts" / "split_t2v_matched" / "stage1" / "splits.json"
V1_SCORES = ROOT / "artifacts" / "split_t2v_matched" / "stage1" / "scores.jsonl"
STAGE0_SCORES = (ROOT / "artifacts" / "split_t2v_matched" / "stage0_t16"
                 / "scores.jsonl")
ARMS = ("genvid_fake", "genvideo_fake")
N_BOOT = 10000
BOOT_SEED = 1624


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def _scores_path(out: Path) -> Path:
    return out / "scores.jsonl"


def _load_scores(path: Path) -> list:
    if not path.exists():
        print(f"ERROR: {path} missing — run --score first", file=sys.stderr)
        return []
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"FAIL: corrupt {path} line {i}: {e}", file=sys.stderr)
            return []
    keys = [(r["arm"], r["generator"], r["filename"]) for r in rows]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    if dup:
        print(f"FAIL: duplicate score keys: {dup[:5]}", file=sys.stderr)
        return []
    return rows


def _job_keys(rows: list) -> list:
    return [(r["arm"], r["generator"], r["filename"]) for r in rows]


# ---------------------------------------------------------------- prepare

def prepare(out: Path) -> int:
    if _scores_path(out).exists():
        print(f"FAIL: {_scores_path(out)} already exists; refusing to re-prepare",
              file=sys.stderr)
        return 2
    for p in (SPEC, V1_SPLITS, V1_SCORES, STAGE0_SCORES):
        if not p.exists():
            print(f"FAIL: missing input {p}", file=sys.stderr)
            return 2

    v1 = _load_json(V1_SPLITS)
    rows = []
    for entry in v1["genvidbench_fakes"]:
        rows.append({**entry, "arm": "genvid_fake"})
    for entry in v1["genvideo_fakes"]:
        rows.append({**entry, "arm": "genvideo_fake"})

    for r in rows:
        p = Path(r["path"])
        if not p.exists():
            print(f"FAIL: missing clip {p}", file=sys.stderr)
            return 2
        sha = _sha256(p)
        if sha != r["sha256"]:
            print(f"FAIL: clip hash drift {p}: {r['sha256']} -> {sha}",
                  file=sys.stderr)
            return 2
    keys = _job_keys(rows)
    dup = [k for k, c in Counter(keys).items() if c > 1]
    if dup:
        print(f"FAIL: duplicate rows: {dup[:5]}", file=sys.stderr)
        return 2

    stage0 = [json.loads(l) for l in STAGE0_SCORES.read_text().splitlines()
              if l.strip()]
    reals = [r for r in stage0 if r.get("arm") == "stage0_real"]
    if len(reals) != 1000:
        print(f"FAIL: expected 1000 Stage 0 reals, got {len(reals)}",
              file=sys.stderr)
        return 2
    if any(r.get("score") is None for r in reals):
        print("FAIL: Stage 0 real arm has unavailable rows", file=sys.stderr)
        return 2

    counts = Counter(r["arm"] for r in rows)
    doc = {
        "stage": "diag_t16_crossfps",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe_hash": st.split_recipe_hash(),
        "spec": str(SPEC),
        "inputs": {
            "v1_splits": {"path": str(V1_SPLITS), "sha256": _sha256(V1_SPLITS)},
            "v1_scores": {"path": str(V1_SCORES), "sha256": _sha256(V1_SCORES)},
            "stage0_scores": {"path": str(STAGE0_SCORES),
                              "sha256": _sha256(STAGE0_SCORES)},
        },
        "counts": dict(sorted(counts.items())),
        "n_rows": len(rows),
        "reals": {"n": len(reals), "source": "stage0_t16 stage0_real",
                  "scores_sha256": _sha256(STAGE0_SCORES)},
        "rows": rows,
    }
    _write_json(out / "rows.json", doc)
    print(f"prepare: {len(rows)} rows "
          f"({counts.get('genvid_fake')} genvid_fake, "
          f"{counts.get('genvideo_fake')} genvideo_fake) + 1000 sealed reals")
    print(f"recipe_hash {doc['recipe_hash']}")
    return 0


def write_seal(out: Path) -> int:
    rows_path = out / "rows.json"
    if not rows_path.exists():
        print("FAIL: rows.json missing — run --prepare", file=sys.stderr)
        return 2
    if _scores_path(out).exists():
        print("FAIL: scores exist; refusing to seal after scoring",
              file=sys.stderr)
        return 2
    doc = _load_json(rows_path)
    lines = [
        "DIAG_T16_CROSSFPS seal",
        f"created_at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"git_head {_git_head()}",
        f"recipe_hash {st.split_recipe_hash()}",
        f"spec {_sha256(SPEC)} {SPEC}",
        f"module {_sha256(BENCH / 'split_t16_temporal.py')}",
        f"exam {_sha256(Path(__file__).resolve())}",
        f"rows.json {_sha256(rows_path)}",
    ]
    for name, meta in sorted(doc["inputs"].items()):
        lines.append(f"{name} {meta['sha256']} {meta['path']}")
    lines.append(f"n_rows {doc['n_rows']}")
    (out / "seal.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def seal_problems(out: Path) -> list:
    seal = out / "seal.txt"
    if not seal.exists():
        return ["seal.txt missing"]
    fields = {}
    for line in seal.read_text().splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            fields[parts[0]] = parts[1]
    p = []
    if fields.get("recipe_hash") != st.split_recipe_hash():
        p.append("recipe_hash mismatch")
    if fields.get("rows.json") != _sha256(out / "rows.json"):
        p.append("rows.json changed since seal")
    if fields.get("module") != _sha256(BENCH / "split_t16_temporal.py"):
        p.append("module changed since seal")
    if fields.get("exam") != _sha256(Path(__file__).resolve()):
        p.append("exam changed since seal")
    if fields.get("spec") and fields["spec"].split(" ")[0] != _sha256(SPEC):
        p.append("spec changed since seal")
    if fields.get("n_rows") != str(_load_json(out / "rows.json")["n_rows"]):
        p.append("n_rows mismatch")
    return p


def _require_seal(out: Path) -> bool:
    probs = seal_problems(out)
    if probs:
        for p in probs:
            print(f"FAIL: {p}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------- score

def score(out: Path) -> int:
    if not _require_seal(out):
        return 2
    doc = _load_json(out / "rows.json")
    jobs = doc["rows"]
    recipe_hash = st.split_recipe_hash()
    scores_path = _scores_path(out)
    log_path = out / "score.log"

    done = {}
    if scores_path.exists():
        for i, line in enumerate(scores_path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"FAIL: corrupt {scores_path} line {i}: {e}",
                      file=sys.stderr)
                return 2
            if row.get("recipe_hash") != recipe_hash:
                print(f"FAIL: stale cache row {i}", file=sys.stderr)
                return 2
            key = (row["arm"], row["generator"], row["filename"])
            if key in done:
                print(f"FAIL: duplicate key {key}", file=sys.stderr)
                return 2
            done[key] = row

    out_rows = []
    t0 = time.time()
    with open(scores_path, "a") as f, open(log_path, "a") as log:
        for i, job in enumerate(jobs):
            key = (job["arm"], job["generator"], job["filename"])
            path = Path(job["path"])
            if not path.exists():
                print(f"FAIL: missing clip {path}", file=sys.stderr)
                return 2
            sha = _sha256(path)
            if sha != job["sha256"]:
                print(f"FAIL: clip changed since prepare: {path}",
                      file=sys.stderr)
                return 2
            if key in done:
                if done[key].get("sha256") != sha:
                    print(f"FAIL: clip changed since scoring: {path}",
                          file=sys.stderr)
                    return 2
                out_rows.append(done[key])
                continue
            rec = {"arm": job["arm"], "generator": job["generator"],
                   "label": int(job["label"]), "path": str(path),
                   "filename": job["filename"], "sha256": sha,
                   "recipe_hash": recipe_hash, "source": "new"}
            rec.update(st.score_clip(path))
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                line = (f"[{i+1}/{len(jobs)}] {job['arm']}/"
                        f"{job['generator']}/{job['filename']} "
                        f"score={rec.get('score')} "
                        f"reason={rec.get('unavailable_reason')}")
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
            out_rows.append(rec)
    dt = time.time() - t0
    print(f"score: {len(out_rows)} rows in {dt/60:.1f} min")
    return 0


# ---------------------------------------------------------------- metrics

def _auc(fakes: list, reals: list, key: str = "score"):
    from sklearn.metrics import roc_auc_score
    f = [r[key] for r in fakes if r.get(key) is not None]
    r = [r[key] for r in reals if r.get(key) is not None]
    if not f or not r:
        return None
    y = [1] * len(f) + [0] * len(r)
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, f + r))


def _bootstrap_auc_ci(fakes: list, reals: list,
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    from sklearn.metrics import roc_auc_score
    f = np.asarray([r["score"] for r in fakes if r.get("score") is not None],
                   dtype=np.float64)
    r = np.asarray([r["score"] for r in reals if r.get("score") is not None],
                   dtype=np.float64)
    if f.size == 0 or r.size == 0:
        return {"n_boot": 0, "seed": seed, "ci95": None, "note": "empty arm"}
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(f.size), np.zeros(r.size)])
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        fs = rng.choice(f, size=f.size, replace=True)
        rs = rng.choice(r, size=r.size, replace=True)
        vals[i] = roc_auc_score(y, np.concatenate([fs, rs]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"n_boot": int(n_boot), "seed": int(seed),
            "ci95": [round(float(lo), 6), round(float(hi), 6)],
            "note": "stratified percentile bootstrap; report-only"}


def _spearman(rows: list) -> dict:
    pairs = [(r["fps"], r["score"]) for r in rows
             if r.get("score") is not None and r.get("fps") is not None]
    if len(pairs) < 3 or len({p[0] for p in pairs}) < 2:
        return {"rho": None, "n": len(pairs),
                "note": "constant or insufficient fps"}
    try:
        from scipy.stats import spearmanr
    except Exception:
        return {"rho": None, "n": len(pairs), "note": "scipy unavailable"}
    rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
    return {"rho": round(float(rho), 6), "n": len(pairs)}


def _band(auc) -> tuple:
    if auc is None:
        return "INCONCLUSIVE", "no AUC"
    if auc >= 0.85:
        return "GATE_HAS_SIGNAL", "justifies Stage 1 T=16 spend"
    if auc >= 0.70:
        return "PARTIAL", "below Stage 1 S3 bar; decide on GenVideo arm"
    return "NO_TRANSFER_EVIDENCE", "recommend closing the SPLIT line"


def verdict(out: Path) -> int:
    if not _require_seal(out):
        return 2
    doc = _load_json(out / "rows.json")
    rows = _load_scores(_scores_path(out))
    if not rows:
        return 2
    if sorted(_job_keys(rows)) != sorted(_job_keys(doc["rows"])):
        got, want = set(_job_keys(rows)), set(_job_keys(doc["rows"]))
        print(f"FAIL: row set mismatch: {len(got - want)} foreign, "
              f"{len(want - got)} missing", file=sys.stderr)
        return 2

    stage0 = [json.loads(l) for l in STAGE0_SCORES.read_text().splitlines()
              if l.strip()]
    reals = [r for r in stage0 if r.get("arm") == "stage0_real"]

    per_arm = {}
    for arm in ARMS:
        a = [r for r in rows if r["arm"] == arm]
        scored = [r for r in a if r.get("score") is not None]
        reasons = Counter(r.get("unavailable_reason") for r in a
                          if r.get("score") is None)
        per_gen = {}
        for gen in sorted({r["generator"] for r in a}):
            g = [r for r in scored if r["generator"] == gen]
            per_gen[gen] = {
                "n": len([r for r in a if r["generator"] == gen]),
                "n_scored": len(g),
                "auc_vs_stage0_real": _auc(g, reals) if g else None,
                "fps_median": (float(np.median([r["fps"] for r in g]))
                               if g else None),
            }
        per_arm[arm] = {
            "n": len(a),
            "n_scored": len(scored),
            "n_unavailable": len(a) - len(scored),
            "unavailable_reasons": dict(sorted(reasons.items())),
            "auc_vs_stage0_real": _auc(scored, reals),
            "auc_ci95": _bootstrap_auc_ci(scored, reals),
            "per_generator": per_gen,
            "fps_spearman": _spearman(scored),
        }

    primary = per_arm["genvid_fake"]["auc_vs_stage0_real"]
    band, action = _band(primary)

    metrics = {
        "stage": "diag_t16_crossfps",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "recipe_hash": st.split_recipe_hash(),
        "negatives": {"arm": "stage0_real", "n": len(reals),
                      "mean_score": float(np.mean([r["score"] for r in reals])),
                      "median_score": float(np.median([r["score"] for r in reals]))},
        "per_arm": per_arm,
        "primary": {"arm": "genvid_fake", "auc": primary, "band": band},
        "v1_t8_context": {"genvideo_auc": 0.513919, "genvidbench_auc": 0.546810,
                          "note": "different negative class (3 fps MSR-VTT); "
                                  "context only, not a paired delta"},
    }
    result = {
        "decision": f"DIAG_T16_CROSSFPS_{band}",
        "primary_auc": primary,
        "band": band,
        "precommitted_action": action,
        "booked": 0,
        "court": 64.8,
        "ood_video": 1.2,
        "n_rows": len(rows),
        "n_foreign": 0,
        "n_missing": 0,
        "non_claims": [
            "not a Stage 1 result; S1-S8 do not apply",
            "two variables changed vs v1 (frame budget, real-arm content+fps)",
            "no fresh-data evidence",
            "no real-guard / FPR claim",
        ],
    }
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "verdict.json", result)
    print(json.dumps(result, indent=1))
    for arm in ARMS:
        d = per_arm[arm]
        print(f"{arm}: n={d['n']} scored={d['n_scored']} "
              f"auc={d['auc_vs_stage0_real']} ci={d['auc_ci95']['ci95']}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = Path(a.out) if a.out else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    if a.prepare:
        return prepare(out)
    if a.seal:
        return write_seal(out)
    if a.score:
        return score(out)
    if a.verdict:
        return verdict(out)
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
