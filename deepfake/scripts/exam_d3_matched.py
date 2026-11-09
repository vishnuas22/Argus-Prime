"""D3 content-matched T2V re-measure — Phase 0/D3 spec Steps 7–9.

Pre-registered bars (fixed BEFORE measurement):
  G1 AUC >= 0.90: 300 GenVideo-Val fakes vs 300 MSRVTT eval reals.
  G2 AUC >= 0.85: 434 GenVidBench fakes vs the same 300 eval reals.
  G3 catch >= 15/27 issued audit fixtures at tau.
  G4 FPR <= 0.10 on the 300 eval reals.
Measurement only: CPU-only (no MPS), no wiring, no pointer/stacker edits.

  .venv/bin/python deepfake/scripts/exam_d3_matched.py --prepare
  .venv/bin/python deepfake/scripts/exam_d3_matched.py --smoke
  .venv/bin/python deepfake/scripts/exam_d3_matched.py --score
  .venv/bin/python deepfake/scripts/exam_d3_matched.py --verdict
  .venv/bin/python deepfake/scripts/exam_d3_matched.py            # prepare+score+verdict
"""
import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
for p in ("deepfake", "deepfake/bench", "deepfake/scripts"):
    if str(WS / p) not in sys.path:
        sys.path.insert(0, str(WS / p))

from exam_d3_t2v import recipe_hash, fixture_jobs, list_clips  # noqa: E402
import d3_temporal as dt  # noqa: E402

GENVIDEO = WS / "data" / "raw" / "genvideo_val"
REAL_DIR = GENVIDEO / "real_msrvtt"
GENVIDBENCH = WS / "data" / "raw" / "genvidbench"
PRIOR = WS / "artifacts" / "d3_t2v" / "eval"
DEFAULT_OUT = WS / "artifacts" / "d3_t2v_matched"
LOG = WS / "logs" / "d3_matched.log"

ENCODER = "xclip-b16"
TARGET_RECIPE = "1ba8154ca03c4732"
SEED_CALIB = 42
SEED_EVAL = 43
SEED_FAKE = 42
N_CALIB = 300
N_EVAL = 300
N_FAKE = 300
N_GENVID_FAKE = 434
N_FIXTURE = 32
N_FIXTURE_ISSUED = 27

PRIMARY_GENS = ("Sora", "WildScrape", "Crafter")
FALLBACK_GENS = ("Gen2", "HotShot", "Lavie", "ModelScope", "MoonValley",
                 "MorphStudio", "Show_1")
GENVID_DIRS = ("sora", "kling", "opentora")
PRIMARY_DIRS = tuple(g.lower() for g in PRIMARY_GENS)
FALLBACK_DIRS = tuple(g.lower() for g in FALLBACK_GENS)
DEPLOYED = {
    "p6v11/stacker.joblib": WS / "artifacts" / "p6v11" / "stacker.joblib",
    "p6vid_v4/stacker.joblib": WS / "artifacts" / "p6vid_v4" / "stacker.joblib",
    "p4/audio_probe.joblib": WS / "artifacts" / "p4" / "audio_probe.joblib",
}


def _iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _date() -> str:
    return time.strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clips(d: Path) -> list:
    return list_clips(d) if d.exists() else []


def _key(row: dict) -> tuple:
    return (row["arm"], row["generator"], row["filename"])


def _deployed_hashes() -> dict:
    return {rel: _sha256(p) for rel, p in DEPLOYED.items() if p.exists()}


def guard(out: Path, force: bool) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    blocked = [p for p in (out / "metrics.json", out / "verdict.json")
               if p.exists()]
    if blocked and not force:
        print(f"ERROR: {', '.join(str(p) for p in blocked)} already exists — "
              f"pass --force", file=sys.stderr)
        return False
    return True


def _select_fakes(genvideo: Path) -> tuple:
    selected = {}
    shortfall = N_FAKE
    for gen in PRIMARY_GENS:
        clips = _clips(genvideo / gen.lower())
        take = (random.Random(SEED_FAKE).sample(clips, min(100, len(clips)))
                if clips else [])
        selected[gen.lower()] = take
        shortfall -= len(take)

    pools = {}
    for gen in FALLBACK_GENS:
        clips = _clips(genvideo / gen.lower())
        random.Random(SEED_FAKE).shuffle(clips)
        pools[gen.lower()] = clips
    order = [g for g in FALLBACK_DIRS if pools[g]]
    i = 0
    while shortfall > 0 and any(pools[g] for g in order):
        gen = order[i % len(order)]
        i += 1
        if pools[gen]:
            selected.setdefault(gen, []).append(pools[gen].pop(0))
            shortfall -= 1

    usable = sorted(g for g, v in selected.items() if v)
    total = sum(len(v) for v in selected.values())
    return selected, usable, total


def prepare(out: Path, real_dir: Path = REAL_DIR,
            genvideo: Path = GENVIDEO) -> int:
    real = _clips(real_dir)
    if len(real) < N_CALIB + N_EVAL:
        print(f"FAIL: {len(real)} real clips in {real_dir}, need >= "
              f"{N_CALIB + N_EVAL}", file=sys.stderr)
        return 1
    calib = random.Random(SEED_CALIB).sample(real, N_CALIB)
    calib_paths = set(calib)
    calib_names = {p.name for p in calib}
    pool = [p for p in real if p not in calib_paths]
    ev = random.Random(SEED_EVAL).sample(pool, N_EVAL)
    if calib_paths & set(ev):
        print("FAIL: real split path overlap", file=sys.stderr)
        return 1
    if calib_names & {p.name for p in ev}:
        print("FAIL: real split filename overlap", file=sys.stderr)
        return 1

    selected, usable, total = _select_fakes(genvideo)
    if total != N_FAKE:
        print(f"FAIL: {total} GenVideo-Val fakes selected, need {N_FAKE}",
              file=sys.stderr)
        return 1
    if len(usable) < 3:
        print(f"FAIL: only {len(usable)} usable fake dirs, need >= 3",
              file=sys.stderr)
        return 1

    fx = fixture_jobs()
    n_issued = sum(1 for r in fx if r["issued"])
    if len(fx) != N_FIXTURE or n_issued != N_FIXTURE_ISSUED:
        print(f"FAIL: fixtures {len(fx)} rows / {n_issued} issued, expected "
              f"{N_FIXTURE}/{N_FIXTURE_ISSUED}", file=sys.stderr)
        return 1

    splits = {
        "created_at": _iso(),
        "corpus": {"genvideo_val": str(genvideo), "real_dir": str(real_dir),
                   "genvidbench": str(GENVIDBENCH)},
        "seeds": {"calib": SEED_CALIB, "eval": SEED_EVAL, "fake": SEED_FAKE},
        "algorithm": ("reals: Random(42).sample(sorted,300) then "
                      "Random(43).sample(sorted remaining,300); fakes: "
                      "Random(42).sample(sorted primary dir, min(100,n)); "
                      "shortfall round-robin over fallback dirs (seeded 42, "
                      "one clip at a time, sorted-then-shuffled pools)"),
        "real_pool": len(real),
        "n_calib": len(calib),
        "n_eval": len(ev),
        "n_fake": total,
        "zero_overlap": True,
        "zero_overlap_by_filename": not (calib_names & {p.name for p in ev}),
        "fake_dirs": usable,
        "fake_counts": {g: len(selected[g]) for g in usable},
        "fixture_rows": len(fx),
        "fixture_issued": n_issued,
        "calib": [{"filename": p.name, "path": str(p)} for p in sorted(calib)],
        "eval_real": [{"filename": p.name, "path": str(p)} for p in sorted(ev)],
        "fakes": [{"generator": g, "filename": p.name, "path": str(p)}
                  for g in usable for p in sorted(selected[g])],
        "fixtures": [{"generator": r["generator"], "filename": r["filename"],
                      "path": str(r["path"]), "issued": bool(r["issued"])}
                     for r in sorted(fx, key=lambda r: (r["generator"],
                                                        r["filename"]))],
    }
    sp = out / "splits.json"
    sp.write_text(json.dumps(splits, indent=2))

    manifest = {
        "exam": "D3 content-matched T2V re-measure",
        "created_at": _iso(),
        "out": str(out),
        "encoder": ENCODER,
        "recipe_hash": recipe_hash(ENCODER),
        "corpora": {
            "genvideo_val": {
                "path": str(genvideo), "real_dir": str(real_dir),
                "license": "CC-BY-NC-4.0", "role": "EXAM-ONLY",
                "counts": {g: len(_clips(genvideo / g))
                           for g in PRIMARY_DIRS + FALLBACK_DIRS},
                "real_clips": len(real),
            },
            "genvidbench": {
                "path": str(GENVIDBENCH),
                "counts": {g: len(_clips(GENVIDBENCH / g))
                           for g in GENVID_DIRS},
            },
            "fixtures": {"rows": len(fx), "issued": n_issued},
        },
        "deployed_artifact_sha256": _deployed_hashes(),
        "wiring": "none",
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"PREPARE OK: calib={len(calib)} eval={len(ev)} fake={total} "
          f"dirs={len(usable)} {dict((g, len(selected[g])) for g in usable)} "
          f"fixtures={len(fx)}/{n_issued} -> {sp}")
    return 0


def _load_splits(out: Path) -> dict:
    p = out / "splits.json"
    if not p.exists():
        print(f"ERROR: {p} missing — run --prepare first", file=sys.stderr)
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"FAIL: corrupt {p}: {e}", file=sys.stderr)
        return {}


def _prior_rows() -> list:
    p = PRIOR / "scores.jsonl"
    if not p.exists():
        return []
    rows = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"WARN: ignoring corrupt prior line {i}: {e}", file=sys.stderr)
            return []
    return rows


def _cache_eligible() -> tuple:
    current = recipe_hash(ENCODER)
    if current != TARGET_RECIPE:
        return False, f"recipe_hash {current} != {TARGET_RECIPE}"
    p = PRIOR / "metrics.json"
    if not p.exists():
        return False, "prior metrics.json missing"
    try:
        prior = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        return False, f"prior metrics.json corrupt: {e}"
    if prior.get("recipe_hash") != current:
        return False, (f"prior metrics recipe_hash {prior.get('recipe_hash')} "
                       f"!= {current}")
    return True, "recipe matched"


def _reuse_map(prior_rows: list) -> dict:
    reuse = {}
    for r in prior_rows:
        if r["arm"] == "eval_fake":
            reuse[("genvid_fake", r["generator"], r["filename"])] = r
        elif r["arm"] == "fixture":
            reuse[("fixture", r["generator"], r["filename"])] = r
    return reuse


def _spotcheck(reuse: dict, job_keys: set = None) -> tuple:
    keys = sorted(k for k, r in reuse.items()
                  if r.get("std") is not None
                  and (job_keys is None or k in job_keys))
    if len(keys) < 3:
        return False, "fewer than 3 reusable scored clips"
    picks = [keys[0], keys[len(keys) // 2], keys[-1]]
    for key in picks:
        row = reuse[key]
        res = dt.d3_std(Path(row["path"]), ENCODER)
        if (res["std"] is None or row.get("n_frames") != res["n_frames"]
                or abs(float(res["std"]) - float(row["std"])) > 1e-6):
            return False, (f"spot-check drift on {key}: prior "
                           f"{row.get('std')}/{row.get('n_frames')} vs "
                           f"{res['std']}/{res['n_frames']}")
    return True, picks


def genvid_jobs() -> list:
    prior = [r for r in _prior_rows() if r["arm"] == "eval_fake"]
    pairs = sorted({(r["generator"], r["filename"]) for r in prior})
    if len(pairs) != N_GENVID_FAKE:
        print(f"FAIL: prior cache has {len(pairs)} GenVidBench clips, need "
              f"{N_GENVID_FAKE}; refusing to re-sample", file=sys.stderr)
        return []
    missing = [f"{g}/{f}" for g, f in pairs
               if not (GENVIDBENCH / g / f).exists()]
    if missing:
        print(f"FAIL: {len(missing)} prior GenVidBench clips missing, "
              f"e.g. {missing[:3]}", file=sys.stderr)
        return []
    return [{"arm": "genvid_fake", "generator": g, "label": 1,
             "path": GENVIDBENCH / g / f, "filename": f}
            for g, f in pairs]


def build_jobs(splits: dict) -> list:
    for key, n in (("calib", N_CALIB), ("eval_real", N_EVAL),
                   ("fakes", N_FAKE)):
        if len(splits.get(key, [])) != n:
            print(f"FAIL: splits {key}={len(splits.get(key, []))}, "
                  f"expected {n}", file=sys.stderr)
            return []
    jobs = []
    for e in splits["calib"]:
        jobs.append({"arm": "calib", "generator": "msrvtt", "label": 0,
                     "path": Path(e["path"]), "filename": e["filename"]})
    for e in splits["eval_real"]:
        jobs.append({"arm": "eval_real", "generator": "msrvtt", "label": 0,
                     "path": Path(e["path"]), "filename": e["filename"]})
    for e in splits["fakes"]:
        jobs.append({"arm": "genvideo_fake", "generator": e["generator"],
                     "label": 1, "path": Path(e["path"]),
                     "filename": e["filename"]})
    gv = genvid_jobs()
    if not gv:
        return []
    jobs += gv
    fx = fixture_jobs()
    n_issued = sum(1 for r in fx if r["issued"])
    if len(fx) != N_FIXTURE or n_issued != N_FIXTURE_ISSUED:
        print(f"FAIL: fixtures {len(fx)} rows / {n_issued} issued, expected "
              f"{N_FIXTURE}/{N_FIXTURE_ISSUED}", file=sys.stderr)
        return []
    jobs += fx
    return jobs


def score_rows(jobs: list, scores_path: Path, reuse: dict = None,
               log_path: Path = LOG) -> list:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    jkeys = [_key(j) for j in jobs]
    dup = [k for k, c in Counter(jkeys).items() if c > 1]
    if dup:
        print(f"FAIL: duplicate job keys: {dup[:5]}", file=sys.stderr)
        return []

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
                return []
            key = _key(row)
            if key in done:
                print(f"FAIL: duplicate key in {scores_path}: {key}",
                      file=sys.stderr)
                return []
            done[key] = row

    out = []
    with open(scores_path, "a") as f, open(log_path, "a") as log:
        for i, job in enumerate(jobs):
            key = _key(job)
            if key in done:
                out.append(done[key])
                continue
            path = Path(job["path"])
            if not path.exists():
                print(f"FAIL: missing clip {path}", file=sys.stderr)
                return []
            rec = {"arm": job["arm"], "generator": job["generator"],
                   "label": job["label"], "path": str(path),
                   "filename": job["filename"], "sha256": _sha256(path)}
            if reuse and key in reuse:
                prior = reuse[key]
                rec.update({"std": prior.get("std"),
                            "n_frames": prior.get("n_frames"),
                            "unavailable_reason": prior.get("unavailable_reason"),
                            "source": "reuse"})
            else:
                res = dt.d3_std(path, ENCODER)
                rec.update({"std": res["std"], "n_frames": res["n_frames"],
                            "unavailable_reason": res["unavailable_reason"],
                            "source": "new"})
            if "issued" in job:
                rec["issued"] = bool(job["issued"])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                line = (f"[{i+1}/{len(jobs)}] {job['arm']}/{job['generator']}/"
                        f"{job['filename']} std={rec['std']} src={rec['source']}")
                print(line, flush=True)
                log.write(line + "\n")
            log.flush()
            out.append(rec)

    keys = [_key(r) for r in out]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    if dup:
        print(f"FAIL: duplicate keys after scoring: {dup[:5]}", file=sys.stderr)
        return []
    return out


def score(out: Path) -> int:
    if dt._device() != "cpu":
        print("FAIL: D3 is CPU-only; refusing non-CPU device", file=sys.stderr)
        return 1
    splits = _load_splits(out)
    if not splits:
        return 1
    jobs = build_jobs(splits)
    if not jobs:
        return 1

    reuse = {}
    ok, why = _cache_eligible()
    if ok:
        prior = _prior_rows()
        ok, why = _cache_coverage(prior)
        if ok:
            candidate = _reuse_map(prior)
            spot_ok, detail = _spotcheck(candidate, {_key(j) for j in jobs})
            if spot_ok:
                reuse = candidate
                print(f"cache reuse ENABLED ({why}); spot-check {detail}",
                      flush=True)
            else:
                print(f"cache reuse DISABLED: {detail}", flush=True)
        else:
            print(f"cache reuse DISABLED: {why}", flush=True)
    else:
        print(f"cache reuse DISABLED: {why}", flush=True)

    rows = score_rows(jobs, out / "scores.jsonl", reuse)
    if not rows:
        return 1
    counts = Counter(r["arm"] for r in rows)
    sources = Counter(r["source"] for r in rows)
    print(f"SCORE OK: rows={len(rows)} arms={dict(counts)} "
          f"sources={dict(sources)} -> {out / 'scores.jsonl'}")
    return 0


def _cache_coverage(prior: list) -> tuple:
    keys = [_key(r) for r in prior]
    if len(set(keys)) != len(keys):
        return False, "prior cache has duplicate (arm,generator,filename) keys"
    n_fake = sum(1 for r in prior if r["arm"] == "eval_fake")
    n_fix = sum(1 for r in prior if r["arm"] == "fixture")
    if n_fake != N_GENVID_FAKE:
        return False, f"prior has {n_fake} eval_fake rows, need {N_GENVID_FAKE}"
    if n_fix != N_FIXTURE:
        return False, f"prior has {n_fix} fixture rows, need {N_FIXTURE}"
    return True, "prior cache covers 434+32"


def smoke(out: Path) -> int:
    if dt._device() != "cpu":
        print("FAIL: D3 is CPU-only; refusing non-CPU device", file=sys.stderr)
        return 1
    splits = _load_splits(out)
    if splits:
        calib = [Path(e["path"]) for e in splits["calib"][:4]]
        ev = [Path(e["path"]) for e in splits["eval_real"][:4]]
        fx = [{"generator": e["generator"], "filename": e["filename"],
               "path": Path(e["path"]), "issued": e["issued"]}
              for e in splits["fixtures"][:2]]
    else:
        real = _clips(REAL_DIR)
        if len(real) < 8:
            print(f"FAIL: {len(real)} real clips for smoke", file=sys.stderr)
            return 1
        calib = random.Random(SEED_CALIB).sample(real, 4)
        pool = [p for p in real if p not in set(calib)]
        ev = random.Random(SEED_EVAL).sample(pool, 4)
        fx = fixture_jobs()[:2]
    if len(calib) != 4 or len(ev) != 4 or len(fx) != 2:
        print("FAIL: smoke selections incomplete", file=sys.stderr)
        return 1

    jobs = []
    for gen in PRIMARY_GENS:
        for p in _clips(GENVIDEO / gen.lower())[:2]:
            jobs.append({"arm": "genvideo_fake", "generator": gen.lower(),
                         "label": 1, "path": p, "filename": p.name})
    for p in calib:
        jobs.append({"arm": "calib", "generator": "msrvtt", "label": 0,
                     "path": p, "filename": p.name})
    for p in ev:
        jobs.append({"arm": "eval_real", "generator": "msrvtt", "label": 0,
                     "path": p, "filename": p.name})
    for r in fx:
        jobs.append({"arm": "fixture", "generator": r["generator"],
                     "label": 1, "path": Path(r["path"]),
                     "filename": r["filename"], "issued": bool(r["issued"])})
    if len(jobs) != 16:
        print(f"FAIL: smoke has {len(jobs)} clips, expected 16", file=sys.stderr)
        return 1

    rows = score_rows(jobs, out / "smoke" / "scores.jsonl")
    if not rows:
        return 1
    if any(r.get("std") is None for r in rows):
        bad = [r["filename"] for r in rows if r.get("std") is None]
        print(f"FAIL: smoke null std: {bad}", file=sys.stderr)
        return 1
    print(f"SMOKE OK — {len(rows)} rows -> {out / 'smoke' / 'scores.jsonl'}")
    return 0


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
    dup = [k for k, c in Counter(_key(r) for r in rows).items() if c > 1]
    if dup:
        print(f"FAIL: duplicate score keys: {dup[:5]}", file=sys.stderr)
        return []
    return rows


def _avail(rows: list, arm: str) -> list:
    return [r for r in rows if r["arm"] == arm and r.get("std") is not None]


def _auc(fakes: list, reals: list):
    from sklearn.metrics import roc_auc_score
    if not fakes or not reals:
        return None
    y = [1] * len(fakes) + [0] * len(reals)
    s = [-r["std"] for r in fakes] + [-r["std"] for r in reals]
    return float(roc_auc_score(y, s))


def _ap(fakes: list, reals: list):
    from sklearn.metrics import average_precision_score
    if not fakes or not reals:
        return None
    y = [1] * len(fakes) + [0] * len(reals)
    s = [-r["std"] for r in fakes] + [-r["std"] for r in reals]
    return float(average_precision_score(y, s))


def verdict(out: Path) -> int:
    rows = _load_scores(out / "scores.jsonl")
    if not rows:
        return 1
    expected = {"calib": N_CALIB, "eval_real": N_EVAL,
                "genvideo_fake": N_FAKE, "genvid_fake": N_GENVID_FAKE,
                "fixture": N_FIXTURE}
    counts = Counter(r["arm"] for r in rows)
    if set(counts) != set(expected):
        print(f"FAIL: score arms {sorted(counts)} != {sorted(expected)}",
              file=sys.stderr)
        return 1
    for arm, n in expected.items():
        if counts[arm] != n:
            print(f"FAIL: arm {arm} has {counts[arm]} rows, expected {n}",
                  file=sys.stderr)
            return 1

    calib = [r["std"] for r in rows
             if r["arm"] == "calib" and r.get("std") is not None]
    if not calib:
        print("FAIL: no available calib scores", file=sys.stderr)
        return 1
    tau_raw = float(np.percentile(calib, 10))
    tau = round(tau_raw, 6)

    unavailable = {}
    decode = {}
    decode_raw = {}
    for arm in expected:
        arm_rows = [r for r in rows if r["arm"] == arm]
        n_bad = sum(1 for r in arm_rows if r.get("std") is None)
        unavailable[arm] = n_bad
        decode_raw[arm] = n_bad / len(arm_rows)
        decode[arm] = round(decode_raw[arm], 6)
    any_decode = any(v > 0.10 for v in decode_raw.values())

    er = _avail(rows, "eval_real")
    g1f = _avail(rows, "genvideo_fake")
    g2f = _avail(rows, "genvid_fake")
    g1 = _auc(g1f, er)
    g2 = _auc(g2f, er)

    fixture_rows = [r for r in rows if r["arm"] == "fixture"]
    issued = [r for r in fixture_rows if r.get("issued")]
    if len(issued) != N_FIXTURE_ISSUED:
        print(f"FAIL: {len(issued)} issued fixtures in scores, expected "
              f"{N_FIXTURE_ISSUED}", file=sys.stderr)
        return 1
    catch = sum(1 for r in issued
                if r.get("std") is not None and r["std"] < tau)
    fpr = (float(np.mean([r["std"] < tau for r in er])) if er else None)

    bars = {
        "G1": {"bar": 0.90, "auc": round(g1, 6) if g1 is not None else None,
               "n_fake": len(g1f), "n_real": len(er),
               "pass": g1 is not None and g1 >= 0.90},
        "G2": {"bar": 0.85, "auc": round(g2, 6) if g2 is not None else None,
               "n_fake": len(g2f), "n_real": len(er),
               "pass": g2 is not None and g2 >= 0.85},
        "G3": {"bar": 15, "catch": catch, "n_issued": len(issued),
               "pass": catch >= 15},
        "G4": {"bar": 0.10, "fpr": round(fpr, 6) if fpr is not None else None,
               "n_real": len(er),
               "pass": fpr is not None and fpr <= 0.10},
    }

    per_generator = {}
    for arm in ("genvideo_fake", "genvid_fake"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        per_generator[arm] = {}
        for gen in sorted({r["generator"] for r in arm_rows}):
            gf = [r for r in arm_rows
                  if r["generator"] == gen and r.get("std") is not None]
            auc = _auc(gf, er)
            ap = _ap(gf, er)
            per_generator[arm][gen] = {
                "n": sum(1 for r in arm_rows if r["generator"] == gen),
                "n_available": len(gf),
                "auc": round(auc, 6) if auc is not None else None,
                "ap": round(ap, 6) if ap is not None else None,
            }

    fake_counts = {}
    for arm in ("genvideo_fake", "genvid_fake"):
        fake_counts[arm] = dict(Counter(r["generator"] for r in rows
                                        if r["arm"] == arm))

    current_hashes = _deployed_hashes()
    mp = out / "MANIFEST.json"
    if not mp.exists():
        print(f"FAIL: {mp} missing — run --prepare to record the deployed "
              f"artifact baseline before --verdict", file=sys.stderr)
        return 1
    try:
        baseline = json.loads(mp.read_text()).get("deployed_artifact_sha256")
    except json.JSONDecodeError as e:
        print(f"FAIL: unreadable {mp}: {e}", file=sys.stderr)
        return 1
    if not baseline:
        print(f"FAIL: {mp} has no deployed_artifact_sha256 baseline",
              file=sys.stderr)
        return 1
    unchanged = current_hashes == baseline
    baseline_source = "MANIFEST.json"

    if any_decode:
        decision = "INCONCLUSIVE_DECODE"
    elif not bars["G1"]["pass"]:
        decision = "D3_CLOSED_G1_MISS"
    elif not bars["G4"]["pass"]:
        decision = "D3_REAL_GUARD_FAIL"
    elif not bars["G2"]["pass"] or not bars["G3"]["pass"]:
        decision = "D3_PAPER_ONLY_NOT_TRANSFERABLE"
    else:
        decision = "PASS_PENDING_WIRING_SPEC"

    metrics = {
        "exam": "D3 content-matched T2V re-measure",
        "date": _date(),
        "encoder": ENCODER,
        "recipe_hash": recipe_hash(ENCODER),
        "tau": tau_raw,
        "tau_rounded": tau,
        "n_calib": counts["calib"],
        "n_eval_real": counts["eval_real"],
        "n_genvideo_fake": counts["genvideo_fake"],
        "n_genvid_fake": counts["genvid_fake"],
        "n_fixture": counts["fixture"],
        "n_fixture_issued": len(issued),
        "n_unavailable": unavailable,
        "decode_fail_rate": decode,
        "bars": bars,
        "per_generator": per_generator,
        "fake_counts": fake_counts,
        "fixture_counts": {"n_total": len(fixture_rows),
                           "n_issued": len(issued),
                           "catch_issued": catch,
                           "catch_all": sum(
                               1 for r in fixture_rows
                               if r.get("std") is not None and r["std"] < tau)},
        "deployed_artifact_sha256": current_hashes,
        "deployed_artifact_sha256_baseline": baseline,
        "deployed_artifact_sha256_baseline_source": baseline_source,
        "deployed_artifact_sha256_unchanged": unchanged,
        "decision": decision,
        "wiring": "none",
        "license": "CC-BY-NC-4.0",
    }
    verdict_doc = {
        "decision": decision,
        "bars": bars,
        "tau": tau,
        "recipe_hash": metrics["recipe_hash"],
        "wiring": "none",
        "license": "CC-BY-NC-4.0",
        "deployed_artifact_sha256_unchanged": unchanged,
        "date": _date(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "verdict.json").write_text(json.dumps(verdict_doc, indent=2))

    print(f"G1 AUC={bars['G1']['auc']} (bar 0.90) "
          f"{'PASS' if bars['G1']['pass'] else 'FAIL'}")
    print(f"G2 AUC={bars['G2']['auc']} (bar 0.85) "
          f"{'PASS' if bars['G2']['pass'] else 'FAIL'}")
    print(f"G3 catch={bars['G3']['catch']}/{bars['G3']['n_issued']} (bar 15) "
          f"{'PASS' if bars['G3']['pass'] else 'FAIL'}")
    print(f"G4 FPR={bars['G4']['fpr']} (bar 0.10) "
          f"{'PASS' if bars['G4']['pass'] else 'FAIL'}")
    print(f"tau={tau} (p10={tau_raw}) decision={decision}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="D3 content-matched T2V exam (CPU-only; no wiring)")
    ap.add_argument("--prepare", action="store_true",
                    help="write splits.json + MANIFEST.json")
    ap.add_argument("--smoke", action="store_true",
                    help="16-clip pipeline smoke into <out>/smoke")
    ap.add_argument("--score", action="store_true",
                    help="score all arms into <out>/scores.jsonl (resumable)")
    ap.add_argument("--verdict", action="store_true",
                    help="recompute metrics.json + verdict.json from scores")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--force", action="store_true",
                    help="resume/append and overwrite metrics/verdict")
    args = ap.parse_args()
    out = Path(args.out)

    if args.smoke:
        if args.prepare or args.score or args.verdict:
            ap.error("--smoke is standalone")
        if not guard(out, args.force):
            return 1
        return smoke(out)

    if not any((args.prepare, args.score, args.verdict)):
        args.prepare = args.score = args.verdict = True
    if not guard(out, args.force):
        return 1
    if args.prepare:
        rc = prepare(out)
        if rc:
            return rc
    if args.score:
        rc = score(out)
        if rc:
            return rc
    if args.verdict:
        rc = verdict(out)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
