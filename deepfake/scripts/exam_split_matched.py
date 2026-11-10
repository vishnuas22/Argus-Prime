"""SPLIT matched T2V exam — pre-registration artifacts/split_prereg/SPEC.md.

Stage 0 (default): GenBuster++ video/real vs video/fake, AUC only.
  PASS >= 0.70 · HOLD 0.60-0.70 · KILL < 0.60.
  Mandatory controls: per-clip motion metric, 100-bin motion-rebalanced AUC,
  motion-only AUC, fps/score correlation, D3 reference context 0.4843.

Stage 1 (--stage 1, only after Stage 0 PASS): fresh 300 calib + 700 eval
reals, 300 GenVideo fakes across 10 generators, 434 GenVidBench fakes,
32 fixtures / 27 issued, 7% real-only calibration. S2-S8 gates are frozen.
S6 adds 700 JPEG-q80 compressed-real forwards (the spec's 1,766-forward
arithmetic omits this arm; recorded, bars unchanged).

Measurement only: CPU-only, no wiring, no training, no pointer movement.
No D3 score or tau reuse; D3 scores.jsonl is read ONLY for SHA-256 exclusion.

  .venv/bin/python deepfake/scripts/exam_split_matched.py --prepare
  .venv/bin/python deepfake/scripts/exam_split_matched.py --smoke
  .venv/bin/python deepfake/scripts/exam_split_matched.py --score
  .venv/bin/python deepfake/scripts/exam_split_matched.py --verdict
"""
import argparse
import csv
import hashlib
import json
import math
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

import split_temporal as st  # noqa: E402

GENBUSTER = WS / "data" / "raw" / "genbuster_plusplus" / "extracted" / "video"
GENBUSTER_ZIP = WS / "data" / "raw" / "genbuster_plusplus" / \
    "GenBuster-Bench-plusplus.zip"
GENVIDEO = WS / "data" / "raw" / "genvideo_val"
REAL_DIR = GENVIDEO / "real_msrvtt"
GENVIDBENCH = WS / "data" / "raw" / "genvidbench"
D3_DIR = WS / "artifacts" / "d3_t2v_matched"
FIXTURE_MANIFEST = WS / "artifacts" / "audit_20261106" / "endpoint_manifest.csv"
FIXTURE_RESULTS = WS / "artifacts" / "audit_20261106" / "endpoint_results.csv"
DEFAULT_OUT = WS / "artifacts" / "split_t2v_matched"
LOG0 = WS / "logs" / "split_stage0.log"
LOG1 = WS / "logs" / "split_stage1.log"

N_STAGE0 = 1000
N_CALIB = 300
N_EVAL = 700
N_FAKE = 300
N_GENVID_FAKE = 434
N_FIXTURE = 32
N_FIXTURE_ISSUED = 27
N_SMOKE = 16
N_BINS = 100
ALPHA = 0.07
SEEDS = {"calib": 142, "eval": 143, "fake": 142, "genvidbench": 142}
GENS = ("sora", "wildscrape", "crafter", "gen2", "hotshot", "lavie",
        "modelscope", "moonvalley", "morphstudio", "show_1")
GENVID_DIRS = ("sora", "kling", "opentora")
D3_REFERENCE_AUC = 0.4843
D3_REFERENCE_SOURCE = ("artifacts/audit_20261106 motion-bias audit, "
                       "GenBuster++ run; context only, not a gate")
GENVIDBENCH_TAKE = {"sora": 51, "kling": 263, "opentora": 120}
DEPLOYED = {
    "p6v11/stacker.joblib": WS / "artifacts" / "p6v11" / "stacker.joblib",
    "p6vid_v4/stacker.joblib": WS / "artifacts" / "p6vid_v4" / "stacker.joblib",
    "p4/audio_probe.joblib": WS / "artifacts" / "p4" / "audio_probe.joblib",
}


class Stage1Infeasible(RuntimeError):
    pass


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


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"corrupt {path}: {e}") from e


def list_clips(d: Path) -> list:
    """Case-insensitive .mp4, macOS junk excluded (D3 harness convention)."""
    return sorted(q for q in d.iterdir() if q.is_file()
                  and q.suffix.lower() == ".mp4"
                  and not q.name.startswith("._"))


def _clip_entry(path: Path, label: int, arm: str, generator: str) -> dict:
    return {"arm": arm, "label": int(label), "generator": generator,
            "path": str(path), "filename": path.name,
            "sha256": _sha256(path)}


def _dedupe_by_sha(entries: list) -> tuple:
    seen, out, dups = set(), [], []
    for e in entries:
        if e["sha256"] in seen:
            dups.append(e)
            continue
        seen.add(e["sha256"])
        out.append(e)
    return out, dups


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


def _scores_path(out: Path) -> Path:
    return out / "scores.jsonl"


def _recipe_ok(rows: list, label: str) -> bool:
    want = st.split_recipe_hash()
    bad = [r for r in rows if r.get("recipe_hash") != want]
    if bad:
        print(f"FAIL: {label} has {len(bad)} rows scored under a different "
              f"recipe_hash (e.g. {bad[0].get('recipe_hash')} != {want}); "
              f"refusing to mix recipes", file=sys.stderr)
        return False
    return True


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


def score_rows(jobs: list, scores_path: Path, log_path: Path) -> list:
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    recipe_hash = st.split_recipe_hash()
    jkeys = [(j["arm"], j["generator"], j["filename"]) for j in jobs]
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
            if row.get("recipe_hash") != recipe_hash:
                print(f"FAIL: stale score cache {scores_path}: row {i} "
                      f"recipe_hash {row.get('recipe_hash')} != "
                      f"{recipe_hash}; use a new --out or remove the file",
                      file=sys.stderr)
                return []
            key = (row["arm"], row["generator"], row["filename"])
            if key in done:
                print(f"FAIL: duplicate key in {scores_path}: {key}",
                      file=sys.stderr)
                return []
            done[key] = row

    out = []
    with open(scores_path, "a") as f, open(log_path, "a") as log:
        for i, job in enumerate(jobs):
            key = (job["arm"], job["generator"], job["filename"])
            path = Path(job["path"])
            if not path.exists():
                print(f"FAIL: missing clip {path}", file=sys.stderr)
                return []
            sha = _sha256(path)
            if key in done:
                row = done[key]
                if row.get("sha256") != sha:
                    print(f"FAIL: clip changed since scoring: {path} "
                          f"({row.get('sha256')} -> {sha})", file=sys.stderr)
                    return []
                out.append(row)
                continue
            rec = {"arm": job["arm"], "generator": job["generator"],
                   "label": int(job["label"]), "path": str(path),
                   "filename": job["filename"], "sha256": sha,
                   "recipe_hash": recipe_hash, "source": "new"}
            if "issued" in job:
                rec["issued"] = bool(job["issued"])
            if job.get("jpeg_q80"):
                rec["launder"] = "frame_jpeg_q80"
            res = st.score_clip(path, jpeg_q80=bool(job.get("jpeg_q80")))
            rec.update(res)
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
            out.append(rec)

    keys = [(r["arm"], r["generator"], r["filename"]) for r in out]
    dup = [k for k, c in Counter(keys).items() if c > 1]
    if dup:
        print(f"FAIL: duplicate keys after scoring: {dup[:5]}",
              file=sys.stderr)
        return []
    return out


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


def _motion_rebalanced(rows: list, n_bins: int = N_BINS) -> dict:
    usable = [r for r in rows
              if r.get("score") is not None and r.get("motion") is not None]
    if not usable:
        return {"auc": None, "n_bins_requested": n_bins, "n_bins_nonempty": 0,
                "n_used": 0, "n_skipped": 0, "bins": []}
    order = sorted(range(len(usable)),
                   key=lambda i: (usable[i]["motion"], usable[i]["arm"],
                                  usable[i]["filename"]))
    chunks = np.array_split(np.asarray(order), n_bins)
    aucs, bins = [], []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        br = [usable[int(i)] for i in chunk]
        y = [r["label"] for r in br]
        s = [r["score"] for r in br]
        rec = {"n": len(br), "n_real": sum(1 for v in y if v == 0),
               "n_fake": sum(1 for v in y if v == 1),
               "motion_min": float(min(r["motion"] for r in br)),
               "motion_max": float(max(r["motion"] for r in br))}
        if len(set(y)) < 2:
            rec["auc"] = None
            bins.append(rec)
            continue
        from sklearn.metrics import roc_auc_score
        rec["auc"] = float(roc_auc_score(y, s))
        aucs.append(rec["auc"])
        bins.append(rec)
    return {"auc": float(np.mean(aucs)) if aucs else None,
            "n_bins_requested": n_bins,
            "n_bins_nonempty": len(bins),
            "n_used": len(aucs),
            "n_skipped": len(bins) - len(aucs),
            "tie_break": "motion, arm, filename",
            "bins": bins}


def _corr(rows: list, xkey: str = "fps", ykey: str = "score") -> dict:
    pairs = [(float(r[xkey]), float(r[ykey])) for r in rows
             if r.get(xkey) is not None and r.get(ykey) is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "pearson": None, "spearman": None,
                "note": "fewer than 3 pairs"}
    if len({x for x, _ in pairs}) < 2:
        return {"n": len(pairs), "pearson": None, "spearman": None,
                "note": "constant x (source fps)"}
    x = np.asarray([p[0] for p in pairs], dtype=np.float64)
    y = np.asarray([p[1] for p in pairs], dtype=np.float64)
    pearson = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 else None
    try:
        from scipy.stats import spearmanr
        sp = spearmanr(x, y).statistic
        spearman = float(sp) if np.isfinite(sp) else None
    except Exception:
        spearman = None
    return {"n": len(pairs), "pearson": pearson, "spearman": spearman}


def _fps_summary(rows: list) -> dict:
    vals = [float(r["fps"]) for r in rows if r.get("fps") is not None]
    if not vals:
        return {"n": 0, "unique": [], "min": None, "max": None}
    return {"n": len(vals), "unique": sorted({round(v, 6) for v in vals}),
            "min": min(vals), "max": max(vals)}


def _wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return None
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def _arm_rows(rows: list, arm: str) -> list:
    return [r for r in rows if r["arm"] == arm]


def _avail(rows: list) -> list:
    return [r for r in rows if r.get("score") is not None]


# ---------------------------------------------------------------- Stage 0

def prepare_stage0(out: Path) -> int:
    real_dir, fake_dir = GENBUSTER / "real", GENBUSTER / "fake"
    if not real_dir.is_dir() or not fake_dir.is_dir():
        print(f"FAIL: GenBuster++ video dirs missing: {real_dir} / {fake_dir}",
              file=sys.stderr)
        return 1
    real, fake = list_clips(real_dir), list_clips(fake_dir)
    if not real or not fake:
        print(f"FAIL: GenBuster++ video counts real={len(real)} "
              f"fake={len(fake)}", file=sys.stderr)
        return 1
    verified = len(real) == N_STAGE0 and len(fake) == N_STAGE0
    if not verified:
        print(f"WARNING: GenBuster++ counts real={len(real)} fake={len(fake)} "
              f"!= {N_STAGE0}/{N_STAGE0}; using exact verified counts",
              file=sys.stderr)

    real_e = [_clip_entry(p, 0, "stage0_real", "genbuster_real") for p in real]
    fake_e = [_clip_entry(p, 1, "stage0_fake", "genbuster_fake") for p in fake]
    real_e, real_dups = _dedupe_by_sha(real_e)
    fake_e, fake_dups = _dedupe_by_sha(fake_e)
    if real_dups or fake_dups:
        print(f"FAIL: duplicate SHA-256 inside a Stage 0 arm "
              f"(real={len(real_dups)}, fake={len(fake_dups)})",
              file=sys.stderr)
        return 1

    archive = None
    if GENBUSTER_ZIP.exists():
        archive = {"path": str(GENBUSTER_ZIP),
                   "bytes": GENBUSTER_ZIP.stat().st_size,
                   "sha256": _sha256(GENBUSTER_ZIP)}

    splits = {
        "stage": 0,
        "created_at": _iso(),
        "corpus": {"video": str(GENBUSTER), "real_dir": str(real_dir),
                   "fake_dir": str(fake_dir), "license": "MIT",
                   "role": "measurement-only", "archive": archive},
        "expected_counts": {"real": N_STAGE0, "fake": N_STAGE0},
        "counts": {"real": len(real_e), "fake": len(fake_e)},
        "expected_counts_verified": verified,
        "clips": real_e + fake_e,
    }
    _write_json(out / "splits.json", splits)

    manifest = {
        "exam": "SPLIT Stage 0 motion-unbiased gate",
        "created_at": _iso(),
        "out": str(out),
        "stage": 0,
        "encoder": st.ENCODER,
        "recipe_hash": st.split_recipe_hash(),
        "code_sha256": st.CODE_SHA,
        "policy": st.POLICY,
        "corpora": {"genbuster_plusplus": {
            "path": str(GENBUSTER), "license": "MIT",
            "role": "measurement-only", "video_only": True,
            "counts": {"real": len(real_e), "fake": len(fake_e)},
            "archive": archive}},
        "deployed_artifact_sha256": _deployed_hashes(),
        "wiring": "none",
    }
    _write_json(out / "MANIFEST.json", manifest)
    print(f"PREPARE OK: stage=0 real={len(real_e)} fake={len(fake_e)} "
          f"verified={verified} -> {out / 'splits.json'}")
    return 0


def _stage0_jobs(splits: dict) -> list:
    return [{"arm": e["arm"], "generator": e["generator"],
             "label": e["label"], "path": Path(e["path"]),
             "filename": e["filename"], "sha256": e["sha256"]}
            for e in splits["clips"]]


def score_stage0(out: Path) -> int:
    if st._device() != "cpu":
        print("FAIL: SPLIT is CPU-only; refusing non-CPU device",
              file=sys.stderr)
        return 1
    sp = out / "splits.json"
    if not sp.exists():
        print(f"ERROR: {sp} missing — run --prepare first", file=sys.stderr)
        return 1
    try:
        splits = _load_json(sp)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    jobs = _stage0_jobs(splits)
    if not jobs:
        print("FAIL: no Stage 0 jobs", file=sys.stderr)
        return 1
    rows = score_rows(jobs, _scores_path(out), LOG0)
    if not rows:
        return 1
    counts = Counter(r["arm"] for r in rows)
    print(f"SCORE OK: stage=0 rows={len(rows)} arms={dict(counts)} "
          f"-> {_scores_path(out)}")
    return 0


def verdict_stage0(out: Path) -> int:
    sp = out / "splits.json"
    if not sp.exists():
        print(f"ERROR: {sp} missing — run --prepare first", file=sys.stderr)
        return 1
    try:
        splits = _load_json(sp)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    rows = _load_scores(_scores_path(out))
    if not rows:
        return 1
    if not _recipe_ok(rows, str(_scores_path(out))):
        return 1
    expected = {"stage0_real": splits["counts"]["real"],
                "stage0_fake": splits["counts"]["fake"]}
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

    unavailable, decode = {}, {}
    for arm in expected:
        arm_rows = _arm_rows(rows, arm)
        n_bad = sum(1 for r in arm_rows if r.get("score") is None)
        unavailable[arm] = n_bad
        decode[arm] = n_bad / len(arm_rows) if arm_rows else 0.0

    avail = _avail(rows)
    reals = [r for r in avail if r["label"] == 0]
    fakes = [r for r in avail if r["label"] == 1]
    auc = _auc(fakes, reals)
    mr = _motion_rebalanced(avail)
    motion_auc = _auc(fakes, reals, key="motion")
    motion_inverted = (1.0 - motion_auc) if motion_auc is not None else None
    fps_corr = {
        "pooled": _corr(avail),
        "real": _corr(reals),
        "fake": _corr(fakes),
    }

    inconclusive = any(v > 0.10 for v in decode.values())
    if inconclusive:
        decision = "SPLIT_STAGE0_INCONCLUSIVE_DECODE"
    elif auc is None:
        decision = "SPLIT_STAGE0_NO_AUC"
    elif auc >= 0.70:
        decision = "SPLIT_MOTION_PASS"
    elif auc >= 0.60:
        decision = "SPLIT_MOTION_HOLD"
    else:
        decision = "SPLIT_MOTION_FAIL"

    bars = {"pass_min": 0.70, "hold_min": 0.60, "kill_below": 0.60}
    metrics = {
        "exam": "SPLIT Stage 0 motion-unbiased gate",
        "stage": 0,
        "date": _date(),
        "created_at": _iso(),
        "encoder": st.ENCODER,
        "recipe_hash": st.split_recipe_hash(),
        "code_sha256": st.CODE_SHA,
        "policy": st.POLICY,
        "corpus": splits.get("corpus", {}),
        "counts": {"real": len(reals), "fake": len(fakes),
                   "n_scored": len(rows), "n_available": len(avail),
                   "n_unavailable": len(rows) - len(avail)},
        "n_unavailable": unavailable,
        "decode_fail_rate": {k: round(v, 6) for k, v in decode.items()},
        "auc": round(auc, 6) if auc is not None else None,
        "bars": bars,
        "decision": decision,
        "motion_controls": {
            "motion_rebalanced_auc": mr,
            "motion_only_auc": {
                "higher_motion_fake": round(motion_auc, 6)
                if motion_auc is not None else None,
                "higher_motion_real": round(motion_inverted, 6)
                if motion_inverted is not None else None,
                "note": "negative-control detector; not a gate",
            },
            "fps_correlation": fps_corr,
            "fps_distribution": {"pooled": _fps_summary(avail),
                                 "real": _fps_summary(reals),
                                 "fake": _fps_summary(fakes)},
        },
        "d3_reference": {"auc": D3_REFERENCE_AUC,
                         "source": D3_REFERENCE_SOURCE,
                         "is_gate": False,
                         "d3_policy_reused": False},
        "wiring": "none",
        "license": "MIT (GenBuster++ video; measurement-only)",
    }
    verdict_doc = {
        "decision": decision,
        "stage": 0,
        "auc": metrics["auc"],
        "bars": bars,
        "motion_rebalanced_auc": mr["auc"],
        "motion_only_auc": metrics["motion_controls"]["motion_only_auc"],
        "fps_correlation": fps_corr,
        "d3_reference_auc": D3_REFERENCE_AUC,
        "recipe_hash": metrics["recipe_hash"],
        "date": _date(),
        "wiring": "none",
    }
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "verdict.json", verdict_doc)

    print(f"Stage 0 AUC={metrics['auc']} (PASS>=0.70, HOLD>=0.60, "
          f"KILL<0.60) decision={decision}")
    print(f"  motion-rebalanced AUC={mr['auc']} "
          f"used={mr['n_used']}/{mr['n_bins_nonempty']} bins")
    print(f"  motion-only AUC={motion_auc} "
          f"inverted={motion_inverted} (negative control)")
    print(f"  fps corr pooled={fps_corr['pooled']} real={fps_corr['real']}")
    print(f"  D3 reference AUC={D3_REFERENCE_AUC} (context only)")
    return 0


# ---------------------------------------------------------------- Stage 1

def _d3_exclusions(d3_dir: Path) -> tuple:
    p = d3_dir / "scores.jsonl"
    if not p.exists():
        raise Stage1Infeasible(
            f"D3 exclusion source missing: {p}. Stage 1 requires SHA-256 "
            f"exclusions from the D3 scores; refusing to substitute.")
    rows = _load_scores(p)
    if not rows:
        raise Stage1Infeasible(f"unreadable D3 exclusion source: {p}")
    arms = ("calib", "eval_real", "genvideo_fake")
    per_arm, all_sha = {}, set()
    for arm in arms:
        shas = {r["sha256"] for r in rows if r["arm"] == arm
                and r.get("sha256")}
        per_arm[arm] = len(shas)
        all_sha |= shas
    expected = {"calib": N_CALIB, "eval_real": N_CALIB, "genvideo_fake": N_FAKE}
    for arm, n in expected.items():
        if per_arm[arm] != n:
            raise Stage1Infeasible(
                f"D3 exclusion arm {arm} has {per_arm[arm]} SHA-256 values, "
                f"expected {n}; refusing to guess the exclusion set")
    return all_sha, per_arm, str(p)


def _real_pool(real_dir: Path, excluded: set) -> tuple:
    clips = list_clips(real_dir)
    entries = [_clip_entry(p, 0, "calib", "msrvtt") for p in clips]
    kept = [e for e in entries if e["sha256"] not in excluded]
    kept, dups = _dedupe_by_sha(kept)
    return kept, len(entries) - len(kept) - len(dups), dups


def _fake_pools(genvideo: Path, excluded: set) -> tuple:
    pools, stats = {}, {}
    for gen in GENS:
        d = genvideo / gen
        if not d.is_dir():
            pools[gen], stats[gen] = [], {"on_disk": 0, "excluded": 0,
                                          "duplicate_sha": 0,
                                          "available": 0}
            continue
        entries = [_clip_entry(p, 1, "genvideo_fake", gen)
                   for p in list_clips(d)]
        kept = [e for e in entries if e["sha256"] not in excluded]
        kept, dups = _dedupe_by_sha(kept)
        pools[gen] = kept
        stats[gen] = {"on_disk": len(entries),
                      "excluded": len(entries) - len(kept) - len(dups),
                      "duplicate_sha": len(dups),
                      "available": len(kept)}
    return pools, stats


def _select_reals(pool: list) -> tuple:
    if len(pool) < N_CALIB + N_EVAL:
        raise Stage1Infeasible(
            f"real pool {len(pool)} < {N_CALIB + N_EVAL} after SHA-256 "
            f"exclusion")
    calib = random.Random(SEEDS["calib"]).sample(pool, N_CALIB)
    calib_sha = {e["sha256"] for e in calib}
    rest = [e for e in pool if e["sha256"] not in calib_sha]
    ev = random.Random(SEEDS["eval"]).sample(rest, N_EVAL)
    if calib_sha & {e["sha256"] for e in ev}:
        raise Stage1Infeasible("calib/eval SHA-256 overlap")
    return calib, ev


def _select_genvideo_fakes(pools: dict) -> tuple:
    rng = random.Random(SEEDS["fake"])
    selected, remaining = {}, {}
    for gen in GENS:
        pool = list(pools.get(gen, []))
        rng.shuffle(pool)
        take = min(N_FAKE // len(GENS), len(pool))
        selected[gen] = pool[:take]
        remaining[gen] = pool[take:]
    shortfall = N_FAKE - sum(len(v) for v in selected.values())
    order = [g for g in GENS if remaining[g]]
    i = 0
    while shortfall > 0 and any(remaining[g] for g in order):
        gen = order[i % len(order)]
        i += 1
        if remaining[gen]:
            selected[gen].append(remaining[gen].pop(0))
            shortfall -= 1
    total = sum(len(v) for v in selected.values())
    if total != N_FAKE:
        raise Stage1Infeasible(
            f"only {total} GenVideo fakes selectable after D3 SHA-256 "
            f"exclusion, need {N_FAKE}")
    empty = [g for g in GENS if not selected[g]]
    return selected, empty


def _select_genvidbench(genvidbench: Path) -> tuple:
    pools = {d: list_clips(genvidbench / d) for d in GENVID_DIRS
             if (genvidbench / d).is_dir()}
    missing = [d for d in GENVID_DIRS if d not in pools]
    if missing:
        raise Stage1Infeasible(f"GenVidBench dirs missing: {missing}")
    selected = []
    remaining = N_GENVID_FAKE
    for d in ("sora", "kling"):
        take = min(GENVIDBENCH_TAKE[d], len(pools[d]), remaining)
        selected += [{"generator": d, "path": p} for p in pools[d][:take]]
        remaining -= take
    need = remaining
    if need > len(pools["opentora"]):
        raise Stage1Infeasible(
            f"GenVidBench opentora {len(pools['opentora'])} < {need}")
    opentora = random.Random(SEEDS["genvidbench"]).sample(
        pools["opentora"], need)
    selected += [{"generator": "opentora", "path": p} for p in opentora]
    entries = [_clip_entry(Path(e["path"]), 1, "genvid_fake",
                           e["generator"]) for e in selected]
    counts = Counter(e["generator"] for e in entries)
    return entries, dict(counts)


def fixture_jobs() -> list:
    if not FIXTURE_MANIFEST.exists() or not FIXTURE_RESULTS.exists():
        raise Stage1Infeasible(
            f"fixture sources missing: {FIXTURE_MANIFEST} / "
            f"{FIXTURE_RESULTS}")
    man = [r for r in csv.DictReader(open(FIXTURE_MANIFEST))
           if r["kind"] == "video" and int(r["label"]) == 1]
    abstained = set()
    for r in csv.DictReader(open(FIXTURE_RESULTS)):
        if (r["kind"] == "video" and int(r["label"]) == 1
                and r["abstained"].strip().lower() in ("true", "1")):
            abstained.add(r["path"])
    rows = []
    for r in man:
        p = Path(r["path"])
        if not p.exists():
            print(f"WARN: fixture missing on disk: {p}", file=sys.stderr)
            continue
        rows.append({"arm": "fixture", "generator": r["family"],
                     "label": 1, "path": str(p), "filename": p.name,
                     "issued": r["path"] not in abstained,
                     "sha256": _sha256(p)})
    return rows


def prepare_stage1(out: Path, real_dir: Path, genvideo: Path,
                   genvidbench: Path, d3_dir: Path,
                   allow_empty_strata: bool) -> int:
    for d in (real_dir, genvideo, genvidbench):
        if not d.is_dir():
            print(f"FAIL: Stage 1 data path missing: {d}", file=sys.stderr)
            return 1
    try:
        excluded, excl_counts, excl_source = _d3_exclusions(d3_dir)
    except Stage1Infeasible as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    real_pool, n_real_excluded, real_dups = _real_pool(real_dir, excluded)
    try:
        calib, ev = _select_reals(real_pool)
    except Stage1Infeasible as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    try:
        fake_pools, fake_stats = _fake_pools(genvideo, excluded)
        selected, empty = _select_genvideo_fakes(fake_pools)
    except Stage1Infeasible as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    if empty and not allow_empty_strata:
        detail = "; ".join(
            f"{g}: available={fake_stats[g]['available']} "
            f"(on_disk={fake_stats[g]['on_disk']}, "
            f"excluded={fake_stats[g]['excluded']})" for g in empty)
        print(
            "FAIL: Stage 1 stratification infeasible under SPEC §5 "
            f"SHA-256 exclusions. Empty strata: {detail}. "
            "The pre-registration requires 300 GenVideo fakes stratified "
            "across 10 generators. Refusing to substitute silently. "
            "Options: (a) amend the pre-registration, or (b) pass "
            "--allow-empty-strata to record the realized 9-stratum split "
            "with the empty stratum named in splits.json/metrics.json.",
            file=sys.stderr)
        return 1
    if empty:
        print(f"WARNING: Stage 1 realized strata omit {empty}; recorded in "
              f"splits.json (--allow-empty-strata)", file=sys.stderr)

    try:
        genvid, genvid_counts = _select_genvidbench(genvidbench)
        fx = fixture_jobs()
    except Stage1Infeasible as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    n_issued = sum(1 for r in fx if r["issued"])
    if len(fx) != N_FIXTURE or n_issued != N_FIXTURE_ISSUED:
        print(f"FAIL: fixtures {len(fx)} rows / {n_issued} issued, expected "
              f"{N_FIXTURE}/{N_FIXTURE_ISSUED}", file=sys.stderr)
        return 1

    fake_entries = [e for g in GENS for e in selected[g]]
    fake_entries.sort(key=lambda e: (e["generator"], e["filename"]))
    splits = {
        "stage": 1,
        "created_at": _iso(),
        "corpus": {"genvideo_val": str(genvideo),
                   "real_dir": str(real_dir),
                   "genvidbench": str(genvidbench),
                   "fixtures_manifest": str(FIXTURE_MANIFEST),
                   "fixtures_results": str(FIXTURE_RESULTS)},
        "seeds": SEEDS,
        "algorithm": (
            "reals: pool filtered by D3 SHA-256 then deduped by SHA-256; "
            "Random(calib seed).sample(sorted,300); Random(eval seed).sample("
            "remaining sorted,700). fakes: per-generator shuffle(seed), take "
            "min(30,available), shortfall round-robin over remaining pools. "
            "GenVidBench: all sora<=51 + all kling<=263 + Random(seed).sample("
            "opentora, remainder)."),
        "exclusions": {"source": excl_source, "by": "sha256",
                       "counts": excl_counts,
                       "n_unique": len(excluded),
                       "also_excludes_d3_calib_reals": True},
        "real_pool": {"on_disk": len(list_clips(real_dir)),
                      "after_exclusion": len(real_pool),
                      "excluded": n_real_excluded,
                      "duplicate_sha_removed": len(real_dups)},
        "n_calib": len(calib),
        "n_eval": len(ev),
        "n_fake": len(fake_entries),
        "n_genvid_fake": len(genvid),
        "n_fixture": len(fx),
        "n_fixture_issued": n_issued,
        "fake_generator_counts": {g: len(selected[g]) for g in GENS},
        "fake_generator_stats": fake_stats,
        "empty_strata": empty,
        "strata_requested": len(GENS),
        "strata_realized": sum(1 for g in GENS if selected[g]),
        "allow_empty_strata": bool(allow_empty_strata),
        "genvidbench_counts": genvid_counts,
        "calib": calib,
        "eval_real": ev,
        "genvideo_fakes": fake_entries,
        "genvidbench_fakes": genvid,
        "fixtures": fx,
    }
    _write_json(out / "splits.json", splits)

    manifest = {
        "exam": "SPLIT Stage 1 content-matched GenVideo rerun",
        "created_at": _iso(),
        "out": str(out),
        "stage": 1,
        "encoder": st.ENCODER,
        "recipe_hash": st.split_recipe_hash(),
        "code_sha256": st.CODE_SHA,
        "policy": st.POLICY,
        "seeds": SEEDS,
        "corpora": {
            "genvideo_val": {"path": str(genvideo),
                             "real_dir": str(real_dir),
                             "license": "CC-BY-NC-4.0",
                             "role": "EXAM-ONLY"},
            "genvidbench": {"path": str(genvidbench),
                            "license": "CC-BY-NC-4.0",
                            "role": "EXAM-ONLY"},
            "fixtures": {"rows": len(fx), "issued": n_issued},
        },
        "forwards": {
            "planned": N_CALIB + N_EVAL + N_FAKE + N_GENVID_FAKE
                       + N_FIXTURE + N_EVAL,
            "spec_estimate": 1766,
            "note": ("S6 compressed-real arm adds 700 forwards; the SPEC "
                     "§5 1,766 arithmetic omits it. Bars unchanged."),
        },
        "deployed_artifact_sha256": _deployed_hashes(),
        "wiring": "none",
    }
    _write_json(out / "MANIFEST.json", manifest)
    print(f"PREPARE OK: stage=1 calib={len(calib)} eval={len(ev)} "
          f"fake={len(fake_entries)} {splits['fake_generator_counts']} "
          f"genvid={len(genvid)} {genvid_counts} fixtures={len(fx)}/"
          f"{n_issued} empty_strata={empty} -> {out / 'splits.json'}")
    return 0


def _stage1_jobs(splits: dict) -> list:
    jobs = []
    for e in splits["calib"]:
        jobs.append({"arm": "calib", "generator": e["generator"],
                     "label": 0, "path": Path(e["path"]),
                     "filename": e["filename"]})
    for e in splits["eval_real"]:
        jobs.append({"arm": "eval_real", "generator": e["generator"],
                     "label": 0, "path": Path(e["path"]),
                     "filename": e["filename"]})
    for e in splits["genvideo_fakes"]:
        jobs.append({"arm": "genvideo_fake", "generator": e["generator"],
                     "label": 1, "path": Path(e["path"]),
                     "filename": e["filename"]})
    for e in splits["genvidbench_fakes"]:
        jobs.append({"arm": "genvid_fake", "generator": e["generator"],
                     "label": 1, "path": Path(e["path"]),
                     "filename": e["filename"]})
    for e in splits["fixtures"]:
        jobs.append({"arm": "fixture", "generator": e["generator"],
                     "label": 1, "path": Path(e["path"]),
                     "filename": e["filename"],
                     "issued": bool(e["issued"])})
    for e in splits["eval_real"]:
        jobs.append({"arm": "eval_real_jpeg80",
                     "generator": e["generator"], "label": 0,
                     "path": Path(e["path"]), "filename": e["filename"],
                     "jpeg_q80": True})
    return jobs


def score_stage1(out: Path) -> int:
    if st._device() != "cpu":
        print("FAIL: SPLIT is CPU-only; refusing non-CPU device",
              file=sys.stderr)
        return 1
    sp = out / "splits.json"
    if not sp.exists():
        print(f"ERROR: {sp} missing — run --prepare --stage 1 first",
              file=sys.stderr)
        return 1
    try:
        splits = _load_json(sp)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    jobs = _stage1_jobs(splits)
    expected = {"calib": N_CALIB, "eval_real": N_EVAL,
                "genvideo_fake": N_FAKE, "genvid_fake": N_GENVID_FAKE,
                "fixture": N_FIXTURE, "eval_real_jpeg80": N_EVAL}
    got = Counter(j["arm"] for j in jobs)
    if dict(got) != expected:
        print(f"FAIL: Stage 1 jobs {dict(got)} != {expected}",
              file=sys.stderr)
        return 1
    rows = score_rows(jobs, _scores_path(out), LOG1)
    if not rows:
        return 1
    counts = Counter(r["arm"] for r in rows)
    print(f"SCORE OK: stage=1 rows={len(rows)} arms={dict(counts)} "
          f"-> {_scores_path(out)}")
    return 0


def _smoke_ok(out: Path) -> tuple:
    p = out / "smoke" / "scores.jsonl"
    if not p.exists():
        return False, f"{p} missing"
    rows = _load_scores(p)
    if not _recipe_ok(rows, str(p)):
        return False, "smoke rows have a stale recipe_hash"
    if len(rows) != N_SMOKE:
        return False, f"smoke has {len(rows)} rows, expected {N_SMOKE}"
    bad = []
    for r in rows:
        if (r.get("score") is None or r.get("ttr") is None
                or r.get("lsmi") is None
                or r.get("feature_shape") != [st.T, st.PATCHES, st.DIM]):
            bad.append(r["filename"])
    if bad:
        return False, f"non-finite/invalid smoke rows: {bad[:5]}"
    return True, f"{len(rows)}/16 finite [8,196,768]"


def verdict_stage1(out: Path) -> int:
    sp = out / "splits.json"
    if not sp.exists():
        print(f"ERROR: {sp} missing — run --prepare --stage 1 first",
              file=sys.stderr)
        return 1
    try:
        splits = _load_json(sp)
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    rows = _load_scores(_scores_path(out))
    if not rows:
        return 1
    if not _recipe_ok(rows, str(_scores_path(out))):
        return 1
    expected = {"calib": N_CALIB, "eval_real": N_EVAL,
                "genvideo_fake": N_FAKE, "genvid_fake": N_GENVID_FAKE,
                "fixture": N_FIXTURE, "eval_real_jpeg80": N_EVAL}
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

    calib = _avail(_arm_rows(rows, "calib"))
    if len(calib) != N_CALIB:
        print(f"FAIL: {len(calib)} available calib scores, expected "
              f"{N_CALIB}", file=sys.stderr)
        return 1
    calib_scores = [r["score"] for r in calib]
    tau = float(np.quantile(calib_scores, 1.0 - ALPHA, method="higher"))

    er = _avail(_arm_rows(rows, "eval_real"))
    gvf = _avail(_arm_rows(rows, "genvideo_fake"))
    gbf = _avail(_arm_rows(rows, "genvid_fake"))
    fx = _arm_rows(rows, "fixture")
    issued = [r for r in fx if r.get("issued")]
    jr = _avail(_arm_rows(rows, "eval_real_jpeg80"))

    unavailable, decode = {}, {}
    for arm in expected:
        arm_rows = _arm_rows(rows, arm)
        n_bad = sum(1 for r in arm_rows if r.get("score") is None)
        unavailable[arm] = n_bad
        decode[arm] = n_bad / len(arm_rows) if arm_rows else 0.0
    decode_fail = any(v > 0.10 for v in decode.values())

    auc_gv = _auc(gvf, er)
    auc_gb = _auc(gbf, er)
    fpr = (float(np.mean([r["score"] >= tau for r in er])) if er else None)
    fpr_k = int(sum(1 for r in er if r["score"] >= tau))
    fpr_ci = _wilson(fpr_k, len(er))
    catch = sum(1 for r in issued
                if r.get("score") is not None and r["score"] >= tau)
    fpr_j = (float(np.mean([r["score"] >= tau for r in jr])) if jr else None)
    fpr_j_k = int(sum(1 for r in jr if r["score"] >= tau))
    fpr_j_ci = _wilson(fpr_j_k, len(jr))
    recall_gv = (float(np.mean([r["score"] >= tau for r in gvf]))
                 if gvf else None)
    recall_gb = (float(np.mean([r["score"] >= tau for r in gbf]))
                 if gbf else None)
    calib_fpr = float(np.mean([s >= tau for s in calib_scores]))

    smoke_ok, smoke_detail = _smoke_ok(out)
    s0 = {"pass": bool(smoke_ok), "detail": smoke_detail}
    s1 = {"pass": bool(len(calib) == N_CALIB and np.isfinite(tau)),
          "bar": "tau = real-only quantile 7% (method=higher)",
          "alpha": ALPHA, "n_calib": len(calib), "tau": tau,
          "calib_fpr_at_tau": round(calib_fpr, 6)}
    s2 = {"pass": auc_gv is not None and auc_gv >= 0.90,
          "bar": 0.90, "auc": round(auc_gv, 6) if auc_gv is not None else None,
          "n_fake": len(gvf), "n_real": len(er)}
    s3 = {"pass": auc_gb is not None and auc_gb >= 0.85,
          "bar": 0.85, "auc": round(auc_gb, 6) if auc_gb is not None else None,
          "n_fake": len(gbf), "n_real": len(er)}
    s4 = {"pass": fpr is not None and fpr <= 0.10,
          "bar": 0.10, "fpr": round(fpr, 6) if fpr is not None else None,
          "k": fpr_k, "n_real": len(er),
          "wilson95": [round(x, 6) for x in fpr_ci] if fpr_ci else None}
    s5 = {"pass": catch >= 15, "bar": 15, "catch": catch,
          "n_issued": len(issued), "n_fixture": len(fx)}
    s6 = {"pass": fpr_j is not None and fpr_j <= 0.12,
          "bar": 0.12, "fpr": round(fpr_j, 6) if fpr_j is not None else None,
          "k": fpr_j_k, "n_real": len(jr),
          "wilson95": [round(x, 6) for x in fpr_j_ci] if fpr_j_ci else None,
          "launder": "frame_jpeg_q80", "tau_unchanged": True}
    s7 = {"pass": recall_gv is not None and recall_gv >= 0.50,
          "bar": 0.50,
          "recall_genvideo": round(recall_gv, 6)
          if recall_gv is not None else None,
          "recall_genvidbench": round(recall_gb, 6)
          if recall_gb is not None else None,
          "n_fake": len(gvf)}
    mr = _motion_rebalanced(er + gvf)
    fps_corr = {"pooled": _corr(er + gvf), "real": _corr(er),
                "fake": _corr(gvf)}
    s8 = {"pass": mr["auc"] is not None,
          "bar": "report-only", "motion_rebalanced_auc": mr["auc"],
          "n_used_bins": mr["n_used"], "n_skipped_bins": mr["n_skipped"],
          "fps_correlation": fps_corr}

    bars = {"S0": s0, "S1": s1, "S2": s2, "S3": s3, "S4": s4,
            "S5": s5, "S6": s6, "S7": s7, "S8": s8}
    if decode_fail:
        decision = "SPLIT_CLOSED_DECODE"
    elif not s0["pass"]:
        decision = "SPLIT_CLOSED_S0"
    elif not s1["pass"]:
        decision = "SPLIT_CLOSED_S1"
    elif not s2["pass"]:
        decision = "SPLIT_CLOSED_S2"
    elif not s3["pass"]:
        decision = "SPLIT_CLOSED_S3"
    elif not s4["pass"]:
        decision = "SPLIT_CLOSED_S4"
    elif not s5["pass"]:
        decision = "SPLIT_CLOSED_S5"
    elif not s6["pass"]:
        decision = "SPLIT_CLOSED_S6"
    elif not s7["pass"]:
        decision = "SPLIT_CLOSED_S7"
    elif not s8["pass"]:
        decision = "SPLIT_CLOSED_S8"
    else:
        decision = "SPLIT_PASS_PENDING_WIRING_SPEC"

    per_generator = {}
    for arm, pool in (("genvideo_fake", gvf), ("genvid_fake", gbf)):
        arm_rows = _arm_rows(rows, arm)
        per_generator[arm] = {}
        for gen in sorted({r["generator"] for r in arm_rows}):
            gf = [r for r in pool if r["generator"] == gen]
            per_generator[arm][gen] = {
                "n": sum(1 for r in arm_rows if r["generator"] == gen),
                "n_available": len(gf),
                "auc": round(_auc(gf, er), 6)
                if _auc(gf, er) is not None else None,
                "recall_at_tau": round(float(np.mean(
                    [r["score"] >= tau for r in gf])), 6) if gf else None,
            }

    current_hashes = _deployed_hashes()
    mp = out / "MANIFEST.json"
    if not mp.exists():
        print(f"FAIL: {mp} missing — run --prepare to record the deployed "
              f"artifact baseline", file=sys.stderr)
        return 1
    try:
        baseline = _load_json(mp).get("deployed_artifact_sha256")
    except RuntimeError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if not baseline:
        print(f"FAIL: {mp} has no deployed_artifact_sha256 baseline",
              file=sys.stderr)
        return 1
    unchanged = current_hashes == baseline

    metrics = {
        "exam": "SPLIT Stage 1 content-matched GenVideo rerun",
        "stage": 1,
        "date": _date(),
        "created_at": _iso(),
        "encoder": st.ENCODER,
        "recipe_hash": st.split_recipe_hash(),
        "code_sha256": st.CODE_SHA,
        "policy": st.POLICY,
        "seeds": splits.get("seeds", SEEDS),
        "counts": {"calib": len(calib), "eval_real": len(er),
                   "genvideo_fake": len(gvf), "genvid_fake": len(gbf),
                   "fixture": len(fx), "fixture_issued": len(issued),
                   "eval_real_jpeg80": len(jr)},
        "n_unavailable": unavailable,
        "decode_fail_rate": {k: round(v, 6) for k, v in decode.items()},
        "tau": tau,
        "tau_alpha": ALPHA,
        "tau_method": "higher",
        "bars": bars,
        "per_generator": per_generator,
        "fake_generator_counts": splits.get("fake_generator_counts", {}),
        "empty_strata": splits.get("empty_strata", []),
        "strata_realized": splits.get("strata_realized"),
        "motion_controls": {"motion_rebalanced_auc": mr,
                            "fps_correlation": fps_corr,
                            "fps_distribution": {
                                "pooled": _fps_summary(er + gvf),
                                "real": _fps_summary(er),
                                "fake": _fps_summary(gvf)}},
        "forwards": {"planned": 2466, "spec_estimate": 1766,
                     "note": "S6 adds 700 compressed-real forwards"},
        "deployed_artifact_sha256": current_hashes,
        "deployed_artifact_sha256_baseline": baseline,
        "deployed_artifact_sha256_unchanged": unchanged,
        "decision": decision,
        "wiring": "none",
        "license": "CC-BY-NC-4.0 (GenVideo-Val, GenVidBench; EXAM-ONLY)",
    }
    verdict_doc = {
        "decision": decision,
        "stage": 1,
        "bars": bars,
        "tau": tau,
        "recipe_hash": metrics["recipe_hash"],
        "empty_strata": metrics["empty_strata"],
        "deployed_artifact_sha256_unchanged": unchanged,
        "wiring": "none",
        "date": _date(),
    }
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "verdict.json", verdict_doc)

    for key in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"):
        b = bars[key]
        state = "PASS" if b["pass"] else "FAIL"
        print(f"{key} {state} {json.dumps({k: v for k, v in b.items() if k != 'pass'})}")
    print(f"tau={tau} decision={decision}")
    return 0


# ---------------------------------------------------------------- smoke

def _smoke_jobs_stage0(out: Path) -> list:
    sp = out / "splits.json"
    if sp.exists():
        try:
            splits = _load_json(sp)
            real = [e for e in splits["clips"] if e["label"] == 0][:8]
            fake = [e for e in splits["clips"] if e["label"] == 1][:8]
            return [{"arm": e["arm"], "generator": e["generator"],
                     "label": e["label"], "path": Path(e["path"]),
                     "filename": e["filename"]} for e in real + fake]
        except RuntimeError:
            pass
    real = list_clips(GENBUSTER / "real")[:8]
    fake = list_clips(GENBUSTER / "fake")[:8]
    return ([{"arm": "stage0_real", "generator": "genbuster_real",
              "label": 0, "path": p, "filename": p.name} for p in real]
            + [{"arm": "stage0_fake", "generator": "genbuster_fake",
                "label": 1, "path": p, "filename": p.name} for p in fake])


def _smoke_jobs_stage1(out: Path) -> list:
    sp = out / "splits.json"
    if sp.exists():
        try:
            splits = _load_json(sp)
            calib = splits["calib"][:8]
            fakes, seen = [], set()
            for e in splits["genvideo_fakes"]:
                if e["generator"] in seen:
                    continue
                seen.add(e["generator"])
                fakes.append(e)
                if len(fakes) == 8:
                    break
            if len(fakes) < 8:
                fakes = splits["genvideo_fakes"][:8]
            return ([{"arm": "calib", "generator": e["generator"],
                      "label": 0, "path": Path(e["path"]),
                      "filename": e["filename"]} for e in calib]
                    + [{"arm": "genvideo_fake", "generator": e["generator"],
                        "label": 1, "path": Path(e["path"]),
                        "filename": e["filename"]} for e in fakes])
        except RuntimeError:
            pass
    real = list_clips(REAL_DIR)[:8]
    fakes = []
    for gen in GENS:
        d = GENVIDEO / gen
        if d.is_dir() and list_clips(d):
            p = list_clips(d)[0]
            fakes.append({"arm": "genvideo_fake", "generator": gen,
                          "label": 1, "path": p, "filename": p.name})
        if len(fakes) == 8:
            break
    return ([{"arm": "calib", "generator": "msrvtt", "label": 0,
              "path": p, "filename": p.name} for p in real] + fakes)


def smoke(out: Path, stage: int) -> int:
    if st._device() != "cpu":
        print("FAIL: SPLIT is CPU-only; refusing non-CPU device",
              file=sys.stderr)
        return 1
    jobs = _smoke_jobs_stage0(out) if stage == 0 else _smoke_jobs_stage1(out)
    if len(jobs) != N_SMOKE:
        print(f"FAIL: smoke has {len(jobs)} clips, expected {N_SMOKE}",
              file=sys.stderr)
        return 1
    rows = score_rows(jobs, out / "smoke" / "scores.jsonl",
                      out / "smoke" / "smoke.log")
    if not rows:
        return 1
    ok, detail = _smoke_ok(out)
    if not ok:
        print(f"FAIL: {detail}", file=sys.stderr)
        return 1
    print(f"SMOKE OK — {detail} -> {out / 'smoke' / 'scores.jsonl'}")
    return 0


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SPLIT matched T2V exam (CPU-only; no wiring)")
    ap.add_argument("--stage", type=int, choices=(0, 1), default=0,
                    help="0 = GenBuster++ motion gate; 1 = GenVideo rerun")
    ap.add_argument("--prepare", action="store_true",
                    help="write splits.json + MANIFEST.json")
    ap.add_argument("--smoke", action="store_true",
                    help="16-clip pipeline smoke into <out>/smoke")
    ap.add_argument("--score", action="store_true",
                    help="score all arms into <out>/scores.jsonl (resumable)")
    ap.add_argument("--verdict", action="store_true",
                    help="write metrics.json + verdict.json from scores")
    ap.add_argument("--out", default=None,
                    help="default: artifacts/split_t2v_matched/stage<stage>")
    ap.add_argument("--force", action="store_true",
                    help="resume/append and overwrite metrics/verdict")
    ap.add_argument("--allow-empty-strata", action="store_true",
                    help="Stage 1 only: write a split with named empty "
                         "strata instead of failing (never silent)")
    ap.add_argument("--real-dir", default=str(REAL_DIR))
    ap.add_argument("--genvideo-dir", default=str(GENVIDEO))
    ap.add_argument("--genvidbench-dir", default=str(GENVIDBENCH))
    ap.add_argument("--d3-dir", default=str(D3_DIR),
                    help="D3 scores.jsonl for SHA-256 exclusion only")
    args = ap.parse_args()

    out = Path(args.out) if args.out else DEFAULT_OUT / f"stage{args.stage}"
    if args.smoke:
        if args.prepare or args.score or args.verdict:
            ap.error("--smoke is standalone")
        if not guard(out, args.force):
            return 1
        return smoke(out, args.stage)

    if not any((args.prepare, args.score, args.verdict)):
        ap.error("choose --prepare, --smoke, --score, or --verdict")
    if not guard(out, args.force):
        return 1
    if args.prepare:
        if args.stage == 0:
            rc = prepare_stage0(out)
        else:
            rc = prepare_stage1(out, Path(args.real_dir),
                                Path(args.genvideo_dir),
                                Path(args.genvidbench_dir),
                                Path(args.d3_dir),
                                args.allow_empty_strata)
        if rc:
            return rc
    if args.score:
        rc = score_stage0(out) if args.stage == 0 else score_stage1(out)
        if rc:
            return rc
    if args.verdict:
        rc = verdict_stage0(out) if args.stage == 0 else verdict_stage1(out)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
