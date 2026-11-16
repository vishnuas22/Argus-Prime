"""SPLIT Stage 0 T=16 exam — pre-registration artifacts/split_prereg/SPEC_T16_STAGE0.md.

GenBuster++ video/real vs video/fake, 1,000 / 1,000, T=16 over a 2.0 s
deterministic t=0 window, AUC only.

  PASS >= 0.70 · HOLD 0.60-0.70 · KILL < 0.60
  >10% unavailable in either arm -> INCONCLUSIVE_DECODE

Mandatory controls: per-clip motion metric, 100-bin motion-rebalanced AUC,
motion-only AUC (both polarities), source-fps/score correlation, v1 T=8
Stage 0 reference (context only), D3 reference (context only).

Measurement only: CPU-only, no wiring, no training, no pointer movement,
no commit. The frozen v1 T=8 SPLIT module and its exam are never imported.

  .venv/bin/python deepfake/scripts/exam_split_stage0_t16.py --prepare
  .venv/bin/python deepfake/scripts/exam_split_stage0_t16.py --seal
  .venv/bin/python deepfake/scripts/exam_split_stage0_t16.py --pilot
  .venv/bin/python deepfake/scripts/exam_split_stage0_t16.py --smoke
  .venv/bin/python deepfake/scripts/exam_split_stage0_t16.py --score
  .venv/bin/python deepfake/scripts/exam_split_stage0_t16.py --verdict
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
for p in ("deepfake", "deepfake/bench", "deepfake/scripts"):
    if str(WS / p) not in sys.path:
        sys.path.insert(0, str(WS / p))

import split_t16_temporal as st  # noqa: E402

GENBUSTER = WS / "data" / "raw" / "genbuster_plusplus" / "extracted" / "video"
GENBUSTER_ZIP = WS / "data" / "raw" / "genbuster_plusplus" / \
    "GenBuster-Bench-plusplus.zip"
DEFAULT_OUT = WS / "artifacts" / "split_t2v_matched" / "stage0_t16"
SPEC_PATH = WS / "artifacts" / "split_prereg" / "SPEC_T16_STAGE0.md"
MODULE_PATH = WS / "deepfake" / "bench" / "split_t16_temporal.py"
EXAM_PATH = Path(__file__).resolve()
TEST_PATH = WS / "tests" / "test_split_t16_temporal.py"
LOG = WS / "logs" / "split_stage0_t16.log"

N_STAGE0 = 1000
N_PILOT = 20
N_PILOT_ARM = 10
N_SMOKE = 16
N_SMOKE_ARM = 8
N_BINS = 100
N_BOOT = 10000
BOOT_SEED = 20261116
PILOT_STOP_H = 3.0
ARMS = {"stage0_real": 0, "stage0_fake": 1}
V1_REFERENCE = {
    "auc": 0.70244,
    "motion_rebalanced_auc": 0.736812,
    "motion_only_auc_higher_motion_fake": 0.495853,
    "motion_only_auc_higher_motion_real": 0.504147,
    "runtime_s": 1439.12,
    "runtime_wall_min": 24.7,
    "source": ("artifacts/split_t2v_matched/stage0/metrics.json; "
               "frozen v1 T=8 Stage 0, context only, not a gate"),
    "is_gate": False,
}
D3_REFERENCE_AUC = 0.4843
D3_REFERENCE_SOURCE = ("artifacts/audit_20261106 motion-bias audit, "
                       "GenBuster++ run; context only, not a gate")


class ExamError(RuntimeError):
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
        raise ExamError(f"corrupt {path}: {e}") from e


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


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=WS,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _xclip_files() -> tuple:
    """Locate the cached HF snapshot files used by _get_encoder."""
    hubs = []
    if os.environ.get("HF_HUB_CACHE"):
        hubs.append(Path(os.environ["HF_HUB_CACHE"]))
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" /
                                             "huggingface")))
    hubs.append(home / "hub")
    hubs.append(Path.home() / ".cache" / "huggingface" / "hub")
    seen = []
    for hub in hubs:
        if hub in seen:
            continue
        seen.append(hub)
        snaps = hub / "models--microsoft--xclip-base-patch16" / "snapshots"
        if not snaps.is_dir():
            continue
        cands = sorted(d for d in snaps.iterdir()
                       if (d / "config.json").is_file()
                       and (d / "model.safetensors").is_file())
        if cands:
            snap = cands[-1]
            return snap / "config.json", snap / "model.safetensors"
    raise ExamError("cached microsoft/xclip-base-patch16 config.json + "
                    "model.safetensors not found in the HF hub cache")


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
            if sha != job.get("sha256"):
                print(f"FAIL: clip changed since prepare: {path} "
                      f"({job.get('sha256')} -> {sha})", file=sys.stderr)
                return []
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
            res = st.score_clip(path)
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


def _bootstrap_auc_ci(fakes: list, reals: list,
                      n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """Stratified percentile bootstrap, report-only (cannot change verdict)."""
    from sklearn.metrics import roc_auc_score
    f = np.asarray([r["score"] for r in fakes if r.get("score") is not None],
                   dtype=np.float64)
    r = np.asarray([r["score"] for r in reals if r.get("score") is not None],
                   dtype=np.float64)
    if f.size == 0 or r.size == 0:
        return {"n_boot": 0, "seed": seed, "ci95": None,
                "note": "empty arm"}
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
    note = None
    if np.std(y) == 0:
        return {"n": len(pairs), "pearson": None, "spearman": None,
                "note": "constant y (score)"}
    pearson = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(pearson):
        pearson = None
        note = "pearson non-finite"
    try:
        from scipy.stats import spearmanr
        sp = spearmanr(x, y).statistic
        spearman = float(sp) if np.isfinite(sp) else None
    except Exception:
        spearman = None
    if spearman is None and note is None:
        note = "spearman unavailable/non-finite"
    out = {"n": len(pairs), "pearson": pearson, "spearman": spearman}
    if note:
        out["note"] = note
    return out


def _fps_summary(rows: list) -> dict:
    vals = [float(r["fps"]) for r in rows if r.get("fps") is not None]
    if not vals:
        return {"n": 0, "unique": [], "min": None, "max": None}
    return {"n": len(vals), "unique": sorted({round(v, 6) for v in vals}),
            "min": min(vals), "max": max(vals)}


def _arm_rows(rows: list, arm: str) -> list:
    return [r for r in rows if r["arm"] == arm]


def _avail(rows: list) -> list:
    return [r for r in rows if r.get("score") is not None]


def _expected_rows(splits: dict) -> tuple:
    """Derive the exact expected score-row set from splits.json."""
    expected, problems = {}, []
    for e in splits["clips"]:
        arm = e.get("arm")
        if arm not in ARMS:
            problems.append(f"unknown arm {arm!r} for {e.get('filename')}")
            continue
        label = ARMS[arm]
        if int(e.get("label", -1)) != label:
            problems.append(f"clip label {e.get('label')} != arm-derived "
                            f"{label} for {e.get('filename')}")
        key = (arm, e.get("generator"), e.get("filename"))
        if key in expected:
            problems.append(f"duplicate expected key {key}")
        expected[key] = {"path": e.get("path"), "sha256": e.get("sha256"),
                         "label": label}
    return expected, problems


def _validate_score_rows(rows: list, splits: dict) -> list:
    """Exact row-set binding: no duplicates, no foreign/missing keys, and
    path/sha256/label equality for every expected key. Arm counts follow."""
    expected, problems = _expected_rows(splits)
    actual, dups = {}, []
    for r in rows:
        key = (r.get("arm"), r.get("generator"), r.get("filename"))
        if key in actual:
            dups.append(key)
        actual[key] = r
    if dups:
        problems.append(f"duplicate score keys: {dups[:5]}")
    foreign = sorted(set(actual) - set(expected), key=repr)
    missing = sorted(set(expected) - set(actual), key=repr)
    if foreign:
        problems.append(f"foreign score keys: {foreign[:5]}")
    if missing:
        problems.append(f"missing score keys: {missing[:5]}")
    for key in sorted(set(expected) & set(actual), key=repr):
        want, got = expected[key], actual[key]
        for field in ("path", "sha256", "label"):
            if got.get(field) != want[field]:
                problems.append(f"{field} mismatch for {key}: "
                                f"{got.get(field)!r} != {want[field]!r}")
    exp_counts = Counter(k[0] for k in expected)
    act_counts = Counter(k[0] for k in actual)
    if exp_counts != act_counts:
        problems.append(f"arm counts {dict(act_counts)} != "
                        f"{dict(exp_counts)}")
    return problems


# ---------------------------------------------------------------- prepare

def prepare(out: Path) -> int:
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
        "t": st.T,
        "window_s": st.WINDOW_S,
        "created_at": _iso(),
        "spec": str(SPEC_PATH),
        "corpus": {"video": str(GENBUSTER), "real_dir": str(real_dir),
                   "fake_dir": str(fake_dir), "license": "MIT",
                   "role": "measurement-only", "video_only": True,
                   "archive": archive},
        "expected_counts": {"real": N_STAGE0, "fake": N_STAGE0},
        "counts": {"real": len(real_e), "fake": len(fake_e)},
        "expected_counts_verified": verified,
        "clips": real_e + fake_e,
    }
    _write_json(out / "splits.json", splits)

    manifest = {
        "exam": "SPLIT Stage 0 T=16 motion gate (paper frame budget)",
        "created_at": _iso(),
        "out": str(out),
        "stage": 0,
        "encoder": st.ENCODER,
        "encoder_hf": st.ENCODER_HF,
        "native_num_frames": st.NATIVE_NUM_FRAMES,
        "grouping_policy": st.GROUPING,
        "recipe_hash": st.split_recipe_hash(),
        "code_sha256": st.CODE_SHA,
        "policy": st.POLICY,
        "corpora": {"genbuster_plusplus": {
            "path": str(GENBUSTER), "license": "MIT",
            "role": "measurement-only", "video_only": True,
            "counts": {"real": len(real_e), "fake": len(fake_e)},
            "archive": archive}},
        "wiring": "none",
    }
    _write_json(out / "MANIFEST.json", manifest)
    print(f"PREPARE OK: stage=0 T={st.T} window={st.WINDOW_S}s "
          f"real={len(real_e)} fake={len(fake_e)} verified={verified} "
          f"-> {out / 'splits.json'}")
    return 0


# ---------------------------------------------------------------- seal

def _input_hashes_lines(splits: dict) -> list:
    return [f"{e['sha256']}  {e['path']}"
            for e in sorted(splits["clips"], key=lambda e: e["path"])]


def write_seal(out: Path) -> int:
    blocked = [p for p in (out / "scores.jsonl", out / "metrics.json",
                           out / "verdict.json") if p.exists()]
    if blocked:
        print(f"FAIL: seal must precede scoring; found "
              f"{', '.join(str(p) for p in blocked)} — refusing to re-seal",
              file=sys.stderr)
        return 1
    sp = out / "splits.json"
    mp = out / "MANIFEST.json"
    if not sp.exists() or not mp.exists():
        print(f"ERROR: {sp} / {mp} missing — run --prepare first",
              file=sys.stderr)
        return 1
    try:
        splits = _load_json(sp)
        manifest = _load_json(mp)
        config_path, weights_path = _xclip_files()
    except ExamError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if manifest.get("recipe_hash") != st.split_recipe_hash():
        print("FAIL: MANIFEST recipe_hash != current recipe; re-run "
              "--prepare --force", file=sys.stderr)
        return 1

    for e in splits["clips"]:
        p = Path(e["path"])
        if not p.exists():
            print(f"FAIL: clip missing at seal time: {p}", file=sys.stderr)
            return 1
        if _sha256(p) != e["sha256"]:
            print(f"FAIL: clip changed since prepare: {p}", file=sys.stderr)
            return 1

    lines = _input_hashes_lines(splits)
    ih = out / "input_hashes.txt"
    ih.write_text("\n".join(lines) + "\n")

    archive = (splits.get("corpus") or {}).get("archive") or {}
    seal = {
        "created_utc": _iso(),
        "git_head": _git_head(),
        "spec_sha256": _sha256(SPEC_PATH),
        "module_sha256": _sha256(MODULE_PATH),
        "exam_sha256": _sha256(EXAM_PATH),
        "test_sha256": _sha256(TEST_PATH),
        "splits_sha256": _sha256(sp),
        "manifest_sha256": _sha256(mp),
        "input_hashes_sha256": _sha256(ih),
        "corpus_archive_sha256": archive.get("sha256"),
        "xclip_config_sha256": _sha256(config_path),
        "xclip_weights_sha256": _sha256(weights_path),
        "xclip_config_path": str(config_path),
        "xclip_weights_path": str(weights_path),
        "recipe_hash": st.split_recipe_hash(),
        "native_num_frames": st.NATIVE_NUM_FRAMES,
        "grouping_policy": st.GROUPING,
        "counts": dict(splits["counts"]),
        "n_input_hashes": len(lines),
        "policy": st.POLICY,
    }
    _write_json(out / "seal.txt", seal)
    print(f"SEAL OK: recipe={seal['recipe_hash']} "
          f"inputs={seal['input_hashes_sha256']} "
          f"module={seal['module_sha256']} exam={seal['exam_sha256']} "
          f"test={seal['test_sha256']} -> {out / 'seal.txt'}")
    return 0


def seal_problems(out: Path) -> list:
    """Return the list of reasons the current files do not match the seal."""
    p = out / "seal.txt"
    if not p.exists():
        return [f"{p} missing — run --seal first"]
    try:
        seal = _load_json(p)
    except ExamError as e:
        return [str(e)]
    problems = []
    checks = (
        ("spec_sha256", SPEC_PATH),
        ("module_sha256", MODULE_PATH),
        ("exam_sha256", EXAM_PATH),
        ("test_sha256", TEST_PATH),
        ("splits_sha256", out / "splits.json"),
        ("manifest_sha256", out / "MANIFEST.json"),
    )
    for key, path in checks:
        if not path.exists():
            problems.append(f"{key}: {path} missing")
            continue
        if _sha256(path) != seal.get(key):
            problems.append(f"{key} mismatch: {path} changed since seal")
    ih = out / "input_hashes.txt"
    if not ih.exists():
        problems.append(f"{ih} missing")
    elif _sha256(ih) != seal.get("input_hashes_sha256"):
        problems.append("input_hashes_sha256 mismatch: file changed since seal")
    if seal.get("recipe_hash") != st.split_recipe_hash():
        problems.append("recipe_hash mismatch: recipe changed since seal")
    if seal.get("native_num_frames") != st.NATIVE_NUM_FRAMES:
        problems.append("native_num_frames changed since seal")
    try:
        splits = _load_json(out / "splits.json")
    except ExamError as e:
        problems.append(str(e))
        return problems
    if dict(seal.get("counts") or {}) != dict(splits.get("counts") or {}):
        problems.append("seal counts != splits counts")
    archive = (splits.get("corpus") or {}).get("archive") or {}
    if archive.get("sha256") != seal.get("corpus_archive_sha256"):
        problems.append("corpus_archive_sha256 mismatch")
    try:
        config_path, weights_path = _xclip_files()
        if _sha256(config_path) != seal.get("xclip_config_sha256"):
            problems.append("xclip_config_sha256 mismatch")
        if _sha256(weights_path) != seal.get("xclip_weights_sha256"):
            problems.append("xclip_weights_sha256 mismatch")
    except ExamError as e:
        problems.append(str(e))
    if ih.exists() and len(ih.read_text().splitlines()) != seal.get(
            "n_input_hashes"):
        problems.append("input_hashes.txt line count != seal")
    return problems


def _require_seal(out: Path) -> bool:
    problems = seal_problems(out)
    if problems:
        print("FAIL: seal invalid — refusing to score:", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        return False
    return True


# ---------------------------------------------------------------- jobs

def _jobs_from_clips(clips: list) -> list:
    return [{"arm": e["arm"], "generator": e["generator"],
             "label": e["label"], "path": Path(e["path"]),
             "filename": e["filename"], "sha256": e["sha256"]}
            for e in clips]


def _load_splits(out: Path) -> dict:
    sp = out / "splits.json"
    if not sp.exists():
        raise ExamError(f"{sp} missing — run --prepare first")
    return _load_json(sp)


def _pilot_jobs(splits: dict) -> list:
    real = [e for e in splits["clips"] if e["label"] == 0]
    fake = [e for e in splits["clips"] if e["label"] == 1]
    sel = real[:N_PILOT_ARM] + fake[:N_PILOT_ARM]
    return _jobs_from_clips(sel)


def _smoke_jobs(splits: dict) -> list:
    real = [e for e in splits["clips"] if e["label"] == 0][:N_SMOKE_ARM]
    fake = [e for e in splits["clips"] if e["label"] == 1][:N_SMOKE_ARM]
    return _jobs_from_clips(real + fake)


# ---------------------------------------------------------------- pilot

def pilot(out: Path) -> int:
    if st._device() != "cpu":
        print("FAIL: SPLIT T=16 is CPU-only; refusing non-CPU device",
              file=sys.stderr)
        return 1
    if not _require_seal(out):
        return 1
    try:
        jobs = _pilot_jobs(_load_splits(out))
    except ExamError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if len(jobs) != N_PILOT:
        print(f"FAIL: pilot has {len(jobs)} clips, expected {N_PILOT}",
              file=sys.stderr)
        return 1
    rows, lats = [], []
    for i, job in enumerate(jobs):
        t0 = time.perf_counter()
        res = st.score_clip(job["path"])
        wall = time.perf_counter() - t0
        lats.append(wall)
        rows.append({"filename": job["filename"], "arm": job["arm"],
                     "label": job["label"], "latency_s": round(wall, 3),
                     "score": res.get("score"),
                     "fps": res.get("fps"),
                     "unavailable_reason": res.get("unavailable_reason")})
        print(f"[pilot {i + 1}/{len(jobs)}] {job['arm']}/{job['filename']} "
              f"{wall:.3f}s score={res.get('score')} "
              f"reason={res.get('unavailable_reason')}", flush=True)
    lat = np.asarray(lats, dtype=np.float64)
    median = float(np.median(lat))
    p90 = float(np.percentile(lat, 90))
    mean = float(np.mean(lat))
    proj_mean = mean * N_STAGE0 * 2
    proj_median = median * N_STAGE0 * 2
    tripped = proj_mean > PILOT_STOP_H * 3600.0
    doc = {
        "created_at": _iso(),
        "recipe_hash": st.split_recipe_hash(),
        "n": len(rows),
        "selection": "first_10_sorted_per_arm",
        "latency_s": {"median": round(median, 3), "p90": round(p90, 3),
                      "mean": round(mean, 3), "min": round(float(lat.min()), 3),
                      "max": round(float(lat.max()), 3)},
        "projected_wall_s": round(proj_mean, 1),
        "projected_wall_h": round(proj_mean / 3600.0, 3),
        "projected_wall_s_median": round(proj_median, 1),
        "projected_wall_h_median": round(proj_median / 3600.0, 3),
        "stop_rule_h": PILOT_STOP_H,
        "tripped": bool(tripped),
        "rows": rows,
    }
    _write_json(out / "pilot" / "pilot.json", doc)
    print(f"PILOT median={median:.3f}s p90={p90:.3f}s mean={mean:.3f}s "
          f"projected={proj_mean / 3600.0:.3f}h "
          f"(median-based {proj_median / 3600.0:.3f}h) "
          f"stop_rule={PILOT_STOP_H}h tripped={tripped}")
    if tripped:
        print(f"STOP: projected wall time {proj_mean / 3600.0:.3f}h > "
              f"{PILOT_STOP_H}h — do not start the full run; return for "
              f"approval", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------- smoke

def smoke(out: Path) -> int:
    if st._device() != "cpu":
        print("FAIL: SPLIT T=16 is CPU-only; refusing non-CPU device",
              file=sys.stderr)
        return 1
    if not _require_seal(out):
        return 1
    try:
        jobs = _smoke_jobs(_load_splits(out))
    except ExamError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    if len(jobs) != N_SMOKE:
        print(f"FAIL: smoke has {len(jobs)} clips, expected {N_SMOKE}",
              file=sys.stderr)
        return 1
    sp = out / "smoke" / "scores.jsonl"
    if sp.exists():
        sp.unlink()
    rows = score_rows(jobs, sp, out / "smoke" / "smoke.log")
    if not rows:
        return 1
    want_shape = [st.T, st.PATCHES, st.DIM]
    bad = [r["filename"] for r in rows
           if (r.get("score") is None or r.get("ttr") is None
               or r.get("lsmi") is None or r.get("motion") is None
               or r.get("feature_shape") != want_shape)]
    if bad:
        print(f"FAIL: non-finite/invalid smoke rows: {bad[:5]}",
              file=sys.stderr)
        return 1
    first = rows[0]
    repeat = st.score_clip(Path(first["path"]))
    keys = ("score", "ttr", "lsmi", "motion", "frame_indices", "n_decoded",
            "fps", "feature_shape")
    diff = [k for k in keys if first.get(k) != repeat.get(k)]
    if diff:
        print(f"FAIL: smoke determinism broken on {first['filename']}: "
              f"{diff}", file=sys.stderr)
        return 1
    doc = {
        "created_at": _iso(),
        "recipe_hash": st.split_recipe_hash(),
        "n": len(rows),
        "patch_shape": want_shape,
        "all_finite": True,
        "determinism": {"filename": first["filename"],
                        "fields": list(keys), "identical": True},
        "pass": True,
    }
    _write_json(out / "smoke" / "smoke.json", doc)
    print(f"SMOKE OK — {len(rows)}/{N_SMOKE} finite {want_shape}; "
          f"deterministic repeat on {first['filename']} -> "
          f"{out / 'smoke' / 'smoke.json'}")
    return 0


# ---------------------------------------------------------------- score

def score(out: Path) -> int:
    if st._device() != "cpu":
        print("FAIL: SPLIT T=16 is CPU-only; refusing non-CPU device",
              file=sys.stderr)
        return 1
    if not _require_seal(out):
        return 1
    try:
        splits = _load_splits(out)
    except ExamError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    jobs = _jobs_from_clips(splits["clips"])
    if len(jobs) != sum(splits["counts"].values()):
        print(f"FAIL: job count {len(jobs)} != splits counts "
              f"{splits['counts']}", file=sys.stderr)
        return 1
    rows = score_rows(jobs, _scores_path(out), LOG)
    if not rows:
        return 1
    counts = Counter(r["arm"] for r in rows)
    print(f"SCORE OK: stage=0 T=16 rows={len(rows)} arms={dict(counts)} "
          f"-> {_scores_path(out)}")
    return 0


# ---------------------------------------------------------------- verdict

def verdict(out: Path) -> int:
    if not _require_seal(out):
        return 1
    try:
        splits = _load_splits(out)
    except ExamError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    rows = _load_scores(_scores_path(out))
    if not rows:
        return 1
    if not _recipe_ok(rows, str(_scores_path(out))):
        return 1
    row_problems = _validate_score_rows(rows, splits)
    if row_problems:
        print("FAIL: score rows do not match the sealed split:",
              file=sys.stderr)
        for item in row_problems[:10]:
            print(f"  - {item}", file=sys.stderr)
        return 1
    expected = {"stage0_real": splits["counts"]["real"],
                "stage0_fake": splits["counts"]["fake"]}

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
    boot = _bootstrap_auc_ci(fakes, reals)
    mr = _motion_rebalanced(avail)
    motion_auc = _auc(fakes, reals, key="motion")
    motion_inverted = (1.0 - motion_auc) if motion_auc is not None else None
    fps_corr = {
        "pooled": _corr(avail),
        "real": _corr(reals),
        "fake": _corr(fakes),
    }

    bars = {"pass_min": 0.70, "hold_min": 0.60, "kill_below": 0.60}
    inconclusive = any(v > 0.10 for v in decode.values())
    if inconclusive or auc is None:
        decision = "SPLIT_T16_STAGE0_INCONCLUSIVE_DECODE"
    elif auc >= 0.70:
        decision = "SPLIT_T16_STAGE0_PASS"
    elif auc >= 0.60:
        decision = "SPLIT_T16_STAGE0_HOLD"
    else:
        decision = "SPLIT_T16_STAGE0_KILL"

    ci = boot.get("ci95")
    ci_crosses = None
    if ci:
        ci_lo, ci_hi = float(ci[0]), float(ci[1])
        ci_crosses = (ci_lo < 0.70 < ci_hi) or (ci_lo < 0.60 < ci_hi)

    try:
        seal = _load_json(out / "seal.txt")
    except ExamError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    metrics = {
        "exam": "SPLIT Stage 0 T=16 motion gate (paper frame budget)",
        "stage": 0,
        "date": _date(),
        "created_at": _iso(),
        "encoder": st.ENCODER,
        "encoder_hf": st.ENCODER_HF,
        "native_num_frames": st.NATIVE_NUM_FRAMES,
        "grouping_policy": st.GROUPING,
        "recipe_hash": st.split_recipe_hash(),
        "code_sha256": st.CODE_SHA,
        "policy": st.POLICY,
        "spec_sha256": seal.get("spec_sha256"),
        "splits_sha256": seal.get("splits_sha256"),
        "manifest_sha256": seal.get("manifest_sha256"),
        "seal_input_hashes_sha256": seal.get("input_hashes_sha256"),
        "corpus": splits.get("corpus", {}),
        "expected_counts_verified": splits.get("expected_counts_verified"),
        "counts": {"real": len(reals), "fake": len(fakes),
                   "n_scored": len(rows), "n_available": len(avail),
                   "n_unavailable": len(rows) - len(avail)},
        "n_unavailable": unavailable,
        "decode_fail_rate": {k: round(v, 6) for k, v in decode.items()},
        "auc": auc,
        "auc_rounded_6": round(auc, 6) if auc is not None else None,
        "auc_bootstrap": boot,
        "ci_crosses_bar": ci_crosses,
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
        "v1_t8_reference": V1_REFERENCE,
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
        "auc_rounded_6": metrics["auc_rounded_6"],
        "auc_bootstrap": boot,
        "ci_crosses_bar": ci_crosses,
        "bars": bars,
        "motion_rebalanced_auc": mr["auc"],
        "motion_only_auc": metrics["motion_controls"]["motion_only_auc"],
        "fps_correlation": fps_corr,
        "v1_t8_reference_auc": V1_REFERENCE["auc"],
        "d3_reference_auc": D3_REFERENCE_AUC,
        "recipe_hash": metrics["recipe_hash"],
        "date": _date(),
        "wiring": "none",
    }
    _write_json(out / "metrics.json", metrics)
    _write_json(out / "verdict.json", verdict_doc)

    print(f"Stage 0 T=16 AUC={metrics['auc']} "
          f"(PASS>=0.70, HOLD>=0.60, KILL<0.60) decision={decision}")
    print(f"  bootstrap 95% CI={ci} (report-only; crosses_bar={ci_crosses})")
    print(f"  motion-rebalanced AUC={mr['auc']} "
          f"used={mr['n_used']}/{mr['n_bins_nonempty']} bins")
    print(f"  motion-only AUC={motion_auc} "
          f"inverted={motion_inverted} (negative control)")
    print(f"  fps corr pooled={fps_corr['pooled']} real={fps_corr['real']}")
    print(f"  v1 T=8 reference AUC={V1_REFERENCE['auc']} (context only)")
    print(f"  D3 reference AUC={D3_REFERENCE_AUC} (context only)")
    return 0


# ---------------------------------------------------------------- CLI

def main() -> int:
    ap = argparse.ArgumentParser(
        description="SPLIT Stage 0 T=16 exam (CPU-only; no wiring)")
    ap.add_argument("--prepare", action="store_true",
                    help="write splits.json + MANIFEST.json (no scoring)")
    ap.add_argument("--seal", action="store_true",
                    help="write input_hashes.txt + seal.txt")
    ap.add_argument("--pilot", action="store_true",
                    help="20-clip latency pilot with the 3 h stop rule")
    ap.add_argument("--smoke", action="store_true",
                    help="16-clip pipeline smoke into <out>/smoke")
    ap.add_argument("--score", action="store_true",
                    help="score all 2,000 clips into <out>/scores.jsonl")
    ap.add_argument("--verdict", action="store_true",
                    help="write metrics.json + verdict.json from scores")
    ap.add_argument("--out", default=None,
                    help="default: artifacts/split_t2v_matched/stage0_t16")
    ap.add_argument("--force", action="store_true",
                    help="overwrite prepare/seal/metrics or resume scoring")
    args = ap.parse_args()

    out = Path(args.out) if args.out else DEFAULT_OUT
    if not any((args.prepare, args.seal, args.pilot, args.smoke,
                args.score, args.verdict)):
        ap.error("choose --prepare, --seal, --pilot, --smoke, --score, "
                 "or --verdict")
    if not guard(out, args.force):
        return 1
    if args.prepare:
        if (out / "splits.json").exists() and not args.force:
            print(f"ERROR: {out / 'splits.json'} already exists — pass "
                  f"--force to re-prepare", file=sys.stderr)
            return 1
        rc = prepare(out)
        if rc:
            return rc
    if args.seal:
        if (out / "seal.txt").exists() and not args.force:
            print(f"ERROR: {out / 'seal.txt'} already exists — pass --force",
                  file=sys.stderr)
            return 1
        rc = write_seal(out)
        if rc:
            return rc
    if args.pilot:
        rc = pilot(out)
        if rc:
            return rc
    if args.smoke:
        rc = smoke(out)
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
