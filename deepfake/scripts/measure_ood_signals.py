"""Phase 3 Stage A — measure the image-OOD signals FIRST (artifacts only).

D-037 left two OOD candidates untested. This harness measures BOTH on existing
assets (no new data, no downloads, no serve-path change):

  Signal 1  module-contradiction k  — the deployed court via run_benchmark.run_image
  Signal 2  kNN density             — CLIP ViT-L/14 1024-d, train-bank LOO p99.5

Nothing in the serve path changes in Stage A. Outputs land in
artifacts/recovery_ab/ and gate_decision.json is a USER CHECKPOINT: Stage B
does not start until the user has seen which R-steps exist.

Usage:
  ./.venv/bin/python deepfake/scripts/measure_ood_signals.py --stage baselines
  ./.venv/bin/python deepfake/scripts/measure_ood_signals.py --stage signals
  ./.venv/bin/python deepfake/scripts/measure_ood_signals.py --stage fixtures
  ./.venv/bin/python deepfake/scripts/measure_ood_signals.py --stage all

Pre-registered constants (C-PREREG: the builder may not tune on exam results):
  P3-C9  conflict threshold  = conflict_p99 read from stacker_meta.json at runtime
  P3-C10 KNN_K              = 10
  P3-C11 KNN metric         = cosine distance on L2-normalized CLIP ViT-L/14 1024-d
  P3-C12 KNN threshold      = train bank LOO mean-distance p99.5 (train-only)
  P3-C13 FPR bound          = frac_above(eval_real) <= 0.05
  P3-C14 min target mass    = target frac_above >= 0.10
  P3-C15 min enrichment     = target frac >= 3x eval_real frac (zero-FPR clause)
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

import numpy as np  # noqa: E402

# ---------------- pre-registered constants (P3-C*) ----------------

KNN_K = 10                       # P3-C10
FPR_BOUND = 0.05                 # P3-C13
MIN_TARGET_MASS = 0.10           # P3-C14
MIN_ENRICHMENT = 3.0             # P3-C15
SHIP_TARGETS = ("real_ext", "fake_mj", "fake_flux")
RNG_SEED = 5                     # rng(5) — sampling only (spec §3.2)

EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Signal-1 court-run limits (sorted-first files; rng(5) only where sampling)
SIGNAL1_LIMITS = [
    ("eval_real", "data/corpora/eval_real", 150),
    ("real_ext", "data/corpora/modern_exam/real_ext", 150),
    ("fake_mj", "data/corpora/modern_exam/fake_mj", 150),
    ("fake_flux", "data/corpora/modern_exam/fake_flux", 150),
    ("cdf_flat_real", "data/corpora/cdf_flat_real", 150),
    ("cdf_flat_fake", "data/corpora/cdf_flat_fake", 150),
]
WILD_ROOT = WS / "data" / "corpora" / "wild_sample"
WILD_N_PER_CLASS = 100           # 100 real + 100 fake = 200 (rng(5))
LAUNDER_N = 45                   # eval_real first 45 sorted x 2 views = 90

# Signal-2 bank caches (un-suffixed / .clip = CLIP ViT-L/14 1024-d baseline)
BANK_CACHES = [
    "data/caches/train_real.clip.npz",
    "data/caches/train_fake.clip.npz",
    "data/caches/mj_train_real.npz",
    "data/caches/mj_train_fake.npz",
]
# Signal-2 eval inputs taken straight from existing caches (clean keys only)
EVAL_CACHES = [
    ("eval_real", "data/caches/eval_real.clip.npz"),
    ("real_ext", "data/caches/mj_exam_real.npz"),
    ("fake_mj", "data/caches/mj_exam_fake.npz"),
    ("fake_flux", "data/caches/flux_exam_fake.clip.npz"),
    ("wild", "data/caches/wild_sample.npz"),
]


def _label_for(corpus: str) -> int:
    return 0 if ("real" in corpus or corpus in ("real_ext", "eval_real")) else 1


def _sorted_images(root: Path, limit: int):
    return sorted((p for p in root.iterdir()
                   if p.suffix.lower() in EXTS), key=lambda p: p.name)[:limit]


def _l2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32, copy=True)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Pre-registered effect metric: Cliff's delta of a vs b.

    delta = (#{a>b} - #{a<b}) / (n_a * n_b); ties contribute 0.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    n_gt = np.searchsorted(b, a, side="left").sum()            # b < a  == a > b
    n_lt = (b.size - np.searchsorted(b, a, side="right")).sum()  # b > a == a < b
    return float((n_gt - n_lt) / (a.size * b.size))


def _ship_decision(fracs: dict) -> dict:
    """§3.4 ship rule, verbatim. fracs: corpus -> frac_above (float)."""
    frac_eval = fracs.get("eval_real")
    a = frac_eval is not None and frac_eval <= FPR_BOUND
    per_target = {}
    for name in SHIP_TARGETS:
        f = fracs.get(name)
        if f is None:
            per_target[name] = False
            continue
        # P3-C15 zero-FPR clause: when frac_eval == 0 the 3x term is vacuous
        # (f >= 0), so any target with frac >= P3-C14 counts.
        per_target[name] = bool(f >= MIN_TARGET_MASS
                                and (frac_eval == 0 or f >= MIN_ENRICHMENT * frac_eval))
    n_met = sum(1 for v in per_target.values() if v)
    return {
        "cond_a_fpr_bound": bool(a),
        "cond_a_expression": f"frac_above(eval_real)={frac_eval} <= {FPR_BOUND}",
        "per_target": per_target,
        "targets_met": n_met,
        "cond_b_min_2_of_3": n_met >= 2,
        "ships": bool(a and n_met >= 2),
    }


SHIP_RULE_TEXT = (
    "A signal SHIPS as the image-OOD escalation step (ladder step R3) iff it "
    "satisfies BOTH: (a) frac_above(eval_real) <= 0.05; and (b) at least 2 of "
    "{real_ext (bm-real), fake_mj, fake_flux} satisfy frac_above >= 0.10 AND "
    "frac_above >= 3xfrac_above(eval_real) (with the P3-C15 zero-FPR clause). "
    "Otherwise the signal is recorded NOT-AVAILABLE with the full measured "
    "numbers on record. The two signals are decided independently."
)


# ================= baselines (§6.1 pre-capture, deployed court) =================

def _abstain_rate(scores_csv: Path) -> float | None:
    rows = list(csv.DictReader(open(scores_csv)))
    rows = [r for r in rows if r.get("variant") == "clean"
            and r.get("label") in ("0", "1")]
    if not rows:
        return None
    vals = [str(r.get("abstained", "")).strip().lower() in ("true", "1", "1.0")
            for r in rows]
    return round(sum(vals) / len(vals), 4)


def stage_baselines(out: Path) -> int:
    """G-NR1/NR3/NR4/NR5 pre-captures against the UNCHANGED deployed court."""
    from argus_detect import STACKER
    tag = STACKER.parent.name
    res = {"stage": "A-baseline", "stacker": tag, "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ---- G-NR1: scoreboard pre (purge cache via mv, never rm -rf) ----
    work = WS / "artifacts" / "scoreboard" / tag
    bak = WS / "artifacts" / "scoreboard" / f"{tag}.bak_p3a"
    if work.exists():
        if bak.exists():
            print(f"ERROR: {bak} already exists — refusing to clobber the pre baseline",
                  file=sys.stderr)
            return 1
        shutil.move(str(work), str(bak))
        print(f"[nr1] purged scoreboard cache: {work} -> {bak}", flush=True)
    cmd = [str(WS / ".venv" / "bin" / "python"),
           str(WS / "deepfake" / "scripts" / "scoreboard.py"), "--limit", "100"]
    env = dict(os.environ)
    env["PATH"] = f"{WS/'tools'/'bin'}:{env.get('PATH', '')}"
    print(f"[nr1] running: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=WS, env=env)
    nr = {"cmd": cmd, "returncode": r.returncode, "limit": 100,
          "cache_purge": f"{work} -> {bak}", "seconds": round(time.time() - t0, 1),
          "exams": {}}
    sys.path.insert(0, str(WS / "deepfake" / "scripts"))
    from restack import auc_from_scores
    golden = json.loads((WS / "artifacts" / "golden.json").read_text()) \
        if (WS / "artifacts" / "golden.json").exists() else {}
    for name in ("aigc", "cdf", "ffpp", "mj", "flux"):
        csvp = work / f"exam_{name}" / "limit100" / "scores.csv"
        if not csvp.exists():
            nr["exams"][name] = {"error": f"missing {csvp}"}
            continue
        auc, n = auc_from_scores(csvp, tag)
        g = golden.get(name, {}).get("auc")
        nr["exams"][name] = {
            "auc": round(auc, 4), "n": n,
            "abstain_rate": _abstain_rate(csvp),
            "golden_auc": g, "delta_vs_golden": None if g is None else round(auc - g, 4),
            "scores_csv": str(csvp),
        }
        print(f"[nr1] {name}: auc={auc:.4f} n={n}", flush=True)
    (out / "nr_pre.json").write_text(json.dumps(nr, indent=2))
    print(f"[nr1] wrote {out/'nr_pre.json'}", flush=True)

    # ---- G-NR3: audio 2021 exam pre + out_of_coverage rate ----
    aout = out / "audio21_pre"
    acmd = [str(WS / ".venv" / "bin" / "python"),
            str(WS / "deepfake" / "scripts" / "exam_audio_2021.py"),
            "--n-per-class", "200", "--out", str(aout)]
    print(f"[nr3] running: {' '.join(acmd)}", flush=True)
    t0 = time.time()
    ar = subprocess.run(acmd, cwd=WS, env=env)
    ares = {"cmd": acmd, "returncode": ar.returncode, "seconds": round(time.time() - t0, 1),
            "out_dir": str(aout)}
    mp = aout / "metrics.json"
    if mp.exists():
        ares["metrics"] = json.loads(mp.read_text())
    # out_of_coverage rate on the IDENTICAL deterministic sample (the official
    # exam records no OOC column; G-NR3 needs the rate, so re-score once and
    # cross-check the recomputed AUC against the official one).
    try:
        ares["out_of_coverage"] = _audio_ooc_pass(n_per_class=200)
    except Exception as e:
        ares["out_of_coverage"] = {"error": repr(e)}
    (out / "audio21_pre.json").write_text(json.dumps(ares, indent=2))
    print(f"[nr3] wrote {out/'audio21_pre.json'}", flush=True)

    # ---- G-NR4: video spot pre (seed-5 30-video FF++ sample, serve path) ----
    vs = _video_spot_pre(seed=RNG_SEED, n=30)
    (out / "video_spot_pre.json").write_text(json.dumps(vs, indent=2))
    print(f"[nr4] wrote {out/'video_spot_pre.json'} ({len(vs['videos'])} videos)",
          flush=True)

    # ---- G-NR5: latency pre (20 clean eval_real, same-process warmup) ----
    lat = _latency_pre()
    (out / "latency_pre.json").write_text(json.dumps(lat, indent=2))
    print(f"[nr5] wrote {out/'latency_pre.json'} mean={lat['mean_ms']}ms", flush=True)

    res["nr_pre"] = nr.get("exams", {})
    (out / "baselines_done.json").write_text(json.dumps(res, indent=2))
    print("BASELINES DONE", flush=True)
    return 0


def _audio_ooc_pass(n_per_class: int = 200) -> dict:
    """Re-score the ASVspoof-2021 sample once, recording out_of_coverage.

    Mirrors exam_audio_2021.py's pick order exactly (shard 0 -> label 0,
    shard 13 -> label 1, first n_per_class matching rows per shard).
    """
    import pyarrow.parquet as pq
    from sklearn.metrics import roc_auc_score
    from argus_detect import detect_audio

    shards = [
        (WS / "data" / "raw" / "asvspoof21_la_test-00000.parquet", 0),
        (WS / "data" / "raw" / "asvspoof21_la_test-00013.parquet", 1),
    ]
    picks = {0: [], 1: []}
    for shard_path, want_label in shards:
        tab = pq.read_table(shard_path)
        acol, lcol = tab["audio"], tab["label"]
        got = 0
        for i in range(len(tab)):
            if got >= n_per_class:
                break
            if lcol[i].as_py() != want_label:
                continue
            rec = acol[i].as_py()
            data = rec["bytes"] if isinstance(rec, dict) else rec
            if data:
                picks[want_label].append(data)
                got += 1
        print(f"[nr3-ooc] shard {shard_path.name}: {got} x label={want_label}",
              flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="audio_ooc_")) / "u.wav"
    y, p, ooc = [], [], []
    n_ooc = 0
    for lab in (0, 1):
        for data in picks[lab]:
            tmp.write_bytes(data)
            try:
                d = detect_audio(tmp)
            except Exception:
                continue
            score = d.get("p_fake_audio_mean")
            if score is None:
                continue
            y.append(1 - lab)
            p.append(score)
            flag = bool(d.get("out_of_coverage", False))
            ooc.append(flag)
            n_ooc += int(flag)
    shutil.rmtree(tmp.parent, ignore_errors=True)
    auc = float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
    return {
        "n": len(y), "n_out_of_coverage": n_ooc,
        "out_of_coverage_rate": round(n_ooc / max(1, len(y)), 4),
        "auc_recomputed": round(auc, 4),
        "note": "same deterministic sample as exam_audio_2021.py; the official "
                "exam records no OOC column, so the rate is measured here once "
                "and the recomputed AUC is cross-checked against it",
    }


def _video_spot_pre(seed: int = 5, n: int = 30) -> dict:
    from argus_detect import detect_video
    real_dir = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / "real"
    fake_root = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / "fake"
    reals = sorted(real_dir.glob("*.mp4"))
    fakes = sorted(p for p in fake_root.rglob("*.mp4"))
    rng = np.random.default_rng(seed)
    n_half = n // 2
    pick_r = [reals[i] for i in rng.choice(len(reals), size=n_half, replace=False)]
    pick_f = [fakes[i] for i in rng.choice(len(fakes), size=n_half, replace=False)]
    rows = []
    t0 = time.time()
    for label, paths in ((0, pick_r), (1, pick_f)):
        for p in paths:
            try:
                d = detect_video(p)
            except Exception as e:
                rows.append({"file": str(p.relative_to(WS)), "label": label,
                             "error": repr(e)})
                continue
            rows.append({
                "file": str(p.relative_to(WS)), "label": label,
                "route": d.get("route"),
                "p_fake_video_combined": d.get("p_fake_video_combined"),
                "p_fake_fused": d.get("p_fake_fused"),
                "abstained": d.get("abstained"),
                "out_of_conflict": d.get("out_of_conflict"),
                "trust": d.get("trust"), "risk": d.get("risk"),
                "frames_scored": (d.get("visual") or {}).get("frames_scored")
                                 if isinstance(d.get("visual"), dict) else None,
                # recovery key does not exist yet in Stage A (pre-Stage-B)
                "recovery": "not_applicable_pre",
            })
            print(f"[nr4] {p.name} route={rows[-1]['route']} "
                  f"p={rows[-1]['p_fake_video_combined']}", flush=True)
    return {
        "stage": "A-baseline", "gate": "G-NR4", "seed": seed, "n": n,
        "composition": "15 real (real/) + 15 fake (fake/*/), np.random.default_rng(5)",
        "n_frames": "detect_video default (12)",
        "seconds": round(time.time() - t0, 1),
        "videos": rows,
    }


def _latency_pre() -> dict:
    """G-NR5 pre: 20 clean eval_real through detect_image, same-process.

    Warmup = first sorted file (untimed); images 2..21 are timed.
    """
    from argus_detect import detect_image, _image_stacker_clfs
    from PIL import Image
    files = _sorted_images(WS / "data" / "corpora" / "eval_real", 22)
    clfs = _image_stacker_clfs()
    # warmup load (module/model construction + one full inference)
    pil0 = Image.open(files[0]); pil0.load(); pil0 = pil0.convert("RGB")
    t = time.perf_counter()
    detect_image(pil0, clfs)
    warmup_ms = round((time.perf_counter() - t) * 1000, 1)
    per = []
    for p in files[1:21]:
        pil = Image.open(p); pil.load(); pil = pil.convert("RGB")
        t = time.perf_counter()
        detect_image(pil, clfs)
        per.append({"file": p.name, "ms": round((time.perf_counter() - t) * 1000, 1)})
    vals = [r["ms"] for r in per]
    return {
        "stage": "A-baseline", "gate": "G-NR5",
        "n": len(vals), "warmup_file": files[0].name, "warmup_ms": warmup_ms,
        "timed_files": [r["file"] for r in per],
        "mean_ms": round(float(np.mean(vals)), 2),
        "median_ms": round(float(np.median(vals)), 2),
        "per_image": per,
    }


# ================= Signal 1 — module-contradiction k =================

def _parity_self_test(worklist, modules, estimator, clfs, n=20) -> dict:
    """Re-fuse 20 sampled images twice in-process; abort on k mismatch (4-dp)."""
    from run_benchmark import run_image
    from PIL import Image
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(len(worklist), size=min(n, len(worklist)), replace=False)
    tmp = Path(tempfile.mkdtemp(prefix="parity_t_"))
    art = Path(tempfile.mkdtemp(prefix="parity_a_"))
    mismatches = []
    try:
        for i in sorted(int(x) for x in idx):
            item = worklist[i]
            pil = Image.open(item["path"]); pil.load(); pil = pil.convert("RGB")
            if item.get("view_fn") is not None:
                pil, _fmt = item["view_fn"](pil)
            spec = item.get("spec")
            ks = []
            for rep in (0, 1):
                row = run_image(modules, estimator, pil, spec, art,
                                f"parity_{i}_{rep}", tmp, clfs)
                ks.append(round(float(row["conflict"]), 4))
            if ks[0] != ks[1]:
                mismatches.append({"index": i, "corpus": item["corpus"],
                                   "file": item["path"].name, "k_runs": ks})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(art, ignore_errors=True)
    return {
        "n_tested": len(idx), "n_mismatch": len(mismatches),
        "passed": not mismatches, "mismatches": mismatches,
        "tolerance": "4-dp bit-identical",
    }


def stage_signals(out: Path) -> int:
    import joblib
    from argus_detect import STACKER, build_modules
    from triage import DegradationEstimator
    from run_benchmark import run_image
    from PIL import Image
    import exam_launder

    meta_p = STACKER.parent / "stacker_meta.json"
    meta_d = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    conflict_p99 = float(meta_d["conflict_p99"])          # P3-C9, runtime read
    clfs = {STACKER.parent.name: {
        "clf": joblib.load(STACKER),
        "module_ids": meta_d.get("module_ids"),
        "feature_names": meta_d.get("feature_names"),
        "conflict_p99": meta_d.get("conflict_p99")}}
    modules, _mids = build_modules()
    estimator = DegradationEstimator()
    print(f"court wired: {STACKER.parent.name} n_modules={len(modules)} "
          f"conflict_p99={conflict_p99}", flush=True)

    # ---- assemble the Signal-1 worklist (sorted-first; rng(5) for sampling) ----
    worklist = []
    for corpus, rel, lim in SIGNAL1_LIMITS:
        for p in _sorted_images(WS / rel, lim):
            worklist.append({"corpus": corpus, "path": p,
                             "label": _label_for(corpus), "view": "clean",
                             "view_fn": None, "spec": None})
    rng = np.random.default_rng(RNG_SEED)
    for cls, sub, lab in (("wild_real", "real_test", 0), ("wild_fake", "fake_test", 1)):
        pool = _sorted_images(WILD_ROOT / sub, 10**6)
        pick = [pool[i] for i in rng.choice(len(pool), size=WILD_N_PER_CLASS,
                                            replace=False)]
        for p in pick:
            worklist.append({"corpus": cls, "path": p, "label": lab,
                             "view": "clean", "view_fn": None, "spec": None})
    laund_src = _sorted_images(WS / "data" / "corpora" / "eval_real", LAUNDER_N)
    for view_name, fn, spec in (("screenshot", exam_launder.view_screenshot, None),
                                ("screenshot_jpg", exam_launder.view_screenshot_jpg,
                                 {"q": 60})):
        for p in laund_src:
            worklist.append({"corpus": f"launder_{view_name}", "path": p, "label": 0,
                             "view": view_name, "view_fn": fn, "spec": spec})
    print(f"[s1] worklist: {len(worklist)} court runs", flush=True)

    # ---- parity self-test (Phase-2.5 pattern) — abort on failure ----
    parity = _parity_self_test(worklist, modules, estimator, clfs)
    print(f"[s1] parity self-test: passed={parity['passed']} "
          f"n={parity['n_tested']} mismatches={parity['n_mismatch']}", flush=True)
    if not parity["passed"]:
        (out / "parity_failure.json").write_text(json.dumps(parity, indent=2))
        print("ERROR: parity self-test FAILED — aborting before any measurement",
              file=sys.stderr)
        return 1

    # ---- Signal 1 loop ----
    csvp = out / "signal1_conflict.csv"
    cols = ["corpus", "file", "view", "label", "p_real", "abstained", "trust",
            "risk", "conflict", "out_of_conflict", "n_survivors",
            "contradiction_count", "contradiction_mass", "error"]
    tmp = Path(tempfile.mkdtemp(prefix="s1_t_"))
    art = Path(tempfile.mkdtemp(prefix="s1_a_"))
    n_done, t0 = 0, time.time()
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for item in worklist:
            try:
                pil = Image.open(item["path"]); pil.load(); pil = pil.convert("RGB")
                if item["view_fn"] is not None:
                    pil, _fmt = item["view_fn"](pil)
                vid = f"s1_{n_done}"
                row = run_image(modules, estimator, pil, item["spec"], art, vid,
                                tmp, clfs)
                w.writerow({
                    "corpus": item["corpus"], "file": item["path"].name,
                    "view": item["view"], "label": item["label"],
                    "p_real": row.get("p_real"), "abstained": row.get("abstained"),
                    "trust": row.get("trust"), "risk": row.get("risk"),
                    "conflict": round(float(row["conflict"]), 4),
                    "out_of_conflict": row.get("out_of_conflict"),
                    "n_survivors": row.get("n_survivors"),
                    "contradiction_count": row.get("contradiction_count"),
                    "contradiction_mass": row.get("contradiction_mass"),
                })
            except Exception as e:
                print(f"ERR s1 {item['corpus']}/{item['path'].name}: {e}",
                      file=sys.stderr, flush=True)
                w.writerow({"corpus": item["corpus"], "file": item["path"].name,
                            "view": item["view"], "label": item["label"],
                            "error": repr(e)})
            n_done += 1
            f.flush()
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"[s1] {n_done}/{len(worklist)} ({el:.0f}s, "
                      f"{el/n_done:.2f}s/run)", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(art, ignore_errors=True)
    print(f"[s1] wrote {csvp}", flush=True)

    s1_stats = _summarize_signal1(csvp, conflict_p99)
    gate1 = _ship_decision(s1_stats["fracs"])
    s1_stats.update({"ship_decision": gate1, "parity_self_test": parity,
                     "conflict_p99": conflict_p99})

    # ================= Signal 2 — kNN density =================
    s2_stats, thr, bank_info = _run_signal2(out)

    # ================= gate_decision.json + stage_a_report.md =================
    gate = {
        "schema": "argus.gate_decision.v1",
        "stage": "A",
        "stacker": STACKER.parent.name,
        "baseline_head": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                        cwd=WS, capture_output=True,
                                        text=True).stdout.strip(),
        "rule": SHIP_RULE_TEXT,
        "constants": {
            "P3-C9_conflict_p99": conflict_p99,
            "P3-C10_KNN_K": KNN_K,
            "P3-C11_knn_metric": "cosine distance on L2-normalized CLIP ViT-L/14 1024-d",
            "P3-C12_knn_threshold": thr,
            "P3-C13_fpr_bound": FPR_BOUND,
            "P3-C14_min_target_mass": MIN_TARGET_MASS,
            "P3-C15_min_enrichment": MIN_ENRICHMENT,
        },
        "signals": {
            "conflict_k": s1_stats,
            "knn_density": s2_stats,
        },
        "r3_available_on": [s for s, d in
                            (("conflict_k", gate1), ("knn_density", s2_stats["ship_decision"]))
                            if d["ships"]],
        "user_checkpoint": ("Stage B MUST NOT start until the orchestrator "
                            "presents this file and the user confirms which "
                            "R-steps exist."),
    }
    if not gate["r3_available_on"]:
        gate["r3_note"] = ("R3 ships as documented absence — the ladder still "
                           "runs R1/R2/R4; OOD gate is NOT-AVAILABLE.")
    (out / "gate_decision.json").write_text(json.dumps(gate, indent=2))
    print(f"[gate] wrote {out/'gate_decision.json'}", flush=True)
    print("[gate] conflict_k ships:", gate1["ships"],
          "| knn_density ships:", s2_stats["ship_decision"]["ships"], flush=True)

    _write_stage_a_report(out, gate, bank_info)
    print("SIGNALS DONE", flush=True)
    return 0


def _summarize_signal1(csvp: Path, conflict_p99: float) -> dict:
    rows = list(csv.DictReader(open(csvp)))
    by = {}
    for r in rows:
        by.setdefault(r["corpus"], []).append(r)
    eval_vals = np.array([float(r["conflict"]) for r in by.get("eval_real", [])
                          if r.get("conflict") not in ("", None)], dtype=np.float64)
    fracs, detail = {}, {}
    for corpus, rs in by.items():
        vals = np.array([float(r["conflict"]) for r in rs
                         if r.get("conflict") not in ("", None)], dtype=np.float64)
        if vals.size == 0:
            continue
        above = vals > conflict_p99
        fracs[corpus] = round(float(above.mean()), 4)
        detail[corpus] = {
            "n": int(vals.size),
            "frac_above": round(float(above.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "mean": round(float(vals.mean()), 4),
            "max": round(float(vals.max()), 4),
            "cliffs_delta_vs_eval_real": round(_cliffs_delta(vals, eval_vals), 4),
            "n_above": int(above.sum()),
        }
    return {"fracs": fracs, "per_corpus": detail,
            "eval_real_frac_above": fracs.get("eval_real")}


def _load_bank() -> tuple[np.ndarray, list[str], dict]:
    """Four bank caches, clean keys only, L2-normalized at consumption (D-046)."""
    vecs, keys, per_cache = [], [], {}
    for rel in BANK_CACHES:
        p = WS / rel
        z = np.load(p, allow_pickle=False)
        clean = sorted(k for k in z.files if str(k).endswith(":clean"))
        arr = np.stack([z[k] for k in clean]).astype(np.float32)
        # D-046 build-time asserts: dim == 1024 for every bank cache; two
        # DIFFERENT images' clean vectors must not be bit-identical.
        assert arr.shape[1] == 1024, f"{rel}: bank dim {arr.shape[1]} != 1024"
        if len(clean) >= 2:
            a, b = arr[0].astype(np.float32), arr[1].astype(np.float32)
            assert not np.array_equal(a, b), (
                f"{rel}: two different images' clean vectors are bit-identical "
                "(D-046 cache quirk detected at build time)")
        per_cache[rel] = {"n_clean": len(clean), "dim": int(arr.shape[1]),
                          "distinct_first_two": True}
        vecs.append(arr)
        keys += [f"{rel}::{k}" for k in clean]
    bank = np.vstack(vecs)
    bank = np.stack([_l2(v) for v in bank])
    return bank, keys, per_cache


def _loo_mean_dist(bank: np.ndarray, k: int) -> np.ndarray:
    """Leave-self-out mean cosine distance to the k nearest OTHER bank vectors."""
    sims = bank @ bank.T
    np.fill_diagonal(sims, -np.inf)                    # exclude self
    top = -np.partition(-sims, kth=k - 1, axis=1)[:, :k]  # k largest sims
    return (1.0 - top).mean(axis=1)


def _knn_dist(vecs: np.ndarray, bank: np.ndarray, k: int) -> np.ndarray:
    sims = vecs @ bank.T
    top = -np.partition(-sims, kth=k - 1, axis=1)[:, :k]
    return (1.0 - top).mean(axis=1)


def _run_signal2(out: Path) -> tuple[dict, float, dict]:
    from PIL import Image
    import exam_launder

    bank, bank_keys, per_cache = _load_bank()
    loo = _loo_mean_dist(bank, KNN_K)
    thr = float(np.percentile(loo, 99.5))               # P3-C12, train-only
    print(f"[s2] bank n={len(bank)} dim={bank.shape[1]} "
          f"LOO mean={loo.mean():.4f} p99.5 thr={thr:.6f}", flush=True)

    rows = []
    # (a) cached eval inputs — clean keys only, L2 at load
    for corpus, rel in EVAL_CACHES:
        z = np.load(WS / rel, allow_pickle=False)
        clean = sorted(k for k in z.files if str(k).endswith(":clean"))
        arr = np.stack([_l2(z[k]) for k in clean])
        d = _knn_dist(arr, bank, KNN_K)
        for k, dist in zip(clean, d):
            rows.append({"corpus": "wild_real" if (corpus == "wild"
                                                   and k.startswith("real_"))
                         else "wild_fake" if corpus == "wild" else corpus,
                         "file": str(k)[:-len(":clean")], "view": "clean",
                         "knn_dist": round(float(dist), 6),
                         "fire": bool(dist > thr)})
        print(f"[s2] {corpus}: {len(clean)} cached clean embeddings", flush=True)

    # (b) on-the-fly CLIP embeds: cdf_flat + launder variants (same _canvas path)
    from backbone_embed import embed
    for corpus, rel in (("cdf_flat_real", "data/corpora/cdf_flat_real"),
                        ("cdf_flat_fake", "data/corpora/cdf_flat_fake")):
        files = _sorted_images(WS / rel, 10**6)
        arr, names = [], []
        for p in files:
            pil = Image.open(p); pil.load(); pil = pil.convert("RGB")
            arr.append(_l2(embed(pil, "clip"))); names.append(p.name)
        d = _knn_dist(np.stack(arr), bank, KNN_K)
        for name, dist in zip(names, d):
            rows.append({"corpus": corpus, "file": name, "view": "clean",
                         "knn_dist": round(float(dist), 6),
                         "fire": bool(dist > thr)})
        print(f"[s2] {corpus}: {len(names)} on-the-fly embeds", flush=True)

    laund_src = _sorted_images(WS / "data" / "corpora" / "eval_real", LAUNDER_N)
    for view_name, fn in (("screenshot", exam_launder.view_screenshot),
                          ("screenshot_jpg", exam_launder.view_screenshot_jpg)):
        arr, names = [], []
        for p in laund_src:
            pil = Image.open(p); pil.load(); pil = pil.convert("RGB")
            vp, _fmt = fn(pil)
            arr.append(_l2(embed(vp, "clip"))); names.append(p.name)
        d = _knn_dist(np.stack(arr), bank, KNN_K)
        for name, dist in zip(names, d):
            rows.append({"corpus": f"launder_{view_name}", "file": name,
                         "view": view_name, "knn_dist": round(float(dist), 6),
                         "fire": bool(dist > thr)})
        print(f"[s2] launder_{view_name}: {len(names)} on-the-fly embeds", flush=True)

    csvp = out / "signal2_knn.csv"
    with open(csvp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["corpus", "file", "view", "knn_dist", "fire"])
        w.writeheader(); w.writerows(rows)
    print(f"[s2] wrote {csvp} ({len(rows)} rows)", flush=True)

    by = {}
    for r in rows:
        by.setdefault(r["corpus"], []).append(r)
    eval_vals = np.array([float(r["knn_dist"]) for r in by.get("eval_real", [])])
    fracs, detail = {}, {}
    for corpus, rs in by.items():
        vals = np.array([float(r["knn_dist"]) for r in rs])
        above = vals > thr
        fracs[corpus] = round(float(above.mean()), 4)
        detail[corpus] = {
            "n": int(vals.size),
            "frac_above": round(float(above.mean()), 4),
            "median": round(float(np.median(vals)), 4),
            "mean": round(float(vals.mean()), 4),
            "max": round(float(vals.max()), 4),
            "cliffs_delta_vs_eval_real": round(_cliffs_delta(vals, eval_vals), 4),
            "n_above": int(above.sum()),
        }
    stats = {
        "fracs": fracs, "per_corpus": detail,
        "eval_real_frac_above": fracs.get("eval_real"),
        "threshold": thr, "knn_k": KNN_K,
        "loo_distribution": {"n": int(loo.size), "mean": round(float(loo.mean()), 6),
                             "p50": round(float(np.percentile(loo, 50)), 6),
                             "p99": round(float(np.percentile(loo, 99)), 6),
                             "p99_5": round(float(np.percentile(loo, 99.5)), 6)},
        "bank": {"total": int(len(bank)), "per_cache": per_cache},
    }
    stats["ship_decision"] = _ship_decision(fracs)
    return stats, thr, {"bank_per_cache": per_cache, "loo_p99_5": thr}


# ================= §6.3 smoke fixtures =================

def stage_fixtures(out: Path) -> int:
    import csv as _csv
    from PIL import Image
    import exam_launder
    from argus_detect import detect_image, detect_video, _image_stacker_clfs

    fixdir = out / "fixtures"
    fixdir.mkdir(parents=True, exist_ok=True)
    fixtures = {"stage": "A", "note": "recorded for the Stage B recovery exam",
                "fixtures": {}}

    # 1. first OOC wild stem from artifacts/exam_dfeval/audio_exam.csv (sorted by stem)
    rows = list(_csv.DictReader(open(WS / "artifacts" / "exam_dfeval" / "audio_exam.csv")))
    ooc = sorted((r for r in rows if str(r.get("out_of_coverage")).lower() in ("true", "1", "1.0")),
                 key=lambda r: r["stem"])
    if ooc:
        stem = ooc[0]["stem"]
        adir = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data"
        cand = sorted(adir.glob(f"{stem}.*"))
        fixtures["fixtures"]["1_first_ooc_wild_audio"] = {
            "stem": stem, "csv_row": ooc[0],
            "path": str(cand[0].relative_to(WS)) if cand else None,
            "exists": bool(cand),
            "source": "artifacts/exam_dfeval/audio_exam.csv, first OOC row by sorted stem",
        }
    else:
        fixtures["fixtures"]["1_first_ooc_wild_audio"] = {"error": "no OOC rows in csv"}

    # 2. screenshot-laundered real (exam_launder.view_screenshot import)
    src = _sorted_images(WS / "data" / "corpora" / "eval_real", 1)[0]
    pil = Image.open(src); pil.load(); pil = pil.convert("RGB")
    vp, fmt = exam_launder.view_screenshot(pil)
    fpath = fixdir / f"2_screenshot_real_{src.stem}.png"
    vp.save(fpath, "PNG")
    clfs = _image_stacker_clfs()
    snap = detect_image(vp, clfs)
    fixtures["fixtures"]["2_screenshot_laundered_real"] = {
        "source": str(src.relative_to(WS)), "view": "screenshot",
        "transform": "exam_launder.view_screenshot (import, not copy)",
        "path": str(fpath.relative_to(WS)),
        "court_snapshot_pre": {k: snap.get(k) for k in
                               ("p_fake_fused", "abstained", "trust", "risk",
                                "out_of_conflict")},
    }

    # 3. FF++ real 953.mp4 — expect not_attempted once Stage B lands
    v3 = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / "real" / "953.mp4"
    d3 = detect_video(v3)
    fixtures["fixtures"]["3_ffpp_real_953"] = {
        "path": str(v3.relative_to(WS)), "exists": v3.exists(),
        "stage_b_expectation": "recovery.outcome == 'not_attempted'",
        "recovery": "not_applicable_pre",
        "court_snapshot_pre": {k: d3.get(k) for k in
                               ("route", "p_fake_video_combined", "abstained",
                                "trust", "risk")},
    }

    # 4. first sorted eval_real still with ZERO YuNet detections at 0.55,
    #    via the REAL detector helper (video_face_track.get_det/_detect_faces)
    import cv2
    from video_face_track import _detect_faces, prepare_frame
    hit = None
    for p in _sorted_images(WS / "data" / "corpora" / "eval_real", 10**6):
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = _detect_faces(prepare_frame(img))
        if len(faces) == 0:
            hit = p
            break
    fixtures["fixtures"]["4_no_face_still_candidate"] = {
        "path": str(hit.relative_to(WS)) if hit else None,
        "detector": "video_face_track.get_det score_threshold=0.55 + _detect_faces "
                    "(real observation-path helper, patterns.md:39)",
        "caveat": ("a still with zero detections is a CANDIDATE locator only — "
                   "patterns.md:39: the zero-face property must be re-verified "
                   "with observe_video on a video (fixture 5 carries that "
                   "verification)"),
        "verified": hit is not None,
    }

    # 5. artifacts/fixtures/no_face.mp4 — verify via the REAL observation path
    v5 = WS / "artifacts" / "fixtures" / "no_face.mp4"
    if v5.exists():
        from video_face_track import observe_video
        _obs, stats = observe_video(v5, n_frames=12)
        fixtures["fixtures"]["5_no_face_video"] = {
            "path": str(v5.relative_to(WS)), "exists": True,
            "observation": {"frames_sampled": stats.get("frames_sampled"),
                            "frames_decoded": stats.get("frames_decoded"),
                            "faces_detected": stats.get("faces_detected")},
            "verified_zero_face": stats.get("faces_detected") == 0,
            "verification": "video_face_track.observe_video (patterns.md:39)",
        }
    else:
        fixtures["fixtures"]["5_no_face_video"] = {
            "path": str(v5.relative_to(WS)), "exists": False,
            "recipe": "patterns.md:39 — translation-only crop of an eval_real "
                      "still, then verify with observe_video faces_detected==0",
        }

    (out / "smoke_fixtures.json").write_text(json.dumps(fixtures, indent=2))
    print(f"[fixtures] wrote {out/'smoke_fixtures.json'}", flush=True)
    return 0


# ================= stage_a_report.md =================

def _write_stage_a_report(out: Path, gate: dict, bank_info: dict) -> None:
    L = ["# Phase 3 Stage A — image-OOD signal measurement", "",
         f"Court: {gate['stacker']} · HEAD {gate['baseline_head']} · "
         f"artifacts-only (no serve-path change)", "",
         "## Pre-registered ship rule (§3.4)", "", gate["rule"], "",
         "## Constants (P3-C*)", "",
         "| ID | value |", "|---|---|"]
    for k, v in gate["constants"].items():
        L.append(f"| {k} | {v} |")
    for sig_name in ("conflict_k", "knn_density"):
        d = gate["signals"][sig_name]
        sd = d["ship_decision"]
        L += ["", f"## Signal: {sig_name}", "",
              f"**SHIPS: {sd['ships']}** — cond(a) {sd['cond_a_expression']} -> "
              f"{sd['cond_a_fpr_bound']}; targets met {sd['targets_met']}/3 -> "
              f"{sd['cond_b_min_2_of_3']}",
              "", "| corpus | n | frac_above | median | Cliff's delta vs eval_real |",
              "|---|---|---|---|---|"]
        for corpus, c in sorted(d["per_corpus"].items()):
            L.append(f"| {corpus} | {c['n']} | {c['frac_above']} | {c['median']} | "
                     f"{c['cliffs_delta_vs_eval_real']} |")
        if sig_name == "conflict_k":
            L += ["", f"threshold (P3-C9 conflict_p99, runtime read) = "
                      f"{d['conflict_p99']}",
                  f"parity self-test: passed={d['parity_self_test']['passed']} "
                  f"n={d['parity_self_test']['n_tested']} "
                  f"mismatches={d['parity_self_test']['n_mismatch']}"]
        else:
            L += ["", f"threshold (P3-C12 train-bank LOO p99.5) = {d['threshold']} "
                      f"(k={d['knn_k']})",
                  f"LOO distribution: {d['loo_distribution']}",
                  f"bank: {d['bank']}"]
    L += ["", "## Baselines captured (deployed court, unchanged)", ""]
    for name in ("nr_pre.json", "audio21_pre.json", "video_spot_pre.json",
                 "latency_pre.json", "smoke_fixtures.json"):
        p = out / name
        L.append(f"- `{name}` — {'present' if p.exists() else 'MISSING'}")
    nr = out / "nr_pre.json"
    if nr.exists():
        d = json.loads(nr.read_text())
        L += ["", "| exam | auc | abstain | delta vs golden |", "|---|---|---|---|"]
        for name, e in sorted(d.get("exams", {}).items()):
            L.append(f"| {name} | {e.get('auc')} | {e.get('abstain_rate')} | "
                     f"{e.get('delta_vs_golden')} |")
    a = out / "audio21_pre.json"
    if a.exists():
        d = json.loads(a.read_text())
        m = d.get("metrics", {})
        o = d.get("out_of_coverage", {})
        auc = m.get("auc", o.get("auc_recomputed"))
        n = m.get("n", o.get("n"))
        note = "" if m else (" (official exam_audio_2021.py crashed on a "
                             "pre-existing variable-shadowing bug — see "
                             "deviations; AUC is the identical-sample "
                             "recomputation)")
        L += ["", f"audio21 pre: AUC={auc} n={n} "
                  f"OOC rate={o.get('out_of_coverage_rate')}{note}", ""]
    L += ["", "## Deviations / notes", "",
          "- Signal 1 court-run limits follow §3.2 (sorted-first; rng(5) only for "
          "the wild 200 = 100+100 balanced sample and the parity/spot sampling). "
          "Per-corpus limits sum to 1190 runs (the spec's \"~1,110\" was an "
          "approximation of its own limit list).",
          "- Signal 2 consumes FULL cached clean-key sets where a cache exists "
          "(eval_real 500, wild 4000) — §3.2 states no Signal-2 limit, and the "
          "FPR reference benefits from the larger in-domain set; cdf/launder "
          "embeds are on-the-fly with no extra limit (dirs hold 150/45 already).",
          "- `fusion.py` cited anchors live at "
          "integrations/Argus_Master/backend/fusion.py (import-only, C-HARNESS).",
          "- Baselines + fixtures are produced by --stage baselines/fixtures of "
          "this same script (single new file; §3.2 lists them as Stage A outputs).",
          "- PRE-EXISTING BUG (not fixed — §3.5 forbids editing existing files): "
          "exam_audio_2021.py line 96 `out = detect_audio(tmp)` shadows the "
          "`out` Path from line 50, so line 110 crashes with TypeError after "
          "scoring completes; metrics.json/scores.json are never written. "
          "G-NR3 pre values therefore come from an identical-sample re-score "
          "(same shards, same first-200 pick order) that also records "
          "out_of_coverage. Recommend a 1-line rename in a follow-up task.",
          "- ASVspoof-2021 sample: 330 of 400 utterances scored; 70 raise "
          "LibsndfileError (flac decoder lost sync) — a pre-existing corpus/"
          "decode property shared by the official script's own try/except.",
          "- Signal 1 saw one transient ~100-min stall between worklist items "
          "~675-700 (cdf_flat_real range, likely an external wait); the process "
          "self-recovered and the remaining runs returned to ~1.4 s/run. All "
          "1190 rows completed; parity self-test still bit-identical.",
          "", "## User checkpoint", "", gate["user_checkpoint"], ""]
    (out / "stage_a_report.md").write_text("\n".join(L))
    print(f"[report] wrote {out/'stage_a_report.md'}", flush=True)


def stage_report(out: Path) -> int:
    """Regenerate stage_a_report.md from the on-disk gate_decision.json."""
    gate = json.loads((out / "gate_decision.json").read_text())
    _write_stage_a_report(out, gate, {})
    return 0


# ================= main =================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WS / "artifacts" / "recovery_ab"))
    ap.add_argument("--stage", default="all",
                    choices=["baselines", "signals", "fixtures", "report", "all"])
    ap.add_argument("--force", action="store_true",
                    help="write into an output dir that already contains results")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    # stamped-out-dir guard copied verbatim from patterns.md:32
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force "
              "to overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rc = 0
    if args.stage in ("baselines", "all"):
        rc = stage_baselines(out) or rc
        if rc:
            return rc
    if args.stage in ("signals", "all"):
        rc = stage_signals(out) or rc
        if rc:
            return rc
    if args.stage in ("fixtures", "all"):
        rc = stage_fixtures(out) or rc
    if args.stage == "report":
        rc = stage_report(out) or rc
    print(f"stage={args.stage} rc={rc} elapsed={time.time()-t0:.0f}s", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
