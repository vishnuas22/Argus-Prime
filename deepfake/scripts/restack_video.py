"""Video restack pipeline — dump -> train -> gate -> verdict (spec item 3.3).

  DUMP    video_dump --split train (limit 1500) and --split val (limit 300);
          plus the D-021 domain diet: dump_cdf = --corpus celebdf (limit 1000,
          official CDF test list excluded — 5b.5); audio on a deterministic
          200-video subsample only (decision #3)
  TRAIN   train_video_stacker on dump_train + dump_val + dump_cdf ->
          work/candidate/{stacker.joblib, stacker_meta.json,
          drift_calibration.json}
  GATE    A: full official FF++ test split (700 videos, one pass) — stacked
            >= 0.90 AND stacked >= naive_combined - 0.005 AND per-method bars
            (Deepfakes/Face2Face/FaceSwap >= 0.88, NeuralTextures >= 0.60)
          B: leave-one-method-out (4 feature-level retrains) >= 0.70 each
          C: Celeb-DF cross-dataset stacked >= 0.75 (0.70-0.75 = escalate)
  VERDICT PROMOTE iff A and B and C; --promote --promote-to <tag> copies the
          candidate to artifacts/<tag>/ (explicit tag required, never
          auto-wired). A gate miss ESCALATES to the user — no autonomous
          waiver file is ever written (decision #2).

Usage:
  .venv/bin/python deepfake/scripts/restack_video.py [--stage all|dump|train|gate|verdict]
      [--smoke] [--work artifacts/restack_video_v2] [--embedder arcface]
      [--promote --promote-to p6vid_v2]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
PY = WS / ".venv" / "bin" / "python"
SCRIPTS = WS / "deepfake" / "scripts"

GATES = {
    "ffpp": {"min_auc": 0.90, "vs_naive_tol": -0.005,
             # spec per-method bars (not a flat floor): the three identity-
             # visible forgeries must clear 0.88; NeuralTextures (mouth-
             # reenactment, weakest identity signal) keeps the 0.60 floor.
             "min_per_method": {"Deepfakes": 0.88, "Face2Face": 0.88,
                                "FaceSwap": 0.88, "NeuralTextures": 0.60}},
    "lomo": {"min_auc": 0.70},          # per held-out method (4 folds)
    "celebdf": {"min_auc": 0.75, "warn_auc": 0.70},
    # <0.75 = FAIL, 0.70-0.75 = escalate to the user (decision #2 — no auto-waive)
}
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
ESCALATE_BAND = (0.85, 0.90)            # FF++ stacked in this band => escalate


def run(cmd, **kw):
    print(f"[run] {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=WS, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r


def stage_dump(work, smoke, embedder):
    limits = (40, 20) if smoke else (1500, 300)
    audio = 0 if smoke else 200
    for split, lim in zip(("train", "val"), limits):
        out = work / f"dump_{split}"
        if (out / "scores.csv").exists():
            print(f"[dump] {split}: exists, skipped", flush=True)
        else:
            cmd = [PY, SCRIPTS / "video_dump.py", "--split", split, "--limit", lim,
                   "--audio-limit", audio, "--out", work, "--embedder", embedder]
            if audio == 0:
                cmd.append("--no-audio")
            run(cmd)
        if split == "train":
            # D-021 diet stage BETWEEN the train-dump and the val-dump (5b.6)
            stage_dump_cdf(work, smoke, embedder)


def stage_dump_cdf(work, smoke, embedder):
    """D-021 domain diet (5b.5): Celeb-DF TRAIN videos (test list excluded by
    video_dump.enumerate_celebdf) so the stacker sees both verdict spaces on
    BOTH domains — an FF++-only diet would teach 'spec always wins' and risk
    gate C. SIZE=1000 (500/500) per the user's binding decision."""
    out = work / "dump_cdf"
    if (out / "scores.csv").exists():
        print("[dump] cdf: exists, skipped", flush=True)
        return
    lim = 40 if smoke else 1000
    run([PY, SCRIPTS / "video_dump.py", "--split", "train", "--corpus", "celebdf",
         "--limit", lim, "--no-audio", "--out", work, "--embedder", embedder])


def stage_train(work, embedder):
    out = work / "candidate"
    if (out / "stacker.joblib").exists():
        print("[train] candidate exists, skipped", flush=True)
        return
    # D-021: ALL three dumps are the diet — a missing dump_cdf would silently
    # train FF++-only (the exact stale-diet failure class 5b.5 exists to fix).
    csvs = []
    for name in ("dump_train", "dump_val", "dump_cdf"):
        p = work / name / "scores.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} — run the dump stage first")
        csvs.append(str(p))
    run([PY, SCRIPTS / "train_video_stacker.py",
         "--csv", *csvs, "--out", out, "--embedder", embedder])
    meta = json.loads((out / "stacker_meta.json").read_text())
    print(f"[train] candidate: val_auc={meta['val_auc']} "
          f"features={len(meta['feature_names'])} embedder={embedder}", flush=True)


def _gate_exam(out_dir, smoke, build_cmd):
    """Run an exam once; reuse its on-disk metrics afterwards."""
    if not (out_dir / "metrics.json").exists():
        run(build_cmd(out_dir))
    return json.loads((out_dir / "metrics.json").read_text())


def stage_gate(work, smoke, embedder):
    cand = work / "candidate"
    cal = cand / "drift_calibration.json"
    # full = the official FF++ test split: 140 reals + 4x140 fakes = 700.
    # (A half-sample gate (per-method 70) flipped the delta-vs-naive sign and
    # hid the Deepfakes bar crossing — n is part of the gate contract.)
    nr, pm, cdf_lim = ((8, 4, 30) if smoke else (140, 140, 0))
    gate_dir = work / "gate_ffpp"

    ffpp = _gate_exam(gate_dir, smoke, lambda o: [
        PY, SCRIPTS / "exam_video_ffpp.py", "--n-real", nr, "--per-method", pm,
        "--frames", 12, "--video-stacker", cand, "--drift-cal", cal,
        "--embedder", embedder, "--out", o])
    auc = ffpp.get("auc", {})
    stacked, naive = auc.get("stacked"), auc.get("naive_combined")
    bars = GATES["ffpp"]["min_per_method"]
    per_meth_ok = all(
        (ffpp.get("per_method", {}).get(m, {}).get("stacked") or 0)
        >= bars[m] for m in METHODS) if not smoke else True
    delta = None if (stacked is None or naive is None) else stacked - naive
    ffpp_pass = bool(stacked is not None and naive is not None
                     and stacked >= GATES["ffpp"]["min_auc"]
                     and delta >= GATES["ffpp"]["vs_naive_tol"] and per_meth_ok)
    g_ffpp = {"auc_stacked": stacked, "auc_naive": naive,
              "auc_identity": auc.get("identity_video_score"),
              "delta_vs_naive": None if delta is None else round(delta, 4),
              "per_method": ffpp.get("per_method", {}),
              "min_per_method": bars,
              "embedder": ffpp.get("embedder"),
              "n": ffpp.get("n_scored"),
              "pass": ffpp_pass}
    print(f"[gate] ffpp: stacked={stacked} naive={naive} delta={delta} "
          f"-> {'PASS' if ffpp_pass else 'FAIL'}", flush=True)

    # --- Gate B: leave-one-method-out (pure feature-level) -----------------
    import pandas as pd
    from train_video_stacker import fit_stacker
    from video_features import row_to_X
    train_df = pd.read_csv(work / "dump_train" / "scores.csv")
    test_df = pd.read_csv(gate_dir / "features.csv")
    reals = test_df[test_df["label"] == 0]
    lomo, lomo_ok = {}, True
    for m in METHODS:
        sub = train_df[~((train_df["label"] == 1) & (train_df["method"] == m))]
        held = test_df[test_df["method"] == m]
        if smoke:
            held = held.head(8)
        eval_df = pd.concat([held, reals], ignore_index=True)
        clf, _va, _X, _y = fit_stacker(sub)
        X = np.vstack([row_to_X(r) for r in eval_df.to_dict("records")])
        p = clf.predict_proba(X)[:, 1]
        y = eval_df["label"].astype(int).to_numpy()
        from sklearn.metrics import roc_auc_score
        a = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
        lomo[m] = {"auc": None if a is None else round(a, 4),
                   "n_fake": int((y == 1).sum()), "n_real": int((y == 0).sum())}
        ok = a is not None and a >= GATES["lomo"]["min_auc"]
        lomo_ok = lomo_ok and ok
        print(f"[gate] lomo {m}: auc={a} -> {'PASS' if ok else 'FAIL'}", flush=True)
    g_lomo = {"per_method_auc": lomo, "min_auc": GATES["lomo"]["min_auc"],
              "pass": bool(lomo_ok)}

    # --- Gate C: Celeb-DF cross-dataset ------------------------------------
    cdf = _gate_exam(work / "gate_celebdf", smoke, lambda o: [
        PY, SCRIPTS / "exam_video_celebdf.py", "--frames", 12,
        "--stacker", cand, "--drift-cal", cal, "--embedder", embedder, *(
            ["--limit", cdf_lim] if cdf_lim else []), "--out", o])
    c_auc = cdf.get("auc", {}).get("stacked")
    c_verdict = ("PASS" if c_auc is not None and c_auc >= GATES["celebdf"]["min_auc"]
                 else "WARN_ESCALATE" if c_auc is not None
                 and c_auc >= GATES["celebdf"]["warn_auc"] else "FAIL")
    g_cdf = {"auc_stacked": c_auc, "auc_naive": cdf.get("auc", {}).get("naive"),
             "auc_identity": cdf.get("auc", {}).get("identity"), "n": cdf.get("n"),
             "embedder": cdf.get("embedder"),
             "verdict": c_verdict, "pass": c_verdict == "PASS",
             "warn": c_verdict == "WARN_ESCALATE"}
    print(f"[gate] celebdf: stacked={c_auc} -> {c_verdict}", flush=True)

    gates = {"ffpp": g_ffpp, "lomo": g_lomo, "celebdf": g_cdf,
             "gates_spec": GATES, "smoke": bool(smoke),
             "embedder": embedder,
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (work / "gates.json").write_text(json.dumps(gates, indent=2))
    return gates


def stage_verdict(work, promote, promote_to):
    gp = work / "gates.json"
    if not gp.exists():
        raise SystemExit("no gate results — run the gate stage first")
    gates = json.loads(gp.read_text())
    if gates.get("smoke"):
        print("[verdict] gates.json is from a --smoke run — verdict skipped "
              "(run the full gate stage before any verdict)", flush=True)
        return 0
    a, b, c = (gates["ffpp"]["pass"], gates["lomo"]["pass"], gates["celebdf"]["pass"])
    stacked = gates["ffpp"]["auc_stacked"]
    print(f"\nVERDICT: {'PROMOTE' if (a and b and c) else 'HOLD'}  "
          f"(ffpp={a} [stacked={stacked}, naive={gates['ffpp']['auc_naive']}], "
          f"lomo={b}, celebdf={c} [{gates['celebdf']['verdict']}])", flush=True)
    if stacked is not None and ESCALATE_BAND[0] <= stacked < ESCALATE_BAND[1]:
        print(f"  ESCALATE: FF++ stacked AUC {stacked} is in the "
              f"{ESCALATE_BAND[0]}-{ESCALATE_BAND[1]} waiver band — per user "
              f"decision #2 this is NOT waived autonomously; user decides.",
              flush=True)
    if gates["celebdf"]["warn"]:
        print(f"  ESCALATE: Celeb-DF stacked AUC {gates['celebdf']['auc_stacked']} "
              f"is in the 0.70-0.75 warn band — user decides.", flush=True)
    if not (a and b and c):
        print("  HOLD: gates incomplete/failed — no promotion, no waiver written.",
              flush=True)
        return 2
    if promote:
        if not promote_to:
            raise SystemExit("--promote requires --promote-to <tag>")
        dst = WS / "artifacts" / promote_to
        dst.mkdir(parents=True, exist_ok=True)
        for name in ("stacker.joblib", "stacker_meta.json", "drift_calibration.json"):
            shutil.copy(work / "candidate" / name, dst / name)
        print(f"promoted -> {dst} (wire argus_detect.VIDEO_STACKER = "
              f"artifacts/{promote_to}/stacker.joblib to activate)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "dump", "train", "gate", "verdict"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--work", default=None,
                    help="work dir (default artifacts/restack_video_v2[_smoke]; "
                         "MUST be fresh — _gate_exam reuses existing metrics.json "
                         "and stage_dump skips existing scores)")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--promote-to", default=None)
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="arcface",
                    help="identity embedder for ALL stages — dump/train/gate "
                         "must match, and the serve path reads it from "
                         "stacker_meta.embedder (phase default: arcface, D-042)")
    args = ap.parse_args()
    if args.promote and not args.promote_to:
        raise SystemExit("--promote requires --promote-to <tag>")

    work = Path(args.work) if args.work else (WS / "artifacts" /
        ("restack_video_v2_smoke" if args.smoke else "restack_video_v2"))
    work.mkdir(parents=True, exist_ok=True)
    stages = (["dump", "train", "gate", "verdict"] if args.stage == "all"
              else [args.stage])
    for st in stages:
        if st == "dump":
            stage_dump(work, args.smoke, args.embedder)
        elif st == "train":
            stage_train(work, args.embedder)
        elif st == "gate":
            stage_gate(work, args.smoke, args.embedder)
        elif st == "verdict":
            return stage_verdict(work, args.promote, args.promote_to)
    return 0


if __name__ == "__main__":
    sys.exit(main())
