"""Restack pipeline — the reusable evidence-channel intake path (D-024).

Stages (resumable via --stage):
  1. DUMP   score the stacker training corpora through the FULL active
            channel set (--channels all, no ladder, clean variant only)
  2. TRAIN  retrain the stacker on the fresh dumps (feature space = active ids)
  3. GATE   A/B old vs candidate stacker on held-out gate sets in ONE pass:
            aigc (eval_real/eval_fake), cdf (face exam), ffpp (face exam)
  4. VERDICT  PROMOTE if AIGC + CDF gates hold; ffpp reported as gain signal.
            --promote copies the candidate to artifacts/<promote-to>/ (explicit
            tag required via --promote-to; never auto-wired).

Usage:
  .venv/bin/python deepfake/scripts/restack.py [--gate-limit 100] [--smoke]
  .venv/bin/python deepfake/scripts/restack.py --stage gate
  .venv/bin/python deepfake/scripts/restack.py --stage verdict --promote --promote-to p6v10
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
PY = WS / ".venv" / "bin" / "python"
BENCH = WS / "deepfake" / "bench" / "run_benchmark.py"
def _wired_stacker() -> Path:
    """Incumbent stacker = the wired court's stacker, never a hardcoded tag."""
    import sys
    sys.path.insert(0, str(WS / "deepfake"))
    from argus_detect import STACKER
    return STACKER


OLD_STACKER = None  # resolved lazily in main() (imports are heavy at module load)

# (dump name, real dir, fake dir, per-class limit) — mirrors the stkv2 recipe
CORPORA = [
    ("train_fake", "data/corpora/train_real", "data/corpora/train_fake", 50),
    ("cdf", "data/corpora/cdf_flat_real", "data/corpora/cdf_flat_fake", 150),
    ("xgen_sana_tr", "data/corpora/train_real", "data/corpora/xgen_sana_tr", 42),
    ("xgen_sdxl_tr", "data/corpora/train_real", "data/corpora/xgen_sdxl_tr", 43),
    # modern-generator domains so the stacker learns specialist behavior
    # on out-of-family inputs (D-026 lesson: FLUX gate caught MJ-probe drift)
    ("mj_train", "data/corpora/mj_train/real", "data/corpora/mj_train/fake", 100),
    ("flux_train", "data/corpora/mj_train/real", "data/corpora/flux_train/fake", 100),
    # user-input-distribution domain: screenshot views (D-032 r6) — stacker must
    # learn that canvas letterboxing + chrome + q60 re-encode is NOT fake evidence
    ("shot_aug", "data/corpora/shot_aug/real", "data/corpora/shot_aug/fake", 100),
    # Phase 4 generator-intake domains (spec B.3, D-049): aigenimages2026
    # CC-BY-4.0 train shards (build_intake_train.py) + B.1 self-generated
    # FLUX.2-klein corpus; shared real side = sha256-deduped aigen train
    # reals (gen_train/real, P4-C19 1000-cap, P4-C22-clean)
    ("flux-1-pro", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/flux-1-pro", 100),
    ("gpt4o", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/gpt4o", 100),
    ("hidream", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/hidream", 100),
    ("ideogram3", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/ideogram3", 100),
    ("imagen4", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/imagen4", 100),
    ("nanao-banana", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/nanao-banana", 100),
    ("seedream3-0", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/seedream3-0", 100),
    ("flux2-klein-4b", "data/corpora/gen_train/real",
     "data/corpora/gen_intake_train/flux2-klein-4b", 100),
]

GATES = [  # (gate name, real dir, fake dir)
    ("aigc", "data/corpora/eval_real", "data/corpora/eval_fake"),
    ("cdf", "data/corpora/cdf_flat_real", "data/corpora/cdf_flat_fake"),
    ("ffpp", None, None),  # flat sample dirs built from ffpp_faces/test
    ("mj", "data/corpora/modern_exam/real_ext", "data/corpora/modern_exam/fake_mj"),
    ("flux", "data/corpora/modern_exam/real_ext", "data/corpora/modern_exam/fake_flux"),
    # Phase 4 intake-domain gates (spec B.3, D-027: every intake domain is
    # gated on its Stage A exam slug; flux2-klein-4b gates on its
    # SELF-GENERATED exam — disclosed in-domain regression gate, not a
    # generalization test; D-049: TOL -0.005 like the other modern gates)
    ("flux-1-pro", "data/corpora/gen_exam/flux-1-pro/real",
     "data/corpora/gen_exam/flux-1-pro/fake"),
    ("gpt4o", "data/corpora/gen_exam/gpt4o/real",
     "data/corpora/gen_exam/gpt4o/fake"),
    ("hidream", "data/corpora/gen_exam/hidream/real",
     "data/corpora/gen_exam/hidream/fake"),
    ("ideogram3", "data/corpora/gen_exam/ideogram3/real",
     "data/corpora/gen_exam/ideogram3/fake"),
    ("imagen4", "data/corpora/gen_exam/imagen4/real",
     "data/corpora/gen_exam/imagen4/fake"),
    ("nanao-banana", "data/corpora/gen_exam/nanao-banana/real",
     "data/corpora/gen_exam/nanao-banana/fake"),
    ("seedream3-0", "data/corpora/gen_exam/seedream3-0/real",
     "data/corpora/gen_exam/seedream3-0/fake"),
    ("flux2-klein-4b", "data/corpora/gen_exam/flux2-klein-4b/real",
     "data/corpora/gen_exam/flux2-klein-4b/fake"),
]
# gate tolerances (auc_new - auc_old); ffpp informational + regression guard
TOL = {"aigc": -0.005, "cdf": -0.010, "ffpp": -1.0, "mj": -0.005, "flux": -0.005,
       # Phase 4 intake gates (D-049)
       "flux-1-pro": -0.005, "gpt4o": -0.005, "hidream": -0.005,
       "ideogram3": -0.005, "imagen4": -0.005, "nanao-banana": -0.005,
       "seedream3-0": -0.005, "flux2-klein-4b": -0.005}


def run_bench(real, fake, out, limit, stackers):
    env = dict(os.environ)
    env["PATH"] = f"{WS/'tools'/'bin'}:{env['PATH']}"
    cmd = [str(PY), str(BENCH), "--real", str(WS/real), "--fake", str(WS/fake),
           "--out", str(out), "--no-ladder", "--channels", "all", "--limit", str(limit)]
    for s in stackers:
        cmd += ["--with-stacker", str(s)]
    r = subprocess.run(cmd, env=env, cwd=WS)
    if r.returncode != 0:
        raise SystemExit(f"benchmark failed: {' '.join(cmd)}")


def stage_dump(work, smoke):
    limits = 4 if smoke else None
    for name, real, fake, lim in CORPORA:
        out = work / f"dump_{name}"
        if (out / "scores.csv").exists():
            print(f"[dump] {name}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        run_bench(real, fake, out, limits or lim, stackers=[])
        print(f"[dump] {name}: done in {(time.time()-t0)/60:.1f} min", flush=True)


def stage_train(work):
    out = work / "candidate"
    if (out / "stacker.joblib").exists():
        print("[train] candidate exists, skipped", flush=True)
        return
    csvs = [str(work / f"dump_{name}" / "scores.csv") for name, *_ in CORPORA]
    cmd = [str(PY), str(WS / "deepfake/scripts/train_stacker.py"),
           "--csv", *csvs, "--out", str(out)]
    r = subprocess.run(cmd, cwd=WS)
    if r.returncode != 0:
        raise SystemExit("stacker training failed")
    meta = json.loads((out / "stacker_meta.json").read_text())
    print(f"[train] candidate: val_auc={meta['val_auc']} modules={meta['module_ids']}",
          flush=True)


def build_ffpp_gate(work, gate_limit):
    """Flat sample dirs of face crops (copies) from the held-out FF++ test split."""
    real_dst, fake_dst = work / "ffpp_real", work / "ffpp_fake"
    if real_dst.exists() and fake_dst.exists():
        return real_dst, fake_dst
    real_dst.mkdir(parents=True, exist_ok=True)
    fake_dst.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    src_real = sorted((WS / "data/corpora/ffpp_faces/test/real").rglob("frame_*.jpg"))
    for i, p in enumerate(rng.permutation(src_real)[:gate_limit]):
        shutil.copy(p, real_dst / f"r{i:04d}_{p.parent.name}.jpg")
    methods = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    per = max(1, gate_limit // len(methods))
    k = 0
    for m in methods:
        srcs = sorted((WS / f"data/corpora/ffpp_faces/test/{m}").rglob("frame_*.jpg"))
        for p in rng.permutation(srcs)[:per]:
            shutil.copy(p, fake_dst / f"f{k:04d}_{p.parent.name}.jpg")
            k += 1
    print(f"[gate] ffpp sets: {len(list(real_dst.iterdir()))} real / "
          f"{len(list(fake_dst.iterdir()))} fake", flush=True)
    return real_dst, fake_dst


def auc_from_scores(csv_path, tag):
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r.get("variant") == "clean"
            and r.get("label") in ("0", "1")]
    y = np.array([int(r["label"]) for r in rows])
    s = -np.array([float(r[f"p_real_{tag}"]) for r in rows])
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)), len(y)


def _bool_rate(csv_path, col):
    rows = list(csv.DictReader(open(csv_path)))
    vals = [str(r.get(col, "")).strip().lower() in ("true", "1", "1.0") for r in rows
            if r.get("variant") == "clean" and r.get("label") in ("0", "1")]
    return round(sum(vals) / max(1, len(vals)), 4)


def stage_gate(work, gate_limit, smoke, old_stacker):
    cand = work / "candidate" / "stacker.joblib"
    old_tag = Path(old_stacker).parent.name
    cand_tag = cand.parent.name
    gl = 3 if smoke else gate_limit
    results = {}
    for name, real, fake in GATES:
        out = work / f"gate_{name}"
        if name == "ffpp":
            real, fake = build_ffpp_gate(work, gl)
        if not (out / "scores.csv").exists():
            run_bench(real, fake, out, gl, stackers=[old_stacker, cand])
        auc_o, n = auc_from_scores(out / "scores.csv", old_tag)
        auc_c, _ = auc_from_scores(out / "scores.csv", cand_tag)
        ok = auc_c >= auc_o + TOL[name]
        results[name] = {"auc_old": round(auc_o, 4), "auc_new": round(auc_c, 4),
                         "delta": round(auc_c - auc_o, 4), "n": n, "pass": bool(ok),
                         "abstained_rate_old": _bool_rate(out / "scores.csv",
                                                          f"abstained_{old_tag}"),
                         "abstained_rate_new": _bool_rate(out / "scores.csv",
                                                          f"abstained_{cand_tag}"),
                         "out_of_conflict_rate_new": _bool_rate(
                             out / "scores.csv", f"out_of_conflict_{cand_tag}")}
        print(f"[gate] {name}: old={auc_o:.4f} new={auc_c:.4f} "
              f"delta={auc_c-auc_o:+.4f} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-limit", type=int, default=100)
    ap.add_argument("--stage", default="all",
                    choices=["all", "dump", "train", "gate", "verdict"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--work", default=None, help="work dir (default artifacts/restack_r1)")
    ap.add_argument("--old", default=None,
                    help="incumbent stacker for A/B gates (default: argus_detect.STACKER)")
    ap.add_argument("--promote-to", default=None,
                    help="artifact dir name for promotion (required with "
                         "--promote; e.g. p6v10)")
    ap.add_argument("--promote", action="store_true",
                    help="on PASS, copy candidate to artifacts/<promote-to>/")
    args = ap.parse_args()
    if args.promote and not args.promote_to:
        raise SystemExit("--promote requires --promote-to <tag>")

    old_stacker = Path(args.old) if args.old else _wired_stacker()
    print(f"[main] incumbent stacker: {old_stacker}", flush=True)
    old_tag = old_stacker.parent.name
    work = Path(args.work) if args.work else (
        WS / "artifacts" / ("restack_smoke" if args.smoke else "restack_r1"))
    work.mkdir(parents=True, exist_ok=True)
    stages = (["dump", "train", "gate", "verdict"] if args.stage == "all"
              else [args.stage])
    gates = None
    for st in stages:
        if st == "dump":
            stage_dump(work, args.smoke)
        elif st == "train":
            stage_train(work)
        elif st == "gate":
            gates = stage_gate(work, args.gate_limit, args.smoke, old_stacker)
            (work / "gates.json").write_text(json.dumps(gates, indent=2))
        elif st == "verdict":
            gp = work / "gates.json"
            if gates is None and gp.exists():
                gates = json.loads(gp.read_text())
            if gates is None:
                raise SystemExit("no gate results — run the gate stage first")
            # verdict over ALL gates (original 5 + Phase 4 intake); ffpp's
            # TOL -1.0 keeps it non-binding, so this is behavior-preserving
            # for the pre-Phase-4 configuration
            promote = all(gates[g]["pass"] for g in gates)
            verdict = "PROMOTE" if promote else "HOLD"
            print(f"\nVERDICT: {verdict}  (aigc={gates['aigc']['pass']}, "
                  f"cdf={gates['cdf']['pass']}, mj={gates['mj']['pass']}, "
                  f"flux={gates['flux']['pass']}, ffpp_gain="
                  f"{gates['ffpp']['delta']:+.4f}, mj_gain={gates['mj']['delta']:+.4f})",
                  flush=True)
            if promote and args.promote:
                dst = WS / "artifacts" / args.promote_to
                dst.mkdir(parents=True, exist_ok=True)
                shutil.copy(work / "candidate" / "stacker.joblib", dst / "stacker.joblib")
                shutil.copy(work / "candidate" / "stacker_meta.json",
                            dst / "stacker_meta.json")
                print(f"promoted -> {dst} (wire argus_detect STACKER to activate)",
                      flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
