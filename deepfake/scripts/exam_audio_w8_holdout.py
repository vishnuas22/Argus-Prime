"""W8 S1 exam — held-out generator generalization for the p12 specialist.

Scores the 14 SEALED holdout generators (840 utts, spec §2.2 — never seen in
specialist training) through the specialist bundle, using the CACHED clean-view
vectors (parity-proven extractor; `:clean:` views only = serve fidelity — the
aug views are training augmentation, not court input).

Utterance aggregation mirrors detect_audio exactly (argus_detect.py:365):
  p_mean  = mean(window probs);  z_mean = mean(window z-norms)
  ooc     = z_mean > bundle coverage_threshold;  abstain = ooc
  catch   = (not ooc) and p_mean > 0.5          (abstention counts as a miss)

Bars (PRE-REGISTERED 2026-10-30, spec §4 S1 — BINDING, D-046):
  aggregate catch >= 0.55  AND  per-generator catch >= 0.30 for >= 10 of 14

Usage: ./.venv/bin/python deepfake/scripts/exam_audio_w8_holdout.py \
    [--bundle artifacts/p12_specialist/audio_probe.joblib] [--out <dir>] [--force]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

WS = Path(__file__).resolve().parents[2]
HOLDOUT = WS / "artifacts/generator_intake/w8_diet_sealed/holdout_w8.jsonl"
CACHE = WS / "data/caches/asv_w7_mlaad.npz"
LABELS = WS / "data/caches/asv_w7_mlaad.labels.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="artifacts/p12_specialist/audio_probe.joblib")
    ap.add_argument("--out", default="artifacts/exam_audio_w8_holdout_p12specialist")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = WS / args.out
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: out dir non-empty: {out} (use --force)", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(WS / args.bundle)
    mu, sd = bundle["mu"], bundle["sd"]
    clf, thresh = bundle["clf"], bundle.get("coverage_threshold")
    if not thresh:
        print("ERROR: bundle has no coverage_threshold (run inject_audio_coverage)",
              file=sys.stderr)
        return 1

    hold = [json.loads(l) for l in HOLDOUT.read_text().splitlines() if l.strip()]
    gen_of_uid = {r["uid"]: r["path"].split("data/raw/mlaad/")[-1].split("/")[2]
                  for r in hold}
    z = np.load(CACHE)
    labels = json.loads(LABELS.read_text())

    # group clean-view window vectors per uid
    by_uid = defaultdict(list)
    for k in z.files:
        uid, view, _win = k.split(":")
        if view == "clean" and uid in gen_of_uid:
            by_uid[uid].append(k)
    n_vec = sum(len(v) for v in by_uid.values())
    print(f"holdout: {len(by_uid)} uids, {n_vec} clean-view vectors, "
          f"threshold={thresh}", flush=True)

    rows, misses = [], 0
    catch_by_gen = defaultdict(lambda: [0, 0])   # gen -> [catches, total]
    for uid, keys in sorted(by_uid.items()):
        H = np.stack([z[k] for k in keys]).astype(np.float32)
        Xs = (H - mu) / sd
        p = clf.predict_proba(Xs)[:, 1]
        zn = np.linalg.norm(Xs, axis=1)
        p_mean, z_mean = float(p.mean()), float(zn.mean())
        ooc = bool(z_mean > thresh)
        catch = bool((not ooc) and p_mean > 0.5)
        g = gen_of_uid[uid]
        catch_by_gen[g][1] += 1
        catch_by_gen[g][0] += int(catch)
        rows.append({"uid": uid, "generator": g, "n_windows": len(keys),
                     "p_mean": round(p_mean, 4), "z_mean": round(z_mean, 1),
                     "ooc": ooc, "catch": catch})
        misses += int(not catch)

    per_gen = {g: {"catch": round(c / n, 4), "n": n} for g, (c, n)
               in sorted(catch_by_gen.items())}
    agg = 1.0 - misses / len(rows)
    gens_ge_030 = sum(1 for d in per_gen.values() if d["catch"] >= 0.30)
    metrics = {
        "exam": "S1 held-out generator generalization (specialist-alone)",
        "n_utts": len(rows), "n_vectors": n_vec,
        "aggregate_catch": round(agg, 4),
        "gens_ge_0.30": gens_ge_030,
        "per_generator": per_gen,
        "abstain_rate": round(sum(r["ooc"] for r in rows) / len(rows), 4),
        "bars": {"aggregate_catch >= 0.55": agg >= 0.55,
                 "gens_ge_0.30 >= 10/14": gens_ge_030 >= 10},
        "bundle": args.bundle,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    with open(out / "scores.csv", "w") as f:
        f.write("uid,generator,n_windows,p_mean,z_mean,ooc,catch\n")
        for r in rows:
            f.write(f"{r['uid']},{r['generator']},{r['n_windows']},{r['p_mean']},"
                    f"{r['z_mean']},{r['ooc']},{r['catch']}\n")
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_generator"}, indent=1),
          flush=True)
    for g, d in per_gen.items():
        print(f"  {g:38s} catch={d['catch']:.3f} n={d['n']}", flush=True)
    all_pass = all(metrics["bars"].values())
    print(f"S1 VERDICT: {'PASS' if all_pass else 'FAIL'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
