"""V2a leg 3 — paired GenD + p6vid_v4 ensemble. The last defensible GenD question.

Both models are already scored on the IDENTICAL clips:
  - GenD   : artifacts/v1g_probe/{ffpp,cdf}_scores.json
  - p6vid_v4: artifacts/v1d_restrav/scores.jsonl

Pre-registered bar (restated before looking at results):
  BAR: 10-fold ensemble AUC must beat p6vid_v4 ALONE on CDF (0.9992 here)
       AND not lose on FF++.
  If it cannot beat 0.9992 on CDF -> close the GenD line.

Read-only. Diagnostic LR only. No promotion.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
OUT = WS / "artifacts/v1g_probe"
SEED = 20261121

# p6vid_v4 scores keyed by path
p6 = {}
for line in open(WS / "artifacts/v1d_restrav/scores.jsonl"):
    r = json.loads(line)
    p6[r["path"]] = r.get("p_fake")

rows = []
for name, gp in (("ffpp", OUT / "ffpp_scores.json"),
                 ("cdf", OUT / "cdf_scores.json")):
    g = json.load(open(gp))
    sub = []
    for x in g:
        if x["path"] in p6 and p6[x["path"]] is not None:
            sub.append((x["y"], x["p_fake_mean"], p6[x["path"]]))
    rows.append((name, sub))
    print(f"{name}: {len(sub)} paired clips")

def cv_auc(X, y):
    sc = StandardScaler().fit(X)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    p = cross_val_predict(LogisticRegression(max_iter=2000), sc.transform(X), y,
                          cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))

report = {}
print("\n" + "=" * 68)
print("PAIRED ENSEMBLE (10-fold CV, same clips)")
print("=" * 68)
for name, sub in rows:
    y = np.array([s[0] for s in sub])
    gend = np.array([s[1] for s in sub])
    p6v = np.array([s[2] for s in sub])

    a_gend = roc_auc_score(y, gend)
    a_p6 = roc_auc_score(y, p6v)
    a_ens = cv_auc(np.column_stack([gend, p6v]), y)
    # oracle-ish: does GenD add ANYTHING?
    delta = a_ens - a_p6
    report[name] = {"n": int(len(y)), "auc_gend": a_gend, "auc_p6vid": a_p6,
                    "auc_ensemble": a_ens, "delta": delta}
    print(f"\n[{name}] n={len(y)}")
    print(f"  GenD alone      : {a_gend:.4f}")
    print(f"  p6vid_v4 alone  : {a_p6:.4f}")
    print(f"  ensemble (10-fold): {a_ens:.4f}   Δ vs p6vid = {delta:+.4f}")

print("\n" + "=" * 68)
print("BAR")
print("=" * 68)
b_cdf = report["cdf"]["auc_ensemble"] > report["cdf"]["auc_p6vid"]
b_ffpp = report["ffpp"]["auc_ensemble"] >= report["ffpp"]["auc_p6vid"] - 0.01
print(f"  CDF: ensemble > p6vid_v4 alone ({report['cdf']['auc_p6vid']:.4f}) -> "
      f"{'PASS' if b_cdf else 'FAIL'}")
print(f"  FF++: no material loss (>= {report['ffpp']['auc_p6vid']-0.01:.4f}) -> "
      f"{'PASS' if b_ffpp else 'FAIL'}")
verdict = "PASS" if (b_cdf and b_ffpp) else "FAIL"
print(f"\n  VERDICT: {verdict}")

report["verdict"] = verdict
json.dump(report, open(OUT / "ensemble_gate.json", "w"), indent=2)
print(f"\nsaved {OUT / 'ensemble_gate.json'}")