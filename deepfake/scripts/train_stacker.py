"""Train the P6 fusion stacker: replaces reliability-weighted voting.

Learns P(fake | module evidence scores, reliabilities, capacity) from TRAIN-side
score dumps only (never eval rows). HistGradientBoosting + sigmoid calibration.
Usage:
  ./.venv/bin/python deepfake/scripts/train_stacker.py \
      --csv artifacts/stk_train_sd15/scores.csv artifacts/stk_train_x/scores.csv \
      --out artifacts/p6
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

MODULE_IDS = ["metadata", "compression_history", "spectral_probe",
              "real_distribution_probe", "perturbation_probe",
              "learned_aigc_probe", "learned_face_probe"]
CAPS = ["HIGH", "MODERATE", "LOW"]


def featurize(df: pd.DataFrame):
    cols, names = [], []
    for m in MODULE_IDS:
        v = df[m].astype(float).fillna(0.0).to_numpy()
        r = df[f"{m}_rel"].astype(float).fillna(0.0).to_numpy()
        cols += [v, r]
        names += [f"{m}_e", f"{m}_r"]
    cap = pd.get_dummies(df["capacity"]).reindex(columns=CAPS, fill_value=0).to_numpy()
    return np.column_stack(cols + [cap]), names + [f"cap_{c}" for c in CAPS]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/p6")
    args = ap.parse_args()

    dfs = []
    for p in args.csv:
        d = pd.read_csv(p)
        keep = d[d["variant"] == "clean"] if "variant" in d else d
        dfs.append(keep[keep["label"].isin([0, 1])])
        print(f"{p}: {len(keep)} rows")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["sha"])
    X, names = featurize(df)
    y = df["label"].astype(int).to_numpy()
    print(f"stacker dataset: {X.shape}, fakes={y.sum()}")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)
    base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=11)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_tr, y_tr)
    p = clf.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y_va, p)
    fpr, tpr, _ = roc_curve(y_va, p)
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    print(f"VAL auc={auc:.4f} eer~{(fpr[i]+1-tpr[i])/2:.4f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=11),
        method="sigmoid", cv=3).fit(X, y)
    joblib.dump(final, out / "stacker.joblib")
    (out / "stacker_meta.json").write_text(json.dumps({
        "model": "HistGradientBoosting+sigmoid calibration",
        "feature_names": names,
        "module_ids": MODULE_IDS,
        "n_rows": int(len(X)), "val_auc": round(float(auc), 4),
        "trained_on": args.csv,
        "caveat": "trained on train-side corpora only; eval rows never seen",
    }, indent=2))
    print(f"saved {out/'stacker.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
