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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from channels import active_ids  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "integrations" / "Argus_Master" / "backend"))
from fusion import RELIABILITY_FLOOR  # noqa: E402

CAPS = ["HIGH", "MODERATE", "LOW"]


def featurize(df: pd.DataFrame, module_ids: list):
    cols, names = [], []
    for m in module_ids:
        v = df[m].astype(float)
        r = df[f"{m}_rel"].astype(float)
        mask = v.isna() | r.isna() | (r < RELIABILITY_FLOOR)
        v = v.mask(mask, 0.0).fillna(0.0).to_numpy()
        r = r.mask(mask, 0.0).fillna(0.0).to_numpy()
        cols += [v, r]
        names += [f"{m}_e", f"{m}_r"]
    cap = pd.get_dummies(df["capacity"]).reindex(columns=CAPS, fill_value=0).to_numpy()
    extra, extra_names = [], []
    for col in ("n_survivors", "contradiction_count", "contradiction_mass"):
        if col not in df.columns:
            raise SystemExit(f"{col!r} missing from dump CSVs — re-dump with updated run_benchmark")
        extra.append(df[col].astype(float).fillna(0.0).to_numpy())
        extra_names.append(col)
    return (np.column_stack(cols + [cap] + extra),
            names + [f"cap_{c}" for c in CAPS] + extra_names)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/p6")
    ap.add_argument("--modules", default=None,
                    help="comma-separated module ids; default = channels.active_ids() "
                         "(core + every registered channel with a joblib on disk)")
    args = ap.parse_args()
    MODULE_IDS = ([m.strip() for m in args.modules.split(",")] if args.modules
                  else active_ids())

    dfs = []
    for p in args.csv:
        d = pd.read_csv(p)
        keep = d[d["variant"] == "clean"] if "variant" in d else d
        dfs.append(keep[keep["label"].isin([0, 1])])
        print(f"{p}: {len(keep)} rows")
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["sha"])
    if "conflict" not in df.columns:
        raise SystemExit("dump CSVs lack 'conflict' column — re-dump")
    conflict_p99 = float(np.quantile(df["conflict"].astype(float), 0.99))
    X, names = featurize(df, MODULE_IDS)
    y = df["label"].astype(int).to_numpy()
    print(f"stacker dataset: {X.shape}, fakes={y.sum()}")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)
    base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=11)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_tr, y_tr)
    p = clf.predict_proba(X_va)[:, 1]
    p_top = np.maximum(p, 1.0 - p)
    thr = float(np.quantile(p_top, 0.10))
    alpha_cal = float(np.clip(1.0 - thr, 0.02, 0.60))
    conformal_flagged = bool(thr < 0.40)
    print(f"[b3] conformal alpha={alpha_cal} p10={thr} calib_n={len(X_va)} "
          f"flagged={conformal_flagged}")
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
        "conflict_p99": conflict_p99,
        "conflict_p99_n": int(len(df)),
        "trained_on": args.csv,
        "caveat": "trained on train-side corpora only; eval rows never seen",
        "conformal_alpha": round(alpha_cal, 6),
        "conformal_p10_top_prob": round(thr, 6),
        "conformal_calib_n": int(len(X_va)),
        "conformal_method": "split-conformal: 1-P10(max(P_FAKE,1-P_FAKE)) on the val split predicted by the X_tr-fit model; D-035 train-only",
        "conformal_flagged": conformal_flagged,
    }, indent=2))
    print(f"saved {out/'stacker.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
