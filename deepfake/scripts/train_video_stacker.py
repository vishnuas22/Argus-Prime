"""Train the VIDEO stacker (spec item 3.2) — mirror of train_stacker.py's
recipe, video feature space.

  .venv/bin/python deepfake/scripts/train_video_stacker.py \
      --csv artifacts/restack_video/dump_train/scores.csv \
            artifacts/restack_video/dump_val/scores.csv \
      --out artifacts/p6vid_v1

Laws kept verbatim from the image recipe:
  - rows SORTED BY sha before any split (order-sensitivity law, memory/patterns:34)
  - stratified holdout, random_state=11
  - HistGradientBoosting(max_iter=200, lr=0.08, rs=11) in CalibratedClassifierCV
    (method="sigmoid", cv=3)

ALSO fits the identity-drift calibration (item 2.2) from TRAIN-REAL rows and
writes drift_calibration.json beside the model — the drift module refuses to
vote without it (fail-closed). spike_thresh comes from the train dump's sidecar
(exactly the threshold its drift_spikes feature was built with) so the stacker
feature and the module agree bit-for-bit.
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
from video_channels import fit_drift_calibration  # noqa: E402
from video_features import FEATURE_NAMES_FULL, row_to_X  # noqa: E402

WS = Path(__file__).resolve().parents[2]
DFD_SPLIT_MANIFEST = WS / "data" / "raw" / "ffpp_dfd" / "split.json"
# W4b lever A (pre-registered, user-sealed 2026-10-31): the exact rational
# n_DFD-fakes/n_DFD-reals from the receipted W4 train append — up-weights
# DFD-family real rows to balance the family the stacker actually sees.
DFD_REAL_WEIGHT = 593 / 191


def dfd_weights(df: pd.DataFrame):
    """Per-row sample weights (W4b §2): DFD-family real rows carry
    DFD_REAL_WEIGHT, every other row 1.0. Family rule is a pure function of
    the row + the split-law artifact; call only AFTER the dedupe + sha-sort
    so the weight vector stays index-aligned with X/y by construction.
    Returns (w, audit) — audit carries the family-match counts the weight
    receipt needs."""
    dfd_sources = set(json.loads(DFD_SPLIT_MANIFEST.read_text()))
    is_dfd = (df["method"] == "DeepFakeDetection") | (
        df["file"].map(lambda f: Path(f).stem).isin(dfd_sources))
    y = df["label"].astype(int)
    w = np.where((y == 0) & is_dfd, DFD_REAL_WEIGHT, 1.0)
    audit = {"n_dfd_fakes": int((is_dfd & (y == 1)).sum()),
             "n_dfd_reals_weighted": int((is_dfd & (y == 0)).sum())}
    return w, audit


def fit_stacker(df: pd.DataFrame):
    """Rows -> (calibrated clf fit on ALL rows, val_auc, X, y).

    The single recipe used by train_video_stacker AND restack_video's LOMO
    retrains (leave-one-method-out gate) — one recipe, one feature space.
    W4c: the trainer builds X in the FULL space (FEATURE_NAMES_FULL — the
    24 deployed features + the 8 SCENE features)."""
    X = np.vstack([row_to_X(r, FEATURE_NAMES_FULL)
                   for r in df.to_dict("records")])
    y = df["label"].astype(int).to_numpy()
    w, _audit = dfd_weights(df)
    X_tr, X_va, y_tr, y_va, w_tr, _w_va = train_test_split(
        X, y, w, test_size=0.2, stratify=y, random_state=11)
    base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                          random_state=11)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(
        X_tr, y_tr, sample_weight=w_tr)
    auc = float(roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1]))
    final = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                       random_state=11),
        method="sigmoid", cv=3).fit(X, y, sample_weight=w)
    return final, auc, X, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="artifacts/p6vid_v1")
    ap.add_argument("--cal-csv", default=None,
                    help="CSV whose real rows fit the drift calibration "
                         "(default: the FIRST --csv, i.e. the train dump)")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="sface",
                    help="must match the embedder the dump used (recorded in "
                         "the sidecar; the serve path reads stacker_meta)")
    args = ap.parse_args()
    from video_face_track import set_embedder
    # guard: the dump's embedder must be the train's embedder — drift features
    # (and identity clustering) are embedder-dependent, and stacker_meta is
    # what the serve path will honour.
    sidecar = Path(args.csv[0]).parent / "drift_spike.json"
    if sidecar.exists():
        dumped = json.loads(sidecar.read_text()).get("embedder")
        if dumped and dumped != args.embedder:
            raise SystemExit(f"embedder mismatch: dump used '{dumped}' but "
                             f"--embedder {args.embedder} — re-dump or match")
    set_embedder(args.embedder)
    cal_csv = Path(args.cal_csv or args.csv[0])

    dfs = []
    for p in args.csv:
        d = pd.read_csv(p)
        d = d[d["label"].isin([0, 1])]
        dfs.append(d)
        print(f"{p}: {len(d)} rows")
    df = pd.concat(dfs, ignore_index=True)
    _w_raw, raw_audit = dfd_weights(df)
    n_raw = len(df)
    df = df.drop_duplicates(subset=["sha256_12"])
    n_dropped = n_raw - len(df)
    df = df.sort_values("sha256_12", kind="mergesort").reset_index(drop=True)
    missing = [c for c in FEATURE_NAMES_FULL if c not in df.columns]
    if missing:
        raise SystemExit(f"dump CSVs lack feature columns {missing} — re-dump")
    print(f"video stacker dataset: {df.shape}, fakes={int(df['label'].sum())}")
    _w, audit = dfd_weights(df)
    reconciled = (n_raw == len(df) + n_dropped
                  and audit["n_dfd_fakes"] <= raw_audit["n_dfd_fakes"]
                  and audit["n_dfd_reals_weighted"]
                  <= raw_audit["n_dfd_reals_weighted"])
    print(f"weight audit: weight_rule=dfd_real_upweight k={DFD_REAL_WEIGHT} "
          f"n_rows={len(df)} n_dropped_dupes={n_dropped} "
          f"n_dfd_fakes={audit['n_dfd_fakes']} "
          f"(pre-dedupe {raw_audit['n_dfd_fakes']}) "
          f"n_dfd_reals_weighted={audit['n_dfd_reals_weighted']} "
          f"(pre-dedupe {raw_audit['n_dfd_reals_weighted']}) -> "
          f"{'RECONCILED' if reconciled else 'UNRECONCILED'}", flush=True)
    if audit["n_dfd_reals_weighted"] == 0 or not reconciled:
        raise SystemExit("weight audit FAILED — STOP (W4b: a silent-empty or "
                         "unreconciled DFD family match makes the retrain a "
                         "no-op)")

    # --- drift calibration from TRAIN-REAL rows only (D-035 law) -----------
    cal_df = pd.read_csv(cal_csv)
    cal_rows = cal_df[(cal_df["label"] == 0) & (cal_df["n_drift_tracks"] > 0)]
    cal = fit_drift_calibration(cal_rows.to_dict("records"))
    sidecar = cal_csv.parent / "drift_spike.json"
    if sidecar.exists():  # exact threshold the dump's feature used
        cal["spike_thresh"] = float(json.loads(sidecar.read_text())["spike_thresh"])
    if cal["n_videos"] < 300:
        print(f"WARN: drift calibration fit on {cal['n_videos']} videos "
              f"(spec wants >=300)", flush=True)
    if not cal["p50"] < cal["p90"]:
        raise SystemExit(f"drift calibration invalid: p50={cal['p50']} p90={cal['p90']}")

    final, auc, X, _y = fit_stacker(df)
    fpr, tpr, _ = roc_curve(df["label"].astype(int),
                            final.predict_proba(X)[:, 1])
    i = int(np.argmin(np.abs(fpr - (1 - tpr))))
    print(f"VAL auc={auc:.4f} eer~{(fpr[i] + 1 - tpr[i]) / 2:.4f} (holdout-reported "
          f"auc is the fit_stacker internal 80/20 split)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out / "stacker.joblib")
    from video_face_track import EMBEDDER_NAME
    (out / "drift_calibration.json").write_text(json.dumps(cal, indent=2))
    (out / "stacker_meta.json").write_text(json.dumps({
        "model": "HistGradientBoosting+sigmoid calibration",
        "feature_names": FEATURE_NAMES_FULL,
        "feature_space": "video_scene",
        "n_rows": int(len(X)), "val_auc": round(float(auc), 4),
        "weight_rule": "dfd_real_upweight",
        "k": DFD_REAL_WEIGHT,
        "n_dropped_dupes": int(n_dropped),
        "n_dfd_fakes": audit["n_dfd_fakes"],
        "n_dfd_reals_weighted": audit["n_dfd_reals_weighted"],
        "trained_on": args.csv,
        "drift_calibration": str(out / "drift_calibration.json"),
        "drift_calibration_fit": cal,
        "frame_budget": 12,
        "embedder": EMBEDDER_NAME,
        "caveat": ("trained on the listed dumps only; FF++ test split and "
                   "Celeb-DF List_of_testing_videos never seen "
                   "(Celeb-DF train-diet videos ARE in-distribution when provided); "
                   "DFD exam-split sources (data/raw/ffpp_dfd/split.json) are "
                   "held out of every dump when the DFD diet is present; "
                   "feature space = video_scene (W4c): the 24 deployed "
                   "features + the 8 SCENE full-frame/context features — "
                   "CANNOT serve until the FEATURE_NAMES flip"),
    }, indent=2))
    print(f"saved {out / 'stacker.joblib'} (val_auc={auc:.4f}, "
          f"drift cal n={cal['n_videos']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
