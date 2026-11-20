"""W9 STEP 4b (D-079 SEALED): train the fused stacker (p13_fused).

FUSION LAW (spec §2 — ABSOLUTE):
- ONE calibrated stacker: HistGradientBoostingClassifier(max_iter=200,
  learning_rate=0.08, random_state=11) inside CalibratedClassifierCV(
  method="sigmoid", cv=3) — byte-mirrors train_stacker.py:78-79.
- Scores enter as FEATURES ONLY. max()-composition, OR-of-fake-calls,
  mean-of-probes, gate-overwrite, abstention-as-vote: BANNED (mechanism-a).
- Stacker trains on TRAIN-split rows ONLY; val-split rows stay held out and
  feed the val AUC receipt + the conformal threshold (train-only, D-035 law:
  thr = P10(max(p, 1-p)) on val predictions — train_stacker.py:82-84 pattern).
- p4 snapshot COPIED into the bundle with sha256 receipt; artifacts/p4 itself
  is NEVER touched.

Usage:
  ./.venv/bin/python deepfake/scripts/train_audio_w9_stacker.py \
      --rows artifacts/p13_fused --out artifacts/p13_fused
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])


def eer_from(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    d = fpr - (1 - tpr)
    i = int(np.argmin(np.abs(d)))
    if 0 < i < len(fpr) and d[i] != d[i - 1]:
        w = -d[i - 1] / (d[i] - d[i - 1])
        return float(fpr[i - 1] + w * (fpr[i] - fpr[i - 1]))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="artifacts/p13_fused")
    ap.add_argument("--out", default="artifacts/p13_fused")
    args = ap.parse_args()
    rows = Path(args.rows)

    meta_in = json.loads((rows / "rows_meta.json").read_text())
    feat_names = meta_in["feature_names"]
    if any(n.startswith("prosody_") for n in feat_names):
        raise SystemExit("prosody features present — Channel D was CUT at sealing "
                         "(D-079) — STOP")
    df_tr = pd.read_csv(rows / "rows_train.csv")
    df_va = pd.read_csv(rows / "rows_val.csv")
    X_tr = df_tr[feat_names].to_numpy(dtype=float)
    y_tr = df_tr["label"].to_numpy(dtype=int)
    X_va = df_va[feat_names].to_numpy(dtype=float)
    y_va = df_va["label"].to_numpy(dtype=int)
    print(f"stacker dataset: train {X_tr.shape} fakes={int(y_tr.sum())} | "
          f"val {X_va.shape} fakes={int(y_va.sum())}", flush=True)

    base = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                          random_state=11)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3).fit(X_tr, y_tr)

    p_va = clf.predict_proba(X_va)[:, 1]
    auc = float(roc_auc_score(y_va, p_va))
    eer = eer_from(y_va, p_va)
    p_top = np.maximum(p_va, 1.0 - p_va)
    thr = float(np.quantile(p_top, 0.10))
    alpha_cal = float(np.clip(1.0 - thr, 0.02, 0.60))
    conformal_flagged = bool(thr < 0.40)
    print(f"VAL auc={auc:.4f} eer={eer:.4f} | conformal p10={thr:.4f} "
          f"alpha={alpha_cal:.4f} flagged={conformal_flagged}", flush=True)

    out = Path(args.out)
    # Bundle file carries the fusion flag itself: the dormant branch in
    # argus_detect.detect_audio does joblib.load(AUDIO_PROBE) and routes on
    # bundle.get("fusion") — so the pointer target must BE a dict with that
    # key. load_court unwraps ["stacker"] (audio_fusion.py contract).
    joblib.dump({"fusion": "feature-level-only", "stacker": clf},
                out / "audio_court.joblib")

    # diet manifest copied into the bundle (spec §6 bundle layout)
    diet_sha = hashlib.sha256(
        (WS / "artifacts/generator_intake/w9_diet_sealed/manifest.jsonl").read_bytes()
    ).hexdigest()
    shutil.copy2(WS / "artifacts/generator_intake/w9_diet_sealed/manifest.jsonl",
                 out / "DIET_MANIFEST.jsonl")

    # p4 frozen snapshot (incumbent NEVER overwritten; copy + sha receipt)
    snap = out / "p4_snapshot"
    snap.mkdir(exist_ok=True)
    for f in ("audio_probe.joblib", "train_meta.json"):
        shutil.copy2(WS / "artifacts/p4" / f, snap / f)
    p4_sha = hashlib.sha256((snap / "audio_probe.joblib").read_bytes()).hexdigest()
    if p4_sha != meta_in["p4_snapshot_sha256"]:
        raise SystemExit("p4 snapshot sha mismatch vs rows_meta receipt — STOP")

    (out / "stacker_meta.json").write_text(json.dumps({
        "model": "HistGradientBoosting+sigmoid calibration (train_stacker.py:78-79 mirror)",
        "fusion": "feature-level-only",
        "fusion_law": "scores enter as FEATURES only; max()/OR/mean composition and "
                      "gate-overwrite BANNED (mechanism-a, patterns:33); abstention "
                      "never used as a vote",
        "feature_names": feat_names,
        "n_features": len(feat_names),
        "channel_d": "CUT at sealing (D-079) — no prosody features",
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
        "val_auc": round(auc, 4), "val_eer": round(eer, 4),
        "conformal_p10_top_prob": round(thr, 6),
        "conformal_alpha": round(alpha_cal, 6),
        "conformal_calib_n": int(len(X_va)),
        "conformal_method": "split-conformal: P10(max(P_FAKE,1-P_FAKE)) on held-out "
                            "val rows predicted by the train-fit stacker; D-035 train-only",
        "conformal_flagged": conformal_flagged,
        "abstention_rule": "(p4_ooc AND wavlm_ooc) OR (max(p_fused,1-p_fused) <= thr) "
                           "— pre-registered at sealing (D-079 OPEN QUESTION 2)",
        "b_oof": meta_in["b_oof"],
        "p4_snapshot_sha256": p4_sha,
        "diet_manifest_sha256": diet_sha,
        "seeds": {"diet": 42, "extraction": 123, "oof": 11, "stacker": 11},
        "trained_on": "w9_diet_sealed train split only; val rows never seen by the stacker",
        "caveat": "in-diet training; external generalization is the exam story (D1-D6)",
    }, indent=1))
    print(f"saved {out/'audio_court.joblib'} | p4 snapshot sha {p4_sha[:12]} | "
          f"diet sha {diet_sha[:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
