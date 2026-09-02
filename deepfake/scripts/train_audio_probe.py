"""Train P4 audio probe on cached XLS-R embeddings. Reports val EER (ASVspoof standard).
Usage: .venv/bin/python deepfake/scripts/train_audio_probe.py \
    --train data/caches/asv_train.npz --val data/caches/asv_val.npz --out artifacts/p4
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def eer_from(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    d = fpr - (1 - tpr)
    i = int(np.argmin(np.abs(d)))
    if 0 < i < len(fpr) and d[i] != d[i - 1]:
        w = -d[i - 1] / (d[i] - d[i - 1])
        return float(fpr[i - 1] + w * (fpr[i] - fpr[i - 1]))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def load(npz):
    z = np.load(npz)
    labels = json.loads(Path(npz).with_suffix(".labels.json").read_text())
    keys = list(z.files)
    X = np.stack([z[k] for k in keys]).astype(np.float32)
    y = np.array([labels.get(k, None) for k in keys])
    m = np.array([v is not None for v in y])
    keep = [k for k, mk in zip(keys, m) if mk]
    return X[m], y[m].astype(int), keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="artifacts/p4")
    args = ap.parse_args()

    Xtr, ytr, _ = load(args.train)
    Xva, yva, kva = load(args.val)
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xva = (Xva - mu) / sd
    print(f"train={Xtr.shape} fakes={int(ytr.sum())} | val={Xva.shape} fakes={int(yva.sum())}")

    clf = LogisticRegression(C=3.0, max_iter=3000).fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    auc = float(roc_auc_score(yva, p))
    eer = eer_from(yva, p)
    print(f"VAL: auc={auc:.4f} eer={eer:.4f}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "mu": mu, "sd": sd}, out / "audio_probe.joblib")
    (out / "train_meta.json").write_text(json.dumps({
        "model": "logistic on frozen wav2vec2-XLS-R-300m mean-pooled",
        "val_auc": round(auc, 4), "val_eer": round(eer, 4),
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "caveat": "in-domain ASVspoof LA; unknown-attack generalization is the eval-set story",
    }, indent=2))
    print(f"saved {out/'audio_probe.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
