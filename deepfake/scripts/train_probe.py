"""Train the P5-lite AIGC linear probe (UnivFD recipe) on cached CLIP features.

L2-normalized frozen ViT-L/14 features -> logistic regression (C swept by CV).
Saves sklearn Pipeline + metadata JSON with held-out validation metrics.
Usage: .venv/bin/python deepfake/scripts/train_probe.py <real.npz> <fake.npz> --out artifacts/p5
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


def load(npz_path: Path):
    z = np.load(npz_path)
    X = np.stack([z[k] for k in z.files]).astype(np.float32)
    return X, list(z.files)


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
    ap.add_argument("real_npz")
    ap.add_argument("fake_npz", nargs="+")
    ap.add_argument("--out", default="artifacts/p5")
    args = ap.parse_args()

    Xr, nr = load(Path(args.real_npz))
    parts = [load(Path(f)) for f in args.fake_npz]
    Xf = np.concatenate([x for x, _ in parts])
    nf = sum(len(n) for _, n in parts)
    X = np.concatenate([Xr, Xf])
    y = np.array([0] * len(Xr) + [1] * len(Xf))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    print(f"features: {X.shape}, reals={len(Xr)} fakes={len(Xf)}")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.15, stratify=y, random_state=7)
    best_c, best_score = None, -1
    for c in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
        s = cross_val_score(clf, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=7),
                            scoring="roc_auc").mean()
        print(f"C={c}: cv_auc={s:.4f}")
        if s > best_score:
            best_c, best_score = c, s

    clf = LogisticRegression(C=best_c, max_iter=2000).fit(X_tr, y_tr)
    p = clf.predict_proba(X_va)[:, 1]
    val_auc = float(roc_auc_score(y_va, p))
    val_eer = eer_from(y_va, p)
    print(f"VAL: auc={val_auc:.4f} eer={val_eer:.4f} (n={len(y_va)}, C={best_c})")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = LogisticRegression(C=best_c, max_iter=2000).fit(X, y)
    joblib.dump(final, out / "probe.joblib")
    (out / "train_meta.json").write_text(json.dumps({
        "model": "LogisticRegression(L2-normalized CLIP ViT-L/14 penultimate)",
        "C": best_c, "cv_auc_train": round(float(best_score), 4),
        "val_auc": round(val_auc, 4), "val_eer": round(val_eer, 4),
        "n_train_views": int(len(X)), "n_real_files": len(nr), "n_fake_files": int(nf),
        "feature_dim": int(X.shape[1]),
        "views_per_image": ["clean", "jpeg75", "resize75"],
        "caveat": "in-domain training (SD1.5 family); cross-generator transfer untested",
    }, indent=2))
    print(f"saved: {out/'probe.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
