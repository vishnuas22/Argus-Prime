"""Train the inversion probe (SD-VAE round-trip features -> logistic).

v3 (r6/D-032): trained on screenshot-augmented views in addition to the
D-030 laundering views (JPEG q50/q30). Output: artifacts/inversion_probe/.
Usage: ./.venv/bin/python deepfake/scripts/train_inversion_probe.py <train.npz>
"""
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

WS = Path(__file__).resolve().parents[2]


def main():
    train_npz = Path(sys.argv[1]) if len(sys.argv) > 1 else WS / "data/caches/inversion_train_v3.npz"
    ex = np.load(WS / "data/caches/inversion_exam.npz", allow_pickle=True)
    tr = np.load(train_npz, allow_pickle=True)
    F, y = tr["features"], tr["labels"]
    print(f"train: {F.shape}, reals={int((y == 0).sum())} fakes={int((y == 1).sum())}", flush=True)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=3.0, max_iter=2000))
    cv = cross_val_score(clf, F, y, cv=5, scoring="roc_auc")
    print(f"5-fold CV AUC: {cv.mean():.4f} +/- {cv.std():.4f}", flush=True)
    clf.fit(F, y)

    Ef, ey, ep = ex["features"], ex["labels"], ex["paths"].astype(str)
    p = clf.predict_proba(Ef)[:, 1]
    m_mj = np.char.find(ep, "fake_mj") >= 0
    m_fx = np.char.find(ep, "fake_flux") >= 0
    m_sg = np.char.find(ep, "sg2_probe") >= 0
    m_real = np.char.find(ep, "real_ext") >= 0
    print(f"zero-shot FLUX AUC: {roc_auc_score(ey[m_fx | m_real], p[m_fx | m_real]):.4f}", flush=True)
    print(f"zero-shot MJ   AUC: {roc_auc_score(ey[m_mj | m_real], p[m_mj | m_real]):.4f}", flush=True)
    print(f"SG2: {int((p[m_sg] > 0.5).sum())}/{int(m_sg.sum())} detected, mean p={p[m_sg].mean():.3f}", flush=True)
    print(f"external reals FPR: {float((p[m_real] > 0.5).mean()):.3f}", flush=True)

    out = WS / "artifacts" / "inversion_probe"
    out.mkdir(exist_ok=True)
    joblib.dump(clf, out / "inversion_probe.joblib")
    print(f"saved probe -> {out / 'inversion_probe.joblib'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
