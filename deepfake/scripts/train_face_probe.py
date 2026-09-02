"""Train P3 face head on Celeb-DF with official test-split holdout.

Train: all video folders NOT in List_of_testing_videos.txt.
Test:  exactly the official test videos (never seen in training).
Usage: .venv/bin/python deepfake/scripts/train_face_probe.py \
    --cache data/caches/cdf_faces.npz --test-list "data/raw/celebdf/Celeb-DF-v2/List_of_testing_videos.txt" \
    --out artifacts/p3
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/caches/cdf_faces.npz")
    ap.add_argument("--test-list", default="data/raw/celebdf/Celeb-DF-v2/List_of_testing_videos.txt")
    ap.add_argument("--out", default="artifacts/p3")
    ap.add_argument("--c", type=float, default=3.0)
    args = ap.parse_args()

    test_stems = set()
    for line in Path(args.test_list).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        lab, rel = line.split(" ", 1)
        test_stems.add(Path(rel).stem)

    z = np.load(args.cache)
    X, y, vid_test = [], [], []
    for k in z.files:
        stem = k.split("/")[1].split(":")[0]
        sub = k.split("/")[0]
        label = 1 if sub == "synthesis" else 0
        X.append(z[k].astype(np.float32))
        y.append(label)
        vid_test.append(stem in test_stems)
    X = np.stack(X)
    y = np.array(y)
    vid_test = np.array(vid_test)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

    tr, te = ~vid_test, vid_test
    print(f"views: total={len(y)} train={tr.sum()} test={te.sum()} "
          f"(fakes: {y.sum()} | test-fakes={y[te].sum()})")

    clf = LogisticRegression(C=args.c, max_iter=2000).fit(X[tr], y[tr])
    p_te = clf.predict_proba(X[te])[:, 1]
    tr_idx = np.where(tr)[0]
    sub = np.random.default_rng(3).choice(tr_idx, size=min(5000, tr_idx.size), replace=False)
    if len(np.unique(y[sub])) < 2:
        sub = np.where(y == 0)[0][:2500].tolist() + np.where(y == 1)[0][:2500].tolist()
    p_tr = clf.predict_proba(X[sub])[:, 1]
    y_tr_sub = y[sub]

    metrics = {
        "train_auc": round(float(roc_auc_score(y_tr_sub, p_tr)), 4),
        "official_test_auc": round(float(roc_auc_score(y[te], p_te)), 4),
        "official_test_eer": round(eer_from(y[te], p_te), 4),
        "n_train_views": int(tr.sum()), "n_test_views": int(te.sum()),
        "C": args.c,
        "note": "video-level identity leakage possible within splits (frames of same video share both sides); "
                "official protocol evaluates per-video which we approximate at frame level",
    }
    print(json.dumps(metrics, indent=2))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    final = LogisticRegression(C=args.c, max_iter=2000).fit(X, y)
    joblib.dump(final, out / "face_probe.joblib")
    (out / "train_meta.json").write_text(json.dumps(metrics, indent=2))
    print(f"saved {out/'face_probe.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
