"""Full-P3 multi-domain face probe: train {Celeb-DF + WildDeepfake (+ FF++ with --with-ffpp)}, exam on held-out tests.

Usage: .venv/bin/python deepfake/scripts/train_face_multidomain.py --out artifacts/p3_full
       .venv/bin/python deepfake/scripts/train_face_multidomain.py --with-ffpp --out artifacts/p3_ffpp

FF++ domain: data/caches/ffpp_train.npz + ffpp_test.npz built from
data/corpora/ffpp_faces/{train,test} via build_feature_cache.py. Cache keys are
<label>/<video>/frame_XXX.jpg:<view> so load_cache's first-component label rule
applies unchanged. FF++ splits are identity-disjoint from each other and from
Celeb-DF/WildDeepfake.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).parent))
from train_face_probe import eer_from  # noqa: E402


def load_cache(npz):
    z = np.load(npz)
    X = np.stack([z[k] for k in z.files]).astype(np.float32)
    def _lab(k):
        first = k.split("/")[0].lower()
        return 0 if "real" in first else 1
    y = np.array([_lab(k) for k in z.files])
    return X, y, list(z.files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/p3_full")
    ap.add_argument("--c", type=float, default=3.0)
    ap.add_argument("--with-ffpp", action="store_true",
                    help="add FF++ c23 face crops (train) as domain + FF++ test exam")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xc, yc, keys_c = load_cache("data/caches/cdf_faces.npz")
    Xw_tr, yw_tr, _ = load_cache("data/caches/wild_train.npz")
    Xw_te, yw_te, _ = load_cache("data/caches/wild_sample.npz")

    test_stems = set()
    for line in Path("data/raw/celebdf/Celeb-DF-v2/List_of_testing_videos.txt").read_text().splitlines():
        if line.strip():
            test_stems.add(Path(line.strip().split(" ", 1)[1]).stem)

    is_test = np.array([k.split("/")[1].split(":")[0] in test_stems for k in keys_c])
    Xc_tr, yc_tr, Xc_te, yc_te = Xc[~is_test], yc[~is_test], Xc[is_test], yc[is_test]

    X_parts = [Xc_tr, Xw_tr]
    y_parts = [yc_tr, yw_tr]
    domains = ["Celeb-DF(train split)", "WildDeepfake(train sample)"]
    exams = [("cdf_official_test", Xc_te, yc_te), ("wild_test", Xw_te, yw_te)]

    if args.with_ffpp:
        Xf_tr, yf_tr, _ = load_cache("data/caches/ffpp_train.npz")
        Xf_te, yf_te, _ = load_cache("data/caches/ffpp_test.npz")
        X_parts += [Xf_tr]
        y_parts += [yf_tr]
        domains.append("FF++(train split, c23)")
        exams.append(("ffpp_test", Xf_te, yf_te))

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    X_train = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-8)
    print(f"train: cdf={len(yc_tr)} wild={len(yw_tr)}"
          + (f" ffpp={len(yf_tr)}" if args.with_ffpp else "")
          + f" | fakes={int(y_train.sum())}")

    clf = LogisticRegression(C=args.c, max_iter=2000).fit(X_train, y_train)

    results = {}
    for name, Xe, ye in exams:
        Xn = Xe / (np.linalg.norm(Xe, axis=1, keepdims=True) + 1e-8)
        p = clf.predict_proba(Xn)[:, 1]
        auc = float(roc_auc_score(ye, p))
        fpr, tpr, _ = roc_curve(ye, p)
        d = fpr - (1 - tpr)
        i = int(np.argmin(np.abs(d)))
        eer = (fpr[i] + 1 - tpr[i]) / 2
        results[name] = {"auc": round(auc, 4), "eer": round(float(eer), 4), "n": int(len(ye))}
        print(f"{name}: auc={auc:.4f} eer={eer:.4f}")

    final = LogisticRegression(C=args.c, max_iter=2000).fit(X_train, y_train)
    joblib.dump(final, out / "face_probe.joblib")
    (out / "train_meta.json").write_text(json.dumps({
        "training_domains": domains,
        "results": results, "C": args.c,
    }, indent=2))
    print(f"saved {out/'face_probe.joblib'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
