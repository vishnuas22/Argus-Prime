"""W9 STEP 3 (D-079 SEALED): train probe B — frozen WavLM-base-plus + logistic.

Recipe byte-mirrors train_audio_probe.py:46-58 (own mu/sd z-norm fit on TRAIN
vectors only, LogisticRegression(C=3.0, max_iter=3000)) + class_weight="balanced"
(D-063 prior-shift antidote, sealed).

Splits come from the diet manifest (per-source, segment-grouped for MLAAD):
train-split vectors (ALL 3 views — aug robustness) train the probe; val-split
vectors give the NR1 receipt (val AUC floor 0.98, reported all-views AND
clean-view; the bar reads the all-views number — the stricter convention).

OOF law (fusion law §2): B's scores enter the stacker OUT-OF-FOLD for train
rows — 5 folds, seed 11, groups keyed at the SAME granularity as the split
(MLAAD: text segment; LA/RADAR: uid) so no group straddles a fold boundary.
Fold models fit the same recipe on fold-train vectors; fold-val rows are scored
on their CLEAN views (row semantics = serve semantics). Final deployed B = refit
on the full train split; val-split scores come from that refit (honest val).

B's own coverage threshold = P99.5 of per-utterance znorm_mean over TRAIN rows
(clean views, final mu/sd) — own train p99.5, no exam peeking (spec §2 row B),
mirroring p4's bundle construction.

Usage:
  ./.venv/bin/python deepfake/scripts/train_audio_w9_wavlm.py \
      --cache data/caches/w9_wavlm.npz \
      --manifest artifacts/generator_intake/w9_diet_sealed/manifest.jsonl \
      --out artifacts/p13_fused
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

WS = Path("/Users/Dataghost/Argus_Prime")
SEED = 11
N_FOLDS = 5


def eer_from(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    d = fpr - (1 - tpr)
    i = int(np.argmin(np.abs(d)))
    if 0 < i < len(fpr) and d[i] != d[i - 1]:
        w = -d[i - 1] / (d[i] - d[i - 1])
        return float(fpr[i - 1] + w * (fpr[i] - fpr[i - 1]))
    return float((fpr[i] + (1 - tpr[i])) / 2)


def fit_recipe(X, y):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True) + 1e-8
    clf = LogisticRegression(C=3.0, max_iter=3000, class_weight="balanced") \
        .fit((X - mu) / sd, y)
    return clf, mu, sd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/caches/w9_wavlm.npz")
    ap.add_argument("--extra-cache", action="append", default=[],
                    help="W9.1: additional wavlm npz (repeatable; e.g. "
                         "data/caches/w9b_wavlm.npz, data/caches/w9c_wavlm.npz) "
                         "merged in — keys must be disjoint")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="artifacts/p13_fused")
    args = ap.parse_args()

    man = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    split_of, label_of, group_of = {}, {}, {}
    for r in man:
        split_of[r["uid"]] = r["split"]
        label_of[r["uid"]] = r["label"]
        # group granularity == split granularity (fusion law: no group straddles)
        group_of[r["uid"]] = r.get("segment") if r["source"] == "mlaad" else r["uid"]

    z = np.load(args.cache)
    merged = {k: z[k] for k in z.files}
    for ep in args.extra_cache:
        ze = np.load(ep)
        overlap = set(ze.files) & set(merged)
        if overlap:
            # W9.1 construction law: the same audio file can appear in two
            # prefixes (W9's poisoned 'bona' rows are real spoof files). The
            # extra cache (fresher extraction) wins; the diet manifest is the
            # single authoritative label source.
            for k in overlap:
                merged.pop(k, None)
            print(f"merge: {len(overlap)} overlapping keys in {ep} — extra cache "
                  f"wins (same-file vectors; diet label authoritative)", flush=True)
        for k in ze.files:
            merged[k] = ze[k]
    diet_uids = set(split_of)
    keys = [k for k in merged if k.rsplit(":", 2)[0] in diet_uids]
    n_dropped = len(merged) - len(keys)
    if n_dropped:
        print(f"cache: ignored {n_dropped} keys from non-diet uids (retired W9 "
              f"poisoned rows / replaced sample) — diet manifest is authoritative",
              flush=True)
    uid_of = {k: k.rsplit(":", 2)[0] for k in keys}
    view_of = {k: k.rsplit(":", 2)[1] for k in keys}
    X_all = np.stack([merged[k] for k in keys]).astype(np.float32)
    uids = np.array([uid_of[k] for k in keys])
    views = np.array([view_of[k] for k in keys])
    splits = np.array([split_of[u] for u in uids])
    y_all = np.array([label_of[u] for u in uids])
    print(f"vectors: {len(keys)} | train {int((splits == 'train').sum())} | "
          f"val {int((splits == 'val').sum())}", flush=True)

    tr = splits == "train"
    clf, mu, sd = fit_recipe(X_all[tr], y_all[tr])

    # NR1 receipt: val AUC all-views (bar) + clean-view + EER
    va = splits == "val"
    p_va = clf.predict_proba((X_all[va] - mu) / sd)[:, 1]
    auc_all = float(roc_auc_score(y_all[va], p_va))
    cl = va & (views == "clean")
    p_cl = clf.predict_proba((X_all[cl] - mu) / sd)[:, 1]
    auc_clean = float(roc_auc_score(y_all[cl], p_cl))
    eer = eer_from(y_all[va], p_va)
    print(f"NR1 receipt: val AUC all-views {auc_all:.4f} (bar >= 0.98) | "
          f"clean-view {auc_clean:.4f} | EER {eer:.4f}", flush=True)

    # OOF pass: group-keyed 5-fold, seed 11 round-robin deal
    groups = sorted(set(group_of[u] for u in uids[tr]))
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)
    fold_of_group = {g: i % N_FOLDS for i, g in enumerate(groups)}
    # val groups are intentionally absent from fold_of_group (val is never a
    # fold) — default -1 is unreachable because lookups are masked by tr_idx.
    fold_of_vec = np.array([fold_of_group.get(group_of[u], -1) for u in uids])
    tr_idx = np.where(tr)[0]
    oof = {}
    for k in range(N_FOLDS):
        va_f = tr_idx[fold_of_vec[tr_idx] == k]
        tr_f = tr_idx[fold_of_vec[tr_idx] != k]
        if len(np.unique(y_all[tr_f])) < 2:
            raise SystemExit(f"OOF fold {k} degenerate (single class) — STOP")
        clf_f, mu_f, sd_f = fit_recipe(X_all[tr_f], y_all[tr_f])
        cl_f = va_f[views[va_f] == "clean"]
        for ui in np.unique(uids[cl_f]):
            m = uids[cl_f] == ui
            Xs = (X_all[cl_f][m] - mu_f) / sd_f
            oof[str(ui)] = {
                "pfake_mean": round(float(clf_f.predict_proba(Xs)[:, 1].mean()), 6),
                "znorm_mean": round(float(np.linalg.norm(Xs, axis=1).mean()), 4),
                "fold": int(k),
            }
        print(f"fold {k}: train {len(tr_f)} vec / val {len(cl_f)} clean vec | "
              f"classes {np.bincount(y_all[tr_f])}", flush=True)

    # per-utterance clean-view aggregation helper (detect_audio math)
    def utt_scores(idx_mask, clf_, mu_, sd_):
        out = {}
        cl_idx = np.where(idx_mask & (views == "clean"))[0]
        Xs = (X_all[cl_idx] - mu_) / sd_
        ps = clf_.predict_proba(Xs)[:, 1]
        zn = np.linalg.norm(Xs, axis=1)
        for ui in np.unique(uids[cl_idx]):
            m = uids[cl_idx] == ui
            out[str(ui)] = {"pfake_mean": round(float(ps[m].mean()), 6),
                            "znorm_mean": round(float(zn[m].mean()), 4),
                            "n_windows": int(m.sum())}
        return out

    val_scores = utt_scores(va, clf, mu, sd)
    train_final_scores = utt_scores(tr, clf, mu, sd)  # threshold provenance only
    znorm_p995 = float(np.quantile([v["znorm_mean"] for v in train_final_scores.values()], 0.995))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "mu": mu, "sd": sd,
                 "coverage_threshold": znorm_p995, "znorm_p995": znorm_p995,
                 "model": "microsoft/wavlm-base-plus frozen mean-pooled",
                 "recipe": "logistic C=3.0 max_iter=3000 class_weight=balanced "
                           "(train_audio_probe.py:46-58 + D-063)"},
                out / "wavlm_probe.joblib")
    (out / "wavlm_oof_scores.json").write_text(json.dumps(oof))
    (out / "wavlm_val_scores.json").write_text(json.dumps(val_scores))
    meta = {
        "model": "microsoft/wavlm-base-plus frozen mean-pooled (repo corrected from "
                 "facebook/ — dead path, HF 401; same WavLMModel arch hidden 768)",
        "recipe": "logistic C=3.0 max_iter=3000 + class_weight=balanced (D-063)",
        "views_trained": ["clean", "aug1", "aug2"],
        "n_train_vectors": int(tr.sum()), "n_val_vectors": int(va.sum()),
        "val_auc_all_views": round(auc_all, 4),
        "val_auc_clean_view": round(auc_clean, 4),
        "val_eer": round(eer, 4),
        "nr1_bar": "val AUC >= 0.98 (all-views convention, stricter)",
        "nr1_pass": bool(auc_all >= 0.98),
        "oof": {"folds": N_FOLDS, "seed": SEED,
                "grouping": "MLAAD segment / LA+RADAR uid (split granularity)",
                "n_oof_utts": len(oof)},
        "coverage_threshold": round(znorm_p995, 4),
        "coverage_construction": "P99.5 of per-utterance znorm_mean over TRAIN rows "
                                 "(clean views, final mu/sd) — own train p99.5, no exam peeking",
    }
    (out / "wavlm_train_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: meta[k] for k in ("val_auc_all_views", "val_auc_clean_view",
                                           "val_eer", "nr1_pass", "coverage_threshold")}),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
