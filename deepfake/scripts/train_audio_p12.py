"""Train P12_AUDIO_SPECIALIST (W8 modern-TTS specialist) on the sealed W8 diet.

Mirrors train_audio_p11.py (which mirrors p4's recipe) with EXACTLY TWO declared
deltas, both pre-registered in spec artifacts/generator_intake/w8_specialist_spec.md
(PRE-REGISTERED 2026-10-30, user ruling):

  1. DIET FILTER: merged cache keys are filtered to the sealed W8 manifest
     (artifacts/generator_intake/w8_diet_sealed/diet_w8.jsonl). This mechanically
     excludes LA spoof, RADAR spoof, the 14 holdout generators (S1 exam set), and
     anything else not in the specialist diet. Kokoro is excluded by --sources.
  2. CLASS WEIGHTS: LogisticRegression(class_weight='balanced') — the D-063
     mechanism-(a) antidote (unweighted LR absorbed the 70.6% fake prior; the W8
     diet is ~1.25:1 and the specialist must not over-call bona). Declared BEFORE
     training; not an outcome-tuned change.

Everything else is p4/p11 verbatim: standardize (mu/sd from train), C=3.0,
max_iter=3000, group-level 90/10 split (MLAAD groups = segment), bundle
{clf, mu, sd}, train_split export for D-035 coverage injection.

Drop audit: every excluded vector is accounted for by source and reason before
training proceeds (an unexplained drop is an error, not a warning).

Usage: ./.venv/bin/python deepfake/scripts/train_audio_p12.py \
    [--out artifacts/p12_specialist] [--split-seed 42]
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
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
    ap.add_argument("--sources", default="la,radar,mlaad")
    ap.add_argument("--diet", default="artifacts/generator_intake/w8_diet_sealed/diet_w8.jsonl")
    ap.add_argument("--holdout", default="artifacts/generator_intake/w8_diet_sealed/holdout_w8.jsonl")
    ap.add_argument("--out", default="artifacts/p12_specialist")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    diet = [json.loads(l) for l in (Path(args.diet).read_text().splitlines()) if l.strip()]
    holdout = [json.loads(l) for l in (Path(args.holdout).read_text().splitlines()) if l.strip()]
    manifest_uids = {r["uid"] for r in diet}
    holdout_uids = {r["uid"] for r in holdout}
    if manifest_uids & holdout_uids:
        print("ERROR: holdout uids leak into the train manifest", file=sys.stderr)
        return 1
    group_of_uid = {r["uid"]: r.get("segment", r["uid"]) for r in diet}
    print(f"diet: {len(manifest_uids)} train uids | holdout: {len(holdout_uids)} uids (S1 only)",
          flush=True)

    # --- merge per-source caches, then filter to the sealed W8 manifest ---
    X_parts, y_parts, key_parts = [], [], []
    dropped = defaultdict(int)          # source -> count
    dropped_holdout = 0
    kept_by_src = {}
    for s in sources:
        npz = Path(f"data/caches/asv_w7_{s}.npz")
        lab = Path(f"data/caches/asv_w7_{s}.labels.json")
        if not npz.exists():
            print(f"ERROR: cache missing: {npz}", file=sys.stderr)
            return 1
        z = np.load(npz)
        labels = json.loads(lab.read_text())
        keys = list(z.files)
        w7_rows = {r["uid"]: r for r in
                   (json.loads(l) for l in
                    (Path("artifacts/generator_intake/w7_diet_sealed/diet_draft.jsonl")
                     .read_text().splitlines()) if l.strip()) if r["source"] == s}
        kept = []
        for k in keys:
            uid = k.split(":")[0]
            if uid in manifest_uids:
                kept.append(k)
                continue
            dropped[s] += 1
            if uid in holdout_uids:
                dropped_holdout += 1
            elif uid in w7_rows and w7_rows[uid]["label"] == 1:
                pass                      # spoof side excluded by diet law
            else:
                print(f"ERROR: unexplained drop {s}:{uid}", file=sys.stderr)
                return 1
        X_parts.append(np.stack([z[k] for k in kept]).astype(np.float32))
        y_parts.append(np.array([labels[k] for k in kept], dtype=int))
        key_parts.extend(kept)
        kept_by_src[s] = len(kept)
        print(f"{s}: kept {len(kept)} / {len(keys)} vectors (dropped {len(keys) - len(kept)})"
              f" [holdout: {sum(1 for k in keys if k.split(':')[0] in holdout_uids)}]", flush=True)
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    keys = key_parts
    print(f"merged: {X.shape[0]} vectors, dim {X.shape[1]}, fakes={int(y.sum())} "
          f"(dropped total {sum(dropped.values())}, of which holdout {dropped_holdout})",
          flush=True)

    # --- group-level 90/10 split (W7 §2.3.6 law: MLAAD segments never straddle) ---
    import random
    rng = random.Random(args.split_seed)
    uids_by_sg = defaultdict(list)
    groups_by_cell = defaultdict(set)
    uid_label = {r["uid"]: r["label"] for r in diet}
    uid_src = {r["uid"]: r["source"] for r in diet}
    for r in diet:
        uids_by_sg[(r["source"], group_of_uid[r["uid"]])].append(r["uid"])
        groups_by_cell[(r["source"], r["label"])].add(group_of_uid[r["uid"]])
    val_groups = set()
    for cell in sorted(groups_by_cell):
        gs = sorted(groups_by_cell[cell])
        rng.shuffle(gs)
        val_groups.update((cell[0], g) for g in gs[:max(1, int(round(len(gs) * 0.10)))])
    val_uids = set()
    for (src, g), uids in uids_by_sg.items():
        if (src, g) in val_groups:
            val_uids.update(uids)
    is_val = np.array([k.split(":")[0] in val_uids for k in keys])
    Xtr, ytr, Xva, yva = X[~is_val], y[~is_val], X[is_val], y[is_val]
    print(f"S0 split: train={Xtr.shape[0]} (fakes={int(ytr.sum())}) "
          f"val={Xva.shape[0]} (fakes={int(yva.sum())}) | {len(val_uids)} val uids", flush=True)

    # --- recipe: p4 verbatim + the ONE declared delta ---
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd
    clf = LogisticRegression(C=3.0, max_iter=3000,
                             class_weight="balanced").fit(Xtr_s, ytr)
    p = clf.predict_proba(Xva_s)[:, 1]
    auc = float(roc_auc_score(yva, p))
    eer = eer_from(yva, p)
    print(f"S0 VAL: auc={auc:.4f} eer={eer:.4f} (bar: auc >= 0.99)", flush=True)

    # --- save (never touches artifacts/p4 or artifacts/p11_audio) ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "mu": mu, "sd": sd}, out / "audio_probe.joblib")
    diet_sha = hashlib.sha256(Path(args.diet).read_bytes()).hexdigest()[:16]
    (out / "train_meta.json").write_text(json.dumps({
        "model": "logistic on frozen wav2vec2-XLS-R-300m mean-pooled (p4 recipe, C=3.0, "
                 "class_weight='balanced' — the declared W8 delta)",
        "spec": "artifacts/generator_intake/w8_specialist_spec.md (PRE-REGISTERED 2026-10-30)",
        "diet_sha256_16": diet_sha,
        "sources": {s: {"n_vectors": kept_by_src[s]} for s in sources},
        "split": {"seed": args.split_seed, "rule": "group-level 90/10 stratified by "
                  "source x label; MLAAD groups = segment", "n_val_uids": len(val_uids)},
        "s0_val_auc": round(auc, 4), "s0_val_eer": round(eer, 4),
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "dropped_by_source": dict(dropped), "dropped_holdout_vectors": dropped_holdout,
        "coverage": "injected post-hoc by inject_audio_coverage.py (D-035) on the TRAIN split",
    }, indent=1))
    np.savez_compressed(out / "train_split.npz",
                        **{k: X[i].astype(np.float16) for i, k in enumerate(keys) if not is_val[i]})
    (out / "train_split.labels.json").write_text(json.dumps(
        {k: int(y[i]) for i, k in enumerate(keys) if not is_val[i]}))
    print(f"saved {out/'audio_probe.joblib'} (+ train_split for coverage injection)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
