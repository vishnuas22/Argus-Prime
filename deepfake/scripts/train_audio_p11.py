"""Train P11_AUDIO (W7 diverse-corpus audio probe) on merged per-source caches.

Mirrors train_audio_probe.py (p4 recipe) EXACTLY: standardize (mu/sd from train),
LogisticRegression(C=3.0, max_iter=3000), dump {clf, mu, sd} — no class weights, no
new hyperparameters. Recipe fidelity is the experiment; the diet is the treatment.

W7 additions (per pre-registered spec artifacts/generator_intake/w7_audio_retrain_spec.md):
  1. Merges per-source caches asv_w7_{la,kokoro,radar,mlaad}.npz (+ .labels.json).
     Keys are `uid:view:window`, globally unique (MLAAD uids generator-qualified —
     bare basenames collided 555x across generators, fixed in the diet builder).
  2. E1 split is GROUP-level: groups = MLAAD `segment` (bare LibriVox basename) so the
     same underlying text segment never straddles train/val (spec §2.3 item 6); other
     sources group by uid. Stratified 90/10 by source x label, seed recorded.
  3. train_meta.json carries full provenance: diet sha256, per-source vector counts,
     split seed, pointer to each source cache's own train_meta.json (RNG provenance).
  4. Coverage threshold is NOT set here — inject_audio_coverage.py runs after
     (--train on the merged TRAIN split), per D-035.

Usage: ./.venv/bin/python deepfake/scripts/train_audio_p11.py \
    [--sources la,kokoro,radar,mlaad] [--out artifacts/p11_audio] [--split-seed 42]
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
    ap.add_argument("--sources", default="la,kokoro,radar,mlaad")
    ap.add_argument("--out", default="artifacts/p11_audio")
    ap.add_argument("--split-seed", type=int, default=42)
    args = ap.parse_args()
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    # --- merge per-source caches ---
    X_parts, y_parts, key_parts, src_of_key = [], [], [], {}
    per_src = {}
    for s in sources:
        npz = Path(f"data/caches/asv_w7_{s}.npz")
        lab = Path(f"data/caches/asv_w7_{s}.labels.json")
        if not npz.exists():
            print(f"ERROR: cache missing: {npz} — extract {s} first", file=sys.stderr)
            return 1
        z = np.load(npz)
        labels = json.loads(lab.read_text())
        keys = list(z.files)
        X_parts.append(np.stack([z[k] for k in keys]).astype(np.float32))
        y_parts.append(np.array([labels[k] for k in keys], dtype=int))
        key_parts.extend(keys)
        for k in keys:
            if k in src_of_key:
                print(f"ERROR: duplicate cache key across sources: {k}", file=sys.stderr)
                return 1
            src_of_key[k] = s
        per_src[s] = {"n_vectors": len(keys),
                      "train_meta": json.loads(Path(f"data/caches/asv_w7_{s}.train_meta.json").read_text())}
        print(f"{s}: {len(keys)} vectors", flush=True)
    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    keys = key_parts
    print(f"merged: {X.shape[0]} vectors, dim {X.shape[1]}, fakes={int(y.sum())}", flush=True)

    # --- E1 group-level split (spec §2.3 item 6) ---
    diet = [json.loads(l) for l in
            Path("artifacts/generator_intake/w7_diet_sealed/diet_draft.jsonl").read_text().splitlines() if l.strip()]
    group_of_uid = {r["uid"]: r.get("segment", r["uid"]) for r in diet}
    uid_of_key = lambda k: k.split(":")[0]
    missing = sum(1 for k in keys if uid_of_key(k) not in group_of_uid)
    if missing:
        print(f"ERROR: {missing} cache keys not in sealed diet manifest", file=sys.stderr)
        return 1
    import random
    rng = random.Random(args.split_seed)
    # groups: MLAAD -> segment (all generators' versions of a segment stay together);
    # other sources -> uid. Stratify 90/10 over GROUPS within each (source, label) cell.
    uids_by_sg = defaultdict(list)          # (source, group) -> [uids]
    groups_by_cell = defaultdict(set)       # (source, label) -> {groups}
    for r in diet:
        g = group_of_uid[r["uid"]]
        uids_by_sg[(r["source"], g)].append(r["uid"])
        groups_by_cell[(r["source"], r["label"])].add(g)
    val_groups = set()
    for cell in sorted(groups_by_cell):
        gs = sorted(groups_by_cell[cell])
        rng.shuffle(gs)
        val_groups.update((cell[0], g) for g in gs[:max(1, int(round(len(gs) * 0.10)))])
    val_uids = set()
    for (src, g), uids in uids_by_sg.items():
        if (src, g) in val_groups:
            val_uids.update(uids)
    is_val = np.array([uid_of_key(k) in val_uids for k in keys])
    Xtr, ytr, Xva, yva = X[~is_val], y[~is_val], X[is_val], y[is_val]
    print(f"E1 split: train={Xtr.shape[0]} (fakes={int(ytr.sum())}) "
          f"val={Xva.shape[0]} (fakes={int(yva.sum())}) | {len(val_uids)} val uids", flush=True)

    # --- p4 recipe, unchanged ---
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr_s = (Xtr - mu) / sd
    Xva_s = (Xva - mu) / sd
    clf = LogisticRegression(C=3.0, max_iter=3000).fit(Xtr_s, ytr)
    p = clf.predict_proba(Xva_s)[:, 1]
    auc = float(roc_auc_score(yva, p))
    eer = eer_from(yva, p)
    print(f"E1 VAL: auc={auc:.4f} eer={eer:.4f} (bar: auc >= 0.995)", flush=True)

    # --- save (never touches artifacts/p4) ---
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "mu": mu, "sd": sd}, out / "audio_probe.joblib")
    diet_sha = hashlib.sha256(
        Path("artifacts/generator_intake/w7_diet_sealed/diet_draft.jsonl").read_bytes()).hexdigest()[:16]
    (out / "train_meta.json").write_text(json.dumps({
        "model": "logistic on frozen wav2vec2-XLS-R-300m mean-pooled (p4 recipe, C=3.0, unchanged)",
        "spec": "artifacts/generator_intake/w7_audio_retrain_spec.md (PRE-REGISTERED 2026-10-29)",
        "diet_sha256_16": diet_sha,
        "sources": {s: {"n_vectors": per_src[s]["n_vectors"]} for s in sources},
        "split": {"seed": args.split_seed, "rule": "group-level 90/10 stratified by source x label; "
                  "MLAAD groups = segment (spec §2.3.6)", "n_val_uids": len(val_uids)},
        "e1_val_auc": round(auc, 4), "e1_val_eer": round(eer, 4),
        "n_train": int(len(ytr)), "n_val": int(len(yva)),
        "extractor_provenance": {s: per_src[s]["train_meta"] for s in sources},
        "coverage": "injected post-hoc by inject_audio_coverage.py (D-035) on the merged TRAIN split",
    }, indent=1))
    # merged TRAIN-split cache for the coverage injector (threshold must not see val)
    np.savez_compressed(out / "train_split.npz",
                        **{k: X[i].astype(np.float16) for i, k in enumerate(keys) if not is_val[i]})
    (out / "train_split.labels.json").write_text(json.dumps(
        {k: int(y[i]) for i, k in enumerate(keys) if not is_val[i]}))
    print(f"saved {out/'audio_probe.joblib'} (+ train_split for coverage injection)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
