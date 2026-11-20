"""W9 STEP 4a (D-079 SEALED): build stacker rows from the three channels.

Row semantics = serve semantics: CLEAN views only, per-utterance pooling exactly
as detect_audio does it (mean over windows of p and of ||(h-mu)/sd||).

Channels:
  A (p4, FROZEN — never retrained): frozen bundle applied to w9 XLS-R clean
     vectors -> p4_pfake_mean, p4_znorm_mean, p4_ooc (bundle coverage_threshold).
  B (WavLM): TRAIN rows take OUT-OF-FOLD scores (wavlm_oof_scores.json, 5-fold
     seed 11, group-keyed — fusion law); VAL rows take final-refit scores
     (wavlm_val_scores.json). ooc = znorm_mean > B coverage_threshold (train-only
     P99.5 constant, no exam peeking).
  C (spectral, training-free): mean over clean windows of the 71-d raw vector;
     spectral_score = mean |z| vs a real-envelope (mean/std) fit on TRAIN-split
     REAL rows ONLY (D-035 law) — envelope params recorded in rows_meta.json.
Plus n_windows (clean window count) and duration_s (sf.info header-only; LA rows
via parquet bytes -> BytesIO).

Outputs: artifacts/p13_fused/rows_train.csv, rows_val.csv, rows_meta.json.
Usage:
  ./.venv/bin/python deepfake/scripts/build_audio_w9_rows.py \
      --manifest artifacts/generator_intake/w9_diet_sealed/manifest.jsonl \
      --prefix data/caches/w9 --out artifacts/p13_fused
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import soundfile as sf

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
P4_BUNDLE = WS / "artifacts/p4/audio_probe.joblib"


def duration_of(r) -> float:
    if "path" in r:
        info = sf.info(r["path"])
    else:
        if not hasattr(duration_of, "_pf"):
            duration_of._pf = {}
        pf = r["parquet"]
        if pf not in duration_of._pf:
            duration_of._pf[pf] = pd.read_parquet(pf)
        a = duration_of._pf[pf].iloc[r["row"]]["audio"]
        b = a["bytes"] if isinstance(a, dict) else a
        info = sf.info(io.BytesIO(b))
    return info.frames / info.samplerate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prefix", default="data/caches/w9")
    ap.add_argument("--extra-prefix", action="append", default=[],
                    help="W9.1: additional cache prefixes (repeatable; e.g. "
                         "data/caches/w9b, data/caches/w9c) merged in — keys must "
                         "be disjoint; bad ledgers are unioned")
    ap.add_argument("--out", default="artifacts/p13_fused")
    args = ap.parse_args()
    out = Path(args.out)

    man = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    info_of = {r["uid"]: r for r in man}

    def _load_npz(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}

    zx = _load_npz(f"{args.prefix}_xlsr.npz")
    zw = _load_npz(f"{args.prefix}_wavlm.npz")
    zs = _load_npz(f"{args.prefix}_spectral.npz")
    for ep in args.extra_prefix:
        for suf, d in (("xlsr", zx), ("wavlm", zw), ("spectral", zs)):
            e = _load_npz(f"{ep}_{suf}.npz")
            overlap = set(e) & set(d)
            if overlap:
                # W9.1 construction law: extra prefix (fresher extraction) wins
                # on same-file keys; the diet manifest is authoritative for labels.
                for k in overlap:
                    d.pop(k, None)
                print(f"merge: {len(overlap)} overlapping {suf} keys in {ep} — "
                      f"extra prefix wins", flush=True)
            d.update(e)
    p4 = joblib.load(P4_BUNDLE)
    bndl = joblib.load(Path(args.out) / "wavlm_probe.joblib")
    oof = json.loads((Path(args.out) / "wavlm_oof_scores.json").read_text())
    valsc = json.loads((Path(args.out) / "wavlm_val_scores.json").read_text())
    b_thr = float(bndl["coverage_threshold"])

    # group clean-view keys per uid per cache
    def group_clean(z):
        per = {}
        for k in (z.files if hasattr(z, "files") else z):
            uid, view, si = k.rsplit(":", 2)
            if view != "clean":
                continue
            per.setdefault(uid, []).append((int(si[1:]), k))
        return {u: [k for _, k in sorted(v)] for u, v in per.items()}

    cx, cw, cs = group_clean(zx), group_clean(zw), group_clean(zs)
    if not (set(cx) == set(cw) == set(cs)):
        raise SystemExit("cache key sets diverge across xlsr/wavlm/spectral — STOP")
    keep = set(info_of)
    n_drop = len(set(cx) - keep)
    if n_drop:
        print(f"cache: ignored {n_drop} non-diet uids (retired W9 poisoned rows / "
              f"replaced sample) — diet manifest is authoritative", flush=True)
    cx = {u: v for u, v in cx.items() if u in keep}
    cw = {u: v for u, v in cw.items() if u in keep}
    cs = {u: v for u, v in cs.items() if u in keep}
    if not (set(cx) == set(cw) == set(cs)):
        raise SystemExit("filtered cache key sets diverge — STOP")
    uids = sorted(cx)
    # disclosed decode attrition: uids absent from caches must ALL be in the
    # extraction bad ledger; anything unrecorded means an incomplete extraction.
    bad_uids = set(json.loads(
        Path(f"{args.prefix}_xlsr.train_meta.json").read_text())["bad_uids"])
    for ep in args.extra_prefix:
        p_extra = Path(f"{ep}_xlsr.train_meta.json")
        if p_extra.exists():
            bad_uids |= set(json.loads(p_extra.read_text())["bad_uids"])
    absent = [u for u in info_of if u not in cx]
    unrecorded = [u for u in absent if u not in bad_uids]
    if unrecorded:
        raise SystemExit(f"{len(unrecorded)} manifest uids absent from caches but NOT "
                         f"in extraction bad ledger (e.g. {unrecorded[:3]}) — STOP")
    print(f"attrition: {len(absent)} uids absent, all in bad ledger "
          f"({len(bad_uids)} recorded) — disclosed decode attrition", flush=True)

    # channel A: frozen p4 math on XLS-R clean vectors
    a_feats = {}
    for u in uids:
        X = np.stack([zx[k].astype(np.float32) for k in cx[u]])
        Xz = (X - p4["mu"]) / p4["sd"]
        a_feats[u] = {
            "p4_pfake_mean": float(p4["clf"].predict_proba(Xz)[:, 1].mean()),
            "p4_znorm_mean": float(np.linalg.norm(Xz, axis=1).mean()),
            "n_windows": len(cx[u]),
        }
        a_feats[u]["p4_ooc"] = int(a_feats[u]["p4_znorm_mean"] > p4["coverage_threshold"])

    # channel B: OOF (train) / final-refit (val) scores
    b_feats = {}
    for u in uids:
        split = info_of[u]["split"]
        src = oof[u] if split == "train" else valsc[u]
        b_feats[u] = {"wavlm_pfake_mean": src["pfake_mean"],
                      "wavlm_znorm_mean": src["znorm_mean"],
                      "wavlm_ooc": int(src["znorm_mean"] > b_thr),
                      "b_score_source": f"oof_fold{src['fold']}" if split == "train"
                                        else "final_refit"}

    # channel C: pooled spectral + D-035 envelope (train reals only)
    SP_NAMES = ([f"sp_ltas_{i}" for i in range(64)] +
                ["sp_flat_mean", "sp_flat_std", "sp_centroid_var", "sp_hf_ratio",
                 "sp_edge_freq", "sp_frame_corr", "sp_phase_smooth"])
    c_raw = {}
    for u in uids:
        c_raw[u] = np.stack([zs[k].astype(np.float64) for k in cs[u]]).mean(0)
    env_rows = [u for u in uids if info_of[u]["split"] == "train"
                and info_of[u]["label"] == 0]
    E = np.stack([c_raw[u] for u in env_rows])
    env_mean, env_std = E.mean(0), E.std(0) + 1e-9
    np.savez(out / "spectral_envelope.npz", mean=env_mean, std=env_std,
             n=len(env_rows),
             feature_names=np.array(SP_NAMES, dtype=object))
    c_feats = {}
    for u in uids:
        zv = np.abs((c_raw[u] - env_mean) / env_std)
        c_feats[u] = dict(zip(SP_NAMES, c_raw[u])) | {"spectral_score": float(zv.mean())}

    # durations
    # durations — only for cache-present rows (bad-ledger rows have no vectors)
    durs = {r["uid"]: duration_of(r) for r in man if r["uid"] in cx}

    rows = []
    for u in uids:
        r = info_of[u]
        rows.append({"uid": u, "source": r["source"], "label": r["label"],
                     "split": r["split"], "duration_s": round(durs[u], 3),
                     **a_feats[u], **b_feats[u], **c_feats[u]})
    df = pd.DataFrame(rows)
    feat_names = (["p4_pfake_mean", "p4_znorm_mean", "p4_ooc",
                   "wavlm_pfake_mean", "wavlm_znorm_mean", "wavlm_ooc",
                   "spectral_score"] + SP_NAMES + ["n_windows", "duration_s"])
    missing_cols = [c for c in feat_names if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"row feature columns missing: {missing_cols} — STOP")

    df[df["split"] == "train"].to_csv(out / "rows_train.csv", index=False)
    df[df["split"] == "val"].to_csv(out / "rows_val.csv", index=False)
    p4_sha = hashlib.sha256(P4_BUNDLE.read_bytes()).hexdigest()
    (out / "rows_meta.json").write_text(json.dumps({
        "feature_names": feat_names,
        "n_train": int((df["split"] == "train").sum()),
        "n_val": int((df["split"] == "val").sum()),
        "views": "clean only (serve semantics); B trained on all 3 views",
        "b_oof": "train rows: 5-fold seed 11 group-keyed OOF (fusion law); "
                 "val rows: final refit",
        "d035_envelope": {
            "fit_on": "TRAIN-split REAL rows only",
            "n_fit_rows": len(env_rows),
            "spectral_score": "mean |z| over 71 spectral features vs envelope",
        },
        "p4_snapshot_sha256": p4_sha,
        "fusion": "feature-level-only",
    }, indent=1))
    print(f"rows: train {int((df['split']=='train').sum())} / "
          f"val {int((df['split']=='val').sum())} | features {len(feat_names)} "
          f"| p4 sha {p4_sha[:12]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
