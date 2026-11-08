"""Score the sealed 2021 n=330 shard (NR1/NR2 clauses) with a candidate court.

Why this exists (2026-11-07, D-083 addendum 5):
  The D-081 full-2021 design (reals-complete 23,944 + 50,000 seeded fakes) no
  longer contains the registered "first 200 per class" selection that produced
  the incumbent p4 2021 receipt (AUC 0.912, n=330, exam_audio_2021_p4w8).
  The battery's old shortcut ("head(200) decodable per class" from the full
  CSV) therefore drifted to n=400 after the decoder repair and fail-closed.

  This script replays exam_audio_2021.py's EXACT selection law:
    shard 0  : first 200 label-0 (fake) rows with non-empty audio bytes
    shard 13 : first 200 label-1 (real) rows with non-empty audio bytes
    keep only rows the ORIGINAL libsndfile path decodes (the old score loop
    dropped failures and counted idx over successes)
    -> 157 fakes + 173 reals = 330 (verified against the p4w8 scores.json
       idx coverage below; counts must match exactly or this script STOPs).

  The 330 rows are then scored through the candidate court's SERVE path
  (argus_detect.detect_audio with the fused bundle), i.e. the deployed
  semantics the NR1/NR2 clauses are about. The selection is sealed by uid
  sha256 so the battery can verify identity before consuming rows.csv.

NO bars are changed here. This restores the sealed population.

Usage:
  ./.venv/bin/python deepfake/scripts/score_audio_2021_shard330.py \
      --cand p13b_fused [--force]
"""
import argparse
import hashlib
import io
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
SHARD_DIR = WS / "data" / "raw"
REF_SCORES = WS / "artifacts" / "exam_audio_2021_p4w8" / "scores.json"
SEAL = WS / "artifacts" / "exam_audio_2021_shard330" / "selection.json"
N_PER_CLASS = 200
SHARDS = ((0, 0), (13, 1))  # (parquet shard, dataset label): 0=fake, 1=real
EXPECT = {0: 157, 1: 173}   # original successes among the 400 picks


def build_selection():
    """Replay the original picks + old decode filter; returns rows with bytes."""
    import pyarrow.parquet as pq
    import soundfile as sf

    sel = []
    for shard_no, want_label in SHARDS:
        path = SHARD_DIR / f"asvspoof21_la_test-{shard_no:05d}.parquet"
        if not path.exists():
            raise SystemExit(f"missing parquet shard: {path}")
        tab = pq.read_table(path, columns=["audio", "label"])
        acol, lcol = tab["audio"], tab["label"]
        picks = []
        for i in range(len(tab)):
            if len(picks) >= N_PER_CLASS:
                break
            if lcol[i].as_py() != want_label:
                continue
            rec = acol[i].as_py()
            data = rec["bytes"] if isinstance(rec, dict) else rec
            if not data:
                continue
            picks.append((i, data))
        if len(picks) != N_PER_CLASS:
            raise SystemExit(f"shard {shard_no}: picks={len(picks)} != {N_PER_CLASS}")
        kept = 0
        for i, data in picks:
            try:
                # Old score-loop decode law: failures were skipped silently.
                sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
            except Exception:
                continue
            sel.append({"uid": f"shard{shard_no:03d}_row{i:06d}",
                        "shard": shard_no, "dataset_label": want_label,
                        "label": 1 - want_label, "idx": kept, "_bytes": data})
            kept += 1
        print(f"shard {shard_no}: picks={len(picks)} kept={kept} "
              f"failed={len(picks) - kept} label={want_label}", flush=True)
    return sel


def verify_against_registered(sel):
    ref = json.loads(REF_SCORES.read_text())
    ref_idx = {(r["dataset_label"], r["idx"]) for r in ref}
    mine = {(r["dataset_label"], r["idx"]) for r in sel}
    counts = Counter(r["dataset_label"] for r in sel)
    if dict(counts) != EXPECT:
        raise SystemExit(f"selection counts {dict(counts)} != {EXPECT} — STOP")
    if ref_idx != mine:
        raise SystemExit(f"idx coverage drift vs p4w8 scores.json "
                         f"(missing={sorted(ref_idx - mine)[:5]}, "
                         f"extra={sorted(mine - ref_idx)[:5]}) — STOP")
    print(f"selection verified vs p4w8: n={len(sel)} "
          f"real={counts[1]} fake={counts[0]} idx coverage EXACT", flush=True)


def seal_selection(sel):
    uids = sorted(r["uid"] for r in sel)
    digest = hashlib.sha256("\n".join(uids).encode()).hexdigest()
    seal = {"n": len(uids), "n_real": EXPECT[1], "n_fake": EXPECT[0],
            "sha256": digest, "uids": uids,
            "law": "exam_audio_2021.py first-200-per-class picks + old "
                   "libsndfile decode filter (failures dropped)"}
    if SEAL.exists():
        old = json.loads(SEAL.read_text())
        if old.get("sha256") != digest:
            raise SystemExit("selection seal drift — refusing to score a "
                             "different shard (D-046) — STOP")
        print(f"selection seal matches: {digest[:16]}…", flush=True)
    else:
        SEAL.parent.mkdir(parents=True, exist_ok=True)
        SEAL.write_text(json.dumps(seal, indent=2))
        print(f"selection seal written: {digest[:16]}…", flush=True)
    return digest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True,
                    help="candidate dir under artifacts/ (p13_fused / p13b_fused)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="write into a non-empty output dir")
    ap.add_argument("--metrics-only", action="store_true",
                    help="recompute metrics.json from an existing rows.csv (no scoring)")
    args = ap.parse_args()

    bundle = WS / "artifacts" / args.cand / "audio_court.joblib"
    if not bundle.exists():
        raise SystemExit(f"candidate bundle missing: {bundle}")
    out = (Path(args.out) if args.out else
           WS / "artifacts" / f"exam_audio_2021_shard330_{args.cand.replace('_', '')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: {out} already contains results — pass --force", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    sel = build_selection()
    verify_against_registered(sel)
    digest = seal_selection(sel)

    import pandas as pd
    from sklearn.metrics import roc_auc_score

    if args.metrics_only:
        rp = out / "rows.csv"
        if not rp.exists():
            raise SystemExit(f"--metrics-only: {rp} missing — STOP")
        df = pd.read_csv(rp)
        if len(df) != len(sel) or set(df["uid"]) != {r["uid"] for r in sel}:
            raise SystemExit("--metrics-only: rows.csv selection != rebuilt selection — STOP")
        print(f"metrics-only: loaded {len(df)} rows from {rp}", flush=True)
    else:
        import argus_detect
        from argus_detect import detect_audio

        argus_detect.AUDIO_PROBE = bundle
        tmpdir = Path(tempfile.mkdtemp(prefix="shard330_"))
        tmp = tmpdir / "u.wav"
        rows, t0 = [], time.time()
        for k, r in enumerate(sel):
            tmp.write_bytes(r["_bytes"])
            res = detect_audio(tmp, recovery_enabled=False)
            if not res.get("fused"):
                raise SystemExit("bundle did not route through the fused branch — STOP")
            rows.append({"uid": r["uid"], "shard": r["shard"],
                         "label": r["label"],
                         "p4_pfake_mean": res["p4_pfake_mean"],
                         "p_fused": res["p_fake_audio_mean"],
                         "wavlm_pfake_mean": res["wavlm_pfake_mean"],
                         "spectral_score": res["spectral_score"],
                         "abstained": bool(res["abstained"])})
            if (k + 1) % 50 == 0:
                print(f"scored {k + 1}/{len(sel)} ({(k + 1) / (time.time() - t0):.2f} utt/s)",
                      flush=True)
        tmp.unlink(missing_ok=True)
        tmpdir.rmdir()
        df = pd.DataFrame(rows)
        df.to_csv(out / "rows.csv", index=False)
    y = df["label"].values
    metrics = {
        "cand": args.cand,
        "n": len(df),
        "n_real": int((df["label"] == 0).sum()),
        "n_fake": int((df["label"] == 1).sum()),
        "fused_auc": round(float(roc_auc_score(y, df["p_fused"])), 4),
        "wavlm_auc": round(float(roc_auc_score(y, df["wavlm_pfake_mean"])), 4),
        "p4_auc": round(float(roc_auc_score(y, df["p4_pfake_mean"])), 4),
        "abstain_rate": round(float(df["abstained"].mean()), 4),
        "selection_sha256": digest,
        "selection_law": "exam_audio_2021.py first-200-per-class picks + old "
                         "libsndfile decode filter",
        "serve_path": "argus_detect.detect_audio (fused bundle, recovery off)",
        "receipt": str(out.relative_to(WS)),
    }

    # Parity against the D-081 full-2021 CSV on the overlap (path check only).
    full = (WS / "artifacts" / f"exam_audio_2021_full_{args.cand.replace('_', '')}"
            / "rows_2021_full.csv")
    if full.exists():
        f = pd.read_csv(full)
        m = df.merge(f, on="uid", suffixes=("_330", "_full"))
        if len(m):
            lan = int((m["label_330"] == m["label_full"]).sum())
            metrics["parity_vs_full_exam"] = {
                "overlap_n": len(m),
                "label_match": f"{lan}/{len(m)}",
                "max_abs_dp4": round(float((m["p4_pfake_mean"] - m["p_fake_audio_mean"]).abs().max()), 6),
                "max_abs_dfused": round(float((m["p_fused_330"] - m["p_fused_full"]).abs().max()), 6),
                "max_abs_dwavlm": round(float((m["wavlm_pfake_mean_330"] - m["wavlm_pfake_mean_full"]).abs().max()), 6),
            }
            if lan != len(m):
                raise SystemExit("label drift on 330/full overlap — STOP")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("SHARD330:", json.dumps(metrics), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
