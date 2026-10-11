"""Drift-signal gate (spec item 2.3): does SFace identity drift carry class
signal CHEAPLY (no CLIP, no stacker) on the official FF++ test split?

Scores test-split reals vs fakes (all 4 methods) using ONLY drift features
consecutive-embedding L2 steps along identity tracks), reports per-method
ROC-AUC of drift_p95 plus a logistic probe on {mean, p95, spikes}.

PRE-REGISTERED DECISION GATE:
  auc_probe >= 0.65           -> drift feature proceeds as specified
  0.60 <= auc_probe < 0.65    -> keep feature, stacker decides (documented)
  auc_probe < 0.60            -> contingency 2.4 (ArcFace escalation) BEFORE
                                 any stacker training

spike_thresh is fit on TRAIN-split reals only (default --n-cal 100) — the
D-035 calibration law: no exam peeking into the test split for thresholds.

Usage:
  .venv/bin/python deepfake/scripts/exam_video_drift.py \
      [--n-real 140] [--per-method 70] [--n-cal 100] [--frames 12] \
      [--out artifacts/exam_video_drift_<date>] [--force]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_channels import drift_features  # noqa: E402
from video_face_track import cluster_identities, observe_video  # noqa: E402

MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
SPLITS = WS / "data" / "raw" / "ffpp"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def drift_of(path: Path, frames: int, spike=None) -> dict:
    obs, _stats = observe_video(path, frames, with_scores=False)
    tracks = cluster_identities(obs)
    d = drift_features(tracks, spike_thresh=spike)
    d["n_faces"] = len(obs)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=140)
    ap.add_argument("--per-method", type=int, default=70)
    ap.add_argument("--n-cal", type=int, default=100,
                    help="train-split REAL videos used to fit spike_thresh")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="sface",
                    help="identity embedder behind video_face_track's seam "
                         "(arcface = spec contingency 2.4, fires when sface "
                         "drift auc_probe < 0.60)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    from video_face_track import set_embedder
    set_embedder(args.embedder)

    out = Path(args.out) if args.out else WS / "artifacts" / \
        f"exam_video_drift_{time.strftime('%Y%m%d')}"
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    test_pairs = json.load(open(SPLITS / "test.json"))
    train_pairs = json.load(open(SPLITS / "train.json"))
    t0 = time.time()

    # --- calibration half: TRAIN-split reals only -------------------------
    train_ids = {i for p in train_pairs for i in p}
    cal_reals = [v for v in sorted((MIRROR / "real").glob("*.mp4"))
                 if v.stem in train_ids][: args.n_cal]
    cal_vals = [drift_of(v, args.frames)["drift_p95"] for v in cal_reals]
    cal_vals = [v for v in cal_vals if v]
    spike = float(np.quantile(np.asarray(cal_vals, dtype=np.float64), 0.95))
    print(f"[cal] {len(cal_vals)} train reals -> spike_thresh={spike:.6f} "
          f"(p50={np.quantile(cal_vals,0.5):.4f})", flush=True)

    # --- exam half: official TEST split -----------------------------------
    test_ids = {i for p in test_pairs for i in p}
    reals = [v for v in sorted((MIRROR / "real").glob("*.mp4"))
             if v.stem in test_ids][: args.n_real]
    fake_sets = {}
    for m in METHODS:
        pairs = {tuple(p) for p in test_pairs}
        fakes = [v for v in sorted((MIRROR / "fake" / m).glob("*.mp4"))
                 if tuple(v.stem.split("_")) in pairs
                 or tuple(reversed(v.stem.split("_"))) in pairs]
        rng = np.random.default_rng(11)
        fake_sets[m] = [fakes[i] for i in rng.permutation(len(fakes))[: args.per_method]]

    rows = []
    rows_path = out / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                _r = json.loads(line)
                # dir-qualified: the same filename exists in several method
                # dirs (420 rows here cover only 210 distinct filenames)
                done.add((_r.get("method", ""), _r["file"]))
    with open(rows_path, "a") as f:
        jobs = [(v, 0, "real") for v in reals]
        for m in METHODS:
            jobs += [(v, 1, m) for v in fake_sets[m]]
        for i, (v, label, method) in enumerate(jobs):
            if (method, v.name) in done:
                prev = [json.loads(l) for l in rows_path.read_text().splitlines()
                        if l.strip() and json.loads(l)["method"] == method
                        and json.loads(l)["file"] == v.name]
                if prev:
                    rows.append(prev[0])
                continue
            d = drift_of(v, args.frames, spike=spike)
            row = {"file": v.name, "method": method, "label": label, **d}
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            print(f"[{i+1}/{len(jobs)}] {method}/{v.name} p95={d['drift_p95']:.4f} "
                  f"mean={d['drift_mean']:.4f} tracks={d['n_drift_tracks']}", flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    usable = [r for r in rows if r["n_drift_tracks"] > 0]
    y = np.asarray([r["label"] for r in usable])
    X = np.asarray([[r["drift_mean"], r["drift_p95"], float(r["drift_spikes"])]
                    for r in usable], dtype=np.float64)
    s_p95 = np.asarray([r["drift_p95"] for r in usable], dtype=np.float64)
    auc_p95 = float(roc_auc_score(y, s_p95)) if len(set(y)) > 1 else None
    auc_probe = None
    if len(set(y)) > 1 and len(usable) >= 20:
        probe = LogisticRegression(max_iter=2000, random_state=11).fit(X, y)
        auc_probe = float(roc_auc_score(y, probe.predict_proba(X)[:, 1]))

    reals_p = np.asarray([r["drift_p95"] for r in usable if r["label"] == 0])
    fakes_p = np.asarray([r["drift_p95"] for r in usable if r["label"] == 1])
    per_method = {}
    for m in METHODS:
        sub = [r for r in usable if r["method"] in ("real", m)]
        yy = np.asarray([r["label"] for r in sub])
        ss = np.asarray([r["drift_p95"] for r in sub], dtype=np.float64)
        per_method[m] = {
            "auc_p95": round(float(roc_auc_score(yy, ss)), 4) if len(set(yy)) > 1 else None,
            "n_fake": int((yy == 1).sum()), "n_real": int((yy == 0).sum())}
    gate = ("proceed" if (auc_probe or 0) >= 0.65 else
            "stacker_decides" if (auc_probe or 0) >= 0.60 else
            "CONTINGENCY_2.4")
    metrics = {
        "auc_p95": round(auc_p95, 4) if auc_p95 is not None else None,
        "auc_probe": round(auc_probe, 4) if auc_probe is not None else None,
        "per_method": per_method,
        "real_p50": round(float(np.quantile(reals_p, 0.50)), 4) if len(reals_p) else None,
        "real_p90": round(float(np.quantile(reals_p, 0.90)), 4) if len(reals_p) else None,
        "fake_p50": round(float(np.quantile(fakes_p, 0.50)), 4) if len(fakes_p) else None,
        "fake_p95": round(float(np.quantile(fakes_p, 0.95)), 4) if len(fakes_p) else None,
        "n": len(usable),
        "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum()),
        "n_dropped_no_track": len(rows) - len(usable),
        "spike_thresh": round(spike, 6),
        "cal_n": len(cal_vals),
        "frames": args.frames,
        "embedder": args.embedder,
        "gate": gate,
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDRIFT EXAM: auc_p95={metrics['auc_p95']} auc_probe={metrics['auc_probe']} "
          f"n={metrics['n']} ({metrics['n_real']} real / {metrics['n_fake']} fake)",
          flush=True)
    print(f"  real drift_p95 p50={metrics['real_p50']} p90={metrics['real_p90']} | "
          f"fake p50={metrics['fake_p50']} p95={metrics['fake_p95']}", flush=True)
    print(f"  per-method: {json.dumps(per_method)}", flush=True)
    print(f"  DECISION GATE -> {gate}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
