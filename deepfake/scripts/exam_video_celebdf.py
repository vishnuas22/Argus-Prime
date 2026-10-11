"""Celeb-DF-v2 cross-dataset exam (spec item 4.2) — train on FF++ only,
evaluate the FIXED stacker on the official CDF test list. Never trains.

Usage:
  .venv/bin/python deepfake/scripts/exam_video_celebdf.py \
      [--frames 12] [--stacker artifacts/p6vid_v1] [--audio 0] \
      [--out stamped default] [--force]

Official list (List_of_testing_videos.txt): 517 videos — 340 Celeb-synthesis
(label 0 = fake) + 108 Celeb-real + 70 YouTube-real (label 1 = real). The
exam NORMALIZES those to 1 = fake / 0 = real so AUC, FPR and TPR mean the
same thing here as in exam_video_ffpp.py (see parse_list).
Threshold: stacked AUC >= 0.75 PASS / 0.70-0.75 WARN (escalate) / < 0.70 FAIL.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_dump import featurize_video  # noqa: E402
from video_features import FEATURE_NAMES, compute_row, row_to_X  # noqa: E402

CDF = WS / "data" / "raw" / "celebdf" / "Celeb-DF-v2"
LIST = CDF / "List_of_testing_videos.txt"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FEAT_COLS = ["file", "sha256_12", "label", "method", "split", "route",
             "naive_combined", "identity_video_score", "stacked"] + FEATURE_NAMES


def load_stacker(spec, embedder):
    """None when absent/unwired; hard-fail on embedder mismatch (D-040 skew
    class — identity clustering and drift features are embedder-dependent,
    same shape as train_video_stacker's dump guard)."""
    if not spec:
        return None, None
    p = Path(spec)
    if p.is_dir():
        p = p / "stacker.joblib"
    if not p.exists():
        return None, None
    meta_p = p.with_name("stacker_meta.json")
    if meta_p.exists():
        dumped = json.loads(meta_p.read_text()).get("embedder")
        if dumped and dumped != embedder:
            raise SystemExit(f"embedder mismatch: stacker trained with '{dumped}' "
                             f"but --embedder {embedder} — re-train or match")
    import joblib
    return joblib.load(p), p.parent.name


def write_features_csv(rows_path: Path, feat_path: Path, sha_of: dict) -> int:
    """Rebuild features.csv from EVERY row of scores.jsonl (see the FF++ exam).

    The --force resume path appends to scores.jsonl only, so an incrementally
    maintained features.csv silently loses previously-scored rows. Rebuild
    from the full record at the end of every run; image_court rows carry no
    stacker features and are excluded (count printed — never silent)."""
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    n_noface = 0
    with open(feat_path, "w", newline="") as ff:
        w = csv.DictWriter(ff, fieldnames=FEAT_COLS)
        w.writeheader()
        for r in rows:
            if r.get("route") != "video_face":
                n_noface += 1
                continue
            flat = compute_row(r.get("visual"), r.get("drift"), r.get("audio"))
            w.writerow({"file": r["file"],
                        "sha256_12": r.get("sha256_12") or sha_of.get(r["file"], ""),
                        "label": r["label"], "method": r.get("method"),
                        "split": "celebdf", "route": r.get("route"),
                        "naive_combined": r.get("naive_combined"),
                        "identity_video_score": r.get("identity_video_score"),
                        "stacked": "" if r.get("stacked") is None else r["stacked"],
                        **flat})
    print(f"features.csv rebuilt: {len(rows) - n_noface} video_face rows "
          f"({n_noface} image_court rows excluded — no stacker features)",
          flush=True)
    return len(rows) - n_noface


def parse_list():
    """[(path, label)] — NORMALIZED to 1 = fake, 0 = real.

    The CDF list ships 1 = real / 0 = fake, which is the OPPOSITE of every
    other exam script (FF++, drift, t2v all use 1 = fake). Fed straight into
    roc_auc_score with a P(fake) score it yields 1 − AUC: this gate reported
    0.0996 for a model that actually scores 0.9004. Normalize once, here."""
    jobs = []
    for line in LIST.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        label, rel = int(parts[0]), parts[1]
        p = CDF / rel
        if p.exists():
            jobs.append((p, 1 - label))     # 1 = fake
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--stacker", default=None,
                    help="video stacker dir/joblib (default: argus_detect."
                         "VIDEO_STACKER when promoted, else none)")
    ap.add_argument("--audio", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="cap the number of listed videos (smoke)")
    ap.add_argument("--drift-cal", default=None,
                    help="drift calibration JSON (spike_thresh fixes the "
                         "drift_spikes feature; defaults to the promoted cal)")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="sface",
                    help="identity embedder (must match the trained stacker)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from video_face_track import set_embedder
    set_embedder(args.embedder)
    from argus_detect import VIDEO_STACKER
    spec = args.stacker if args.stacker is not None else (
        str(VIDEO_STACKER) if VIDEO_STACKER.exists() else None)
    stacker, stacker_tag = load_stacker(spec, args.embedder)
    cal_path = args.drift_cal
    if cal_path is None:
        guess = VIDEO_STACKER.parent / "drift_calibration.json"
        cal_path = str(guess) if guess.exists() else None
    spike = (float(json.loads(Path(cal_path).read_text())["spike_thresh"])
             if cal_path and Path(cal_path).exists() else None)

    out = (Path(args.out) if args.out else WS / "artifacts" /
           f"exam_video_celebdf_{stacker_tag or 'nostacker'}_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    jobs = parse_list()
    if args.limit:
        jobs = jobs[: args.limit]
    sha_of = {}
    from video_dump import sha12
    for p, _ in jobs:
        sha_of[p.name] = sha12(p)
    jobs.sort(key=lambda j: (j[1], j[0].name))
    audio_set = set(sorted((p.name for p, _ in jobs),
                           key=lambda n: sha_of[n])[: args.audio])
    print(f"{len(jobs)} listed videos present on disk "
          f"({sum(1 for _, l in jobs if l == 0)} real / "
          f"{sum(1 for _, l in jobs if l == 1)} fake), stacker={stacker_tag or 'none'}",
          flush=True)

    rows_path = out / "scores.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["file"])
    t0 = time.time()
    scored = []
    with open(rows_path, "a") as f:
        for i, (vid, label) in enumerate(jobs):
            if vid.name in done:
                prev = [json.loads(l) for l in rows_path.read_text().splitlines()
                        if l.strip() and json.loads(l)["file"] == vid.name]
                if prev:
                    scored.append(prev[0])
                continue
            res = featurize_video(vid, args.frames, with_audio=(vid.name in audio_set),
                                  spike_thresh=spike)
            if res is None:
                row = {"file": vid.name, "label": label, "method": vid.parts[-2],
                       "route": "image_court", "visual": None, "drift": None,
                       "audio": None, "naive_combined": None,
                       "identity_video_score": None, "stacked": None}
                f.write(json.dumps(row) + "\n")
                f.flush()
                scored.append(row)
                print(f"[{i+1}/{len(jobs)}] {vid.name} route=image_court", flush=True)
                continue
            flat = res["row"]
            stacked = (float(stacker.predict_proba(row_to_X(flat).reshape(1, -1))[0, 1])
                       if stacker is not None else None)
            row = {"file": vid.name, "sha256_12": flat["sha256_12"],
                   "label": label, "method": vid.parts[-2],
                   "route": "video_face", "visual": res["visual"], "drift": res["drift"],
                   "audio": {k: v for k, v in (res["audio"] or {}).items()
                             if k != "per_segment"},
                   "naive_combined": flat["naive_combined"],
                   "identity_video_score": flat["identity_video_score"],
                   "stacked": None if stacked is None else round(stacked, 4)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            scored.append(row)
            print(f"[{i+1}/{len(jobs)}] {vid.parts[-2]}/{vid.name} "
                  f"naive={row['naive_combined']} stacked={row['stacked']}", flush=True)

    # features.csv rebuilt from the FULL scores.jsonl before metrics — a
    # resumed gate must never hand Gate B a subset.
    write_features_csv(rows_path, out / "features.csv", sha_of)

    from sklearn.metrics import roc_auc_score
    face = [r for r in scored if r.get("route") == "video_face"]
    aucs = {}
    for name, key in (("naive", "naive_combined"),
                      ("identity", "identity_video_score"),
                      ("stacked", "stacked")):
        pairs = [(r["label"], r[key]) for r in face if r.get(key) is not None]
        if len({p[0] for p in pairs}) > 1:
            aucs[name] = round(float(roc_auc_score([p[0] for p in pairs],
                                                   [p[1] for p in pairs])), 4)
    # operating point at 0.5 on the stacked (or naive) score
    op_key = "stacked" if aucs.get("stacked") is not None else "naive_combined"
    op = [(r["label"], r[op_key]) for r in face if r.get(op_key) is not None]
    y = np.asarray([p[0] for p in op])
    s = np.asarray([p[1] for p in op], dtype=float)
    fpr = float(((s >= 0.5) & (y == 0)).sum() / max(1, (y == 0).sum()))
    tpr = float(((s >= 0.5) & (y == 1)).sum() / max(1, (y == 1).sum()))
    n_total = len(scored)
    n_face = len(face)
    stacked_auc = aucs.get("stacked")
    verdict = ("PASS" if stacked_auc is not None and stacked_auc >= 0.75 else
               "WARN_ESCALATE" if stacked_auc is not None and stacked_auc >= 0.70 else
               "FAIL")
    metrics = {
        "auc": aucs,
        "fpr@0.5": round(fpr, 4), "tpr@0.5": round(tpr, 4),
        "op_on": op_key,
        "rates": {
            "no_face_route": round((n_total - n_face) / max(1, n_total), 4),
            "audio_out_of_coverage": round(sum(1 for r in face
                                               if (r.get("audio") or {}).get("out_of_coverage"))
                                           / max(1, n_total), 4),
            "drift_unavailable": round(sum(1 for r in face
                                           if (r.get("drift") or {}).get("unavailable_reason"))
                                       / max(1, n_face), 4),
            "identity_conflict": round(sum(1 for r in face
                                           if (r.get("visual") or {}).get("identity_conflict"))
                                       / max(1, n_face), 4),
        },
        "n": n_face, "n_total": n_total,
        "stacker": stacker_tag or "none",
        "embedder": args.embedder,
        "threshold": {"pass": 0.75, "warn": 0.70, "observed": stacked_auc,
                      "verdict": verdict},
        "frames": args.frames,
        "wall_min": round((time.time() - t0) / 60, 2),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nCELEB-DF EXAM: {json.dumps(aucs)} n={n_face}/{n_total} "
          f"fpr@0.5={fpr:.4f} tpr@0.5={tpr:.4f} -> {verdict} "
          f"({metrics['wall_min']}min)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
