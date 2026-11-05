"""FF++ video exam — official test split, per-method, A/B, coverage (4.1).

Full official test split by default (140 real + 4x140 fake = 700 videos, 12
frames). Identity-disjoint from any training. Each row carries the full
identity-aware visual dict, the drift dict, the route, and — when a video
stacker is wired or passed — the stacked prediction. Three AUCs on identical
rows: naive_combined (video_score.combined fallback), identity_video_score
(raw identity aggregation verdict), stacked.

Usage: .venv/bin/python deepfake/scripts/exam_video_ffpp.py \
           [--n-real 140] [--per-method 140] [--frames 12] \
           [--audio 200] [--video-stacker <dir>] --out <dir>
Audio is OFF by default (advisory 200-video subsample per user decision);
--audio N scores the first N rows in deterministic order.
Default --out is stamped: artifacts/exam_video_ffpp_<stacker>_<YYYYMMDD>.
A non-empty out dir is refused unless --force (which resumes/appends —
features.csv is rebuilt from the FULL scores.jsonl at the end of every run,
so a resumed gate never loses previously-scored rows).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_dump import featurize_video  # noqa: E402
from video_features import (FEATURE_NAMES, FEATURE_NAMES_FULL, compute_row,
                            row_to_X)  # noqa: E402

MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
SPLITS = WS / "data" / "raw" / "ffpp"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FEAT_COLS = ["file", "sha256_12", "label", "method", "split", "route",
             "naive_combined", "identity_video_score", "stacked"] \
    + FEATURE_NAMES_FULL


def load_stacker(spec, embedder):
    """(clf, tag, names) — names threads into every row_to_X call.

    None when absent/unwired — the honest 'no stacker' path.

    Guards (D-040 skew class): when a stacker IS present, (a) its meta
    embedder must equal the --embedder argument — identity clustering and
    drift features are embedder-dependent, so a mismatched gate would measure
    a model that does not exist (same shape as train_video_stacker's dump
    guard); (b) its meta feature_names must equal video_features.FEATURE_NAMES
    (the deployed space) OR FEATURE_NAMES_FULL (the W4c extended space — the
    only pre-flip readers of a candidate) — anything else still fail-closes;
    a stale candidate must never score a NEW dump in the wrong feature space."""
    if not spec:
        return None, None, None
    p = Path(spec)
    if p.is_dir():
        p = p / "stacker.joblib"
    if not p.exists():
        return None, None, None
    names = FEATURE_NAMES
    meta_p = p.with_name("stacker_meta.json")
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        dumped = meta.get("embedder")
        if dumped and dumped != embedder:
            raise SystemExit(f"embedder mismatch: stacker trained with '{dumped}' "
                             f"but --embedder {embedder} — re-train or match")
        meta_names = meta.get("feature_names")
        if meta_names is not None and meta_names not in (FEATURE_NAMES,
                                                         FEATURE_NAMES_FULL):
            raise SystemExit(
                f"feature-space mismatch: stacker_meta.feature_names is "
                f"{len(meta_names)} dims, expected the deployed "
                f"{len(FEATURE_NAMES)} or the W4c extended "
                f"{len(FEATURE_NAMES_FULL)} space (D-040 skew class); "
                f"re-train or match")
        if meta_names is not None:
            names = meta_names
    import joblib
    return joblib.load(p), p.parent.name, names


def write_features_csv(rows_path: Path, feat_path: Path, sha_of: dict,
                       split: str = "test") -> int:
    """Rebuild features.csv from EVERY row of scores.jsonl.

    The --force resume path appends to scores.jsonl only, so an incrementally
    maintained features.csv silently loses the previously-scored rows — and
    restack_video's Gate B/LOMO reads features.csv. Rebuilding from the full
    record at the end of every run guarantees Gate B sees ALL scored rows.
    image_court rows carry no stacker features (serve routes them to the
    image court, never the stacker) and are excluded; the count is printed
    so the exclusion is never silent."""
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    n_noface = 0
    with open(feat_path, "w", newline="") as ff:
        w = csv.DictWriter(ff, fieldnames=FEAT_COLS)
        w.writeheader()
        for r in rows:
            if r.get("route") != "video_face":
                n_noface += 1
                continue
            # avs (W-AV): rows scored through the AV channel carry their avs
            # dict in scores.jsonl; older rows lack the key -> None -> the
            # missingness zero-fill (identical to their dump-time values).
            flat = compute_row(r.get("visual"), r.get("drift"), r.get("audio"),
                               r.get("avs"))
            w.writerow({"file": r["file"],
                        "sha256_12": r.get("sha256_12") or sha_of.get(
                            (r.get("method", ""), r["file"]), ""),
                        "label": r["label"], "method": r.get("method"),
                        "split": split, "route": r.get("route"),
                        "naive_combined": r.get("naive_combined"),
                        "identity_video_score": r.get("identity_video_score"),
                        "stacked": "" if r.get("stacked") is None else r["stacked"],
                        **flat})
    print(f"features.csv rebuilt: {len(rows) - n_noface} video_face rows "
          f"({n_noface} image_court rows excluded — no stacker features)",
          flush=True)
    return len(rows) - n_noface


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=140)
    ap.add_argument("--per-method", type=int, default=140)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--audio", type=int, default=0,
                    help="run the XLS-R probe on the first N rows "
                         "(deterministic order); 0 = none (default)")
    ap.add_argument("--drift-cal", default=None,
                    help="drift calibration JSON (spike_thresh fixes the "
                         "drift_spikes feature; defaults to the promoted cal)")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="sface",
                    help="identity embedder (must match the trained stacker)")
    ap.add_argument("--no-audio", action="store_true",
                    help="explicit no-audio (kept for compatibility; audio "
                         "is already off unless --audio N is given)")
    ap.add_argument("--video-stacker", default=None,
                    help="video stacker dir/joblib (default: argus_detect."
                         "VIDEO_STACKER when promoted, else none)")
    ap.add_argument("--out", default=None,
                    help="output dir (default artifacts/exam_video_ffpp_<stacker>_<YYYYMMDD>)")
    ap.add_argument("--force", action="store_true",
                    help="run in an output dir that already contains results "
                         "(existing scores.jsonl is kept and appended — resume)")
    args = ap.parse_args()

    from video_face_track import set_embedder
    set_embedder(args.embedder)
    from argus_detect import STACKER, VIDEO_STACKER
    stacker_spec = args.video_stacker if args.video_stacker is not None else (
        str(VIDEO_STACKER) if VIDEO_STACKER.exists() else None)
    stacker, stacker_tag, names = load_stacker(stacker_spec, args.embedder)

    cal_path = args.drift_cal
    if cal_path is None:
        guess = VIDEO_STACKER.parent / "drift_calibration.json"
        cal_path = str(guess) if guess.exists() else None
    spike = (float(json.loads(Path(cal_path).read_text())["spike_thresh"])
             if cal_path and Path(cal_path).exists() else None)

    out = (Path(args.out) if args.out else WS / "artifacts" /
           f"exam_video_ffpp_{stacker_tag or 'nostacker'}_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force to append/resume",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    test_pairs = json.load(open(SPLITS / "test.json"))
    pair_set = {tuple(p) for p in test_pairs}
    ids = {i for p in test_pairs for i in p}
    jobs = []  # (path, label)
    # filter to the official test split FIRST, then cap — slicing first took
    # the first 140 alphabetical files (only 19 of which are test-split) and
    # silently shrank gate A from 140 reals to 19.
    test_reals = [vid for vid in sorted((MIRROR / "real").glob("*.mp4"))
                  if vid.stem in ids]
    for vid in test_reals[: args.n_real]:
        jobs.append((vid, 0))
    import numpy as np
    rng = np.random.default_rng(11)
    for m in METHODS:
        fakes = [v for v in sorted((MIRROR / "fake" / m).glob("*.mp4"))
                 if tuple(v.stem.split("_")) in pair_set
                 or tuple(reversed(v.stem.split("_"))) in pair_set]
        for v in rng.permutation(fakes)[: args.per_method]:
            jobs.append((v, 1))
    # deterministic audio subsample + resume-safe ordering.
    # keyed on (method, name): FF++ method dirs REUSE filenames across
    # methods (patterns.md:40) — a bare-name key collapses same-stem files.
    sha_of = {}
    from video_dump import sha12
    for vid, _ in jobs:
        sha_of[(vid.parent.name, vid.name)] = sha12(vid)
    jobs.sort(key=lambda j: (j[1], j[0].parent.name, j[0].name))
    audio_set = set(sorted((k for k in sha_of),
                           key=lambda k: sha_of[k])[: args.audio])
    print(f"{len(jobs)} videos ({sum(1 for _, l in jobs if l == 0)} real / "
          f"{sum(1 for _, l in jobs if l == 1)} fake), audio subsample "
          f"{len(audio_set)}, stacker={stacker_tag or 'none'}", flush=True)

    rows_path = out / "scores.jsonl"
    done_keys = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                _r = json.loads(line)
                # dir-qualified: method dirs reuse filenames across methods
                done_keys.add((_r.get("method", ""), _r["file"]))
    t0 = time.time()
    scored = []
    with open(rows_path, "a") as f:
        for i, (vid, label) in enumerate(jobs):
            if (vid.parent.name, vid.name) in done_keys:
                prev = [json.loads(l) for l in rows_path.read_text().splitlines()
                        if l.strip() and json.loads(l)["method"] == vid.parent.name
                        and json.loads(l)["file"] == vid.name]
                if prev:
                    scored.append(prev[0])
                continue
            res = featurize_video(vid, args.frames,
                                  with_audio=((vid.parent.name, vid.name) in audio_set),
                                  spike_thresh=spike)
            if res is None:
                row = {"file": vid.name, "label": label, "method": vid.parent.name,
                       "route": "image_court", "visual": None, "drift": None,
                       "audio": None, "naive_combined": None,
                       "identity_video_score": None, "stacked": None}
                f.write(json.dumps(row) + "\n")
                f.flush()
                scored.append(row)
                print(f"[{i+1}/{len(jobs)}] {vid.parent.name}/{vid.name} "
                      f"route=image_court (no face)", flush=True)
                continue
            flat = res["row"]
            stacked = (float(stacker.predict_proba(
                row_to_X(flat, names).reshape(1, -1))[0, 1])
                if stacker is not None else None)
            row = {"file": vid.name, "sha256_12": flat["sha256_12"],
                   "label": label, "method": vid.parent.name,
                   "route": "video_face", "visual": res["visual"],
                   "drift": res["drift"],
                   "audio": {k: v for k, v in (res["audio"] or {}).items()
                             if k != "per_segment"},
                   "naive_combined": flat["naive_combined"],
                   "identity_video_score": flat["identity_video_score"],
                   "stacked": None if stacked is None else round(stacked, 4)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            scored.append(row)
            print(f"[{i+1}/{len(jobs)}] {vid.parent.name}/{vid.name} "
                  f"naive={row['naive_combined']} id={row['identity_video_score']} "
                  f"stacked={row['stacked']}", flush=True)

    # features.csv is rebuilt from the FULL scores.jsonl (all scored rows,
    # including any resumed ones) BEFORE metrics are written — Gate B/LOMO
    # must never consume a subset.
    write_features_csv(rows_path, out / "features.csv", sha_of, split="test")

    from sklearn.metrics import roc_auc_score
    face = [r for r in scored if r.get("route") == "video_face"]
    y = [r["label"] for r in face]
    aucs = {}
    for name, key in (("naive_combined", "naive_combined"),
                      ("identity_video_score", "identity_video_score"),
                      ("stacked", "stacked")):
        s = [r[key] for r in face if r.get(key) is not None]
        yy = [r["label"] for r in face if r.get(key) is not None]
        if len(set(yy)) > 1 and len(s):
            aucs[name] = round(float(roc_auc_score(yy, s)), 4)
    # 5b.9 diagnostic: FF++-specialist verdict space through the SAME
    # combined() shape as naive_combined — a gen-vs-spec gate miss is then
    # self-diagnosing from metrics.json alone.
    from video_score import combined as _combined
    spec_pairs = [(r["label"], _combined((r.get("visual") or {}).get("spec"),
                                         r.get("audio")))
                  for r in face if (r.get("visual") or {}).get("spec")]
    spec_pairs = [p for p in spec_pairs if p[1] is not None]
    if spec_pairs and len({p[0] for p in spec_pairs}) > 1:
        aucs["naive_spec"] = round(float(
            roc_auc_score([p[0] for p in spec_pairs],
                          [p[1] for p in spec_pairs])), 4)
    per_method = {}
    for m in METHODS:
        sub = [r for r in face if r["method"] == m]
        real = [r for r in face if r["label"] == 0]
        block = {"n": len(sub)}
        for name, key in (("naive", "naive_combined"),
                          ("identity", "identity_video_score"),
                          ("stacked", "stacked")):
            pairs = [(r["label"], r[key]) for r in (sub + real)
                     if r.get(key) is not None]
            if len({p[0] for p in pairs}) > 1:
                block[name] = round(float(roc_auc_score([p[0] for p in pairs],
                                                        [p[1] for p in pairs])), 4)
        per_method[m] = block
    n_total = len(scored)
    n_face = len(face)
    rates = {
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
        "abstained": round(sum(1 for r in scored if r.get("naive_combined") is None)
                           / max(1, n_total), 4),
    }
    metrics = {"n_scored": n_face, "n_total": n_total, "auc": aucs,
               "per_method": per_method, "rates": rates,
               "frames": args.frames, "stacker": stacker_tag or "none",
               "embedder": args.embedder,
               "audio_rows": len(audio_set),
               "wall_min": round((time.time() - t0) / 60, 2)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nVIDEO EXAM: {json.dumps(aucs)} (n_scored={n_face}/{n_total}, "
          f"{metrics['wall_min']}min)", flush=True)
    print(f"  per-method: {json.dumps(per_method)}", flush=True)
    print(f"  rates: {json.dumps(rates)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
