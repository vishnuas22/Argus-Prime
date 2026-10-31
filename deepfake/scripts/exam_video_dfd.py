"""DFD video exam — held-out DeepFakeDetection sources (W4, D-066).

Scores the EXAM split of the DFD diet manifest (data/raw/ffpp_dfd/split.json,
produced ONCE by ffpp/dfd_prepare.py): fakes = mirror DeepFakeDetection files
whose (target,sequence) source is an exam source; reals = the SAME sources'
actor originals from data/raw/ffpp_actors (domain-matched — every exam fake
has its source original on the real side). Whole SOURCES are held out: no
exam fake or original is ever dumped into the v3 diet.

Mirrors exam_video_ffpp.py's shape: same out-dir guard verbatim, features.csv
rebuilt from the FULL scores.jsonl at end-of-run (resume law, patterns:44),
same metrics skeleton; additionally records n_fakes/n_reals/n_sources, the
split-manifest path, and the D2 confound-guard stats (real_mean_pf,
real_frac_gt_half — if the diet taught "actor domain = fake", held-out actor
reals read fake-ward and these fire).

Usage: .venv/bin/python deepfake/scripts/exam_video_dfd.py \
           [--frames 12] [--limit N] [--video-stacker <dir>] \
           [--drift-cal <json>] [--embedder arcface] --out <dir>
Default --out is stamped: artifacts/exam_video_dfd_<stacker>_<YYYYMMDD>.
A non-empty out dir is refused unless --force (which resumes/appends —
features.csv is rebuilt from the FULL scores.jsonl at the end of every run).
"""
import argparse
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_dump import featurize_video, sha12  # noqa: E402
from video_features import row_to_X  # noqa: E402

MIRROR_DFD = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / \
    "fake" / "DeepFakeDetection"
ACTORS = WS / "data" / "raw" / "ffpp_actors" / "c23" / "videos"
DFD_MANIFEST = WS / "data" / "raw" / "ffpp_dfd"
DFD_METHOD = "DeepFakeDetection"


def load_stacker(spec, embedder):
    """Same guards as exam_video_ffpp.load_stacker (embedder + feature space)."""
    from exam_video_ffpp import load_stacker as _load
    return _load(spec, embedder)


def enumerate_exam(smoke_limit: int | None = None, progress=print):
    """(jobs, n_sources) — jobs are (path, label, method, source_key)."""
    assignment = json.loads((DFD_MANIFEST / "split.json").read_text())
    sources = json.loads((DFD_MANIFEST / "sources.json").read_text())
    exam = [k for k in sorted(sources) if assignment[k] == "exam"]
    jobs = []
    for k in exam:
        orig = ACTORS / sources[k]["original"]
        if not orig.exists():
            raise SystemExit(f"missing actor original {orig} — run "
                             f"ffpp/download_actors.py first")
        jobs.append((orig, 0, "real", k))
        for name in sources[k]["fakes"]:
            jobs.append((MIRROR_DFD / f"{name}.mp4", 1, DFD_METHOD, k))
    if smoke_limit:
        fakes = [j for j in jobs if j[1] == 1][:smoke_limit]
        reals = [j for j in jobs if j[1] == 0][:max(1, smoke_limit // 2)]
        jobs = reals + fakes
    # deterministic order: reals first, then by method/name (resume-safe)
    jobs.sort(key=lambda j: (j[1], j[2], j[0].name))
    progress(f"[enum] exam: {sum(1 for j in jobs if j[1] == 0)} reals / "
             f"{sum(1 for j in jobs if j[1] == 1)} fakes over "
             f"{len(exam)} held-out sources"
             + (f" — SMOKE limit {smoke_limit}" if smoke_limit else ""))
    return jobs, len(exam)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke cap: first N fakes + N//2 reals "
                         "(deterministic; full exam = the whole exam split)")
    ap.add_argument("--drift-cal", default=None,
                    help="drift calibration JSON (spike_thresh fixes the "
                         "drift_spikes feature; defaults to the promoted cal)")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="sface",
                    help="identity embedder (must match the trained stacker)")
    ap.add_argument("--video-stacker", default=None,
                    help="video stacker dir/joblib (default: argus_detect."
                         "VIDEO_STACKER when promoted, else none)")
    ap.add_argument("--out", default=None,
                    help="output dir (default artifacts/exam_video_dfd_<stacker>_<YYYYMMDD>)")
    ap.add_argument("--force", action="store_true",
                    help="run in an output dir that already contains results "
                         "(existing scores.jsonl is kept and appended — resume)")
    args = ap.parse_args()

    from video_face_track import set_embedder
    set_embedder(args.embedder)
    from argus_detect import STACKER, VIDEO_STACKER
    stacker_spec = args.video_stacker if args.video_stacker is not None else (
        str(VIDEO_STACKER) if VIDEO_STACKER.exists() else None)
    stacker, stacker_tag = load_stacker(stacker_spec, args.embedder)

    cal_path = args.drift_cal
    if cal_path is None:
        guess = VIDEO_STACKER.parent / "drift_calibration.json"
        cal_path = str(guess) if guess.exists() else None
    spike = (float(json.loads(Path(cal_path).read_text())["spike_thresh"])
             if cal_path and Path(cal_path).exists() else None)

    out = (Path(args.out) if args.out else WS / "artifacts" /
           f"exam_video_dfd_{stacker_tag or 'nostacker'}_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force to append/resume",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    jobs, n_sources = enumerate_exam(args.limit)
    sha_of = {}
    for vid, _, method, _src in jobs:
        sha_of[(method, vid.name)] = sha12(vid)
    print(f"{len(jobs)} videos, stacker={stacker_tag or 'none'}, "
          f"spike={spike}", flush=True)

    rows_path = out / "scores.jsonl"
    prev_rows = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                _r = json.loads(line)
                prev_rows[(_r.get("method", ""), _r["file"])] = _r
    t0 = time.time()
    scored = []
    with open(rows_path, "a") as f:
        for i, (vid, label, method, src_key) in enumerate(jobs):
            if (method, vid.name) in prev_rows:
                scored.append(prev_rows[(method, vid.name)])
                continue
            res = featurize_video(vid, args.frames, with_audio=False,
                                  spike_thresh=spike)
            if res is None:
                row = {"file": vid.name, "label": label, "method": method,
                       "source": src_key, "route": "image_court",
                       "visual": None, "drift": None, "audio": None,
                       "naive_combined": None, "identity_video_score": None,
                       "stacked": None}
                f.write(json.dumps(row) + "\n")
                f.flush()
                scored.append(row)
                print(f"[{i+1}/{len(jobs)}] {method}/{vid.name} "
                      f"route=image_court (no face)", flush=True)
                continue
            flat = res["row"]
            stacked = (float(stacker.predict_proba(row_to_X(flat).reshape(1, -1))[0, 1])
                       if stacker is not None else None)
            row = {"file": vid.name, "sha256_12": flat["sha256_12"],
                   "label": label, "method": method, "source": src_key,
                   "route": "video_face", "visual": res["visual"],
                   "drift": res["drift"], "audio": None,
                   "naive_combined": flat["naive_combined"],
                   "identity_video_score": flat["identity_video_score"],
                   "stacked": None if stacked is None else round(stacked, 4)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            scored.append(row)
            print(f"[{i+1}/{len(jobs)}] {method}/{vid.name} "
                  f"naive={row['naive_combined']} "
                  f"id={row['identity_video_score']} stacked={row['stacked']}",
                  flush=True)

    # features.csv rebuilt from the FULL scores.jsonl (resume law, patterns:44)
    from exam_video_ffpp import write_features_csv
    write_features_csv(rows_path, out / "features.csv", sha_of, split="exam")

    from sklearn.metrics import roc_auc_score
    face = [r for r in scored if r.get("route") == "video_face"]
    reals = [r for r in face if r["label"] == 0]
    fakes = [r for r in face if r["label"] == 1]
    aucs = {}
    for name, key in (("naive_combined", "naive_combined"),
                      ("identity_video_score", "identity_video_score"),
                      ("stacked", "stacked")):
        s = [r[key] for r in face if r.get(key) is not None]
        yy = [r["label"] for r in face if r.get(key) is not None]
        if len(set(yy)) > 1 and len(s):
            aucs[name] = round(float(roc_auc_score(yy, s)), 4)
    from video_score import combined as _combined
    spec_pairs = [(r["label"], _combined((r.get("visual") or {}).get("spec"),
                                         r.get("audio")))
                  for r in face if (r.get("visual") or {}).get("spec")]
    spec_pairs = [p for p in spec_pairs if p[1] is not None]
    if spec_pairs and len({p[0] for p in spec_pairs}) > 1:
        aucs["naive_spec"] = round(float(
            roc_auc_score([p[0] for p in spec_pairs],
                          [p[1] for p in spec_pairs])), 4)

    # D2 confound-guard stats: stacked pf on the held-out ACTOR REALS
    real_scores = [r["stacked"] for r in reals if r.get("stacked") is not None]
    real_mean_pf = (round(sum(real_scores) / len(real_scores), 4)
                    if real_scores else None)
    real_frac_gt_half = (round(sum(1 for s in real_scores if s > 0.5)
                               / len(real_scores), 4) if real_scores else None)
    rates = {
        "no_face_route": round((len(scored) - len(face)) / max(1, len(scored)), 4),
        "drift_unavailable": round(sum(1 for r in face
                                       if (r.get("drift") or {}).get("unavailable_reason"))
                                   / max(1, len(face)), 4),
        "identity_conflict": round(sum(1 for r in face
                                       if (r.get("visual") or {}).get("identity_conflict"))
                                   / max(1, len(face)), 4),
        "abstained": round(sum(1 for r in scored if r.get("naive_combined") is None)
                           / max(1, len(scored)), 4),
    }
    metrics = {"n_scored": len(face), "n_total": len(scored),
               "n_fakes": len(fakes), "n_reals": len(reals),
               "n_sources": n_sources,
               "n_sources_scored": len({r.get("source") for r in face
                                        if r.get("source")}),
               "split_manifest": str(DFD_MANIFEST / "split.json"),
               "auc": aucs, "real_mean_pf": real_mean_pf,
               "real_frac_gt_half": real_frac_gt_half,
               "rates": rates, "frames": args.frames,
               "stacker": stacker_tag or "none", "embedder": args.embedder,
               "audio_rows": 0,
               "wall_min": round((time.time() - t0) / 60, 2)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nDFD EXAM: {json.dumps(aucs)} (n_scored={len(face)}/{len(scored)}, "
          f"n_sources={n_sources}, {metrics['wall_min']}min)", flush=True)
    print(f"  D2 guard: real_mean_pf={real_mean_pf} (bar <=0.35), "
          f"real_frac_gt_half={real_frac_gt_half} (bar <=0.25)", flush=True)
    print(f"  rates: {json.dumps(rates)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
