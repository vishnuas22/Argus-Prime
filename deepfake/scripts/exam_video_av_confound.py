"""AV confound exam — D4 narration/music strata (W-AV, D-076 §6).

Scores the EXAM narration/music strata (real-class rows whose audio is
NOT the source's lip-synced speech: Kokoro TTS narration, synthesized music
beds) at the DEPLOYED operating point (stacked pf > 0.5 — argus_detect.py's
verdict rule). These strata exist to catch the stacker learning
"audio present + lips moving = fake": if the diet taught that shortcut,
out-of-family audio reals read fake-ward and this FPR fires (bar <= 0.10).

All rows are label 0 BY DESIGN (outside the threat family). metrics.json
carries fpr_issued, in_coverage_rate, mean_pf, n_rows + a per-stratum
breakdown (narration vs music — informational; the gate is the pooled FPR).

Mirrors exam_video_dfd.py's shape: out-dir guard verbatim, features.csv
rebuilt from the FULL scores.jsonl at end-of-run, same metrics skeleton.

Usage: .venv/bin/python deepfake/scripts/exam_video_av_confound.py \
           [--frames 12] [--limit N] [--video-stacker <dir>] \
           [--drift-cal <json>] [--av-calib <json>] [--embedder arcface] \
           --out <dir>
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
from video_features import AV_FEATURES, row_to_X  # noqa: E402

AV_DIET = WS / "data" / "raw" / "av_diet"
AV_GEN = AV_DIET / "gen"
OPERATING_POINT = 0.5                 # argus_detect verdict rule (deployed)


def load_stacker(spec, embedder):
    from exam_video_ffpp import load_stacker as _load
    return _load(spec, embedder)


def enumerate_confound(limit: int | None = None, progress=print):
    """(jobs, expected_counts) from the generation receipt's EXAM strata."""
    genr = json.loads((AV_DIET / "gen_receipt.json").read_text())
    jobs = []
    for method in ("narration", "music"):
        for name in genr.get(method, {}).get("exam", []):
            p = AV_GEN / method / name
            if not p.exists():
                raise SystemExit(f"confound stratum file missing: {p} — the "
                                 f"generation receipt lists it; run the "
                                 f"generation driver to completion")
            jobs.append((p, 0, method, name))
    jobs.sort(key=lambda j: (j[2], j[0].name))
    if limit:
        jobs = jobs[:limit]
    expected = {m: sum(1 for j in jobs if j[2] == m)
                for m in ("narration", "music")}
    progress(f"[enum] confound exam: {len(jobs)} real-class strata rows "
             f"{expected} (deployed operating point pf > {OPERATING_POINT})")
    return jobs, expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke cap: first N strata rows")
    ap.add_argument("--drift-cal", default=None)
    ap.add_argument("--av-calib", default=None,
                    help="AV coverage calibration JSON (defaults to "
                         "av_calibration.json beside the stacker); REQUIRED")
    ap.add_argument("--av-cache", default=None,
                    help="per-workdir AV result cache (calibration-safe; "
                         "without it every run re-featurizes from scratch)")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="arcface")
    ap.add_argument("--video-stacker", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from video_face_track import set_embedder
    set_embedder(args.embedder)
    from argus_detect import STACKER, VIDEO_STACKER
    stacker_spec = args.video_stacker if args.video_stacker is not None else (
        str(VIDEO_STACKER) if VIDEO_STACKER.exists() else None)
    stacker, stacker_tag, names = load_stacker(stacker_spec, args.embedder)

    cal_path = args.drift_cal
    if cal_path is None:
        base = (Path(stacker_spec) if stacker_spec else VIDEO_STACKER)
        guess = (base if base.is_dir() else base.parent) / "drift_calibration.json"
        cal_path = str(guess) if guess.exists() else None
    spike = (float(json.loads(Path(cal_path).read_text())["spike_thresh"])
             if cal_path and Path(cal_path).exists() else None)

    av_cal_path = args.av_calib
    if av_cal_path is None:
        base = (Path(stacker_spec) if stacker_spec else VIDEO_STACKER)
        guess = (base if base.is_dir() else base.parent) / "av_calibration.json"
        av_cal_path = str(guess) if guess.exists() else None
    if av_cal_path is None or not Path(av_cal_path).exists():
        raise SystemExit("av_calibration.json not found beside the stacker "
                         "(and no --av-calib) — fail-closed (W-AV D-076)")
    av_calib = json.loads(Path(av_cal_path).read_text())

    out = (Path(args.out) if args.out else WS / "artifacts" /
           f"exam_video_av_confound_{stacker_tag or 'nostacker'}_"
           f"{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass "
              f"--force to append/resume", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    jobs, expected = enumerate_confound(args.limit)
    sha_of = {(m, v.name): sha12(v) for v, _, m, _n in jobs}
    print(f"{len(jobs)} videos, stacker={stacker_tag or 'none'}, "
          f"spike={spike}, av_calib={av_cal_path}", flush=True)

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
        for i, (vid, label, method, name) in enumerate(jobs):
            if (method, vid.name) in prev_rows:
                scored.append(prev_rows[(method, vid.name)])
                continue
            res = featurize_video(vid, args.frames, with_audio=True,
                                  spike_thresh=spike, with_av=True,
                                  av_calib=av_calib,
                                  av_cache_dir=(Path(args.av_cache)
                                                if args.av_cache else None))
            if res is None:
                row = {"file": vid.name, "label": label, "method": method,
                       "route": "image_court", "visual": None, "drift": None,
                       "audio": None, "naive_combined": None,
                       "identity_video_score": None, "stacked": None}
                f.write(json.dumps(row) + "\n")
                f.flush()
                scored.append(row)
                print(f"[{i+1}/{len(jobs)}] {method}/{vid.name} "
                      f"route=image_court (no face)", flush=True)
                continue
            flat = res["row"]
            stacked = (float(stacker.predict_proba(
                row_to_X(flat, names).reshape(1, -1))[0, 1])
                if stacker is not None else None)
            row = {"file": vid.name, "sha256_12": flat["sha256_12"],
                   "label": label, "method": method, "route": "video_face",
                   "visual": res["visual"], "drift": res["drift"],
                   "audio": res["audio"],
                   "avs": {k: flat[k] for k in AV_FEATURES},
                   "av_diagnostics": res.get("av_diagnostics"),
                   "naive_combined": flat["naive_combined"],
                   "identity_video_score": flat["identity_video_score"],
                   "stacked": None if stacked is None else round(stacked, 4)}
            f.write(json.dumps(row) + "\n")
            f.flush()
            scored.append(row)
            print(f"[{i+1}/{len(jobs)}] {method}/{vid.name} "
                  f"stacked={row['stacked']}", flush=True)

    from exam_video_ffpp import write_features_csv
    write_features_csv(rows_path, out / "features.csv", sha_of, split="exam")

    face = [r for r in scored if r.get("route") == "video_face"]
    issued = [r for r in face
              if r.get("stacked") is not None and r["stacked"] > OPERATING_POINT]
    fpr = (round(len(issued) / max(1, len(face)), 4)
           if face else None)
    pfs = [r["stacked"] for r in face if r.get("stacked") is not None]
    mean_pf = round(sum(pfs) / len(pfs), 4) if pfs else None
    in_cov = sum(1 for r in face
                 if (r.get("avs") or {}).get("avs_in_coverage") == 1)
    per_stratum = {}
    for m in ("narration", "music"):
        sub = [r for r in face if r.get("method") == m]
        iss = [r for r in sub
               if r.get("stacked") is not None and r["stacked"] > OPERATING_POINT]
        per_stratum[m] = {"n": len(sub), "issued": len(iss),
                          "fpr": round(len(iss) / max(1, len(sub)), 4)}
    metrics = {"n_scored": len(face), "n_total": len(scored),
               "fpr_issued": fpr, "mean_pf": mean_pf,
               "operating_point": OPERATING_POINT,
               "in_coverage_rate": round(in_cov / max(1, len(face)), 4),
               "per_stratum": per_stratum,
               "expected": expected,
               "manifest": str(AV_DIET / "gen_receipt.json"),
               "stacker": stacker_tag or "none", "embedder": args.embedder,
               "frames": args.frames,
               "wall_min": round((time.time() - t0) / 60, 2)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nAV CONFOUND EXAM: fpr_issued={fpr} (bar <=0.10) "
          f"mean_pf={mean_pf} n={len(face)} ({metrics['wall_min']}min)",
          flush=True)
    print(f"  per-stratum: {json.dumps(per_stratum)}", flush=True)
    print(f"  in_coverage={metrics['in_coverage_rate']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
