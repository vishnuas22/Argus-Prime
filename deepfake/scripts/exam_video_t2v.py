"""Modern-generator (T2V) exam — INFORMATIONAL, no Phase 5 threshold (4.4).

Scores the GenVidBench Tier-1 corpus through the FULL detect_video route and
reports, per generator: AUC vs a real reference set AND the route
distribution (face-route vs image-court fallback share). On full-frame T2V
the image-court fallback is EXPECTED to dominate — measuring that share is
the point; it sets next-phase targets and never gates Phase 5.

  .venv/bin/python deepfake/scripts/exam_video_t2v.py \
      [--frames 12] [--n-real 140] [--real-set ffpp|celebdf] \
      [--stacker artifacts/p6vid_v1] [--out stamped] [--force]

Requires: data/raw/genvidbench/MANIFEST.json (fetch_genvidbench.py --tier 1).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

GENVID = WS / "data" / "raw" / "genvidbench"
MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
SPLITS = WS / "data" / "raw" / "ffpp"
CDF = WS / "data" / "raw" / "celebdf" / "Celeb-DF-v2"


def real_jobs(real_set: str, n: int):
    if real_set == "ffpp":
        ids = {i for p in json.load(open(SPLITS / "test.json")) for i in p}
        vids = [v for v in sorted((MIRROR / "real").glob("*.mp4")) if v.stem in ids]
    elif real_set == "celebdf":
        vids = []
        for line in (CDF / "List_of_testing_videos.txt").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2 and int(parts[0]) == 1 and (CDF / parts[1]).exists():
                vids.append(CDF / parts[1])
    else:
        raise SystemExit(f"unknown --real-set {real_set!r}")
    return [(v, 0, "real") for v in vids[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--n-real", type=int, default=140)
    ap.add_argument("--limit-per-gen", type=int, default=0,
                    help="cap clips per generator (0 = every clip on disk; "
                         "Tier-1 is 14k+ clips — subsample for a tractable "
                         "informational pass and report the n you ran)")
    ap.add_argument("--real-set", default="ffpp", choices=["ffpp", "celebdf"])
    ap.add_argument("--stacker", default=None)
    ap.add_argument("--embedder", default="arcface",
                    choices=["sface", "arcface"],
                    help="must match the embedder the stacker was trained on "
                         "(D-042: pipeline default is arcface)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    man_p = GENVID / "MANIFEST.json"
    if not man_p.exists():
        print(f"ERROR: {man_p} missing — run "
              f".venv/bin/python deepfake/scripts/fetch_genvidbench.py --tier 1 "
              f"first (Tier-1 acquisition, spec item 6.1)", file=sys.stderr)
        return 1
    man = json.loads(man_p.read_text())
    gen_dirs = {g: GENVID / g for g in man.get("generators", {}) if (GENVID / g).exists()}
    if not gen_dirs:
        print("ERROR: MANIFEST present but no extracted generator dirs on disk",
              file=sys.stderr)
        return 1

    from argus_detect import detect_video, VIDEO_STACKER
    from video_face_track import set_embedder
    set_embedder(args.embedder)
    spec = args.stacker if args.stacker is not None else (
        str(VIDEO_STACKER) if VIDEO_STACKER.exists() else None)
    if spec and Path(spec).is_dir():          # accept a stage dir too
        spec = str(Path(spec) / "stacker.joblib")
    stacker_tag = Path(spec).parent.name if spec and Path(spec).exists() else None
    if spec and Path(spec).exists():
        # detect_video() reads VIDEO_STACKER itself; point it at --stacker so
        # the pass scores with the same model the gates scored with.
        import argus_detect
        argus_detect.VIDEO_STACKER = Path(spec)
        argus_detect._VID_STACKER_CACHE = {"loaded": False, "clf": None,
                                           "embedder": None}

    out = (Path(args.out) if args.out else WS / "artifacts" /
           f"exam_video_t2v_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    jobs = real_jobs(args.real_set, args.n_real)
    for g, d in sorted(gen_dirs.items()):
        # case-insensitive + junk-excluded: keling ships `.MP4` and `._*`
        clips = sorted(q for q in d.iterdir() if q.is_file()
                       and q.suffix.lower() == ".mp4"
                       and not q.name.startswith("._"))
        if args.limit_per_gen:
            clips = clips[: args.limit_per_gen]
        for v in clips:
            jobs.append((v, 1, g))
    print(f"{len(jobs)} videos; stacker={stacker_tag or 'none'} "
          f"embedder={args.embedder} "
          f"(INFORMATIONAL exam — no Phase 5 threshold)", flush=True)

    rows_path = out / "scores.jsonl"
    done: dict = {}
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                _r = json.loads(line)
                # generator-qualified: clip names are only unique per
                # generator dir (sora/kling/opentora reuse numbering)
                done[(_r.get("generator", ""), _r["file"])] = _r
    t0 = time.time()
    rows = []
    with open(rows_path, "a") as f:
        for i, (vid, label, gen) in enumerate(jobs):
            key = (gen, vid.name)
            if key in done:
                rows.append(done[key])
                continue
            v = detect_video(vid, args.frames)
            row = {"file": vid.name, "label": label, "generator": gen,
                   "route": v.get("route"),
                   "p_fake_fused": v.get("p_fake_fused"),
                   "stacker": v.get("stacker"),
                   "confidence_degraded": v.get("confidence_degraded"),
                   "abstained": v.get("abstained"),
                   "n_identities": v.get("n_identities"),
                   "out_of_conflict": v.get("out_of_conflict")}
            f.write(json.dumps(row) + "\n")
            f.flush()
            rows.append(row)
            print(f"[{i+1}/{len(jobs)}] {gen}/{vid.name} route={row['route']} "
                  f"p={row['p_fake_fused']}", flush=True)

    from sklearn.metrics import roc_auc_score
    real_rows = [r for r in rows if r["label"] == 0 and r.get("p_fake_fused") is not None]
    per_gen, route_all = {}, {}
    # route share over EVERY scored row (fakes + the real reference set) —
    # a fallback share that silently drops the reals would overstate coverage.
    for r in rows:
        route_all[r["route"]] = route_all.get(r["route"], 0) + 1
    for g in sorted(gen_dirs):
        fakes = [r for r in rows if r["generator"] == g
                 and r.get("p_fake_fused") is not None]
        pairs = [(r["label"], r["p_fake_fused"]) for r in fakes] + \
                [(0, r["p_fake_fused"]) for r in real_rows]
        auc = (round(float(roc_auc_score([p[0] for p in pairs],
                                         [p[1] for p in pairs])), 4)
               if len({p[0] for p in pairs}) > 1 else None)
        n = len([r for r in rows if r["generator"] == g])
        dist = {}
        for r in [r for r in rows if r["generator"] == g]:
            dist[r["route"]] = dist.get(r["route"], 0) + 1
        per_gen[g] = {"auc": auc, "n": n, "routes": dist,
                      "image_court_share": round(dist.get("image_court", 0) / max(1, n), 4)}

    metrics = {"informational": True, "auc": per_gen,
               "route_distribution": route_all,
               "route_distribution_note": "all scored rows incl. real reference",
               "real_set": args.real_set, "n_real": len(real_rows),
               "limit_per_gen": args.limit_per_gen,
               "stacker": stacker_tag or "none", "embedder": args.embedder,
               "frames": args.frames,
               "wall_min": round((time.time() - t0) / 60, 2),
               "note": "no Phase 5 gate — sets next-phase targets; full-frame "
                       "T2V is expected to route mostly through the image court"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nT2V EXAM (informational): {json.dumps(per_gen)}", flush=True)
    print(f"  routes: {json.dumps(route_all)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
