"""AV exam — D1 self-gen held-out + D2 AVLips external + D3 integrity (W-AV, D-076).

--corpus av     scores the EXAM split of the av diet (data/raw/av_diet/
                split.json, produced ONCE by build_av_diet.py): fakes =
                gen/fakes/{key}.mp4 (Wav2Lip), reals = gen/reals/{key}.mp4 —
                the SAME sources' matched reals (domain-matched by
                construction). Exam sources are held out entirely: never
                dumped, never trained on. Receipted generation failures
                (no-face honesty §5.5) are excluded.
--corpus avlips scores the AVLips VALIDATION SET — external, never trained
                on anything from it (D2's out-of-family exam: multi-
                generator, perturbed). Reads data/raw/avlips/manifest.json
                {real: [paths], fake: [paths]} written by the acquisition
                step; >2000 videos -> deterministic 1000/class sha-sorted
                seed-11 subsample (cap decision recorded).

Mirrors exam_video_dfd.py's shape: same out-dir guard verbatim (patterns:32),
features.csv rebuilt from the FULL scores.jsonl at end-of-run (patterns:44),
same metrics skeleton. AV specifics: every row runs WITH audio and the AV
channel (with_av=True); the coverage calibration rides the stacker bundle
(av_calibration.json beside the stacker — the serve contract) and is
REQUIRED (fail-closed: without it every row is out-of-coverage and the exam
is meaningless). metrics.json carries the D3 integrity block (source
straddle, per-pair ffprobe parity, dedup, license completeness) — restack's
av gate folds it into av.pass as a hard stop.

Usage: .venv/bin/python deepfake/scripts/exam_video_av.py --corpus av|avlips \
           [--frames 12] [--limit N] [--video-stacker <dir>] \
           [--drift-cal <json>] [--av-calib <json>] [--embedder arcface] \
           --out <dir>
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_dump import featurize_video, sha12  # noqa: E402
from video_features import AV_FEATURES, row_to_X  # noqa: E402

AV_DIET = WS / "data" / "raw" / "av_diet"
AV_GEN = AV_DIET / "gen"
AVLIPS_DIR = WS / "data" / "raw" / "avlips"
AVLIPS_CAP = 1000                    # per class, when >2000 videos (spec §6)
REQUIRED_LICENSES = ("ravdess", "wav2lip", "syncnet", "kokoro")


def load_stacker(spec, embedder):
    from exam_video_ffpp import load_stacker as _load
    return _load(spec, embedder)


def _ffprobe_streams(path: Path) -> dict | None:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,codec_name,width,height,"
                        "r_frame_rate,sample_rate,channels", "-of", "json",
                        str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        streams = json.loads(r.stdout).get("streams", [])
    except json.JSONDecodeError:
        return None
    out = {}
    for s in streams:
        if s.get("codec_type") == "video":
            out["video"] = (s.get("codec_name"), s.get("r_frame_rate"),
                            s.get("width"), s.get("height"))
        elif s.get("codec_type") == "audio":
            out["audio"] = (s.get("codec_name"), s.get("sample_rate"),
                            s.get("channels"))
    return out


def _license_complete(corpus: str) -> tuple[bool, str]:
    p = AV_DIET / "license_receipts.json"
    if not p.exists():
        return False, f"{p} missing"
    try:
        lic = json.loads(p.read_text())
    except json.JSONDecodeError:
        return False, f"{p} is not valid JSON"
    entries = lic.get("assets", lic if isinstance(lic, dict) else {})
    required = list(REQUIRED_LICENSES) + (["avlips"] if corpus == "avlips" else [])
    for name in required:
        e = entries.get(name)
        if not e:
            return False, f"license receipt missing for {name}"
        lid = str(e.get("license", e.get("id", "")))
        if not lid or "UNKNOWN" in lid.upper():
            return False, f"license unknown for {name}"
    return True, "complete"


def enumerate_av_exam(smoke_limit: int | None = None, progress=print):
    """(jobs, n_sources, integrity_pre) — jobs are (path, label, method, key)."""
    splitj = json.loads((AV_DIET / "split.json").read_text())
    genr = json.loads((AV_DIET / "gen_receipt.json").read_text())
    failed = genr.get("failed", {})
    splits = splitj["splits"]
    # D3 straddle: the exam split must be disjoint from train+val BY MANIFEST
    straddle = sorted(set(splits["exam"]) & (set(splits["train"])
                                             | set(splits["val"])))
    jobs = []
    for k in sorted(splits["exam"]):
        if k in failed:
            continue                      # receipted failure: never examined
        f, r = AV_GEN / "fakes" / f"{k}.mp4", AV_GEN / "reals" / f"{k}.mp4"
        if not (f.exists() and r.exists()):
            raise SystemExit(f"exam pair missing for {k} and not receipted "
                             f"as failed — run the generation driver; the "
                             f"exam refuses partial pairs")
        jobs.append((r, 0, "real", k))
        jobs.append((f, 1, "Wav2Lip", k))
    if smoke_limit:
        fakes = [j for j in jobs if j[1] == 1][:smoke_limit]
        reals = [j for j in jobs if j[1] == 0][:max(1, smoke_limit // 2)]
        jobs = reals + fakes
    jobs.sort(key=lambda j: (j[1], j[2], j[0].name))
    n_sources = len({j[3] for j in jobs})
    progress(f"[enum] av exam: {sum(1 for j in jobs if j[1] == 0)} reals / "
             f"{sum(1 for j in jobs if j[1] == 1)} fakes over {n_sources} "
             f"held-out sources ({len(failed)} sources receipted-failed)"
             + (f" — SMOKE limit {smoke_limit}" if smoke_limit else ""))
    return jobs, n_sources, {"straddle_violations": len(straddle),
                             "straddle_keys": straddle[:5]}


def enumerate_avlips(progress=print):
    """External AVLips val set from the acquisition manifest; capped."""
    man_p = AVLIPS_DIR / "manifest.json"
    if not man_p.exists():
        raise SystemExit(f"{man_p} missing — run the AVLips acquisition step "
                         f"first (D2 external exam; fail-closed)")
    man = json.loads(man_p.read_text())
    reals = [Path(p) for p in man.get("real", []) if Path(p).exists()]
    fakes = [Path(p) for p in man.get("fake", []) if Path(p).exists()]
    if len(reals) + len(fakes) > 2 * AVLIPS_CAP:
        cap = AVLIPS_CAP
        reals = sorted(reals, key=lambda p: sha12(p))[:cap]
        fakes = sorted(fakes, key=lambda p: sha12(p))[:cap]
        capped = True
    else:
        capped = False
    jobs = ([(p, 0, "avlips_real", p.name) for p in reals]
            + [(p, 1, "avlips_fake", p.name) for p in fakes])
    jobs.sort(key=lambda j: (j[1], j[0].name))
    progress(f"[enum] avlips: {len(reals)} reals / {len(fakes)} fakes "
             f"(external, never trained)"
             + (f" — CAPPED at {AVLIPS_CAP}/class sha-sorted seed-11"
                if capped else ""))
    return jobs, {"capped": capped}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["av", "avlips"], required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke cap: first N fakes + N//2 reals (corpus=av)")
    ap.add_argument("--drift-cal", default=None,
                    help="drift calibration JSON (defaults to the stacker "
                         "bundle's drift_calibration.json)")
    ap.add_argument("--av-calib", default=None,
                    help="AV coverage calibration JSON (defaults to "
                         "av_calibration.json beside the stacker — the serve "
                         "contract); REQUIRED to exist either way")
    ap.add_argument("--av-cache", default=None,
                    help="per-workdir AV result cache (cached_extract is "
                         "calibration-safe: the cached part is "
                         "calibration-INDEPENDENT; coverage is recomputed at "
                         "use time). Without it every exam run re-featurizes "
                         "from scratch — kill/resume loses hours.")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="arcface",
                    help="identity embedder (must match the trained stacker)")
    ap.add_argument("--video-stacker", default=None,
                    help="video stacker dir/joblib (default: argus_detect."
                         "VIDEO_STACKER when promoted, else none)")
    ap.add_argument("--out", default=None)
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
        base = (Path(stacker_spec) if stacker_spec else VIDEO_STACKER)
        guess = (base if base.is_dir() else base.parent) / "drift_calibration.json"
        cal_path = str(guess) if guess.exists() else None
    spike = (float(json.loads(Path(cal_path).read_text())["spike_thresh"])
             if cal_path and Path(cal_path).exists() else None)

    # AV coverage calibration — the serve contract: it rides the stacker
    # bundle (VIDEO_STACKER.parent). Fail-closed: without it the exam is
    # meaningless (every row out-of-coverage).
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
           f"exam_video_av_{args.corpus}_{stacker_tag or 'nostacker'}_"
           f"{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass "
              f"--force to append/resume", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    integrity = {}
    if args.corpus == "av":
        jobs, n_sources, integrity = enumerate_av_exam(args.limit)
    else:
        jobs, capnote = enumerate_avlips()
        n_sources = len(jobs)
        integrity["capped"] = capnote["capped"]
    sha_of = {}
    for vid, _, method, _k in jobs:
        sha_of[(method, vid.name)] = sha12(vid)
    # D3 dedup: (method, file) must be unique across the exam jobs
    integrity["dedup_clean"] = len(sha_of) == len(jobs)
    # D3 license completeness (hard stop)
    lic_ok, lic_why = _license_complete(args.corpus)
    integrity["license_complete"] = lic_ok
    integrity["license_note"] = lic_why
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
        for i, (vid, label, method, key) in enumerate(jobs):
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
                       "source": key, "route": "image_court",
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
            stacked = (float(stacker.predict_proba(
                row_to_X(flat, names).reshape(1, -1))[0, 1])
                if stacker is not None else None)
            row = {"file": vid.name, "sha256_12": flat["sha256_12"],
                   "label": label, "method": method, "source": key,
                   "route": "video_face", "visual": res["visual"],
                   "drift": res["drift"], "audio": res["audio"],
                   "avs": res["row"] and {k: flat[k] for k in AV_FEATURES},
                   "av_diagnostics": res.get("av_diagnostics"),
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
    for name in ("naive_combined", "identity_video_score", "stacked"):
        s = [r[name] for r in face if r.get(name) is not None]
        yy = [r["label"] for r in face if r.get(name) is not None]
        if len(set(yy)) > 1 and len(s):
            aucs[name] = round(float(roc_auc_score(yy, s)), 4)
    # per-feature AUCs (INFORMATIONAL — never a gate; D-046 bars are sealed)
    per_feature = {}
    for feat in AV_FEATURES:
        s, yy = [], []
        for r in face:
            v = (r.get("avs") or {}).get(feat)
            if v is not None:
                s.append(float(v))
                yy.append(r["label"])
        if len(set(yy)) > 1 and len(s):
            per_feature[feat] = round(float(roc_auc_score(yy, s)), 4)

    # D3 ffprobe parity per matched pair (corpus=av only)
    if args.corpus == "av":
        real_by_key = {r["source"]: r for r in reals}
        parity_violations = []
        for fk in fakes:
            rr = real_by_key.get(fk.get("source"))
            if rr is None:
                continue
            sf = _ffprobe_streams(AV_GEN / "fakes" / fk["file"])
            sr = _ffprobe_streams(AV_GEN / "reals" / rr["file"])
            if sf is None or sr is None:
                parity_violations.append({"source": fk.get("source"),
                                          "reason": "ffprobe_failed"})
            elif sf.get("video") != sr.get("video") or \
                    sf.get("audio") != sr.get("audio"):
                parity_violations.append({"source": fk.get("source"),
                                          "fake": sf, "real": sr})
        integrity["ffprobe_parity_violations"] = len(parity_violations)
        if parity_violations:
            (out / "parity_violations.json").write_text(
                json.dumps(parity_violations, indent=2))
    else:
        integrity["ffprobe_parity_violations"] = 0   # external set: n/a

    in_cov = [1 for r in face if (r.get("avs") or {}).get("avs_in_coverage") == 1]
    in_cov_rate = round(len(in_cov) / max(1, len(face)), 4)
    rates = {
        "no_face_route": round((len(scored) - len(face)) / max(1, len(scored)), 4),
        "in_coverage": in_cov_rate,
        "avs_present": round(sum(1 for r in face
                                 if (r.get("avs") or {}).get("avs_present") == 1)
                             / max(1, len(face)), 4),
        "abstained": round(sum(1 for r in scored
                               if r.get("naive_combined") is None)
                           / max(1, len(scored)), 4),
    }
    metrics = {"corpus": args.corpus,
               "n_scored": len(face), "n_total": len(scored),
               "n_fakes": len(fakes), "n_reals": len(reals),
               "n_sources": n_sources,
               "manifest": str(AV_DIET / "split.json")
               if args.corpus == "av" else str(AVLIPS_DIR / "manifest.json"),
               "auc": aucs, "per_feature_auc": per_feature,
               "integrity": integrity,
               "in_coverage_rate": in_cov_rate,
               "rates": rates, "frames": args.frames,
               "stacker": stacker_tag or "none", "embedder": args.embedder,
               "av_calibration": av_cal_path,
               "wall_min": round((time.time() - t0) / 60, 2)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nAV EXAM [{args.corpus}]: {json.dumps(aucs)} "
          f"(n_scored={len(face)}/{len(scored)}, n_sources={n_sources}, "
          f"{metrics['wall_min']}min)", flush=True)
    print(f"  in_coverage={in_cov_rate} integrity={json.dumps(integrity)}",
          flush=True)
    print(f"  per-feature AUCs (informational): {json.dumps(per_feature)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
