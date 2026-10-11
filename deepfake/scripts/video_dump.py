"""Dump video feature rows for the video stacker (spec item 3.2).

Enumerates videos from data/raw/ffpp/{train,val}.json with the IDENTITY-
DISJOINT split preserved (720/140/140 ids): reals = real/*.mp4 whose stem is
in the split-id set; fakes = fake/<Method>/<a>_<b>.mp4 whose PAIR tuple is in
the split pairs (both directions; pair semantics exactly as exam_video_ffpp).

Usage:
  .venv/bin/python deepfake/scripts/video_dump.py --split train|val \
      [--frames 12] [--limit 1500] [--audio-limit 1200] [--no-audio] \
      [--out artifacts/restack_video] [--drift-cal PATH]

Selection is deterministic: sha-sorted within each class, reals first up to
limit//2, then fakes (so the drift calibration always sees the full real
quota). Append-resumable via the done-set of file names (exam guard pattern).

Audio: only the first --audio-limit rows in sha-sorted order run the XLS-R
probe; every other row gets audio_present=0 (missingness is an explicit
feature, D-035-consistent — no silent train/serve skew).

drift_l2 (per-video pooled embedding steps) is stored so drift_spikes can be
backfilled exactly once the spike threshold is known: either --drift-cal
(gate/test dumps use the trained calibration) or, for the train dump, the
train-real p95 of per-video drift (sidecar drift_spike.json) — the same
definition train_video_stacker uses to fit the calibration.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_channels import drift_features  # noqa: E402
from video_features import FEATURE_NAMES, compute_row  # noqa: E402

MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
SPLITS = WS / "data" / "raw" / "ffpp"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
CSV_COLS = ["file", "sha256_12", "label", "method", "split", "route",
            "naive_combined", "identity_video_score", "drift_l2"] + FEATURE_NAMES


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def enumerate_split(split: str, limit: int, progress=print):
    """Deterministic video selection for one split: (reals, fakes) Path lists."""
    pairs = json.load(open(SPLITS / f"{split}.json"))
    pair_set = {tuple(p) for p in pairs}
    ids = {i for p in pairs for i in p}
    reals = sorted(v for v in (MIRROR / "real").glob("*.mp4") if v.stem in ids)
    fakes = []
    for m in METHODS:
        got = [v for v in sorted((MIRROR / "fake" / m).glob("*.mp4"))
               if tuple(v.stem.split("_")) in pair_set
               or tuple(reversed(v.stem.split("_"))) in pair_set]
        fakes += got
    progress(f"[enum] {split}: {len(reals)} reals / {len(fakes)} fakes available")
    t0 = time.time()
    reals = sorted(((sha12(v), v) for v in reals), key=lambda t: t[0])
    fakes = sorted(((sha12(v), v) for v in fakes), key=lambda t: t[0])
    progress(f"[enum] hashed {len(reals)+len(fakes)} files in "
             f"{time.time()-t0:.0f}s")
    n_real = min(len(reals), limit // 2)
    n_fake = max(0, limit - n_real)
    sel_reals = [v for _, v in reals[:n_real]]
    # fakes balanced across methods, sha-sorted within method
    per = max(1, n_fake // len(METHODS))
    picked = []
    for m in METHODS:
        msel = [v for s, v in fakes if v.parent.name == m][:per]
        picked += msel
    if len(picked) < n_fake:
        have = {v.name for v in picked}
        picked += [v for s, v in fakes if v.name not in have][: n_fake - len(picked)]
    return sel_reals, picked[:n_fake]


def featurize_video(path: Path, frames: int = 12, with_audio: bool = False,
                    spike_thresh: float | None = None):
    """Score one video through visual+drift(+audio) -> flat stacker row.

    Returns None when zero faces are detected (the no-face route; these rows
    never train the stacker — the serve path routes them to the image court)."""
    from video_score import audio_channel, combined, visual_channel_bundle
    vis, tracks, _per_track, _obs = visual_channel_bundle(path, frames)
    if vis is None:
        return None
    d = drift_features(tracks, spike_thresh=spike_thresh)
    l2 = _pooled_l2(tracks)
    aud = audio_channel(path) if with_audio else None
    row = {"file": path.name, "sha256_12": sha12(path), "route": "video_face",
           **compute_row(vis, d, aud)}
    row["naive_combined"] = combined(vis, aud)
    row["identity_video_score"] = vis.get("video_score")
    return {"row": row, "visual": vis, "drift": d, "audio": aud,
            "l2": [round(float(x), 6) for x in l2],
            "naive_combined": row["naive_combined"]}


def _pooled_l2(tracks) -> list:
    """Pooled consecutive-embedding steps of the drift-capable tracks."""
    from video_channels import MIN_TRACK_OBS, track_l2
    pools = [track_l2(t) for t in tracks if t.n >= MIN_TRACK_OBS]
    pools = [p for p in pools if len(p)]
    return np.concatenate(pools).tolist() if pools else []


def _spike_from_rows(rows) -> float | None:
    vals = [float(r["drift_p95_max"]) for r in rows
            if int(r["label"]) == 0 and int(r["n_drift_tracks"]) > 0]
    if len(vals) < 2:
        return None
    return float(np.quantile(np.asarray(vals, dtype=np.float64), 0.95))


def _apply_spikes(rows, spike):
    for r in rows:
        l2 = [float(x) for x in (r.get("drift_l2") or "").split()] if \
            isinstance(r.get("drift_l2"), str) else [float(x) for x in (r.get("drift_l2") or [])]
        r["drift_spikes"] = int(sum(1 for x in l2 if x > spike)) if spike else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--audio-limit", type=int, default=1200)
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--out", default=str(WS / "artifacts" / "restack_video"))
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="sface",
                    help="identity embedder (must match at serve time — "
                         "stacker_meta.embedder records the choice)")
    ap.add_argument("--drift-cal", default=None,
                    help="calibration JSON whose spike_thresh fixes drift_spikes "
                         "(gate/test dumps); train dump fits its own from reals")
    args = ap.parse_args()
    from video_face_track import set_embedder
    set_embedder(args.embedder)
    limit = args.limit or (1500 if args.split == "train" else
                           300 if args.split == "val" else 700)
    out = Path(args.out) / f"dump_{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "scores.csv"
    noface_path = out / "noface.jsonl"

    reals, fakes = enumerate_split(args.split, limit)
    jobs = [(v, 0, "real") for v in reals] + \
           [(v, 1, v.parent.name) for v in fakes]
    jobs = sorted(jobs, key=lambda j: (j[1] != 0, j[0].parent.name, j[0].name))
    # keys are (method, filename): FF++ method dirs REUSE filenames across
    # methods (220 of 780 train-fake names collide), so a bare-filename key
    # silently drops every later method's copy.
    sha_of = {(v.parent.name, v.name): sha12(v) for v, _, _ in jobs}
    audio_set = set(sorted(((v.parent.name, v.name) for v, _, _ in jobs),
                           key=lambda k: sha_of[k])[: 0 if args.no_audio
                                                     else args.audio_limit])

    spike = None
    if args.drift_cal:
        spike = float(json.loads(Path(args.drift_cal).read_text())["spike_thresh"])

    done, rows = set(), []
    if rows_path.exists():
        with open(rows_path) as f:
            for r in csv.DictReader(f):
                done.add((r.get("method", ""), r["file"]))
                rows.append(r)
        print(f"[dump:{args.split}] resume: {len(done)} rows already present",
              flush=True)

    t0 = time.time()
    with open(rows_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        if not rows_path.exists() or rows_path.stat().st_size == 0:
            w.writeheader()
        for i, (vid, label, method) in enumerate(jobs):
            if (method, vid.name) in done:
                continue
            res = featurize_video(vid, args.frames,
                                  with_audio=((method, vid.name) in audio_set),
                                  spike_thresh=spike)
            if res is None:
                with open(noface_path, "a") as nf:
                    nf.write(json.dumps({"file": vid.name, "label": label,
                                         "method": method,
                                         "reason": "no_face"}) + "\n")
                done.add((method, vid.name))
                continue
            r = res["row"]
            r.update(label=label, method=method, split=args.split,
                     drift_l2=" ".join(str(x) for x in res["l2"]))
            if spike:
                _apply_spikes([r], spike)
            else:
                r["drift_spikes"] = 0
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
            f.flush()
            rows.append(r)
            done.add((method, vid.name))
            print(f"[{i+1}/{len(jobs)}] {method}/{vid.name} "
                  f"vid={r['vid_video_score']} drift_p95={r['drift_p95_max']} "
                  f"audio={r['audio_present']} ({(time.time()-t0)/60:.1f}min)",
                  flush=True)

    if spike is None:
        spike = _spike_from_rows(rows)
        if spike is None:
            raise SystemExit("drift spike threshold undetermined (need >=2 usable "
                             "train reals) — check the dump")
        with open(rows_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            w.writeheader()
            for r in rows:
                _apply_spikes([r], spike)
                w.writerow({k: r.get(k, "") for k in CSV_COLS})
    from video_face_track import EMBEDDER_NAME
    (out / "drift_spike.json").write_text(json.dumps(
        {"spike_thresh": round(float(spike), 6),
         "fit_on": f"ffpp {args.split}-split real videos only",
         "embedder": EMBEDDER_NAME}, indent=2))
    n_audio = sum(int(r.get("audio_present", 0) or 0) for r in rows)
    print(f"[dump:{args.split}] done: {len(rows)} rows "
          f"({sum(1 for r in rows if int(r['label'])==0)} real / "
          f"{sum(1 for r in rows if int(r['label'])==1)} fake), "
          f"audio rows={n_audio}, spike_thresh={spike:.6f}, "
          f"{(time.time()-t0)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
