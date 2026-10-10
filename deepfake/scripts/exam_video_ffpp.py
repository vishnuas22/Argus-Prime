"""FF++ video exam — the video branch's first second-dataset validation (D-022).

Samples held-out FF++ test-split videos (identity-disjoint from any training),
runs the three-channel video scorer (visual face head + temporal + XLS-R audio),
reports video-level AUCs.

Usage: .venv/bin/python deepfake/scripts/exam_video_ffpp.py \
           [--n-real 30] [--per-method 8] --out <dir>
Default --out is stamped: artifacts/exam_video_ffpp_<stacker>_<YYYYMMDD>.
A non-empty out dir is refused unless --force (which resumes/appends).
"""
import argparse
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
from video_score import visual_channel, audio_channel, combined  # noqa: E402

MIRROR = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
SPLITS = WS / "data" / "raw" / "ffpp"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-real", type=int, default=30)
    ap.add_argument("--per-method", type=int, default=8)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--out", default=None,
                    help="output dir (default artifacts/exam_video_ffpp_<stacker>_<YYYYMMDD>)")
    ap.add_argument("--force", action="store_true",
                    help="run in an output dir that already contains results "
                         "(existing scores.jsonl is kept and appended — resume)")
    args = ap.parse_args()
    from argus_detect import STACKER
    out = (Path(args.out) if args.out else WS / "artifacts" /
           f"exam_video_ffpp_{STACKER.parent.name}_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force to append/resume",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    test_pairs = json.load(open(SPLITS / "test.json"))
    jobs = []  # (path, label)
    for vid in sorted((MIRROR / "real").glob("*.mp4"))[: args.n_real]:
        if vid.stem in {i for p in test_pairs for i in p}:
            jobs.append((vid, 0))
    rng = __import__("numpy").random.default_rng(11)
    for m in METHODS:
        fakes = [v for v in sorted((MIRROR / "fake" / m).glob("*.mp4"))
                 if tuple(v.stem.split("_")) in {tuple(p) for p in test_pairs}
                 or tuple(reversed(v.stem.split("_"))) in {tuple(p) for p in test_pairs}]
        for v in rng.permutation(fakes)[: args.per_method]:
            jobs.append((v, 1))
    print(f"{len(jobs)} videos ({sum(1 for _, l in jobs if l == 0)} real / "
          f"{sum(1 for _, l in jobs if l == 1)} fake)", flush=True)

    rows_path = out / "scores.jsonl"
    done_keys = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            if line.strip():
                done_keys.add(json.loads(line)["file"])
    y_true, y_comb, y_vis = [], [], []
    with open(rows_path, "a") as f:
        for i, (vid, label) in enumerate(jobs):
            if vid.name in done_keys:
                continue
            vis = visual_channel(vid, args.frames)
            aud = audio_channel(vid)
            comb = combined(vis, aud) if vis else None
            row = {"file": vid.name, "label": label, "method": vid.parent.name,
                   "visual": vis, "audio": {k: v for k, v in (aud or {}).items()
                                            if k != "per_segment"},
                   "combined": comb}
            f.write(json.dumps(row) + "\n")
            f.flush()
            if comb is not None and vis:
                y_true.append(label)
                y_comb.append(comb)
                y_vis.append(vis["mean"])
            print(f"[{i+1}/{len(jobs)}] {vid.parent.name}/{vid.name} "
                  f"comb={comb} vis={vis and vis['mean']}", flush=True)

    from sklearn.metrics import roc_auc_score
    res = {}
    for name, scores in (("combined", y_comb), ("visual_mean", y_vis)):
        if len(set(y_true)) > 1 and len(scores):
            res[name] = round(float(roc_auc_score(y_true, scores)), 4)
    (out / "metrics.json").write_text(json.dumps(
        {"n_scored": len(y_true), **res}, indent=2))
    print(f"\nVIDEO EXAM: {res} (n={len(y_true)})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
