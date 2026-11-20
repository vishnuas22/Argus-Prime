"""V1d stage 2b — resumable p6vid_v4 scoring (deployed config, 12 frames).

Appends one JSON line per clip to artifacts/v1d_restrav/scores.jsonl.
Re-running skips clips already present, so an interrupted run resumes in place.

This is the long pole (~9 s/clip at 12 frames). Run it in the background:
    nohup .venv/bin/python deepfake/scripts/v1d_score.py > /tmp/v1d_score.log 2>&1 &
"""
import argparse
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WS / "deepfake"), str(WS / "deepfake" / "bench"),
                str(WS / "deepfake" / "scripts"),
                str(WS / "integrations" / "Argus_Master" / "backend")]

OUT = WS / "artifacts" / "v1d_restrav"
MANIFEST = OUT / "manifest.json"
SCORES = OUT / "scores.jsonl"


def unique_clips(manifest):
    seen, order = set(), []
    for name, d in manifest["domains"].items():
        for cls in ("real", "fake"):
            for p in d.get(cls, []):
                if p not in seen:
                    seen.add(p)
                    order.append(p)
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    from argus_detect import detect_video

    manifest = json.load(open(MANIFEST))
    clips = unique_clips(manifest)
    print(f"unique clips: {len(clips)}")

    done = set()
    if SCORES.exists():
        for line in open(SCORES):
            try:
                done.add(json.loads(line)["path"])
            except Exception:
                continue
        print(f"resuming: {len(done)} already scored")
    todo = [c for c in clips if c not in done]
    print(f"to do: {len(todo)}  frames={args.frames}")

    if not todo:
        print("nothing to do")
        return

    t0 = time.time()
    with open(SCORES, "a") as fh:
        for i, p in enumerate(todo):
            rec = {"path": p}
            try:
                r = detect_video(Path(p), n_frames=args.frames)
                rec["p_fake"] = r.get("p_fake_fused")
                rec["abstained"] = r.get("abstained")
                rec["route"] = r.get("route")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"[:160]
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if (i + 1) % 10 == 0:
                el = time.time() - t0
                rate = (i + 1) / el
                eta = (len(todo) - i - 1) / rate
                print(f"  [{i+1}/{len(todo)}] {el/60:.1f}m elapsed, "
                      f"{rate:.2f} clip/s, ETA {eta/60:.1f}m", flush=True)
    print(f"scores done in {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()