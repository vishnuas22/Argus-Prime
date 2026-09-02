"""Bulk-download FF++ c23 videos (originals + 4 manipulation methods) from TUM server.

Resumable, parallel, retrying. ~5000 files / ~10GB.
Usage: .venv/bin/python deepfake/scripts/ffpp/bulk_download.py [--workers 8]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen, Request

WS = Path(__file__).resolve().parents[3]
RAW = WS / "data" / "raw" / "ffpp"
OUT = RAW / "videos"
BASE = "https://kaldir.vc.in.tum.de/faceforensics/v3/"
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


def build_list():
    pairs = json.load(open(RAW / "filelist.json"))
    fakes = []
    for m in METHODS:
        for a, b in pairs:
            fakes.append((f"{m}/c23/videos/{a}_{b}.mp4",
                          OUT / "manipulated_sequences" / m / "c23" / "videos" / f"{a}_{b}.mp4"))
            fakes.append((f"{m}/c23/videos/{b}_{a}.mp4",
                          OUT / "manipulated_sequences" / m / "c23" / "videos" / f"{b}_{a}.mp4"))
    ids = sorted({x for p in pairs for x in p})
    origs = [(f"original_sequences/youtube/c23/videos/{i}.mp4",
              OUT / "original_sequences" / "youtube" / "c23" / "videos" / f"{i}.mp4")
             for i in ids]
    return origs + fakes


def fetch(rel: Path, dst: Path, tries: int = 4):
    """Returns True on success, 'gone' on definitive 404, False on transient failure."""
    if dst.exists() and dst.stat().st_size > 0:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + str(rel)
    from urllib.error import HTTPError
    for t in range(tries):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=90) as r, open(dst.with_suffix(".part"), "wb") as f:
                while True:
                    chunk = r.read(1 << 18)
                    if not chunk:
                        break
                    f.write(chunk)
            dst.with_suffix(".part").rename(dst)
            return True
        except HTTPError as e:
            if t == tries - 1:
                print(f"FAIL {rel}: {e}", flush=True)
                return False
            time.sleep(3 * (t + 1))
        except Exception as e:
            if t == tries - 1:
                print(f"FAIL {rel}: {e}", flush=True)
                return False
            time.sleep(2 * (t + 1))
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", type=int, default=4,
                    help="retry passes over failures (server is flaky)")
    args = ap.parse_args()
    jobs = build_list()

    todo = [(r, d) for r, d in jobs if not (d.exists() and d.stat().st_size > 0)]
    print(f"{len(jobs)} files total, {len(todo)} to fetch", flush=True)

    canary = BASE + "manipulated_sequences/Face2Face/c23/videos/071_054.mp4"
    def window_open():
        try:
            req = Request(canary, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-200"})
            return urlopen(req, timeout=30).status in (200, 206)
        except Exception:
            return False

    for p in range(1, args.passes + 1):
        if not todo:
            break
        waited = 0
        while not window_open():
            waited += 300
            print(f"pass {p}: server dark, waiting ({waited}s)...", flush=True)
            if waited >= 3600:
                break
            time.sleep(300)
        if not window_open():
            print(f"pass {p}: skipping, still dark after {waited}s", flush=True)
            continue
        t0 = time.time()
        done = [0]

        def work(job):
            ok = fetch(*job)
            done[0] += 1
            if done[0] % 100 == 0:
                rate = done[0] / (time.time() - t0) * 60
                print(f"[pass {p} {done[0]}/{len(todo)}] {rate:.0f} files/min", flush=True)
            return ok

        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(work, todo))
        gone = sum(1 for ok in results if ok == "gone")
        if gone:
            print(f"pass {p}: {gone} permanent-404 recorded", flush=True)
        failed = [job for job, ok in zip(todo, results) if ok is not True]
        print(f"pass {p}: {len(todo)-len(failed)} fetched, {len(failed)} still failing", flush=True)
        todo = failed
        if todo:
            print("cooling down 30s before next pass...", flush=True)
            time.sleep(30)
    print(f"complete: {sum(1 for _, d in jobs if d.exists() and d.stat().st_size > 0)}/{len(jobs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
