"""W4: download the FF++ ACTOR ORIGINALS needed by the DFD diet (D-066).

The mirror's 1000 DeepFakeDetection fakes cover 318 (target,sequence) sources
(data/raw/ffpp_dfd/sources.json); their actor-original videos are NOT on disk
anywhere (the HF mirror zip ships fakes + youtube reals only). This script
downloads exactly those 318 originals from the official kaldir v3 server
(same TOS the repo already operates under — ffpp_download.py is the
user-approved copy) into data/raw/ffpp_actors/c23/videos/, reusing
bulk_download's fetch machinery (size>0 resume, retry passes, canary window).

ALL THREE splits' originals are downloaded: train/val feed the dumps, exam
feeds exam_video_dfd's real side (domain-matched reals of the held-out
sources). There is NO fakes-only fallback (L-CHANNEL-CONFOUND law): if the
server 404s at scale or the window never opens, this exits non-zero and the
run STOPS.

--probe writes the mandatory ffprobe audio receipt (>=10 mirror DFD fakes +
>=10 downloaded originals): ANY audio stream => exit 1 (ESCALATE before any
dump — the audio_present=0 missingness assumption would be false).

Usage:
  .venv/bin/python deepfake/scripts/ffpp/download_actors.py [--workers 8] [--passes 4]
  .venv/bin/python deepfake/scripts/ffpp/download_actors.py --probe
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen

WS = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bulk_download import BASE, fetch  # noqa: E402

SOURCES = WS / "data" / "raw" / "ffpp_dfd" / "sources.json"
OUT = WS / "data" / "raw" / "ffpp_actors" / "c23" / "videos"
MIRROR_DFD = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / \
    "fake" / "DeepFakeDetection"
CANARY_REL = "original_sequences/actors/c23/videos/01__meeting_serious.mp4"


def build_jobs() -> list:
    sources = json.loads(SOURCES.read_text())
    jobs = []
    for key in sorted(sources):
        name = sources[key]["original"]          # '<target>__<sequence>.mp4'
        rel = f"original_sequences/actors/c23/videos/{name}"
        jobs.append((rel, OUT / name))
    return jobs


def window_open() -> bool:
    try:
        req = Request(BASE + CANARY_REL,
                      headers={"User-Agent": "Mozilla/5.0",
                               "Range": "bytes=0-200"})
        return urlopen(req, timeout=30).status in (200, 206)
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--probe", action="store_true",
                    help="ffprobe audio receipt only (no download)")
    args = ap.parse_args()

    if args.probe:
        return probe_audio()

    jobs = build_jobs()
    todo = [(r, d) for r, d in jobs if not (d.exists() and d.stat().st_size > 0)]
    print(f"{len(jobs)} actor originals needed, {len(todo)} to fetch",
          flush=True)

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
            if done[0] % 25 == 0:
                rate = done[0] / (time.time() - t0) * 60
                print(f"[pass {p} {done[0]}/{len(todo)}] {rate:.1f} files/min",
                      flush=True)
            return ok

        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(work, todo))
        failed = [job for job, ok in zip(todo, results) if ok is not True]
        print(f"pass {p}: {len(todo) - len(failed)} fetched, "
              f"{len(failed)} still failing", flush=True)
        todo = failed
        if todo:
            print("cooling down 30s before next pass...", flush=True)
            time.sleep(30)

    present = [d for _, d in jobs if d.exists() and d.stat().st_size > 0]
    missing = [(r, d) for r, d in jobs
               if not (d.exists() and d.stat().st_size > 0)]
    print(f"complete: {len(present)}/{len(jobs)} actor originals on disk "
          f"(size>0)", flush=True)
    if missing:
        print(f"STOP — {len(missing)} originals NOT fetchable after "
              f"{args.passes} passes (e.g. {missing[:3]}). There is NO "
              f"fakes-only fallback (L-CHANNEL-CONFOUND): escalate, do not "
              f"dump.", flush=True)
        return 1

    manifest = {"base": BASE, "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "n_expected": len(jobs), "n_present": len(present),
                "sha256_12": {}}
    h = hashlib.sha256()
    for _, d in sorted(jobs, key=lambda j: j[1].name):
        with open(d, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        manifest["sha256_12"][d.name] = h.hexdigest()[:12]
        h = hashlib.sha256()
    (WS / "data" / "raw" / "ffpp_actors" / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True))
    print(f"sha256_12 manifest: data/raw/ffpp_actors/manifest.json "
          f"({len(manifest['sha256_12'])} entries)", flush=True)
    return 0


def probe_audio() -> int:
    """ffprobe receipt: >=10 mirror DFD fakes + >=10 downloaded originals."""
    def has_audio(path: Path) -> bool:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True)
        return bool(r.stdout.strip())

    fakes = sorted(MIRROR_DFD.glob("*.mp4"))[:10]
    fakes.append(MIRROR_DFD / "01_02__outside_talking_still_laughing__YVGY8LOK.mp4")
    origs = sorted(OUT.glob("*.mp4"))[:10]
    receipt = {"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "df_fakes": {}, "actor_originals": {}}
    for p in fakes:
        receipt["df_fakes"][p.name] = int(has_audio(p))
    for p in origs:
        receipt["actor_originals"][p.name] = int(has_audio(p))
    n_fake = len(receipt["df_fakes"])
    n_orig = len(receipt["actor_originals"])
    n_audio = sum(receipt["df_fakes"].values()) + sum(receipt["actor_originals"].values())
    receipt["n_probed"] = {"df_fakes": n_fake, "actor_originals": n_orig,
                           "with_audio_stream": n_audio}
    out = WS / "data" / "raw" / "ffpp_actors" / "audio_receipt.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"ffprobe receipt: {n_fake} DFD fakes + {n_orig} actor originals "
          f"probed, {n_audio} with an audio stream -> {out}", flush=True)
    if n_fake < 10 or n_orig < 10:
        print("STOP — receipt quota not met (need >=10 + >=10).", flush=True)
        return 1
    if n_audio:
        print(f"ESCALATE — {n_audio} probed videos CARRY AN AUDIO STREAM: "
              f"the audio_present=0 missingness assumption is FALSE for this "
              f"corpus. Do not dump; escalate per spec §2.", flush=True)
        return 1
    print("zero audio streams — audio_present=0 missingness assumption holds",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
