"""Fetch GenVidBench Tier-1 (modern T2V corpus) — spec item 6.1.

Tier-1 ONLY (~5 GB, no new tools). TIER-2 IS DEFERRED to a later phase by
user decision: `--tier 2` prints a deferred notice and exits 0 without
downloading anything.

  .venv/bin/python deepfake/scripts/fetch_genvidbench.py --tier 1

Pipeline: HuggingFace snapshot of jian-0/GenVidBench (allow_patterns
GenVidBench/6.7m/*) -> verify per-file sizes -> extract archives ->
data/raw/genvidbench/{sora,kling,opentora}/<clip>.mp4 -> MANIFEST.json with
{file, bytes, sha256, source_url, license_note} + n_clips per generator.

IDEMPOTENT (parallel-download friendly): archives the user already placed are
skipped (size-verified); already-extracted clip dirs with a matching MANIFEST
are skipped; a re-run only ever fills in what is missing.

Usage:
  .venv/bin/python deepfake/scripts/fetch_genvidbench.py --tier 1 [--force]
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
DEST = WS / "data" / "raw" / "genvidbench"
HF_REPO = "jian-0/GenVidBench"
HF_PATTERN = "GenVidBench/6.7m/*"
LICENSE_NOTE = "see dataset card"

# (archive name, expected bytes (approx), generator subdir, min clip count)
TIER1 = [
    ("OpenAI_Sora.zip", 0.49e9, "sora", 51),
    ("keling.zip", 1.38e9, "kling", 264),
    ("OpenSora_13800.tar.gz", 3.13e9, "opentora", 13800),
]
MIN_FREE_BYTES = 10 * 1024 ** 3


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _clips(outdir: Path) -> int:
    """Usable clips on disk: real .mp4 (any case), junk excluded."""
    if not outdir.exists():
        return 0
    return sum(1 for p in outdir.iterdir()
               if p.is_file() and p.suffix.lower() == ".mp4"
               and not p.name.startswith("._"))


def _extract(archive: Path, outdir: Path) -> int:
    """Extract real .mp4 entries from a zip/tar; returns number of clips.

    Junk is skipped outright: macOS __MACOSX/ AppleDouble `._*` stubs and
    .DS_Store are not videos (keling.zip carries 263 of them — counting them
    would double the manifest's clip count). Names are matched
    case-insensitively because keling.zip also carries 106 `.MP4` files."""
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0

    def _wanted(name: str) -> bool:
        base = Path(name).name
        return (not name.startswith("__MACOSX/") and "/__MACOSX/" not in name
                and not base.startswith("._") and base != ".DS_Store"
                and base.lower().endswith(".mp4"))

    def _write(src_bytes_read, dst: Path) -> None:
        if not dst.exists():
            with open(dst, "wb") as f:
                src_bytes_read(f)

    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                if info.is_dir() or not _wanted(info.filename):
                    continue
                dst = outdir / Path(info.filename).name
                _write(lambda f, i=info: shutil.copyfileobj(z.open(i), f), dst)
                n += 1
    else:
        with tarfile.open(archive, "r:*") as t:
            for m in t.getmembers():
                if not m.isfile() or not _wanted(m.name):
                    continue
                dst = outdir / Path(m.name).name
                src = t.extractfile(m)
                if src is None:
                    continue
                _write(lambda f, s=src: shutil.copyfileobj(s, f), dst)
                n += 1
    return n


def _locate(archive: Path) -> Path:
    """Root path for an archive, relocating HF's repo-path layout.

    snapshot_download(local_dir=DEST) preserves the repo-relative path
    (GenVidBench/6.7m/<archive>), while extraction reads DEST/<archive>.
    Idempotent: if the archive exists anywhere under DEST (skipping .cache),
    move it to the root once, then return it. Never re-downloads."""
    if archive.exists():
        return archive
    for p in DEST.rglob(archive.name):
        if ".cache" in p.parts or p == archive:
            continue
        archive.parent.mkdir(parents=True, exist_ok=True)
        p.replace(archive)
        print(f"relocated {p.relative_to(DEST)} -> {archive.name}")
        return archive
    return archive


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, default=1, choices=[1, 2])
    ap.add_argument("--force", action="store_true",
                    help="re-extract even when the clip dir looks complete")
    args = ap.parse_args()

    if args.tier == 2:
        print("GenVidBench Tier-2 (~120 GB) is DEFERRED to a later phase "
              "(user decision) — nothing downloaded.")
        return 0

    free = shutil.disk_usage(DEST if DEST.exists() else WS).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"FAIL: only {free/1e9:.1f} GB free, need > 10 GB "
                         f"for Tier-1 extraction")
    DEST.mkdir(parents=True, exist_ok=True)
    man_p = DEST / "MANIFEST.json"

    # 1) source archives (download only when absent — user may place them)
    missing = []
    for name, approx, _gen, _min in TIER1:
        p = _locate(DEST / name)
        if p.exists() and p.stat().st_size > 0.9 * approx:
            print(f"skip download: {name} present ({p.stat().st_size/1e9:.2f} GB)")
        else:
            missing.append(name)
    if missing:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise SystemExit("FAIL: huggingface_hub missing and archives not "
                             f"present: {e}")
        print(f"downloading {len(missing)} archive(s) from {HF_REPO} ...")
        snapshot_download(repo_id=HF_REPO, repo_type="dataset",
                          allow_patterns=[HF_PATTERN], local_dir=str(DEST))

    # 2) verify + extract
    manifest, total_bytes, total_clips = [], 0, 0
    for name, approx, gen, min_clips in TIER1:
        p = _locate(DEST / name)
        if not p.exists():
            print(f"MISSING after download: {name}", file=sys.stderr)
            continue
        nbytes = p.stat().st_size
        if nbytes < 0.5 * approx:
            print(f"FAIL: {name} is {nbytes/1e9:.2f} GB, expected ~{approx/1e9:.2f} GB",
                  file=sys.stderr)
            continue
        outdir = DEST / gen
        # idempotent junk sweep: AppleDouble `._*` / .DS_Store from the archive
        if outdir.exists():
            junk = [q for q in outdir.iterdir() if q.is_file()
                    and (q.name.startswith("._") or q.name == ".DS_Store")]
            for q in junk:
                q.unlink()
            if junk:
                print(f"removed {len(junk)} junk file(s) from {outdir}")
        if args.force or _clips(outdir) == 0:
            _extract(p, outdir)
        # recount from disk — an entry-processed counter would count junk
        n = _clips(outdir)
        print(f"{gen}: {n} clips -> {outdir}")
        if n < min_clips:
            print(f"WARN: {gen} has {n} clips (<{min_clips} expected)", file=sys.stderr)
        total_bytes += nbytes
        total_clips += n
        manifest.append({"file": name, "bytes": nbytes,
                         "sha256": sha256_of(p),
                         "source_url": f"https://huggingface.co/datasets/{HF_REPO}",
                         "license_note": LICENSE_NOTE,
                         "generator": gen, "n_clips": n})

    if total_clips == 0:
        # fail closed: never leave a 0-clip MANIFEST behind (a failed run must
        # not look like a successful acquisition)
        if man_p.exists() and json.loads(man_p.read_text()).get("total_clips") == 0:
            man_p.unlink()
        print("FAIL: 0 clips extracted — no MANIFEST written", file=sys.stderr)
        return 1
    man = {"tier": 1, "repo": HF_REPO, "pattern": HF_PATTERN,
           "files": manifest, "total_bytes": total_bytes,
           "total_clips": total_clips,
           "generators": {m["generator"]: m["n_clips"] for m in manifest},
           "note": "exam-only corpus (never trains, per user scope); "
                   "Tier-2 deferred (user decision); acquisition route = "
                   "D-041 (HF snapshot, user-placed archives honoured)"}
    man_p.write_text(json.dumps(man, indent=2))
    print(f"MANIFEST -> {man_p} (total {total_bytes/1e9:.2f} GB, "
          f"{total_clips} clips: {man['generators']})")
    if total_bytes > 6e9:
        print("WARN: total exceeds the ~6 GB Tier-1 expectation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
