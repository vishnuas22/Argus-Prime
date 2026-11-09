"""Fetch GenVideo-Val (T2V exam corpus) — Phase 0/D3 spec Step 6.

  .venv/bin/python deepfake/scripts/fetch_genvideo_val.py            # fetch + verify + extract
  .venv/bin/python deepfake/scripts/fetch_genvideo_val.py --check    # archive sha + counts + free only
  .venv/bin/python deepfake/scripts/fetch_genvideo_val.py --extract  # archive already verified

IDEMPOTENT: resumes `GenVideo-Val.zip.part` via HTTP Range, never deletes user
data, skips macOS junk (`__MACOSX/`, `._*`, `.DS_Store`) and recounts usable
clips from disk (case-insensitive `.mp4`). License CC-BY-NC-4.0, EXAM-ONLY.
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
DEST = WS / "data" / "raw" / "genvideo_val"
ARCHIVE_NAME = "GenVideo-Val.zip"
URL = ("https://modelscope.cn/datasets/cccnju/Gen-Video/resolve/master/"
       "GenVideo-Val.zip")
EXPECTED_BYTES = 13_934_051_816
EXPECTED_SHA256 = ("aba02ecd57528257249dd38b72b7a34c9d56bc75d4f2541a721b2778f1d864a6")
LICENSE = "CC-BY-NC-4.0"
ROLE = "EXAM-ONLY"

REAL_SUBDIR = "real_msrvtt"
GENERATORS = ("Sora", "WildScrape", "Crafter", "Gen2", "HotShot", "Lavie",
              "ModelScope", "MoonValley", "MorphStudio", "Show_1")
# The published GenVideo-Val archive stores reals directly under
# ``GenVideo-Val/Real/msrvtt_*.mp4`` (no ``video/real_MSRVTT`` segment).
# Match the directory segment, not a fixed nested path, so either release
# layout resolves to the same normalized destination.
REAL_MARKER = "real/"
MIN_FREE_BEFORE_DOWNLOAD = 55e9
MIN_FREE_AFTER_EXTRACT = 25e9
MIN_REALS = 600
MIN_FAKES = 300
MIN_FAKE_DIRS = 3
CHUNK = 8 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _clips(outdir: Path) -> int:
    """Usable clips on disk: real .mp4 (any case), macOS junk excluded."""
    if not outdir.exists():
        return 0
    return sum(1 for p in outdir.iterdir()
               if p.is_file() and p.suffix.lower() == ".mp4"
               and not p.name.startswith("._") and p.name != ".DS_Store")


def _junk(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return (name.startswith("__MACOSX/") or "/__MACOSX/" in name
            or base.startswith("._") or base == ".DS_Store")


def _safe_rel(name: str) -> Path:
    rel = Path(name)
    if rel.is_absolute() or ".." in rel.parts:
        raise RuntimeError(f"unsafe zip member path: {name!r}")
    return rel


def _zip_names(archive: Path) -> list:
    with zipfile.ZipFile(archive) as z:
        return [i.filename for i in z.infolist() if not i.is_dir()]


def _prefix_counts(names: list, target: str, suffix: str = "") -> dict:
    counts: dict = {}
    for name in names:
        if suffix and not name.lower().endswith(suffix):
            continue
        parts = name.split("/")
        for i, seg in enumerate(parts[:-1]):
            if seg.lower() == target.lower():
                pref = "/".join(parts[:i + 1]) + "/"
                counts[pref] = counts.get(pref, 0) + 1
    return counts


def inspect_layout(archive: Path) -> dict:
    """Read the central directory and resolve the real/fake prefixes.

    Returns {"real_prefix": str|None, "real_candidates": [...],
             "gens": {name: prefix}}; never extracts and never guesses a
    missing real prefix (caller stops with exit 5).
    """
    names = [n for n in _zip_names(archive) if not _junk(n)]
    real_prefixes = set()
    for name in names:
        low = name.lower()
        idx = low.find(REAL_MARKER)
        if idx >= 0:
            real_prefixes.add(name[:idx + len(REAL_MARKER)])
    real_candidates = sorted(real_prefixes)
    if not real_candidates:
        candidates = set()
        for name in names:
            parts = name.split("/")[:-1]
            for i, seg in enumerate(parts):
                if "real" in seg.lower():
                    candidates.add("/".join(parts[:i + 1]) + "/")
        if not candidates:
            candidates = {name.split("/")[0] + "/" for name in names
                          if "/" in name}
        real_candidates = sorted(candidates)[:20]

    gens = {}
    for gen in GENERATORS:
        counts = _prefix_counts(names, gen, ".mp4")
        counts = {p: n for p, n in counts.items() if n > 0}
        if not counts:
            continue
        best = sorted(counts, key=lambda p: (-counts[p], p))[0]
        if len(counts) > 1:
            print(f"note: {gen}: {len(counts)} prefix candidates; using {best}",
                  flush=True)
        gens[gen] = best

    return {"real_prefix": real_candidates[0] if len(real_prefixes) == 1 else None,
            "real_prefixes": sorted(real_prefixes),
            "real_candidates": real_candidates,
            "gens": gens}


def extract_selected(archive: Path, layout: dict, staging: Path,
                     dest: Path = None) -> tuple:
    """Extract only real_MSRVTT + the selected generator dirs, junk skipped."""
    selected = []
    if layout["real_prefix"]:
        selected.append((layout["real_prefix"], REAL_SUBDIR))
    for gen, pref in layout["gens"].items():
        selected.append((pref, gen.lower()))
    selected.sort(key=lambda t: -len(t[0]))

    staging.mkdir(parents=True, exist_ok=True)
    n = 0
    nbytes = 0
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if _junk(name) or not name.lower().endswith(".mp4"):
                continue
            match = next((s for s in selected if name.startswith(s[0])), None)
            if match is None:
                continue
            if dest is not None:
                final = dest / match[1] / Path(name).name
                if final.exists() and final.stat().st_size == info.file_size:
                    continue
            dst = staging / _safe_rel(name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if dst.stat().st_size == info.file_size:
                    continue
                raise RuntimeError(f"staging collision with differing file: {dst}")
            with z.open(info) as src, open(dst, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            n += 1
            nbytes += info.file_size

            if n % 100 == 0:
                print(f"  extracted {n} clips ({nbytes / 1e9:.2f} GB)", flush=True)
    return n, nbytes


def normalize(layout: dict, staging: Path, dest: Path) -> int:
    """Move staged files to DEST/real_msrvtt and DEST/<gen_lower>; no overwrite."""
    plan = []
    seen = {}
    dirs = []
    if layout["real_prefix"]:
        dirs.append((layout["real_prefix"], REAL_SUBDIR))
    for gen, pref in layout["gens"].items():
        dirs.append((pref, gen.lower()))

    for prefix, target_sub in dirs:
        base = staging / Path(prefix)
        if not base.exists():
            continue
        for src in sorted(base.rglob("*")):
            if not src.is_file() or src.suffix.lower() != ".mp4":
                continue
            if _junk(src.name) or src.name.startswith("._"):
                continue
            target = dest / target_sub / src.name
            if target in seen and seen[target] != src:
                raise RuntimeError(f"target collision: {seen[target]} and {src} "
                                   f"both -> {target}")
            seen[target] = src
            plan.append((src, target))

    moved = 0
    for src, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != src.stat().st_size:
                raise RuntimeError(f"refusing to overwrite differing file: {target}")
            continue
        src.replace(target)
        moved += 1
    return moved


def recount(dest: Path) -> dict:
    counts = {REAL_SUBDIR: _clips(dest / REAL_SUBDIR)}
    for gen in GENERATORS:
        counts[gen.lower()] = _clips(dest / gen.lower())
    fake_dirs = [g for g in GENERATORS if counts[g.lower()] > 0]
    counts["_total_fakes"] = sum(counts[g.lower()] for g in GENERATORS)
    counts["_fake_dirs"] = len(fake_dirs)
    return counts


def _gates(counts: dict) -> tuple:
    reasons = []
    if counts[REAL_SUBDIR] < MIN_REALS:
        reasons.append(f"reals {counts[REAL_SUBDIR]} < {MIN_REALS}")
    if counts["_total_fakes"] < MIN_FAKES:
        reasons.append(f"fakes {counts['_total_fakes']} < {MIN_FAKES}")
    if counts["_fake_dirs"] < MIN_FAKE_DIRS:
        reasons.append(f"fake dirs {counts['_fake_dirs']} < {MIN_FAKE_DIRS}")
    return (not reasons), reasons


def _print_counts(dest: Path, counts: dict) -> None:
    real = counts[REAL_SUBDIR]
    print(f"real_msrvtt: {real} clips")
    for gen in GENERATORS:
        n = counts[gen.lower()]
        print(f"{gen.lower()}: {n} clips" + ("" if n else " (missing)"))
    print(f"fakes total: {counts['_total_fakes']} clips")
    print(f"usable fake dirs: {counts['_fake_dirs']}")
    print(f"real_msrvtt >= {MIN_REALS}: "
          f"{'PASS' if real >= MIN_REALS else 'FAIL'}")
    print(f"fakes >= {MIN_FAKES}: "
          f"{'PASS' if counts['_total_fakes'] >= MIN_FAKES else 'FAIL'}")
    print(f"fake dirs >= {MIN_FAKE_DIRS}: "
          f"{'PASS' if counts['_fake_dirs'] >= MIN_FAKE_DIRS else 'FAIL'}")


def _bad_path(path: Path, sha: str) -> Path:
    bad = path.with_name(path.name + f".bad-{sha[:8]}")
    i = 2
    while bad.exists():
        bad = path.with_name(path.name + f".bad-{sha[:8]}-{i}")
        i += 1
    return bad


def _finalize_part(part: Path, archive: Path) -> int:
    nbytes = part.stat().st_size
    if nbytes != EXPECTED_BYTES:
        sha = sha256_of(part)
        bad = _bad_path(part, sha)
        part.replace(bad)
        print(f"FAIL: partial is {nbytes} bytes, expected {EXPECTED_BYTES}", file=sys.stderr)
        print(f"  expected sha256 {EXPECTED_SHA256}", file=sys.stderr)
        print(f"  actual   sha256 {sha}", file=sys.stderr)
        print(f"  kept as {bad}", file=sys.stderr)
        return 4
    sha = sha256_of(part)
    if sha != EXPECTED_SHA256:
        bad = _bad_path(part, sha)
        part.replace(bad)
        print(f"FAIL: sha256 mismatch\n  expected {EXPECTED_SHA256}\n"
              f"  actual   {sha}\n  kept as {bad}", file=sys.stderr)
        return 4
    part.replace(archive)
    print(f"ARCHIVE OK: {archive} ({nbytes} bytes, sha256 {sha})")
    return 0


def download(dest: Path, archive: Path, url: str = URL) -> int:
    try:
        import requests
    except ImportError as e:
        print(f"FAIL: {e} (requests is required for downloading)", file=sys.stderr)
        return 1

    free = _free_bytes(dest)
    print(f"free before download: {free / 1e9:.1f} GB")
    if free < MIN_FREE_BEFORE_DOWNLOAD:
        print(f"FAIL: only {free / 1e9:.1f} GB free, need >= "
              f"{MIN_FREE_BEFORE_DOWNLOAD / 1e9:.0f} GB", file=sys.stderr)
        return 2
    dest.mkdir(parents=True, exist_ok=True)
    part = archive.with_name(archive.name + ".part")
    if part.exists() and part.stat().st_size > EXPECTED_BYTES:
        sha = sha256_of(part)
        bad = _bad_path(part, sha)
        part.replace(bad)
        print(f"FAIL: {part.name} is larger than the expected archive\n"
              f"  expected sha256 {EXPECTED_SHA256}\n"
              f"  actual   sha256 {sha}\n  kept as {bad}", file=sys.stderr)
        return 4
    offset = part.stat().st_size if part.exists() else 0
    if offset == EXPECTED_BYTES:
        print(f"complete .part found ({offset} bytes); verifying")
        return _finalize_part(part, archive)

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    print(f"downloading {url}" + (f" (resume at {offset})" if offset else ""),
          flush=True)
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=(30, 300))
    except requests.RequestException as e:
        print(f"FAIL: download request failed: {e}", file=sys.stderr)
        return 1

    with r:
        if offset and r.status_code == 200:
            print("server ignored Range (HTTP 200) — truncating and restarting",
                  flush=True)
            offset = 0
            open(part, "wb").close()
        elif offset and r.status_code == 416:
            if part.stat().st_size == EXPECTED_BYTES:
                return _finalize_part(part, archive)
            print("FAIL: HTTP 416 with an incomplete .part", file=sys.stderr)
            return 1
        elif r.status_code not in (200, 206):
            print(f"FAIL: HTTP {r.status_code} from {url}", file=sys.stderr)
            return 1
        if r.status_code == 206:
            crange = r.headers.get("Content-Range", "")
            if not crange.startswith(f"bytes {offset}-"):
                print(f"FAIL: unexpected Content-Range {crange!r}", file=sys.stderr)
                return 1

        mode = "ab" if offset else "wb"
        written = offset
        last = 0.0
        t0 = time.time()
        try:
            with open(part, mode) as f:
                for chunk in r.iter_content(CHUNK):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    now = time.time()
                    if written - last >= (1 << 30) or now - t0 >= 30:
                        last = written
                        t0 = now
                        print(f"  {written / 1e9:.2f} / "
                              f"{EXPECTED_BYTES / 1e9:.2f} GB", flush=True)
        except requests.RequestException as e:
            print(f"FAIL: interrupted at {written} bytes: {e} "
                  f"(resume with the same command)", file=sys.stderr)
            return 1

    return _finalize_part(part, archive)


def verify_archive(archive: Path, keep_bad: bool = True) -> tuple:
    nbytes = archive.stat().st_size
    if nbytes != EXPECTED_BYTES:
        sha = sha256_of(archive)
        bad = _bad_path(archive, sha)
        if keep_bad:
            archive.replace(bad)
        print(f"FAIL: archive is {nbytes} bytes, expected {EXPECTED_BYTES}\n"
              f"  expected sha256 {EXPECTED_SHA256}\n"
              f"  actual   sha256 {sha}"
              + (f"\n  kept as {bad}" if keep_bad else "\n  left in place"),
              file=sys.stderr)
        return False, 4
    sha = sha256_of(archive)
    if sha != EXPECTED_SHA256:
        bad = _bad_path(archive, sha)
        if keep_bad:
            archive.replace(bad)
        print(f"FAIL: sha256 mismatch\n  expected {EXPECTED_SHA256}\n"
              f"  actual   {sha}"
              + (f"\n  kept as {bad}" if keep_bad else "\n  left in place"),
              file=sys.stderr)
        return False, 4
    print(f"ARCHIVE OK: {archive} ({nbytes} bytes, sha256 {sha})")
    return True, 0


def write_manifest(dest: Path, archive: Path, layout: dict, counts: dict,
                   free_before: int, free_after: int) -> Path:
    manifest = {
        "corpus": "GenVideo-Val",
        "archive": {
            "file": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": EXPECTED_SHA256,
        },
        "source_url": URL,
        "license": LICENSE,
        "role": ROLE,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime(archive.stat().st_mtime)),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "real_prefix": layout["real_prefix"],
        "generator_prefixes": layout["gens"],
        "counts": {k: v for k, v in counts.items() if not k.startswith("_")},
        "total_fakes": counts["_total_fakes"],
        "total_clips": counts[REAL_SUBDIR] + counts["_total_fakes"],
        "usable_fake_dirs": counts["_fake_dirs"],
        "free_bytes_before": free_before,
        "free_bytes_after": free_after,
        "gates": {"reals_min": MIN_REALS, "fakes_min": MIN_FAKES,
                  "fake_dirs_min": MIN_FAKE_DIRS, "pass": True},
    }
    p = dest / "MANIFEST.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2))
    tmp.replace(p)
    return p


def _extract_and_manifest(dest: Path, archive: Path, layout: dict,
                          free_before: int) -> int:
    staging = dest / "extract"
    print(f"real prefix: {layout['real_prefix']}")
    for gen, pref in layout["gens"].items():
        print(f"generator {gen}: {pref}")
    n, nbytes = extract_selected(archive, layout, staging, dest)
    print(f"extracted {n} mp4 members ({nbytes / 1e9:.2f} GB) -> {staging}")
    moved = normalize(layout, staging, dest)
    print(f"normalized {moved} clips -> {dest}")
    counts = recount(dest)
    _print_counts(dest, counts)
    ok, reasons = _gates(counts)
    if not ok:
        print("FAIL: extraction gates failed: " + "; ".join(reasons),
              file=sys.stderr)
        return 6
    free_after = _free_bytes(dest)
    print(f"free after extraction: {free_after / 1e9:.1f} GB")
    if free_after < MIN_FREE_AFTER_EXTRACT:
        print(f"FAIL: only {free_after / 1e9:.1f} GB free, need >= "
              f"{MIN_FREE_AFTER_EXTRACT / 1e9:.1f} GB", file=sys.stderr)
        return 3
    man = write_manifest(dest, archive, layout, counts, free_before, free_after)
    print(f"MANIFEST -> {man}")
    return 0


def check(dest: Path, archive: Path) -> int:
    free = _free_bytes(dest)
    print(f"free: {free / 1e9:.1f} GB")
    if not archive.exists():
        part = archive.with_name(archive.name + ".part")
        if part.exists():
            print(f"ARCHIVE INCOMPLETE: {part} is {part.stat().st_size} bytes "
                  f"(expected {EXPECTED_BYTES})")
        else:
            print(f"ARCHIVE MISSING: {archive}", file=sys.stderr)
        return 1
    ok, rc = verify_archive(archive, keep_bad=False)
    if not ok:
        return rc
    counts = recount(dest)
    _print_counts(dest, counts)
    complete = (dest / "MANIFEST.json").exists()
    if complete:
        ok, reasons = _gates(counts)
        if not ok:
            print("FAIL: extraction gates failed: " + "; ".join(reasons),
                  file=sys.stderr)
            return 6
        print("EXTRACTION OK")
    else:
        print("EXTRACTION INCOMPLETE — run without --check to extract")
    if free < MIN_FREE_AFTER_EXTRACT:
        print(f"FREE >= {MIN_FREE_AFTER_EXTRACT / 1e9:.1f} GB: FAIL "
              f"(measured {free / 1e9:.1f} GB)")
        return 3
    print(f"FREE >= {MIN_FREE_AFTER_EXTRACT / 1e9:.1f} GB: PASS "
          f"(measured {free / 1e9:.1f} GB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch/verify/extract GenVideo-Val (CC-BY-NC-4.0, EXAM-ONLY)")
    ap.add_argument("--check", action="store_true",
                    help="verify archive sha, disk counts and free space only")
    ap.add_argument("--extract", action="store_true",
                    help="extract only (archive already verified)")
    ap.add_argument("--archive", default=None, help="archive path override")
    ap.add_argument("--dest", default=None, help="destination dir override")
    args = ap.parse_args()
    if args.check and args.extract:
        ap.error("--check and --extract are mutually exclusive")

    dest = Path(args.dest).resolve() if args.dest else DEST
    archive = (Path(args.archive).resolve() if args.archive
               else dest / ARCHIVE_NAME)

    if args.check:
        return check(dest, archive)

    free_before = _free_bytes(dest)
    if args.extract:
        if not archive.exists():
            print(f"FAIL: archive missing: {archive}", file=sys.stderr)
            return 1
        try:
            layout = inspect_layout(archive)
        except zipfile.BadZipFile as e:
            print(f"FAIL: not a readable zip: {e}", file=sys.stderr)
            return 1
        if not layout["real_prefix"]:
            print("STOP: real prefix Real/video/real_MSRVTT/ not found; "
                  "candidates:", file=sys.stderr)
            for cand in layout["real_candidates"]:
                print(f"  {cand}", file=sys.stderr)
            return 5
        return _extract_and_manifest(dest, archive, layout, free_before)

    # default: fetch (if needed) + verify + extract
    if archive.exists():
        if archive.with_name(archive.name + ".part").exists():
            print(f"note: leaving {archive.name}.part untouched")
        ok, rc = verify_archive(archive)
        if not ok:
            return rc
    else:
        rc = download(dest, archive)
        if rc:
            return rc

    try:
        layout = inspect_layout(archive)
    except zipfile.BadZipFile as e:
        print(f"FAIL: not a readable zip: {e}", file=sys.stderr)
        return 1
    if not layout["real_prefix"]:
        print("STOP: real prefix Real/video/real_MSRVTT/ not found; "
              "candidates:", file=sys.stderr)
        for cand in layout["real_candidates"]:
            print(f"  {cand}", file=sys.stderr)
        return 5
    return _extract_and_manifest(dest, archive, layout, free_before)


if __name__ == "__main__":
    sys.exit(main())
