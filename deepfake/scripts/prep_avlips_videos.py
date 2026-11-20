#!/usr/bin/env python3
"""W-AV Step 2b — build the AVLips VIDEO-form external exam set (D-077 Option A).

The AVLips v1.0 Drive archive (9.26 GB measured 2026-11-04) is the ONLY public
source of val videos+audio: the shipped validation set is LipFD's preprocessed
composite form (mel-spectrogram band + 5 face crops per PNG — no video, no
audio), unusable by the AV channel. This script:

  1. parses the val sample ids from the composite PNG tree,
  2. selectively extracts ONLY those ids' mp4+wav members from the archive
     (zip = random access; tar/tar.gz/tar.xz = single streaming pass),
  3. muxes each video with its wav when the mp4 carries no audio stream
     (AVLips ships wavs SEPARATELY; av_sync reads the container's stream),
  4. fail-closed asserts: every val id yields a muxed mp4 with BOTH streams,
  5. writes data/raw/avlips/manifest.json {real,fake} + D-050 archive receipt
     + license receipt ("no license stated; research/eval use", user-aware).

--dry-run: parse ids + report the archive's inner layout; extract nothing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
TMP = Path(__import__("os").environ.get("ARGUS_TMP") or __import__("tempfile").gettempdir())


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_streams(p: Path) -> dict | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=codec_type,codec_name,width,height,r_frame_rate",
             "-show_entries", "format=duration", "-of", "json", str(p)],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        j = json.loads(r.stdout)
        streams = j.get("streams", [])
        return {
            "video": next((s for s in streams if s.get("codec_type") == "video"), None),
            "audio": next((s for s in streams if s.get("codec_type") == "audio"), None),
            "duration": float(j.get("format", {}).get("duration", 0) or 0),
        }
    except Exception:
        return None


def parse_val_ids(val_root: Path) -> dict:
    out = {}
    for cls in ("0_real", "1_fake"):
        d = val_root / cls
        ids = sorted({int(p.name.split("_")[0]) for p in d.glob("*.png")})
        out[cls] = ids
    return out


def detect_format(p: Path) -> str:
    with open(p, "rb") as f:
        head = f.read(262)
    if head[:2] == b"PK":
        return "zip"
    if head[:6] == b"\xfd7zXZ\x00":
        return "tar.xz"
    if head[:2] == b"\x1f\x8b":
        return "tar.gz"
    if head[257:262] == b"ustar":
        return "tar"
    return "unknown"


def zip_members(zf, wanted: dict) -> dict:
    """Map (class, id) -> (mp4_name, wav_name) from the zip's central directory."""
    import zipfile
    names = zf.namelist()
    found = {c: {} for c in wanted}
    for n in names:
        parts = n.split("/")
        if len(parts) < 2:
            continue
        cls = next((c for c in ("0_real", "1_fake") if c in parts), None)
        if cls is None:
            continue
        stem = Path(n).stem
        if not stem.isdigit():
            continue
        i = int(stem)
        if i not in wanted[cls]:
            continue
        if n.endswith(".mp4"):
            found[cls].setdefault(i, {})["mp4"] = n
        elif n.endswith(".wav"):
            found[cls].setdefault(i, {})["wav"] = n
    return found


def tar_members(tf, wanted: dict) -> dict:
    import tarfile
    found = {c: {} for c in wanted}
    for m in tf:
        if not m.isfile():
            continue
        parts = m.name.split("/")
        cls = next((c for c in ("0_real", "1_fake") if c in parts), None)
        if cls is None:
            continue
        stem = Path(m.name).stem
        if not stem.isdigit():
            continue
        i = int(stem)
        if i not in wanted[cls]:
            continue
        if m.name.endswith(".mp4"):
            found[cls].setdefault(i, {})["mp4"] = m.name
        elif m.name.endswith(".wav"):
            found[cls].setdefault(i, {})["wav"] = m.name
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=TMP / "avlips_v10.bin")
    ap.add_argument("--val-root", type=Path, default=TMP / "avlips_val_root" / "val")
    ap.add_argument("--out", type=Path, default=WS / "data" / "raw" / "avlips")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.archive.exists():
        print(f"FATAL: archive missing: {args.archive}")
        return 2
    fmt = detect_format(args.archive)
    size = args.archive.stat().st_size
    print(f"archive: {args.archive} ({size:,} B) format={fmt}", flush=True)
    if fmt == "unknown":
        print("FATAL: unrecognized archive format — inspect manually")
        return 2

    wanted = parse_val_ids(args.val_root)
    print(f"val ids: 0_real {len(wanted['0_real'])} "
          f"(max {max(wanted['0_real'])}), "
          f"1_fake {len(wanted['1_fake'])} (max {max(wanted['1_fake'])})", flush=True)

    import zipfile
    import tarfile

    if args.dry_run:
        if fmt == "zip":
            with zipfile.ZipFile(args.archive) as zf:
                names = zf.namelist()
            print(f"members: {len(names)}; first 12:")
            for n in names[:12]:
                print("  ", n)
        else:
            opener = {"tar": tarfile.open, "tar.gz": tarfile.open, "tar.xz": tarfile.open}[fmt]
            mode = "r" if fmt == "tar" else ("r:gz" if fmt == "tar.gz" else "r:xz")
            n = 0
            with opener(args.archive, mode) as tf:
                for m in tf:
                    if n < 12:
                        print("  ", m.name)
                    n += 1
            print(f"members: {n}")
        return 0

    out = args.out
    vids = out / "videos"
    for cls in ("0_real", "1_fake"):
        (vids / cls).mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="avlips_prep_", dir=str(TMP)))

    t0 = time.time()
    results = {"real": [], "fake": [], "skipped": []}
    try:
        if fmt == "zip":
            with zipfile.ZipFile(args.archive) as zf:
                found = zip_members(zf, wanted)
                for cls, tag in (("0_real", "real"), ("1_fake", "fake")):
                    for i in sorted(wanted[cls]):
                        mem = found[cls].get(i, {})
                        if "mp4" not in mem:
                            results["skipped"].append(f"{cls}/{i}: no mp4 member")
                            continue
                        zf.extract(mem["mp4"], staging)
                        src = staging / mem["mp4"]
                        wav_path = None
                        if "wav" in mem:
                            zf.extract(mem["wav"], staging)
                            wav_path = staging / mem["wav"]
                        dst = vids / cls / f"{i}.mp4"
                        ok = _finalize(src, wav_path, dst)
                        (results[tag] if ok else results["skipped"]).append(
                            str(dst.relative_to(WS)) if ok else f"{cls}/{i}: mux failed")
                        src.unlink(missing_ok=True)
                        if wav_path:
                            wav_path.unlink(missing_ok=True)
        else:
            mode = "r" if fmt == "tar" else ("r:gz" if fmt == "tar.gz" else "r:xz")
            opener = {"tar": tarfile.open, "tar.gz": tarfile.open, "tar.xz": tarfile.open}[fmt]
            # pass 1: map wanted (class, id, kind) -> member name
            with opener(args.archive, mode) as tf:
                found = tar_members(tf, wanted)
            # pass 2: extract ALL wanted members to staging (order-independent:
            # a wav may stream after its mp4, so mux only after both are on disk)
            with opener(args.archive, mode) as tf:
                for m in tf:
                    if not m.isfile():
                        continue
                    parts = m.name.split("/")
                    cls = next((c for c in ("0_real", "1_fake") if c in parts), None)
                    if cls is None:
                        continue
                    stem = Path(m.name).stem
                    if not stem.isdigit() or int(stem) not in wanted[cls]:
                        continue
                    kind = "mp4" if m.name.endswith(".mp4") else (
                        "wav" if m.name.endswith(".wav") else None)
                    if kind and kind in found[cls].get(int(stem), {}):
                        tf.extract(m, staging)
            # pass 3: mux loop
            for cls, tag in (("0_real", "real"), ("1_fake", "fake")):
                for i in sorted(wanted[cls]):
                    mem = found[cls].get(i, {})
                    if "mp4" not in mem:
                        results["skipped"].append(f"{cls}/{i}: no mp4 member")
                        continue
                    src = staging / mem["mp4"]
                    wav_path = staging / mem["wav"] if "wav" in mem else None
                    dst = vids / cls / f"{i}.mp4"
                    ok = _finalize(src, wav_path, dst)
                    (results[tag] if ok else results["skipped"]).append(
                        str(dst.relative_to(WS)) if ok else f"{cls}/{i}: mux failed")
                    src.unlink(missing_ok=True)
                    if wav_path:
                        wav_path.unlink(missing_ok=True)
        # final: coverage assert
        missing = []
        for cls, tag in (("0_real", "real"), ("1_fake", "fake")):
            for i in wanted[cls]:
                if not (vids / cls / f"{i}.mp4").exists():
                    missing.append(f"{cls}/{i}")
        if missing:
            print(f"FATAL: {len(missing)} val ids missing muxed video "
                  f"(first 5: {missing[:5]})")
            return 2
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # receipts
    receipt = {
        "archive": str(args.archive),
        "size_bytes": size,
        "sha256": sha256_of(args.archive),
        "format": fmt,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_real": len(results["real"]),
        "n_fake": len(results["fake"]),
        "skipped": results["skipped"][:20],
        "license": "none stated in LipFD repo; research/evaluation use "
                   "(user-aware, D-077)",
        "source": "https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm",
    }
    (out / "archive_receipt.json").write_text(json.dumps(receipt, indent=1))
    manifest = {
        "real": sorted(results["real"]),
        "fake": sorted(results["fake"]),
        "built": receipt["built"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest['real'])} real / {len(manifest['fake'])} fake "
          f"-> {out/'manifest.json'} ({time.time()-t0:.0f}s)", flush=True)
    return 0


def _finalize(src_mp4: Path, wav: Path | None, dst: Path) -> bool:
    """Copy or mux into dst; assert both streams present."""
    st = ffprobe_streams(src_mp4)
    if st is None:
        return False
    if st["audio"] is not None:
        shutil.copy2(src_mp4, dst)
    else:
        if wav is None or not wav.exists():
            return False
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src_mp4), "-i", str(wav),
             "-c:v", "copy", "-c:a", "aac", "-strict", "-2", "-shortest",
             str(dst)], capture_output=True, timeout=300)
        if r.returncode != 0:
            return False
    st2 = ffprobe_streams(dst)
    return bool(st2 and st2["video"] and st2["audio"] and st2["duration"] > 0.5)


if __name__ == "__main__":
    sys.exit(main())
