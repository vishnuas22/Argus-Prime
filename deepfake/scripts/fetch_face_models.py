"""Fetch + verify the face-embedding models (Phase 5 identity embedder).

SFace (MobileFaceNet trained with SFace loss, arXiv 2205.12010) is ArcFace-
family, identity-purpose-built, 128-d, ~13MB inference, and needs ZERO new
Python deps: opencv-python 5.0.0 ships cv2.FaceRecognizerSF which loads it.
This is the default Phase 5 embedder for identity clustering AND drift; the
measured contingency (spec item 2.4 — SFace drift probe AUC < 0.60) swaps in
ArcFace w600k_r50 (512-d, via cv2.dnn, also zero new deps) behind the same
seam.

Idempotent: if the on-disk file already matches the expected sha256,
download is skipped (so a user-placed copy is honored and standalone use
still works).

Usage:
  .venv/bin/python deepfake/scripts/fetch_face_models.py [--force]   # SFace
  .venv/bin/python deepfake/scripts/fetch_face_models.py --arcface   # ArcFace
  .venv/bin/python deepfake/scripts/fetch_face_models.py --all       # both

TRAP: raw.githubusercontent.com/.../face_recognition_sface_2021dec.onnx serves
the 133-byte git-LFS POINTER, not the model. The media.githubusercontent.com
URL below serves the real bytes; we still verify sha256 + size and fail loud
on pointer content.

ArcFace source: insightface publishes NO standalone arcface_w600k_r50.onnx
release asset (v0.7 assets are the packed antelopev2/buffalo_* packages) —
buffalo_l.zip ships the byte-identical model as w600k_r50.onnx (174,383,860
bytes, sha256 4c06341c...619e43). We download buffalo_l, extract ONLY that
member, verify its sha256, and discard the zip.
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
SFACE_URL = ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
             "main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"  # 38,696,353 bytes
SFACE_BYTES = 38_696_353
SFACE = WS / "tools" / "models" / "face_recognition_sface_2021dec.onnx"

ARCFACE_URL = ("https://github.com/deepinsight/insightface/releases/download/"
               "v0.7/buffalo_l.zip")
ARCFACE_ZIP_MEMBER = "w600k_r50.onnx"     # inside buffalo_l.zip
ARCFACE_SHA256 = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"  # 174,383,860 bytes
ARCFACE_BYTES = 174_383_860
ARCFACE = WS / "tools" / "models" / "arcface_w600k_r50.onnx"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch(url: str, dst: Path, expected_sha: str, expected_bytes: int) -> int:
    """Download url -> dst.part, verify size+sha256, atomically rename."""
    part = dst.with_suffix(dst.suffix + ".part")
    if part.exists():
        part.unlink()

    def hook(blocks, bs, total):
        got = blocks * bs
        if total > 0 and got % (64 * bs) < bs:
            print(f"  {got/1e6:.0f}/{total/1e6:.0f} MB", end="\r", flush=True)

    print(f"downloading {url}")
    urllib.request.urlretrieve(url, part, reporthook=hook)
    print()
    size = part.stat().st_size
    if size < 1000:
        head = part.read_bytes()[:200]
        part.unlink()
        raise SystemExit(
            f"FAIL: {dst.name} is {size} bytes — git-LFS pointer, not the model. "
            f"Head: {head!r}")
    if expected_bytes and size != expected_bytes:
        raise SystemExit(f"FAIL: {dst.name} is {size} bytes, expected {expected_bytes}")
    digest = sha256_of(part)
    if digest != expected_sha:
        part.unlink()
        raise SystemExit(f"FAIL: sha256 mismatch for {dst.name}: {digest} != {expected_sha}")
    part.rename(dst)
    print(f"ok: {dst} ({size} bytes, sha256 {digest[:16]}...)")
    return 0


def _ensure(dst: Path, expected_sha: str, force: bool, what: str,
            download) -> int:
    """Idempotent gate shared by both models: skip when sha256 already matches."""
    if dst.exists() and not force:
        digest = sha256_of(dst)
        if digest == expected_sha:
            print(f"skip: {dst.name} already present, sha256 verified "
                  f"({dst.stat().st_size} bytes)")
            return 0
        print(f"warn: {dst} present but sha256 {digest[:16]}... != expected — re-fetching",
              file=sys.stderr)
    print(f"[fetch] {what} -> {dst}")
    return download()


def _fetch_arcface(dst: Path) -> int:
    """Extract the ArcFace w600k_r50 model from the insightface buffalo_l.zip
    release package, verify sha256, discard the zip."""
    zip_part = dst.with_suffix(dst.suffix + ".zip.part")
    if zip_part.exists():
        zip_part.unlink()

    def hook(blocks, bs, total):
        got = blocks * bs
        if total > 0 and got % (64 * bs) < bs:
            print(f"  {got/1e6:.0f}/{total/1e6:.0f} MB", end="\r", flush=True)

    print(f"downloading {ARCFACE_URL}")
    urllib.request.urlretrieve(ARCFACE_URL, zip_part, reporthook=hook)
    print()
    part = dst.with_suffix(dst.suffix + ".part")
    try:
        with zipfile.ZipFile(zip_part) as z:
            # match the member case-insensitively; skip macOS junk (patterns:41)
            member = next((n for n in z.namelist()
                           if Path(n).name.lower() == ARCFACE_ZIP_MEMBER
                           and "__MACOSX" not in n and not Path(n).name.startswith("._")),
                          None)
            if member is None:
                raise SystemExit(f"FAIL: {ARCFACE_ZIP_MEMBER} not found in "
                                 f"buffalo_l.zip — members: {z.namelist()}")
            print(f"extracting {member} ({z.getinfo(member).file_size} bytes)")
            with z.open(member) as src, open(part, "wb") as out:
                for chunk in iter(lambda: src.read(1 << 20), b""):
                    out.write(chunk)
    finally:
        zip_part.unlink(missing_ok=True)
    size = part.stat().st_size
    if size != ARCFACE_BYTES:
        part.unlink()
        raise SystemExit(f"FAIL: {dst.name} is {size} bytes, expected {ARCFACE_BYTES}")
    digest = sha256_of(part)
    if digest != ARCFACE_SHA256:
        part.unlink()
        raise SystemExit(f"FAIL: sha256 mismatch for {dst.name}: {digest} != {ARCFACE_SHA256}")
    part.rename(dst)
    print(f"ok: {dst} ({size} bytes, sha256 {digest[:16]}...)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if sha matches")
    ap.add_argument("--arcface", action="store_true",
                    help="fetch the ArcFace w600k_r50 contingency embedder "
                         "(tools/models/arcface_w600k_r50.onnx) instead of SFace")
    ap.add_argument("--all", action="store_true", help="fetch both SFace and ArcFace")
    args = ap.parse_args()
    SFACE.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    if args.arcface or args.all:
        rc = _ensure(ARCFACE, ARCFACE_SHA256, args.force, "ArcFace (buffalo_l)",
                     lambda: _fetch_arcface(ARCFACE)) or rc
    if not args.arcface or args.all:
        rc = _ensure(SFACE, SFACE_SHA256, args.force, "SFace",
                     lambda: _fetch(SFACE_URL, SFACE, SFACE_SHA256, SFACE_BYTES)) or rc
    return rc


if __name__ == "__main__":
    sys.exit(main())
