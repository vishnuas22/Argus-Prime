"""Bootstrap / verify the runtime artifact bundle (P0a).

The deployed court loads weights that are NOT in git (see .gitignore). This
script is the single entry point for making a fresh clone runnable:

    .venv/bin/python deepfake/bootstrap.py --verify   # check only, no writes
    .venv/bin/python deepfake/bootstrap.py            # fetch missing + verify

Design
------
* Content-addressed: every file is checked against models.lock (sha256 + size).
* Fail-closed: a mismatch or a missing file is an error, never a silent skip.
* Two sources, in order:
    1. a HuggingFace repo, when models.lock records a `bundle` block and
       ARGUS_ARTIFACTS_REPO (or the lock) names it — fetched with
       huggingface_hub at the PINNED revision;
    2. otherwise the existing verified fetchers (tools/models/* via
       deepfake/scripts/fetch_face_models.py) and whatever bundle is already
       placed on disk.
* Idempotent: safe to run repeatedly; --verify never writes.

Exit codes: 0 = all present and verified · 1 = missing/mismatched · 2 = no lock.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[1])
ARTIFACTS_DIR = Path(os.environ.get("ARGUS_ARTIFACTS_DIR") or WS / "artifacts")
LOCK_PATH = WS / "models.lock"

REPO_ENV = "ARGUS_ARTIFACTS_REPO"
REVISION_ENV = "ARGUS_ARTIFACTS_REVISION"


def bundle_target() -> tuple[str | None, str | None]:
    """Resolve (repo_id, revision) from env or the lock's bundle block."""
    lock = {}
    if LOCK_PATH.exists():
        try:
            lock = json.loads(LOCK_PATH.read_text()).get("bundle") or {}
        except json.JSONDecodeError:
            lock = {}
    repo = os.environ.get(REPO_ENV) or lock.get("repo")
    rev = os.environ.get(REVISION_ENV) or lock.get("revision")
    return repo, rev


def fetch_from_hf(missing: list[str], repo: str, revision: str | None) -> list[str]:
    """Download the missing artifacts from a pinned HF revision.

    Files are downloaded into a scratch directory and then copied to their exact
    target path (hf_hub_download joins local_dir with the repo filename, which
    would otherwise double the relative path).

    Returns the list of files that are still missing afterwards.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub not installed; cannot fetch from HF", file=sys.stderr)
        return missing

    scratch = WS / ".hf_download"
    scratch.mkdir(parents=True, exist_ok=True)
    still = []
    for rel in missing:
        src = Path(rel)
        if src.parts and src.parts[0] == "artifacts":
            target = ARTIFACTS_DIR.joinpath(*src.parts[1:])
        else:
            target = WS / src
        try:
            got = Path(hf_hub_download(
                repo_id=repo,
                filename=rel,
                repo_type="model",
                revision=revision,
                local_dir=str(scratch),
            ))
            target.parent.mkdir(parents=True, exist_ok=True)
            got.replace(target)
            print(f"  fetched {rel} -> {target}")
        except Exception as e:
            print(f"  FAILED {rel}: {type(e).__name__}: {e}", file=sys.stderr)
            still.append(rel)
    return still


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lock() -> dict:
    if not LOCK_PATH.exists():
        raise SystemExit(
            f"no models.lock at {LOCK_PATH}\n"
            f"  generate it on the machine that has the artifacts:\n"
            f"    .venv/bin/python deepfake/scripts/make_models_lock.py"
        )
    return json.loads(LOCK_PATH.read_text())


def resolve(rel: str) -> Path:
    """Map a lock-relative path to its on-disk location.

    Paths under artifacts/ honour ARGUS_ARTIFACTS_DIR; everything else is
    repo-relative.
    """
    p = Path(rel)
    if p.parts and p.parts[0] == "artifacts":
        return ARTIFACTS_DIR.joinpath(*p.parts[1:])
    return WS / p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true",
                    help="check only; never fetch or write")
    args = ap.parse_args()

    lock = load_lock()
    artifacts = lock["artifacts"]

    ok, missing, mismatched = [], [], []
    total_bytes = 0
    for rel, meta in artifacts.items():
        p = resolve(rel)
        if not p.exists():
            missing.append((rel, p, meta["bytes"]))
            continue
        size = p.stat().st_size
        digest = sha256_of(p)
        if digest != meta["sha256"] or size != meta["bytes"]:
            mismatched.append((rel, p, meta, size, digest))
            continue
        ok.append(rel)
        total_bytes += size

    print(f"lock      : {LOCK_PATH}")
    print(f"artifacts : {ARTIFACTS_DIR}")
    print(f"verified  : {len(ok)}/{len(artifacts)}  ({total_bytes/1e6:.1f} MB)")

    if missing:
        print(f"\nMISSING ({len(missing)}):")
        for rel, p, size in missing:
            print(f"  {rel}  ({size/1e6:.2f} MB) -> expected at {p}")
    if mismatched:
        print(f"\nMISMATCHED ({len(mismatched)}):")
        for rel, p, meta, size, digest in mismatched:
            print(f"  {rel}")
            print(f"    expected {meta['sha256'][:16]}... {meta['bytes']} B")
            print(f"    actual   {digest[:16]}... {size} B")

    if missing or mismatched:
        if args.verify:
            print("\nVERIFY FAILED (no changes made; --verify)")
            return 1

        # Try to fetch what is missing from the pinned HF bundle.
        repo, rev = bundle_target()
        if missing and repo:
            print(f"\nfetching {len(missing)} missing artifact(s) from "
                  f"{repo} @ {rev or 'main'}")
            missing = [m for m in fetch_from_hf([m[0] for m in missing], repo, rev)
                       if not resolve(m).exists()]

        if missing:
            print(f"\nMISSING ({len(missing)}):")
            for rel in missing:
                rel_path = rel[0] if isinstance(rel, tuple) else rel
                print(f"  {rel_path}")
            print("\nBootstrap incomplete.")
            if not repo:
                print("  no HF bundle configured. Set:")
                print(f"    export {REPO_ENV}=<user>/<repo>")
                print(f"    export {REVISION_ENV}=<pinned-sha>")
                print("  or populate the bundle manually and re-run --verify.")
            else:
                print("  the remaining files are not in the HF bundle.")
                print("  Non-commercial weights are excluded by design — fetch them")
                print("  with: python deepfake/scripts/fetch_face_models.py")
            return 1

        # Re-verify everything after the fetch.
        ok, missing, mismatched = [], [], []
        for rel, meta in artifacts.items():
            p = resolve(rel)
            if not p.exists():
                missing.append((rel, p, meta["bytes"]))
                continue
            if sha256_of(p) != meta["sha256"]:
                mismatched.append((rel, p, meta, p.stat().st_size, sha256_of(p)))
                continue
            ok.append(rel)
        if missing or mismatched:
            print(f"\npost-fetch verify failed: {len(missing)} missing, "
                  f"{len(mismatched)} mismatched", file=sys.stderr)
            return 1

    print(f"\nverified: {len(ok)}/{len(artifacts)}")
    print("\nALL-SHAS-OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())