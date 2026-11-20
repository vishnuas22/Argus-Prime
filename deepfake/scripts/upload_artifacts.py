"""Upload the deployed artifact bundle to HuggingFace (P0a).

Reproducible counterpart to bootstrap.py: bootstrap.py verifies/downloads, this
uploads. Reads models.lock so what gets uploaded is exactly what gets verified.

    .venv/bin/python deepfake/scripts/upload_artifacts.py \
        --repo thedataghost22/Argus_prime [--include-nc]

Non-commercial weights are EXCLUDED by default. tools/models/arcface_w600k_r50.onnx
is insightface (non-commercial); fetch it with deepfake/scripts/fetch_face_models.py
instead of hosting it. Pass --include-nc only if you have accepted redistribution.

Idempotent: re-running re-uploads changed files (HF dedupes by sha256).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
LOCK = WS / "models.lock"

# Non-commercial artifacts: never uploaded without an explicit flag.
NC_ARTIFACTS = {"tools/models/arcface_w600k_r50.onnx"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. thedataghost22/Argus_prime")
    ap.add_argument("--include-nc", action="store_true",
                    help="also upload non-commercial weights (redistribution)")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    lock = json.loads(LOCK.read_text())
    artifacts = lock["artifacts"]

    print(f"repo   : {args.repo}")
    info = api.repo_info(args.repo, repo_type="model")
    print(f"private: {getattr(info, 'private', None)}   current sha: {info.sha}")

    todo = []
    for rel, meta in artifacts.items():
        if rel in NC_ARTIFACTS and not args.include_nc:
            print(f"  skip (NC, not redistributed): {rel}")
            continue
        todo.append((rel, meta))

    total = sum(m["bytes"] for _, m in todo)
    print(f"uploading {len(todo)} files ({total/1e6:.1f} MB)\n")

    for i, (rel, meta) in enumerate(todo, 1):
        p = WS / rel
        if not p.exists():
            print(f"  MISSING {rel}", file=sys.stderr)
            return 1
        digest = sha256_of(p)
        if digest != meta["sha256"]:
            print(f"  SHA MISMATCH {rel}: {digest[:16]} != {meta['sha256'][:16]}",
                  file=sys.stderr)
            return 1
        api.upload_file(path_or_fileobj=str(p), path_in_repo=rel,
                        repo_id=args.repo, repo_type="model",
                        commit_message=f"argus artifact: {rel}")
        print(f"  [{i}/{len(todo)}] {meta['bytes']/1e6:8.2f} MB  {rel}", flush=True)

    files = api.list_repo_files(args.repo, repo_type="model")
    final = api.repo_info(args.repo, repo_type="model")
    print(f"\nuploaded. repo now has {len(files)} files")
    print(f"pinned revision: {final.sha}")
    print("\nnext: record that sha in models.lock as bundle_revision, then")
    print("      ARGUS_ARTIFACTS_REPO=<repo> ARGUS_ARTIFACTS_REVISION=<sha> \\")
    print("      python deepfake/bootstrap.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())