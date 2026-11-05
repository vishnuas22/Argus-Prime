#!/usr/bin/env python3
"""D-050 receipted downloader: RAVDESS video-speech zips (zenodo record 1188976).

W-AV Step 2 acquisition. Downloads the 24 Video_Speech_Actor_XX.zip files,
verifies each against the md5 checksum published by the zenodo API, and writes
a receipt JSON. Idempotent: already-verified files are skipped; partial
downloads resume via curl -C -.

License receipt: CC BY-NC-SA 4.0 (record metadata) — NC term drives D-059
eligibility; SA noted. Receipt includes record DOI version + per-file md5.
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

WS = Path(__file__).resolve().parents[2]
DEST = WS / "data" / "raw" / "av_diet" / "ravdess_zips"
API = "https://zenodo.org/api/records/1188976"
RECEIPT = DEST / "ravdess_download_receipt.json"
PREFIX = "Video_Speech_Actor_"
EXPECTED_N = 24


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    rec = json.load(urlopen(API, timeout=60))
    files = [f for f in rec["files"] if f["key"].startswith(PREFIX)]
    if len(files) != EXPECTED_N:
        print(f"FATAL: expected {EXPECTED_N} {PREFIX}*.zip in record, got {len(files)}", flush=True)
        return 2
    receipt = {
        "api": API,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "doi": rec.get("doi", ""),
        "license": rec.get("metadata", {}).get("license", {}).get("id", "UNKNOWN"),
        "files": [],
    }
    for f in files:
        key, size = f["key"], f["size"]
        url = f["links"]["self"]
        expect = f["checksum"].split(":", 1)[1]  # "md5:hex"
        dest = DEST / key
        ok, got = False, None
        if dest.exists() and dest.stat().st_size == size:
            got = md5_of(dest)
            ok = got == expect
        if not ok:
            # flaky-network law (measured 2026-11-03/04: curl exit 18 partial
            # transfers killed a full run): retry the resumable fetch a few
            # times before failing closed.
            for attempt in range(4):
                r = subprocess.run(["curl", "-sL", "-C", "-", "-o", str(dest),
                                    "--retry", "3", "--retry-delay", "5",
                                    url])
                if r.returncode == 0:
                    break
                print(f"{key}: curl rc={r.returncode} (attempt {attempt + 1}/4) "
                      f"— resuming", flush=True)
                time.sleep(10)
            got = md5_of(dest)
            ok = got == expect and dest.stat().st_size == size
        receipt["files"].append(
            {"key": key, "size": size, "md5_expected": expect, "md5_got": got, "ok": ok}
        )
        n_ok = sum(1 for x in receipt["files"] if x["ok"])
        RECEIPT.write_text(json.dumps(receipt, indent=2))
        print(f"{key}: {'OK' if ok else 'MD5-MISMATCH'} ({n_ok}/{len(files)})", flush=True)
        if not ok:
            print("FATAL: checksum mismatch — failing closed", flush=True)
            return 2
    total = sum(x["size"] for x in receipt["files"])
    print(f"ALL {len(files)} VERIFIED — total {total / 1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
