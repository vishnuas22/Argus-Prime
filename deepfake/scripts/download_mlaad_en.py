"""W7 intake: MLAAD fake/en direct threaded downloader.

Uses the sealed file list (data/raw/mlaad/_filelist.jsonl, 142,857 files) and downloads
each file directly from the resolve endpoint — no repo-tree walking (snapshot_download
walks the full 500k+ file tree client-side before filtering; unusable at this size).
Per-file: skip if exists with exact size; else GET with Range resume; verify size.
License: CC BY-NC-4.0 + NOTICE_NO_COMMERCIAL_USE.txt — court use is non-commercial (D-059b).

Usage: ./.venv/bin/python deepfake/scripts/download_mlaad_en.py [--workers 16]
"""
import argparse
import json
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

BASE = "https://huggingface.co/datasets/mueller91/MLAAD/resolve/main/"
TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()
DEST = Path("data/raw/mlaad")

_tls = threading.local()


def _session() -> requests.Session:
    """Thread-local session — keep-alive connection reuse (urllib's per-file
    TCP+TLS handshake was the bottleneck: 6 files/s at 16 workers, plus local
    DNS resolver churn)."""
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=2, pool_maxsize=2)
        s.mount("https://", adapter)
        s.headers.update({"Authorization": f"Bearer {TOKEN}"})
        _tls.s = s
    return s


def fetch(item: dict) -> tuple[str, bool, str]:
    rel, want = item["path"], item["size"]
    out = DEST / rel
    if out.exists() and out.stat().st_size == want:
        return rel, True, "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    url = BASE + urllib.parse.quote(rel)
    for attempt in range(4):
        try:
            done = out.stat().st_size if out.exists() else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with _session().get(url, headers=headers, timeout=(10, 60), stream=True) as r:
                # HTTP error statuses must RAISE (the old urllib raised on 4xx/5xx and the
                # retry logic handled it; without this check a 429/503 error BODY gets
                # written to disk as if it were content — 15k garbage files caught live).
                r.raise_for_status()
                if done and r.status_code != 206:
                    done = 0  # server ignored Range — restart file
                with out.open("ab" if done else "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if out.stat().st_size == want:
                return rel, True, "ok" if not done else "resumed"
            return rel, False, f"size {out.stat().st_size} != {want}"
        except Exception as e:
            if attempt == 3:
                return rel, False, f"ERR {type(e).__name__}: {e}"
            time.sleep(min(30, 4 * (attempt + 1)))  # gentler backoff for 429/503 class
    return rel, False, "unreachable"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    items = [json.loads(l) for l in (DEST / "_filelist.jsonl").read_text().splitlines() if l.strip()]
    print(f"{len(items)} files to ensure", flush=True)
    ok = skip = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch, it): it["path"] for it in items}
        for i, fu in enumerate(as_completed(futs), 1):
            rel, good, how = fu.result()
            if good:
                ok += 1
                if how == "skip":
                    skip += 1
            else:
                fail += 1
                print(f"FAIL {rel}: {how}", flush=True)
            if i % 2000 == 0:
                rate = i / (time.time() - t0)
                print(f"[{i}/{len(items)}] ok={ok} (skip={skip}) fail={fail} {rate:.0f} files/s", flush=True)
    print(f"done: ok={ok} skip={skip} fail={fail}", flush=True)
    if fail:
        print("DOWNLOAD-INCOMPLETE", flush=True)
        return 1
    total = sum(1 for p in (DEST / "fake" / "en").rglob("*.wav"))
    size = sum(p.stat().st_size for p in (DEST / "fake" / "en").rglob("*.wav"))
    print(f"verified: {total} wav, {size / 1e9:.2f} GB (expect 142857, ~45.4GB)", flush=True)
    print("MLAAD-EN-DONE" if total == 142857 else "TOTAL-MISMATCH", flush=True)
    return 0 if total == 142857 else 1


if __name__ == "__main__":
    sys.exit(main())
