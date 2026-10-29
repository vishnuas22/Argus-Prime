"""W7 intake: MLAAD fake/en snapshot download — PER-GENERATOR (robust against the
repo-level API sibling cap at 100k files; each generator dir's tree listing is complete).

Usage: ./.venv/bin/python deepfake/scripts/intake_mlaad_en.py
Resumable per generator and per file. License: CC BY-NC-4.0 + NOTICE_NO_COMMERCIAL_USE.txt
— court use is non-commercial (D-059b).
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

DEST = Path("data/raw/mlaad")
TOKEN = Path.home().joinpath(".cache/huggingface/token").read_text().strip()


def api(url: str):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOKEN}"})
    return json.load(urllib.request.urlopen(req, timeout=30))


# 1. generator list for fake/en (tree API, non-recursive)
gens = [i["path"].split("/")[-1]
        for i in api("https://huggingface.co/api/datasets/mueller91/MLAAD/tree/main/fake/en")
        if i["type"] == "directory"]
print(f"{len(gens)} generators under fake/en", flush=True)

# 2. per-generator snapshot (resumable; 999 files each expected)
done, t0 = 0, time.time()
for gi, g in enumerate(gens):
    pattern = f"fake/en/{g}/*"
    snapshot_download(
        repo_id="mueller91/MLAAD",
        repo_type="dataset",
        allow_patterns=[pattern],
        local_dir=str(DEST),
        max_workers=8,
        token=TOKEN,
    )
    n = len(list((DEST / "fake" / "en" / g).glob("*.wav"))) if (DEST / "fake" / "en" / g).exists() else 0
    done += 1
    rate = done / max(time.time() - t0, 1e-9)
    print(f"[{done}/{len(gens)}] {g}: {n} files ({rate:.2f} gen/s)", flush=True)

# 3. verify
total = sum(1 for p in (DEST / "fake" / "en").rglob("*.wav"))
size = sum(p.stat().st_size for p in (DEST / "fake" / "en").rglob("*.wav"))
bad = [(g, len(list((DEST / "fake" / "en" / g).glob("*.wav")))) for g in gens
       if len(list((DEST / "fake" / "en" / g).glob("*.wav"))) != 999]
print(f"verified: {total} wav files, {size / 1e9:.2f} GB (expect 142857, ~45.4GB)", flush=True)
if bad:
    print(f"COUNT-MISMATCH in {len(bad)} generators: {bad[:5]}", flush=True)
    sys.exit(1)
if total != 142857:
    print("TOTAL-MISMATCH", flush=True)
    sys.exit(1)
print("MLAAD-EN-DONE", flush=True)
