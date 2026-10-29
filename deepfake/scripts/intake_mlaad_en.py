"""W7 intake: MLAAD fake/en snapshot download (fires after user accepts the HF gate).

Usage: ./.venv/bin/python deepfake/scripts/intake_mlaad_en.py
Resumable (snapshot_download resumes partial files). License: CC BY-NC-4.0 +
NOTICE_NO_COMMERCIAL_USE.txt — court use is non-commercial (D-059b).
"""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

DEST = Path("data/raw/mlaad")

print("downloading mueller91/MLAAD fake/en (~143 generators x 999 files, ~45.4GB)...", flush=True)
path = snapshot_download(
    repo_id="mueller91/MLAAD",
    repo_type="dataset",
    allow_patterns=["fake/en/*"],
    local_dir=str(DEST),
    max_workers=8,
    token=True,
)
print(f"DONE -> {path}", flush=True)

# verify: count + size
n = 0
size = 0
for p in (DEST / "fake" / "en").rglob("*.wav"):
    n += 1
    size += p.stat().st_size
print(f"verified: {n} wav files, {size / 1e9:.2f} GB (expect 142857 files, ~45.4GB)", flush=True)
if n != 142857:
    print("COUNT-MISMATCH", flush=True)
    sys.exit(1)
print("MLAAD-EN-DONE", flush=True)
