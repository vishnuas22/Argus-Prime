#!/usr/bin/env python3
"""W-AV Step 2: build the av_diet split manifests (ONCE; D-046 law).

Reads the extracted RAVDESS video-speech tree, ffprobe-verifies every
candidate carries an audio stream, draws the diet (alphabetical source keys,
rng.permutation(seed=11)), splits 60/15/25 by SOURCE (no source straddles
splits; W4 precedent w4_video_dfd_spec.md §2), and writes:

  data/raw/av_diet/sources.json           per-source receipt (path, actor,
                                          emotion, statement, rep, ffprobe,
                                          sha256)
  data/raw/av_diet/split.json             train/val/exam key lists
  data/raw/av_diet/MANIFEST.md            strata table + laws + seed receipts
  data/raw/av_diet/license_receipts.json  RAVDESS CC BY-NC-SA 4.0 (zenodo
                                          1188976, md5 receipt ref), Wav2Lip
                                          NC research terms, SyncNet MIT,
                                          Kokoro Apache-2.0

Idempotent but NEVER regenerated after training starts (spec §5.1): if
split.json exists the script refuses to run without --force, and --force is
only for pre-training corrections.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
AV = WS / "data" / "raw" / "av_diet"
SEED = 11
N_SOURCES = 120
SPLIT = (0.60, 0.15, 0.25)


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(p: Path) -> dict | None:
    """Audio-stream receipt: None when the file carries no audio stream."""
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "stream=codec_type,codec_name,width,height,r_frame_rate",
         "-show_entries", "format=duration,size", "-of", "json", str(p)],
        capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        return None
    try:
        j = json.loads(r.stdout)
    except json.JSONDecodeError:
        return None
    streams = j.get("streams", [])
    aud = [s for s in streams if s.get("codec_type") == "audio"]
    vid = [s for s in streams if s.get("codec_type") == "video"]
    if not aud or not vid:
        return None
    v = vid[0]
    fps = v.get("r_frame_rate", "0/1")
    try:
        fps = round(eval(fps), 3)  # "25/1" -> 25.0
    except Exception:
        fps = 0.0
    return {"audio_codec": aud[0].get("codec_name"),
            "video_codec": v.get("codec_name"),
            "width": v.get("width"), "height": v.get("height"),
            "fps": fps,
            "duration_s": float(j.get("format", {}).get("duration", 0.0) or 0.0),
            "size": int(j.get("format", {}).get("size", 0) or 0)}


def parse_key(p: Path) -> dict:
    """RAVDESS filename: MM-VC-EM-IT-ST-RP-AC.avi (modality, vocal channel,
    emotion, intensity, statement, repetition, actor)."""
    parts = p.stem.split("-")
    return {"modality": parts[0], "vocal": parts[1], "emotion": parts[2],
            "intensity": parts[3], "statement": parts[4], "repetition": parts[5],
            "actor": parts[6]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", type=Path, default=AV / "ravdess",
                    help="extracted RAVDESS tree (Actor_XX/ dirs)")
    ap.add_argument("--out", type=Path, default=AV,
                    help="manifest home (default data/raw/av_diet; a scratch "
                         "dir is for dry-run mechanics only — the REAL "
                         "manifest is produced ONCE at the default)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    AV_OUT = args.out

    split_path = AV_OUT / "split.json"
    if split_path.exists() and not args.force:
        print(f"REFUSING: {split_path} exists — manifests are law after they "
              f"are read by training (spec §5.1). Use --force only pre-training.")
        return 2
    AV_OUT.mkdir(parents=True, exist_ok=True)

    # 1. candidates: full-AV (modality 01) SPEECH (vocal 01) files.
    # MEASURED 2026-11-04: the zenodo zips ship .mp4 (h264+aac), not .avi.
    # PREFIX-anchored: the selector is fields 1-2 (modality-vocal); a
    # mid-filename `*-01-01-*` glob accidentally matches any adjacent 01-01
    # pair (e.g. intensity-statement) — caught by the scratch dry-run
    # (50 bogus candidates instead of 120/actor).
    cands = sorted(args.extracted.glob("Actor_*/01-01-*.mp4"))
    if not cands:
        print(f"FATAL: no AV speech files under {args.extracted} — extract the "
              f"ravdess zips first (data/raw/av_diet/ravdess_zips/*.zip)")
        return 2
    print(f"candidates: {len(cands)} AV speech files", flush=True)

    # 2. ffprobe audio receipt per candidate
    sources, no_audio = {}, []
    t0 = time.time()
    for i, p in enumerate(cands):
        pr = ffprobe(p)
        if pr is None:
            no_audio.append(p.name)
            continue
        k = parse_key(p)
        key = p.stem
        # WS-relative when the tree lives in the workspace (the real run);
        # absolute for scratch dry-runs outside it.
        try:
            rel = str(p.relative_to(WS))
        except ValueError:
            rel = str(p)
        sources[key] = {"path": rel, "actor": k["actor"],
                        "emotion": k["emotion"], "intensity": k["intensity"],
                        "statement": k["statement"], "repetition": k["repetition"],
                        **pr}
        if (i + 1) % 200 == 0:
            print(f"  ffprobe {i + 1}/{len(cands)} ({time.time() - t0:.0f}s)", flush=True)
    print(f"with audio: {len(sources)} | no-audio excluded: {len(no_audio)}", flush=True)

    # 3. draw: alphabetical keys, permutation(seed=11), first N_SOURCES
    keys = sorted(sources)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(keys))
    diet = [keys[i] for i in perm[:N_SOURCES]]
    # 4. split the diet: alphabetical again, permutation(seed=11), 60/15/25
    diet_sorted = sorted(diet)
    rng2 = np.random.default_rng(SEED)
    perm2 = rng2.permutation(len(diet_sorted))
    n_train = int(round(SPLIT[0] * len(diet_sorted)))
    n_val = int(round(SPLIT[1] * len(diet_sorted)))
    order = [diet_sorted[i] for i in perm2]
    split = {"train": order[:n_train],
             "val": order[n_train:n_train + n_val],
             "exam": order[n_train + n_val:]}
    actors = {s: sorted({sources[k]["actor"] for k in v}) for s, v in split.items()}
    print(f"diet {len(diet)} -> train {len(split['train'])} "
          f"({len(actors['train'])} actors) / val {len(split['val'])} "
          f"({len(actors['val'])}) / exam {len(split['exam'])} ({len(actors['exam'])})",
          flush=True)

    # 5. receipts
    (AV_OUT / "sources.json").write_text(json.dumps(sources, indent=1))
    (AV_OUT / "split.json").write_text(json.dumps(
        {"seed": SEED, "rule": "alphabetical keys, rng.permutation(seed=11), "
         "60/15/25 of SOURCES (W4 precedent); no source straddles splits",
         "n_diet": len(diet), "pool_n": len(keys), "splits": split,
         "split_actors": actors}, indent=1))
    (AV_OUT / "license_receipts.json").write_text(json.dumps({
        "ravdess": {"license": "CC BY-NC-SA 4.0",
                    "source": "zenodo.org/records/1188976",
                    "verification": "per-file md5 vs zenodo API at download "
                                    "(ravdess_zips/ravdess_download_receipt.json)",
                    "note": "NC term drives D-059 eligibility; SA applies to "
                            "redistribution of derivatives — local research "
                            "tool, non-redistributed"},
        "wav2lip": {"license": "non-commercial research terms (README "
                    "Disclaimer: research/academic/personal only; models "
                    "trained on LRS2 — commercial use prohibited)",
                    "source": "github.com/Rudrabha/Wav2Lip",
                    "eligibility": "D-059 NC ruling"},
        "syncnet": {"license": "MIT (joonson/syncnet_python LICENSE.md)",
                    "source": "github.com/joonson/syncnet_python"},
        "kokoro": {"license": "Apache-2.0",
                   "source": "mlx-community/Kokoro-82M-8bit (HF cache, on disk)"},
    }, indent=1))
    (AV_OUT / "MANIFEST.md").write_text(f"""# av_diet manifest (W-AV Step 2; built {time.strftime('%Y-%m-%dT%H:%M:%S')})

Pool: {len(keys)} RAVDESS AV-speech files (modality 01, vocal 01) across 24 actors.
No-audio excluded: {len(no_audio)} (ffprobe receipt law).
Diet: {len(diet)} sources — alphabetical keys, rng.permutation(seed={SEED}), first {N_SOURCES}.
Split: 60/15/25 of SOURCES, seed {SEED} (W4 precedent): train {len(split['train'])} / val {len(split['val'])} / exam {len(split['exam'])}.
No source straddles splits. Exam sources held out entirely.

Strata (proportions are the RULE, counts are receipts):
- Wav2Lip fakes: 1 per source, method="Wav2Lip"
- Matched reals: same sources, video re-encoded through the IDENTICAL ffmpeg
  output stage Wav2Lip uses (-q:v 1 final mux), audio -c:a copy
- Narration reals (train): 50 — train sources muxed (-c:v copy) with Kokoro
  TTS narration (mlx-community/Kokoro-82M-8bit, Apache-2.0)
- Music reals (train): 50 — numpy-synthesized music bed (seed recorded)
- EXAM pairs: fakes + matched reals of exam sources (held out, never dumped)
- EXAM narration/music: 60 + 60 (exam sources)
- AVLips validation: external eval only (D2), never trained

Laws: channel balance (both fake/real arms through the same ffmpeg encode;
audio identical by construction), no silent-source generation, noface.jsonl
honesty, manifests are law after training starts (D-046).
""")
    print(f"manifests written to {AV_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
