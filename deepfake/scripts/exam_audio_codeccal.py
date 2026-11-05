#!/usr/bin/env python3
"""W2 — Audio per-codec calibration exam (Phase-R slot 2), harness v2.

Registered protocol: artifacts/generator_intake/audio_codec_calibration_spec.md
Greenlight: user word 2026-10-19 ("Proceed W2 now" — same pattern as W1).
READ-ONLY for the court: scores the SHIPPED audio court via
argus_detect.detect_audio(recovery_enabled=False) — no court code touched,
no new dependencies, offsets fitted on train reals ONLY (D-035).

v2 protocol corrections (measured during bring-up, disclosed in the report):
  * Fit stratum = the ORIGINAL ASVspoof 2019 LA train shard (the bundle's own
    training corpus, full-length utterances decoded to the court's exact
    regime: 16 kHz mono WAV). NOT the r9 rebuild-diet files — measured
    finding: audio_r9/real asv21_bonafide copies are 1.00 s truncations whose
    zero-padding regime flips the court wholesale (p≈1.0 on train reals) —
    they cannot condition a codec offset; finding escalated separately.
  * Registered "ogg q5" arm substituted by opus128_ogg: this libsndfile
    (1.2.2, system build) CANNOT decode vorbis (fail-closed, measured), but
    DOES decode opus-in-ogg — a reasoned substitution consistent with the
    ASVspoof-5 research line (opus = top realistic codec).
  * In-court decode table (measured): mp3 YES · flac YES · opus-in-ogg YES ·
    vorbis-in-ogg NO (fail-closed) · shard FLAC via disk YES, via BytesIO NO
    (server-side decode quirk documented in the report).

Strata:
  real_fit    : asv21?? no — 2019 LA train BONAFIDE sample (200, seeded 31)
  fake_recall : 2019 LA train SPOOF sample (60) — recall-loss bar only

Arms: wav (control) · mp3_cbr128 · mp3_cbr192 · flac · opus128_ogg ·
      mp3_mono16k (CBR 128k, mono, 16 kHz)
"""
import csv
import io
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

SHARD = WS / "data" / "raw" / "asvspoof" / "data" / "train-00000-of-00001.parquet"
OUT = WS / "artifacts" / "exam_audio_codeccal"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 31

ARMS = {
    "wav": None,
    "mp3_cbr128": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    "mp3_cbr192": ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "opus128_ogg": ["-c:a", "libopus", "-b:a", "128k", "-f", "ogg"],
    "mp3_mono16k": ["-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1", "-ar", "16000", "-f", "mp3"],
}
EXT = {"mp3_cbr128": "mp3", "mp3_cbr192": "mp3", "flac": "flac",
       "opus128_ogg": "ogg", "mp3_mono16k": "mp3", "wav": "wav"}

N_BONA = 200
N_SPOOF = 60
NF = {"mp3_cbr128": ".mp3", "mp3_cbr192": ".mp3", "flac": ".flac",
      "opus128_ogg": ".ogg", "mp3_mono16k": ".mp3"}


def ffmpeg(args_list, src: Path, dst: Path, extra=None):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    *(extra or []), *args_list, str(dst)], check=True)
    return dst


def pick_stems():
    import pyarrow.parquet as pq
    tab = pq.read_table(SHARD, columns=["system_id", "audio"])
    sys_ids = tab["system_id"].to_pylist()
    bona = [i for i, v in enumerate(sys_ids) if v == "-"]
    spoofs = [i for i, v in enumerate(sys_ids) if v != "-"]
    rng = random.Random(SEED)
    return ([("real_fit", i) for i in rng.sample(bona, min(N_BONA, len(bona)))] +
            [("fake_asv19", i) for i in rng.sample(spoofs, min(N_SPOOF, len(spoofs)))])


def main():
    import argus_detect as A
    import pyarrow.parquet as pq
    print("[init] warming audio model...", flush=True)
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    A._AUDIO_MODEL = None
    A._get_audio_model(device)
    print(f"[init] device={device}", flush=True)

    tab = pq.read_table(SHARD, columns=["system_id", "audio"])
    stems = pick_stems()
    smoke = "--smoke" in sys.argv
    if smoke:
        stems = stems[:6]
        print(f"[smoke] {len(stems)} stems", flush=True)
    print(f"[plan] {len(stems)} stems x {len(ARMS)} arms = {len(stems) * len(ARMS)} scores",
          flush=True)

    work = Path(tempfile.mkdtemp(prefix="codec_cal2_"))
    rows = []
    t0 = time.time()
    for i, (stratum, idx) in enumerate(stems, 1):
        rec = tab["audio"][idx].as_py()
        data = rec["bytes"] if isinstance(rec, dict) else rec
        base = work / f"s{idx}"
        base.write_bytes(data)
        for arm in ARMS:
            t_arm = time.time()
            try:
                if arm == "wav":
                    path = ffmpeg(["-c:a", "pcm_s16le", "-f", "wav", "-ar", "16000", "-ac", "1"],
                                  base, work / f"s{idx}_wav.wav")
                else:
                    path = ffmpeg(ARMS[arm], base, work / f"s{idx}_{arm}.{EXT[arm]}")
                res = A.detect_audio(path, recovery_enabled=False)
                rows.append({
                    "stratum": stratum, "shard_idx": idx, "arm": arm,
                    "p_fake": res["p_fake_audio_mean"],
                    "z_mean": res["znorm_mean"],
                    "z_segs": json.dumps(res["per_segment_znorm"]),
                    "ooc": res["out_of_coverage"],
                    # W9 (D-079): additive column — fused-court abstention flag
                    # (legacy bundles: abstained == ooc, so p4 receipts unchanged)
                    "abstained": bool(res.get("abstained", False)),
                    "segs": res["segments_analyzed"],
                    "wall_ms": int((time.time() - t_arm) * 1000),
                })
            except Exception as ex:
                rows.append({"stratum": stratum, "shard_idx": idx, "arm": arm,
                             "error": f"{type(ex).__name__}: {str(ex)[:170]}"})
        if i % 20 == 0 or i == len(stems):
            print(f"[{i}/{len(stems)}] {time.time()-t0:.0f}s", flush=True)

    # ---- short-upload pad probe (bonus registered appendix: D-046 finding) ----
    if "--smoke" not in sys.argv:
        bonidx = [i for (s, i) in stems if s == "real_fit"][:8]
        print(f"[padprobe] 8 bona stems x trims 1.0/2.0/3.0 s", flush=True)
        probs = pq.read_table(SHARD, columns=["audio"])
        for idx in bonidx:
            rec = probs["audio"][idx].as_py()
            data = rec["bytes"] if isinstance(rec, dict) else rec
            base = work / f"p{idx}"
            base.write_bytes(data)
            for sec in ("1.0", "2.0", "3.0", "4.0"):
                t_arm = time.time()
                try:
                    path = ffmpeg(["-c:a", "pcm_s16le", "-f", "wav",
                                   "-ar", "16000", "-ac", "1", "-t", sec],
                                  base, work / f"trim{idx}_{sec}s.wav")
                    res = A.detect_audio(path, recovery_enabled=False)
                    rows.append({
                        "stratum": f"padprobe_{sec}s", "shard_idx": idx, "arm": "wav_trim",
                        "p_fake": res["p_fake_audio_mean"],
                        "z_mean": res["znorm_mean"],
                        "z_segs": json.dumps(res["per_segment_znorm"]),
                        "ooc": res["out_of_coverage"],
                        "segs": res["segments_analyzed"],
                        "wall_ms": int((time.time() - t_arm) * 1000),
                    })
                except Exception as ex:
                    rows.append({"stratum": f"padprobe_{sec}s", "shard_idx": idx, "arm": "wav_trim",
                                 "error": f"{type(ex).__name__}: {str(ex)[:170]}"})

    with open(OUT / "scores.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=["stratum", "shard_idx", "arm", "p_fake",
                                          "z_mean", "z_segs", "ooc", "abstained",
                                          "segs", "wall_ms", "error"])
        w.writeheader()
        w.writerows(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    print(f"[done] {len(rows)} rows ({n_err} errors) -> {OUT}/scores.csv", flush=True)


if __name__ == "__main__":
    main()
