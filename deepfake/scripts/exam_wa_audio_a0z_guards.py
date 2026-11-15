#!/usr/bin/env python
"""W-AUDIO A0z 1B — guard harness (inference only; no training, no wiring).

Scores the released nii-yamagishilab/xls-r-1b-anti-deepfake checkpoint on the
five sealed guard arms of artifacts/wa_audio_a0z_guards/SPEC.md §4-§5:

  asv21la       G1  ASVspoof 2021 LA AUC, n=330   (integrity only)
  itw           G2  ITW AUC, n=600                (anti-regression)
                G3  ITW bona FPR@0.5, n=300       (anti-regression)
  codec_real    G4  codec pooled FPR@0.5, n=1000  (integrity only)
  dfeval_codec  G5  DF-Eval max AUC drop, 5 arms  (robustness)

Loader and head are the frozen A0z ones (exam_wa_audio_a0z.py): CPU only,
8 threads, torch.inference_mode(), model.eval(), 4.0 s @ 16 kHz mono,
librosa.load -> zero-pad/truncate 64,000 -> F.layer_norm, p_fake = softmax
index 0 (class 0 = fake). MPS is never used.

Decision rule: CUT = 0.5, p_fake > 0.5 -> issue fake, p_fake <= 0.5 -> issue
real. There is no abstention and no z-norm gate. `--threshold` and `--model`
are forbidden and abort the process.

Exit codes:
  0  exam complete (whatever the verdict), or summary written
  1  output dir holds a completed/partial exam; pass --resume or --supersede
  2  INCOMPLETE (fail-closed: a decode/score/transcode error occurred)
  3  config hash or manifest drift (resume refused)
  4  seal missing or invalid
  5  pre-scoring gate failed (class order, determinism, loader parity, CPU)

Usage:
  ./.venv/bin/python deepfake/scripts/exam_wa_audio_a0z_guards.py \
      --exam asv21la --out artifacts/wa_audio_a0z_guards/guards/asv21la
  ./.venv/bin/python deepfake/scripts/exam_wa_audio_a0z_guards.py \
      --summary --out artifacts/wa_audio_a0z_guards
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import resource
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import beta, norm
from sklearn.metrics import roc_auc_score

WS = Path(__file__).resolve().parents[2]
GUARDS_DIR = WS / "artifacts" / "wa_audio_a0z_guards"
SPEC = GUARDS_DIR / "SPEC.md"
SEAL = GUARDS_DIR / "seal.txt"
INPUT_HASHES = GUARDS_DIR / "input_hashes.txt"
READOUT = GUARDS_DIR / "read_k1.py"
READOUT_JSON = GUARDS_DIR / "k1_readout.json"
A0Z_OUT = WS / "artifacts" / "wa_audio_a0z_1b"
A0Z_SCORES = A0Z_OUT / "scores.jsonl"
A0Z_SCORES_SHA = (
    "fb6017e140c953af760ecb43e358ae778d8397bbeeba2c4a36c59c279e45d0a8"
)
DFEVAL_AUDIO = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data"
STEMS_CSV = WS / "data" / "dfeval" / "meta" / "matched_stems.csv"
SELECTION = WS / "artifacts" / "exam_audio_2021_shard330" / "selection.json"
SHARD_DIR = WS / "data" / "raw"
ITW_DIR = WS / "data" / "raw" / "itw" / "release_in_the_wild"
ITW_META = ITW_DIR / "meta.csv"
LA2019 = WS / "data" / "raw" / "asvspoof" / "data" / "train-00000-of-00001.parquet"

EXAMS = ("asv21la", "itw", "codec_real", "dfeval_codec")
CODEC_EXAMS = ("codec_real", "dfeval_codec")

CUT = 0.5
WINDOW_S = 4.0
SAMPLE_RATE = 16000
FAKE_INDEX = 0
CLEAN_DFEVAL_AUC = 0.778066

ARMS = {
    "wav": ["-c:a", "pcm_s16le", "-f", "wav", "-ar", "16000", "-ac", "1"],
    "mp3_cbr128": ["-c:a", "libmp3lame", "-b:a", "128k", "-f", "mp3"],
    "mp3_cbr192": ["-c:a", "libmp3lame", "-b:a", "192k", "-f", "mp3"],
    "flac": ["-c:a", "flac", "-f", "flac"],
    "opus128_ogg": ["-c:a", "libopus", "-b:a", "128k", "-f", "ogg"],
    "mp3_mono16k": ["-c:a", "libmp3lame", "-b:a", "128k", "-ac", "1",
                    "-ar", "16000", "-f", "mp3"],
}
CODEC_ARMS = [a for a in ARMS if a != "wav"]
EXT = {"mp3_cbr128": "mp3", "mp3_cbr192": "mp3", "flac": "flac",
       "opus128_ogg": "ogg", "mp3_mono16k": "mp3", "wav": "wav"}

MANIFEST_COLS = ["uid", "exam", "label", "label_name", "source", "arm",
                 "path", "sha256", "bytes", "orig_sr", "orig_duration_s",
                 "selection_rule"]

BAR_G1 = {"pass": 0.900, "hold": 0.880, "direction": ">="}
BAR_G2 = {"pass": 0.900, "hold": 0.880, "direction": ">="}
BAR_G3 = {"pass_max_fp": 0, "hold_max_fp": 3, "n": 300}
BAR_G4 = {"pass": 0.015, "hold": 0.025, "n": 1000}
BAR_G5 = {"pass": 0.05, "hold": 0.10, "n": 703}

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261115
Z95 = 1.959963984540054

sys.path.insert(0, str(WS / "deepfake" / "scripts"))
sys.path.insert(0, str(WS / "artifacts" / "wa_audio_pilot"))

import exam_wa_audio_a0z as A0Z  # noqa: E402


class Abort(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def canonical_sha(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(["git", "-C", str(WS), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as e:
        return f"unavailable: {e!r}"


class RunLog:
    def __init__(self, path: Path, append: bool = False, echo: bool = True):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "a" if append else "w")
        self.echo = echo

    def __call__(self, msg: str) -> None:
        line = f"{iso_now()} {msg}"
        self.f.write(line + "\n")
        self.f.flush()
        if self.echo:
            print(line, flush=True)

    def first_line(self, msg: str) -> None:
        self.f.write(msg + "\n")
        self.f.flush()
        if self.echo:
            print(msg, flush=True)

    def close(self) -> None:
        self.f.close()


def read_seal() -> dict | None:
    if not SEAL.exists():
        return None
    seal = {}
    for line in SEAL.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            seal[k.strip()] = v.strip()
    return seal


def verify_seal() -> str:
    seal = read_seal()
    if seal is None:
        raise Abort(4, f"seal.txt missing: {SEAL} (SPEC §11: no scoring "
                       "before the seal exists)")
    checks = {
        "spec_sha256": sha256_file(SPEC),
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "readout_sha256": sha256_file(READOUT),
        "input_hashes_sha256": sha256_file(INPUT_HASHES),
    }
    for key, actual in checks.items():
        if seal.get(key) != actual:
            raise Abort(4, f"seal mismatch {key}: seal={seal.get(key)} "
                           f"actual={actual} (SPEC §11.4: a change to a sealed "
                           "file voids the seal)")
    return sha256_file(SEAL)


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (lo, hi)


def compute_midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=np.float64)
    out[order] = t
    return out


def paired_delong(scores: np.ndarray, y: np.ndarray) -> dict:
    """Paired DeLong for k=2 score vectors on the same rows (report-only)."""
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    m, n = len(pos), len(neg)
    tx = np.empty((2, m), dtype=np.float64)
    ty = np.empty((2, n), dtype=np.float64)
    tz = np.empty((2, m + n), dtype=np.float64)
    for r in range(2):
        tx[r] = compute_midrank(scores[r][pos])
        ty[r] = compute_midrank(scores[r][neg])
        tz[r] = compute_midrank(scores[r])
    aucs = tz[:, pos].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, pos] - tx) / n
    v10 = 1.0 - (tz[:, neg] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    cov = sx / m + sy / n
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = math.sqrt(max(var, 0.0))
    z = diff / se if se > 0 else float("inf")
    p = 2.0 * (1.0 - float(norm.cdf(abs(z)))) if se > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "diff": diff, "se": se, "z": z, "p": p,
            "diff_ci95": [diff - Z95 * se, diff + Z95 * se]}


def source_meta(path: Path) -> dict:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return {"orig_sr": int(info.samplerate),
                "orig_duration_s": round(float(info.duration), 6)}
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,duration", "-of", "json",
             str(path)], capture_output=True, text=True, check=True)
        st = json.loads(out.stdout)["streams"][0]
        return {"orig_sr": int(st["sample_rate"]),
                "orig_duration_s": round(float(st.get("duration", 0.0)), 6)}
    except Exception:
        return {"orig_sr": None, "orig_duration_s": None}


def ffmpeg_version() -> str:
    out = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                         text=True, check=True)
    return out.stdout.splitlines()[0]


def transcode(src: Path, dst: Path, arm: str) -> None:
    res = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         *ARMS[arm], str(dst)], capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg {arm} failed on {src}: "
                           f"{res.stderr.strip()[:300]}")


def transcode_or_fail(src: Path, dst: Path, arm: str, uid: str,
                      log: RunLog) -> dict:
    """Fail-closed: a transform failure becomes an unscorable manifest row."""
    try:
        transcode(src, dst, arm)
        return {"sha256": sha256_file(dst), "bytes": dst.stat().st_size}
    except Exception as e:
        log(f"  TRANSCODE_FAIL {uid} {arm}: {e}")
        return {"sha256": "", "bytes": 0}


def manifest_row(**kw) -> dict:
    return {k: kw.get(k) for k in MANIFEST_COLS}


def write_manifest(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or list(rows[0]) != MANIFEST_COLS:
        raise Abort(3, f"manifest header drift: {path}")
    return rows


def clip_hash_aggregate(rows: list[dict]) -> str:
    lines = sorted(f"{r['uid']}\t{r['arm']}\t{r['sha256']}" for r in rows)
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ---------------------------------------------------------------- manifests

def build_asv21la(work: Path, log: RunLog,
                  limit: int | None = None) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    sel = json.loads(SELECTION.read_text())
    uids = sel["uids"]
    if len(uids) != 330 or len(set(uids)) != 330:
        raise Abort(3, f"selection.json uids {len(uids)} != 330 unique")
    digest = hashlib.sha256("\n".join(sorted(uids)).encode()).hexdigest()
    if digest != sel["sha256"]:
        raise Abort(3, f"selection list sha {digest} != sealed {sel['sha256']}")

    tabs, colsets = {}, {}
    for shard in (0, 13):
        p = SHARD_DIR / f"asvspoof21_la_test-{shard:05d}.parquet"
        tabs[shard] = pq.read_table(p, columns=["audio", "label"])
        labels = set(tabs[shard]["label"].to_pylist())
        colsets[shard] = labels
        want = 0 if shard == 0 else 1
        if labels != {want}:
            raise Abort(3, f"shard {shard:05d} parquet label column {labels} "
                           f"!= {{{want}}}")
    log(f"asv21la: shard 000 label={sorted(colsets[0])} "
        f"shard 013 label={sorted(colsets[13])}")

    src_dir = work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for uid in (uids if limit is None else uids[:limit]):
        m = re.match(r"^shard(000|013)_row(\d{6})$", uid)
        if not m:
            raise Abort(3, f"bad asv21la uid {uid!r}")
        shard, idx = int(m.group(1)), int(m.group(2))
        tab = tabs[shard]
        if idx >= len(tab):
            raise Abort(3, f"uid {uid} row {idx} >= shard rows {len(tab)}")
        want = 0 if shard == 0 else 1
        parquet_label = tab["label"][idx].as_py()
        if parquet_label != want:
            raise Abort(3, f"{uid}: parquet label {parquet_label} != {want}")
        rec = tab["audio"][idx].as_py()
        data = rec["bytes"] if isinstance(rec, dict) else rec
        if not data:
            raise Abort(2, f"{uid}: empty parquet audio bytes")
        path = src_dir / uid
        path.write_bytes(data)
        label = 1 - parquet_label
        meta = source_meta(path)
        rows.append(manifest_row(
            uid=uid, exam="asv21la", label=label,
            label_name="Fake" if label == 1 else "Real",
            source=f"asvspoof21_la_test-{shard:05d}.parquet#row{idx}",
            arm="native", path=str(path), sha256=sha256_file(path),
            bytes=path.stat().st_size, orig_sr=meta["orig_sr"],
            orig_duration_s=meta["orig_duration_s"],
            selection_rule="selection.json:330_uids_file_order"))
    counts = Counter(r["label"] for r in rows)
    if limit is None and (counts[0], counts[1]) != (173, 157):
        raise Abort(3, f"asv21la labels {dict(counts)} != 173 real / 157 fake")
    return rows, {"selection_sha256": sel["sha256"],
                  "selection_file_sha256": sha256_file(SELECTION),
                  "n": len(rows), "n_real": counts[0], "n_fake": counts[1],
                  "parquet_label_columns": {str(k): sorted(v)
                                            for k, v in colsets.items()},
                  "detector_label": "1 - parquet_label"}


def build_itw(work: Path, log: RunLog,
              limit: int | None = None) -> tuple[list[dict], dict]:
    meta_rows = list(csv.DictReader(open(ITW_META, newline="")))
    if list(meta_rows[0]) != ["file", "speaker", "label"]:
        raise Abort(3, f"ITW meta columns {list(meta_rows[0])}")
    rng = random.Random(42)
    by_class = defaultdict(lambda: defaultdict(list))
    for r in meta_rows:
        by_class[r["label"]][r["speaker"]].append(r["file"])
    picked = []
    for lab in sorted(by_class):
        per_speaker = []
        for spk in sorted(by_class[lab]):
            files = sorted(by_class[lab][spk])
            rng.shuffle(files)
            per_speaker.extend((f, spk) for f in files[:15])
        rng.shuffle(per_speaker)
        take = per_speaker[:300]
        picked.extend((f, lab, spk) for f, spk in take)
    rng.shuffle(picked)

    if len(picked) != 600:
        raise Abort(3, f"ITW picked {len(picked)} != 600")
    counts = Counter(lab for _, lab, _ in picked)
    if (counts["bona-fide"], counts["spoof"]) != (300, 300):
        raise Abort(3, f"ITW class counts {dict(counts)} != 300/300")
    if len({f for f, _, _ in picked}) != 600:
        raise Abort(3, "ITW duplicate file in selection")
    per_spk = Counter(spk for _, _, spk in picked)
    per_spk_class = Counter((spk, lab) for _, lab, spk in picked)
    if max(per_spk_class.values()) > 15:
        raise Abort(3, "ITW speaker cap breached: "
                       f"{per_spk_class.most_common(1)}")

    rows = []
    for fname, lab, spk in (picked if limit is None else picked[:limit]):
        path = ITW_DIR / fname
        if not path.exists():
            raise Abort(2, f"ITW file missing: {path}")
        label = 0 if lab == "bona-fide" else 1
        meta = source_meta(path)
        rows.append(manifest_row(
            uid=fname, exam="itw", label=label,
            label_name="Fake" if label == 1 else "Real",
            source=f"itw/release_in_the_wild/{fname}", arm="native",
            path=str(path), sha256=sha256_file(path),
            bytes=path.stat().st_size, orig_sr=meta["orig_sr"],
            orig_duration_s=meta["orig_duration_s"],
            selection_rule="exam_audio_itw.py:per_class=300,"
                           "speaker_cap=15,seed=42"))

    picked_lines = "\n".join(f"{f}\t{lab}\t{spk}" for f, lab, spk in picked)
    sel_sha = hashlib.sha256(picked_lines.encode()).hexdigest()
    log(f"itw: 600 picked, 300/300, max per (speaker,class) "
        f"{max(per_spk_class.values())}, max per speaker {max(per_spk.values())}, "
        f"selection sha {sel_sha[:16]}…")
    return rows, {"meta_sha256": sha256_file(ITW_META),
                  "selected_set_sha256": sel_sha,
                  "selected_set_sha256_definition":
                      "sha256 of 'file\\tlabel\\tspeaker' lines in selection "
                      "order (after the final rng.shuffle)",
                  "n": len(rows), "n_real": 300, "n_fake": 300,
                  "per_class": 300, "speaker_cap": 15, "seed": 42,
                  "max_per_speaker_class": max(per_spk_class.values()),
                  "max_per_speaker_total": max(per_spk.values()),
                  "speaker_cap_semantics":
                      "applied per (class, speaker), as in "
                      "exam_audio_itw.py; every one of the 54 ITW speakers "
                      "carries both labels, so a total-per-speaker cap of 15 "
                      "is not reachable by the sealed reference algorithm"}


def build_codec_real(work: Path, log: RunLog,
                     limit: int | None = None) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq

    tab = pq.read_table(LA2019, columns=["system_id", "audio"])
    sysids = tab["system_id"].to_pylist()
    bona = [i for i, v in enumerate(sysids) if v == "-"]
    if len(bona) != 2580:
        raise Abort(3, f"2019 LA bona indices {len(bona)} != 2580")
    picked = random.Random(31).sample(bona, 200)
    if len(set(picked)) != 200:
        raise Abort(3, f"codec_real sample not unique: {len(set(picked))}")

    src_dir = work / "src"
    src_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in (picked if limit is None else picked[:limit]):
        uid = f"asv19la_row{idx:06d}"
        rec = tab["audio"][idx].as_py()
        data = rec["bytes"] if isinstance(rec, dict) else rec
        if not data:
            raise Abort(2, f"{uid}: empty parquet audio bytes")
        src = src_dir / uid
        src.write_bytes(data)
        meta = source_meta(src)
        for arm in ARMS:
            dst = work / f"{uid}_{arm}.{EXT[arm]}"
            built = transcode_or_fail(src, dst, arm, uid, log)
            rows.append(manifest_row(
                uid=uid, exam="codec_real", label=0, label_name="Real",
                source=f"asvspoof/train-00000-of-00001.parquet#idx{idx}",
                arm=arm, path=str(dst), sha256=built["sha256"],
                bytes=built["bytes"], orig_sr=meta["orig_sr"],
                orig_duration_s=meta["orig_duration_s"],
                selection_rule='system_id=="-" then '
                               "random.Random(31).sample(bona,200)"))
        log(f"codec_real: {uid} transcoded x{len(ARMS)}")
    log(f"codec_real: {len(rows)} manifest rows, 200 unique indices")
    return rows, {"shard_sha256": sha256_file(LA2019),
                  "n_indices": 200, "n_bona_pool": len(bona), "seed": 31,
                  "arms": list(ARMS), "ffmpeg": ffmpeg_version(),
                  "transcode": "full-length source, then first 4 s loaded; "
                               "no loudness normalisation"}


def build_dfeval_codec(work: Path, log: RunLog,
                       limit: int | None = None) -> tuple[list[dict], dict]:
    rows_csv = list(csv.DictReader(open(STEMS_CSV, newline="")))
    if list(rows_csv[0]) != ["stem", "label"]:
        raise Abort(3, f"matched_stems columns {list(rows_csv[0])}")
    stems = [r["stem"] for r in rows_csv]
    if len(stems) != len(set(stems)):
        raise Abort(3, "duplicate stems in matched_stems.csv")
    present, missing = [], []
    for r in rows_csv:
        label = r["label"]
        if label not in ("Real", "Fake"):
            raise Abort(3, f"bad label {label!r} for {r['stem']}")
        p = DFEVAL_AUDIO / f"{r['stem']}.mp3"
        rec = {"stem": r["stem"], "label": 0 if label == "Real" else 1,
               "label_name": label, "path": p}
        (present if p.exists() else missing).append(rec)
    unlisted = sorted({p.stem for p in DFEVAL_AUDIO.glob("*.mp3")}
                      - {r["stem"] for r in rows_csv})
    if unlisted:
        raise Abort(3, f"{len(unlisted)} mp3s lack labels")
    present.sort(key=lambda r: r["stem"])
    if len(present) != 703:
        raise Abort(3, f"DF-Eval present {len(present)} != 703")
    n_real = sum(r["label"] == 0 for r in present)
    n_fake = sum(r["label"] == 1 for r in present)
    if (n_real, n_fake) != (395, 308):
        raise Abort(3, f"DF-Eval {n_real}/{n_fake} != 395/308")
    if len(missing) != 74:
        raise Abort(3, f"DF-Eval missing {len(missing)} != 74")
    if (sum(r["label"] == 0 for r in missing),
            sum(r["label"] == 1 for r in missing)) != (56, 18):
        raise Abort(3, "DF-Eval missing split != 56 real / 18 fake")

    rows = []
    for rec in (present if limit is None else present[:limit]):
        src = rec["path"]
        meta = source_meta(src)
        for arm in ARMS:
            dst = work / f"{rec['stem']}_{arm}.{EXT[arm]}"
            built = transcode_or_fail(src, dst, arm, rec["stem"], log)
            rows.append(manifest_row(
                uid=rec["stem"], exam="dfeval_codec", label=rec["label"],
                label_name=rec["label_name"],
                source=f"dfeval/audio/audio-data/audio-data/{rec['stem']}.mp3",
                arm=arm, path=str(dst), sha256=built["sha256"],
                bytes=built["bytes"], orig_sr=meta["orig_sr"],
                orig_duration_s=meta["orig_duration_s"],
                selection_rule="matched_stems.csv:703_present_stems"))
        log(f"dfeval_codec: {rec['stem']} transcoded x{len(ARMS)}")
    log(f"dfeval_codec: {len(rows)} manifest rows "
        f"(703 stems x {len(ARMS)} arms)")
    return rows, {"labels_sha256": sha256_file(STEMS_CSV),
                  "n_stems": 703, "n_real": 395, "n_fake": 308,
                  "n_missing": 74, "n_missing_real": 56,
                  "n_missing_fake": 18, "arms": list(ARMS),
                  "ffmpeg": ffmpeg_version(),
                  "transcode": "full-length source, then first 4 s loaded; "
                               "no loudness normalisation"}


BUILDERS = {"asv21la": build_asv21la, "itw": build_itw,
            "codec_real": build_codec_real, "dfeval_codec": build_dfeval_codec}


# ------------------------------------------------------------ pre-score gates

def class_order_gate(log: RunLog) -> dict:
    if sha256_file(A0Z_SCORES) != A0Z_SCORES_SHA:
        raise Abort(5, "A0z scores.jsonl sha differs from the sealed input")
    rows = [json.loads(line) for line in
            A0Z_SCORES.read_text().splitlines() if line.strip()]
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    p = np.array([r["p_fake"] for r in rows], dtype=np.float64)
    z = np.array([r["fake_logit"] - r["real_logit"] for r in rows],
                 dtype=np.float64)
    auc_p = float(roc_auc_score(y, p))
    auc_z = float(roc_auc_score(y, z))
    if abs(auc_p - auc_z) > 1e-9:
        raise Abort(5, f"class-order gate: AUC(p_fake)={auc_p} != "
                       f"AUC(fake-real)={auc_z}")
    dev = 0.0
    for r in rows:
        f, rr = r["fake_logit"], r["real_logit"]
        soft = 1.0 / (1.0 + math.exp(rr - f))
        dev = max(dev, abs(soft - r["p_fake"]))
    if dev > 1e-6:
        raise Abort(5, f"class-order gate: softmax(index 0) deviates {dev} "
                       "from stored p_fake")
    log(f"class-order gate: AUC(p_fake)={auc_p:.9f} AUC(fake-real)={auc_z:.9f} "
        f"diff=0.0 max softmax dev={dev:.2e}")
    return {"auc_p_fake": auc_p, "auc_logit_diff": auc_z,
            "abs_diff": abs(auc_p - auc_z),
            "max_softmax_deviation": dev, "n": len(rows),
            "scores_sha256": A0Z_SCORES_SHA}


def load_model_gate(threads: int, log: RunLog) -> dict:
    t0 = time.perf_counter()
    m = A0Z.load_model(threads)
    if tuple(m["w_fc"].shape) != (2, 1280):
        raise Abort(5, f"proj_fc shape {tuple(m['w_fc'].shape)} != (2,1280)")
    for name, prm in m["model"].named_parameters():
        if prm.device.type != "cpu":
            raise Abort(5, f"non-CPU parameter {name} on {prm.device}")
    if torch.backends.mps.is_available():
        log("mps: available but UNUSED (CPU-only per SPEC §2)")
    log(f"model loaded in {time.perf_counter()-t0:.1f}s: "
        f"repo={m['repo_revision']} weights={m['weights_bytes']} "
        f"sha={m['weights_sha256']} fc={m['fc_shape']} "
        f"mapped={m['mapped']} skipped={m['skipped']}")
    return m


def loader_parity_gate(manifest: list[dict], exam: str, work: Path,
                       log: RunLog) -> dict:
    from timing_pilot import load_audio as pilot_load

    items = []

    def check(name: str, path: Path) -> None:
        a = pilot_load(path, WINDOW_S)
        b, _ = A0Z.load_audio_meta(path, WINDOW_S)
        same = bool(torch.equal(a, b))
        items.append({"check": name, "path": str(path),
                      "bit_identical": same, "shape": list(b.shape)})
        if not same:
            raise Abort(5, f"loader parity FAIL on {name}: {path}")

    check(f"{exam}:manifest[0]", Path(manifest[0]["path"]))

    import pyarrow.parquet as pq

    sel = json.loads(SELECTION.read_text())
    uid = sel["uids"][0]
    shard = int(re.match(r"^shard(\d{3})_row", uid).group(1))
    idx = int(re.match(r"^shard\d{3}_row(\d{6})$", uid).group(1))
    tab = pq.read_table(SHARD_DIR / f"asvspoof21_la_test-{shard:05d}.parquet",
                        columns=["audio"])
    rec = tab["audio"][idx].as_py()
    data = rec["bytes"] if isinstance(rec, dict) else rec
    pq_path = work / f"parity_{uid}"
    pq_path.write_bytes(data)
    check("parquet-decoded clip", pq_path)

    wav_rows = [r for r in manifest if r["arm"] == "wav"]
    if wav_rows:
        check("transcoded wav control", Path(wav_rows[0]["path"]))
    else:
        src = Path(manifest[0]["path"])
        wav_path = work / "parity_control.wav"
        transcode(src, wav_path, "wav")
        check("transcoded wav control", wav_path)

    log(f"loader parity gate: {len(items)} checks, all bit-identical")
    return {"checks": items, "all_bit_identical": True}


def determinism_gate(model, w_fc, b_fc, manifest: list[dict],
                     log: RunLog) -> dict:
    import torch.nn.functional as Fn

    sample = manifest[:20]
    deltas, first = [], []
    with torch.inference_mode():
        for r in sample:
            x, _ = A0Z.load_audio_meta(Path(r["path"]), WINDOW_S)
            emb = model(x).last_hidden_state
            pooled = Fn.adaptive_avg_pool1d(emb.transpose(1, 2), 1).squeeze(-1)
            p1 = float(Fn.softmax(Fn.linear(pooled, w_fc, b_fc).float(),
                                  dim=1)[0, FAKE_INDEX])
            x2, _ = A0Z.load_audio_meta(Path(r["path"]), WINDOW_S)
            emb2 = model(x2).last_hidden_state
            pooled2 = Fn.adaptive_avg_pool1d(emb2.transpose(1, 2), 1).squeeze(-1)
            p2 = float(Fn.softmax(Fn.linear(pooled2, w_fc, b_fc).float(),
                                  dim=1)[0, FAKE_INDEX])
            first.append(p1)
            deltas.append(abs(p1 - p2))
    worst = max(deltas) if deltas else 0.0
    if worst >= 1e-9:
        raise Abort(5, f"determinism gate: max |delta p_fake| = {worst:.3e} "
                       ">= 1e-9")
    log(f"determinism gate: {len(sample)} clips x2, max |delta p_fake|="
        f"{worst:.3e} < 1e-9")
    return {"n_clips": len(sample), "max_abs_delta": worst,
            "threshold": 1e-9, "pass": True,
            "first_pass_p_fake": [round(v, 10) for v in first]}


# ------------------------------------------------------------------ scoring

def score_row(model, w_fc, b_fc, r: dict, config_sha: str) -> dict:
    base = {"uid": r["uid"], "exam": r["exam"], "arm": r["arm"],
            "label": int(r["label"]), "label_name": r["label_name"],
            "path": r["path"], "sha256": r["sha256"],
            "config_sha256": config_sha, "scored_at": iso_now()}
    t0 = time.perf_counter()
    try:
        x, meta = A0Z.load_audio_meta(Path(r["path"]), WINDOW_S)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        emb = model(x).last_hidden_state
        pooled = F.adaptive_avg_pool1d(emb.transpose(1, 2), 1).squeeze(-1)
        logits = F.linear(pooled, w_fc, b_fc)
        t_fwd = time.perf_counter() - t1
        if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
            raise RuntimeError(f"bad logits {tuple(logits.shape)}")
        probs = F.softmax(logits.float(), dim=1)
        if not torch.isfinite(probs).all():
            raise RuntimeError("non-finite softmax")
        return {**base,
                "n_loaded": meta["n_loaded"], "n_samples": meta["n_samples"],
                "padded": meta["padded"],
                "loaded_sr": meta["orig_sr"],
                "loaded_duration_s": meta["orig_duration_s"],
                "fake_logit": round(float(logits[0, A0Z.FAKE_INDEX]), 8),
                "real_logit": round(float(logits[0, A0Z.REAL_INDEX]), 8),
                "p_fake": round(float(probs[0, A0Z.FAKE_INDEX]), 10),
                "p_real": round(float(probs[0, A0Z.REAL_INDEX]), 10),
                "load_s": round(t_load, 6), "forward_s": round(t_fwd, 6),
                "error": None}
    except Exception as e:
        return {**base, "n_loaded": None, "n_samples": None, "padded": None,
                "loaded_sr": None, "loaded_duration_s": None,
                "fake_logit": None, "real_logit": None, "p_fake": None,
                "p_real": None, "load_s": None, "forward_s": None,
                "error": repr(e)}


def load_existing_scores(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise Abort(3, f"scores.jsonl line {i} is not JSON: {e}")
    return rows


# ------------------------------------------------------------------ metrics

def bar_auc(value: float, bar: dict) -> str:
    if value >= bar["pass"]:
        return "PASS"
    if value >= bar["hold"]:
        return "HOLD"
    return "FAIL"


def metrics_asv21la(rows: list[dict]) -> tuple[dict, dict]:
    ok = [r for r in rows if r["error"] is None]
    y = np.array([r["label"] for r in ok], dtype=np.int64)
    p = np.array([r["p_fake"] for r in ok], dtype=np.float64)
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None
    fp = int(((y == 0) & (p > CUT)).sum())
    tp = int(((y == 1) & (p > CUT)).sum())
    n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
    metrics = {
        "guard": "G1", "n": len(ok), "n_real": n_real, "n_fake": n_fake,
        "auc": None if auc is None else round(auc, 6),
        "fpr_at_0.5_report_only": round(fp / n_real, 6) if n_real else None,
        "tpr_at_0.5_report_only": round(tp / n_fake, 6) if n_fake else None,
        "abstention": "none; every decoded clip is issued",
    }
    verdict = {
        "guard": "G1", "metric": "ASVspoof 2021 LA AUC", "n": len(ok),
        "value": metrics["auc"], "bar": ">= 0.900 (HOLD 0.880-0.899)",
        "verdict": "INCOMPLETE" if auc is None else bar_auc(auc, BAR_G1),
        "epistemic_status": "integrity only (corpus is in-training)",
        "note": "a G1 FAIL triggers a parity audit only, never a score rescue "
                "(SPEC §6 A-3, §8 step 5)",
    }
    return metrics, verdict


def metrics_itw(rows: list[dict]) -> tuple[dict, dict]:
    ok = [r for r in rows if r["error"] is None]
    y = np.array([r["label"] for r in ok], dtype=np.int64)
    p = np.array([r["p_fake"] for r in ok], dtype=np.float64)
    auc = float(roc_auc_score(y, p)) if len(set(y.tolist())) == 2 else None
    bona = p[y == 0]
    n_bona = int(len(bona))
    fp = int((bona > CUT).sum())
    cp = clopper_pearson(fp, n_bona)
    w = wilson(fp, n_bona)
    metrics = {
        "guard": "G2+G3", "n": len(ok), "n_real": n_bona,
        "n_fake": int((y == 1).sum()),
        "auc": None if auc is None else round(auc, 6),
        "g3_fp": fp, "g3_n_bona": n_bona,
        "g3_fpr": round(fp / n_bona, 6) if n_bona else None,
        "g3_wilson95": [round(v, 6) for v in w],
        "g3_clopper_pearson95": [round(v, 6) for v in cp],
        "abstention": "none; every decoded clip is issued",
    }
    g2 = ("INCOMPLETE" if auc is None else bar_auc(auc, BAR_G2))
    if n_bona != 300:
        g3 = "INCOMPLETE"
    elif fp <= BAR_G3["pass_max_fp"]:
        g3 = "PASS"
    elif fp <= BAR_G3["hold_max_fp"]:
        g3 = "HOLD"
    else:
        g3 = "FAIL"
    verdict = {
        "guard": "G2+G3",
        "g2": {"metric": "ITW AUC", "n": len(ok),
               "value": metrics["auc"],
               "bar": ">= 0.900 (HOLD 0.880-0.899)", "verdict": g2,
               "epistemic_status": "absolute anti-regression"},
        "g3": {"metric": "ITW bona FPR@0.5", "n": n_bona, "fp": fp,
               "value": metrics["g3_fpr"],
               "bar": "0/300 PASS, 1-3/300 HOLD, >=4/300 FAIL",
               "verdict": g3,
               "clopper_pearson95": [round(v, 6) for v in cp],
               "note": "count bands bind (A-2); the CP upper is report-only "
                       "and cannot change the verdict",
               "epistemic_status": "absolute anti-regression"},
        "verdict": ("INCOMPLETE" if "INCOMPLETE" in (g2, g3)
                    else "FAIL" if "FAIL" in (g2, g3)
                    else "HOLD" if "HOLD" in (g2, g3) else "PASS"),
    }
    return metrics, verdict


def metrics_codec_real(rows: list[dict]) -> tuple[dict, dict]:
    ok = [r for r in rows if r["error"] is None]
    codec = [r for r in ok if r["arm"] in CODEC_ARMS]
    fp = sum(1 for r in codec if r["p_fake"] > CUT)
    n = len(codec)
    fpr = fp / n if n else None
    by_uid = defaultdict(list)
    for r in codec:
        by_uid[r["uid"]].append(r)
    uids = sorted(by_uid)
    fp_u = np.array([sum(1 for r in by_uid[u] if r["p_fake"] > CUT)
                     for u in uids], dtype=np.float64)
    n_u = np.array([len(by_uid[u]) for u in uids], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    vals = np.empty(BOOTSTRAP_N, dtype=np.float64)
    for i in range(BOOTSTRAP_N):
        idx = rng.choice(len(uids), size=len(uids), replace=True)
        vals[i] = fp_u[idx].sum() / n_u[idx].sum()
    lo, hi = np.percentile(vals, [2.5, 97.5])
    per_arm = {}
    for arm in ARMS:
        sub = [r for r in ok if r["arm"] == arm]
        a_fp = sum(1 for r in sub if r["p_fake"] > CUT)
        per_arm[arm] = {"n": len(sub), "fp": a_fp,
                        "fpr": round(a_fp / len(sub), 6) if sub else None}
    metrics = {
        "guard": "G4", "n": n, "fp": fp,
        "fpr": None if fpr is None else round(fpr, 6),
        "fpr_fraction": f"{fp}/{n}",
        "wilson95": [round(v, 6) for v in wilson(fp, n)],
        "clopper_pearson95": [round(v, 6) for v in clopper_pearson(fp, n)],
        "cluster_bootstrap": {
            "n_boot": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED,
            "cluster": "utterance (200 clusters, 5 codec arms each)",
            "fpr_ci95": [round(float(lo), 6), round(float(hi), 6)],
            "fpr_boot_mean": round(float(vals.mean()), 6),
            "fpr_boot_sd": round(float(vals.std(ddof=1)), 6)},
        "per_arm": per_arm,
        "wav_control": per_arm["wav"],
        "pooled_arms": CODEC_ARMS,
        "abstention": "none; every decoded clip is issued",
    }
    if n != BAR_G4["n"]:
        v = "INCOMPLETE"
    elif fpr <= BAR_G4["pass"]:
        v = "PASS"
    elif fpr <= BAR_G4["hold"]:
        v = "HOLD"
    else:
        v = "FAIL"
    verdict = {
        "guard": "G4", "metric": "codec pooled FPR@0.5", "n": n, "fp": fp,
        "value": metrics["fpr"],
        "bar": "<= 1.5% PASS, >1.5-2.5% HOLD, >2.5% FAIL",
        "verdict": v,
        "epistemic_status": "integrity only (real source is in-training)",
        "note": "the <=1.5% bar cannot be met on any non-training real corpus "
                "in this repo (A-4); retained verbatim as a fit check",
    }
    return metrics, verdict


def metrics_dfeval_codec(rows: list[dict], clean: dict) -> tuple[dict, dict]:
    ok = [r for r in rows if r["error"] is None]
    per_arm, delong = {}, {}
    for arm in ARMS:
        sub = [r for r in ok if r["arm"] == arm]
        sub.sort(key=lambda r: r["uid"])
        if len(sub) != BAR_G5["n"]:
            per_arm[arm] = {"n": len(sub), "auc": None, "drop": None}
            continue
        y = np.array([r["label"] for r in sub], dtype=np.int64)
        p = np.array([r["p_fake"] for r in sub], dtype=np.float64)
        auc = float(roc_auc_score(y, p))
        drop = CLEAN_DFEVAL_AUC - auc
        per_arm[arm] = {"n": len(sub), "auc": round(auc, 6),
                        "drop": round(drop, 6),
                        "fpr_at_0.5_report_only": round(float(
                            ((y == 0) & (p > CUT)).sum()) / int((y == 0).sum()),
                            6),
                        "tpr_at_0.5_report_only": round(float(
                            ((y == 1) & (p > CUT)).sum()) / int((y == 1).sum()),
                            6)}
        missing = [r["uid"] for r in sub if r["uid"] not in clean]
        if not missing and len(sub) == len(clean):
            clean_p = np.array([clean[r["uid"]] for r in sub],
                               dtype=np.float64)
            d = paired_delong(np.vstack([p, clean_p]), y)
            delong[arm] = {"auc_arm": round(d["auc_a"], 6),
                           "auc_clean": round(d["auc_b"], 6),
                           "diff": round(d["diff"], 6),
                           "diff_ci95": [round(v, 6) for v in d["diff_ci95"]],
                           "z": round(d["z"], 4), "p": round(d["p"], 6),
                           "note": "paired DeLong vs clean DF-Eval scores; "
                                   "report-only"}
    codec_aucs = [per_arm[a]["auc"] for a in CODEC_ARMS
                  if per_arm[a]["auc"] is not None]
    max_drop = (max(CLEAN_DFEVAL_AUC - a for a in codec_aucs)
                if codec_aucs else None)
    metrics = {
        "guard": "G5", "n_per_arm": BAR_G5["n"],
        "clean_auc": CLEAN_DFEVAL_AUC,
        "per_arm": per_arm, "max_drop": None if max_drop is None
        else round(max_drop, 6),
        "max_drop_arms": CODEC_ARMS,
        "wav_control_excluded_from_max": True,
        "paired_delong_vs_clean": delong,
        "abstention": "none; every decoded clip is issued",
    }
    if max_drop is None:
        v = "INCOMPLETE"
    elif max_drop <= BAR_G5["pass"]:
        v = "PASS"
    elif max_drop <= BAR_G5["hold"]:
        v = "HOLD"
    else:
        v = "FAIL"
    verdict = {
        "guard": "G5", "metric": "max DF-Eval AUC drop under codec",
        "value": metrics["max_drop"], "n_per_arm": BAR_G5["n"],
        "bar": "<= 0.05 PASS, >0.05-0.10 HOLD, >0.10 FAIL",
        "verdict": v, "wav_control": per_arm["wav"],
        "wav_control_note": "reported and excluded from the max; it isolates "
                            "re-decode from codec damage",
        "epistemic_status": "robustness, on the only clean exam",
    }
    return metrics, verdict


METRIC_BUILDERS = {"asv21la": metrics_asv21la, "itw": metrics_itw,
                   "codec_real": metrics_codec_real,
                   "dfeval_codec": metrics_dfeval_codec}


def load_clean_dfeval() -> dict:
    rows = [json.loads(line) for line in
            A0Z_SCORES.read_text().splitlines() if line.strip()]
    return {r["stem"]: float(r["p_fake"]) for r in rows
            if r.get("error") is None}


# --------------------------------------------------------------------- main

def run_exam(args) -> int:
    exam = args.exam
    out = Path(args.out)
    dry = bool(args.dry_run)
    work = out / "work"
    out.mkdir(parents=True, exist_ok=True)

    completed = False
    if (out / "verdict.json").exists():
        try:
            v = json.loads((out / "verdict.json").read_text())
            completed = v.get("status") == "COMPLETE" or v.get("complete") is True
        except Exception:
            completed = False
    existing_scores = load_existing_scores(out / "scores.jsonl")

    if completed and not args.supersede:
        print(f"REFUSING: {out} holds a completed exam; pass --supersede",
              file=sys.stderr)
        return 1
    if existing_scores and not args.resume and not args.supersede:
        print(f"REFUSING: {out}/scores.jsonl has {len(existing_scores)} rows; "
              "pass --resume or --supersede", file=sys.stderr)
        return 1
    if args.supersede and any(out.iterdir()):
        dest = out / f"superseded_{utc_stamp()}"
        dest.mkdir(parents=True, exist_ok=True)
        for item in list(out.iterdir()):
            if item.name.startswith("superseded_"):
                continue
            shutil.move(str(item), str(dest / item.name))
        existing_scores = []
        completed = False

    log = RunLog(out / "run.log", append=bool(args.resume and not args.supersede))
    try:
        if dry:
            seal_sha = "ABSENT_PRE_SEAL_DRY_RUN"
            if SEAL.exists():
                try:
                    seal_sha = verify_seal()
                except Abort as e:
                    log(f"dry-run seal check: {e}")
            log.first_line(f"seal_sha256={seal_sha}")
            log("mode=dry-run (pre-seal validation; no receipt is claimed)")
        else:
            seal_sha = verify_seal()
            log.first_line(f"seal_sha256={seal_sha}")
        log(f"exam={exam} out={out} threads={args.threads} dry_run={dry} "
            f"resume={args.resume} supersede={args.supersede}")
        log(f"spec={sha256_file(SPEC)} harness={sha256_file(Path(__file__).resolve())}")

        t_start = time.perf_counter()
        work.mkdir(parents=True, exist_ok=True)

        manifest_path = out / "manifest.csv"
        meta_path = out / "manifest_meta.json"
        if manifest_path.exists() and meta_path.exists() and not args.supersede:
            manifest = read_manifest(manifest_path)
            build_meta = json.loads(meta_path.read_text())
            log(f"manifest reused: {len(manifest)} rows "
                f"sha256={sha256_file(manifest_path)}")
        else:
            manifest, build_meta = BUILDERS[exam](
                work, log, 10 if dry else None)
            write_manifest(manifest_path, manifest)
            meta_path.write_text(json.dumps(build_meta, indent=2) + "\n")
            log(f"manifest written: {len(manifest)} rows "
                f"sha256={sha256_file(manifest_path)}")

        manifest_sha = sha256_file(manifest_path)
        missing_files = [r for r in manifest
                         if r["sha256"] and not Path(r["path"]).exists()]
        if missing_files:
            raise Abort(3, f"{len(missing_files)} manifest files missing, e.g. "
                           f"{missing_files[0]['path']}")

        arms = sorted({r["arm"] for r in manifest})
        config = {
            "repo": A0Z.REPO, "revision": None, "weights_sha256": None,
            "window_s": WINDOW_S, "sample_rate": SAMPLE_RATE,
            "fake_index": FAKE_INDEX, "cut": CUT, "exam": exam,
            "arms": arms, "manifest_sha256": manifest_sha,
            "code_sha256": sha256_file(Path(__file__).resolve()),
        }
        model_bundle = load_model_gate(args.threads, log)
        config["revision"] = model_bundle["repo_revision"]
        config["weights_sha256"] = model_bundle["weights_sha256"]
        config_sha = canonical_sha(config)
        if existing_scores:
            recorded = {r.get("config_sha256") for r in existing_scores}
            if len(recorded) != 1:
                raise Abort(3, f"scores.jsonl carries {len(recorded)} config "
                               "hashes; refusing to resume")
            rec = recorded.pop()
            if rec != config_sha:
                raise Abort(3, "config hash mismatch: existing rows "
                               f"{rec} != current {config_sha}")

        order = class_order_gate(log)
        gate_rows = [r for r in manifest if r["sha256"]]
        if not gate_rows:
            raise Abort(2, "no scorable manifest rows (all transforms failed)")
        parity = loader_parity_gate(gate_rows, exam, work, log)
        determinism = determinism_gate(model_bundle["model"],
                                       model_bundle["w_fc"],
                                       model_bundle["b_fc"], gate_rows, log)

        done = {(r["uid"], r["arm"], r["sha256"]) for r in existing_scores}
        todo = [r for r in manifest
                if (r["uid"], r["arm"], r["sha256"]) not in done]
        if dry:
            todo = todo[:60]
            log(f"dry-run: scoring at most 60 rows ({len(todo)} selected; "
                f"<=10 source clips)")
        log(f"scoring plan: {len(todo)} rows "
            f"({len(manifest) - len(todo)} already present)")

        scores_path = out / "scores.jsonl"
        n_err = 0
        t_score = time.perf_counter()
        with open(scores_path, "a") as f, torch.inference_mode():
            for i, r in enumerate(todo, 1):
                row = score_row(model_bundle["model"], model_bundle["w_fc"],
                                model_bundle["b_fc"], r, config_sha)
                f.write(json.dumps(row, sort_keys=True) + "\n")
                f.flush()
                if row["error"] is not None:
                    n_err += 1
                    if n_err <= 10:
                        log(f"  error {row['uid']} {row['arm']}: "
                            f"{row['error']}")
                if i % 25 == 0 or i == len(todo):
                    rate = i / max(time.perf_counter() - t_score, 1e-9)
                    log(f"  {i}/{len(todo)} scored ({rate:.2f} rows/s, "
                        f"errors={n_err})")

        rows = load_existing_scores(scores_path)
        if dry:
            rows = rows[-len(todo):] if todo else rows
        n_manifest = len(manifest)
        scored_keys = {(r["uid"], r["arm"], r["sha256"]) for r in rows}
        missing_rows = [r for r in manifest
                        if (r["uid"], r["arm"], r["sha256"]) not in scored_keys]
        err_rows = [r for r in rows if r.get("error") is not None]
        complete = (not missing_rows and not err_rows
                    and len(rows) == n_manifest)
        if dry:
            complete = False

        clean = load_clean_dfeval() if exam == "dfeval_codec" else None
        if exam == "dfeval_codec":
            metrics, verdict = metrics_dfeval_codec(rows, clean)
        else:
            metrics, verdict = METRIC_BUILDERS[exam](rows)

        status = "COMPLETE" if complete else "INCOMPLETE"
        if dry:
            status = "DRY_RUN"
        metrics_out = {
            "exam": exam, "guard": metrics.get("guard"),
            "created_at": iso_now(), "status": status,
            "dry_run": dry, "seal_sha256": seal_sha,
            "decision_rule": "p_fake > 0.5 -> issue fake; <= 0.5 -> issue real",
            "cut": CUT, "abstention": "none",
            "manifest": {
                "path": str(manifest_path.relative_to(WS))
                if manifest_path.is_relative_to(WS) else str(manifest_path),
                "sha256": manifest_sha, "rows": len(manifest),
                "clip_hashes_sha256": clip_hash_aggregate(manifest),
                "build_meta": build_meta},
            "scored": {
                "n_rows": len(rows), "n_manifest": n_manifest,
                "n_missing": len(missing_rows), "n_errors": len(err_rows),
                "errors": err_rows[:50], "complete": complete,
                "scores_jsonl": str(scores_path),
                "scores_sha256": sha256_file(scores_path)
                if scores_path.exists() else None},
            "config": config, "config_sha256": config_sha,
            "gates": {"class_order": order, "loader_parity": parity,
                      "determinism": determinism},
            "metrics": metrics,
            "model": {
                "repo": A0Z.REPO, "revision": model_bundle["repo_revision"],
                "weights_sha256": model_bundle["weights_sha256"],
                "weights_bytes": model_bundle["weights_bytes"],
                "hidden_size": model_bundle["hidden_size"],
                "num_hidden_layers": model_bundle["num_hidden_layers"],
                "fc_shape": model_bundle["fc_shape"],
                "mapped_keys": model_bundle["mapped"],
                "skipped_keys": model_bundle["skipped"],
                "fake_index": A0Z.FAKE_INDEX, "real_index": A0Z.REAL_INDEX,
                "class_order": {"0": "fake", "1": "real"}},
            "preprocessing": {
                "loader": "librosa.load(sr=16000, mono=True, duration=4.0)",
                "pad": "zero-pad to 64000 if shorter; truncate to 64000",
                "normalization": "F.layer_norm(waveform, waveform.shape)",
                "window_s": WINDOW_S, "sample_rate": SAMPLE_RATE,
                "device": "cpu", "threads": int(torch.get_num_threads())},
            "bars": {"G1": BAR_G1, "G2": BAR_G2, "G3": BAR_G3,
                     "G4": BAR_G4, "G5": BAR_G5},
            "versions": {
                "torch": torch.__version__, "numpy": np.__version__,
                "librosa": __import__("librosa").__version__,
                "soundfile": __import__("soundfile").__version__,
                "platform": platform.platform(),
                "threads": int(torch.get_num_threads())},
            "runtime": {
                "wall_s": round(time.perf_counter() - t_start, 3),
                "max_rss_bytes": int(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss),
                "max_rss_gib": round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 2**30, 3)},
            "hashes": {
                "spec_sha256": sha256_file(SPEC),
                "harness_sha256": sha256_file(Path(__file__).resolve()),
                "readout_sha256": sha256_file(READOUT),
                "seal_sha256": seal_sha,
                "git_head": git_head(),
                "a0z_scores_sha256": A0Z_SCORES_SHA},
        }
        metrics_out["metrics"]["verdict"] = verdict
        metrics_path = out / "metrics.json"
        metrics_path.write_text(json.dumps(metrics_out, indent=2) + "\n")

        verdict_out = {
            "exam": exam, "status": status, "dry_run": dry,
            "guard": verdict.get("guard"),
            "verdict": verdict.get("verdict"),
            "detail": verdict,
            "complete": complete,
            "n_scored": len(rows), "n_manifest": n_manifest,
            "n_missing": len(missing_rows), "n_errors": len(err_rows),
            "manifest_sha256": manifest_sha,
            "scores_sha256": metrics_out["scored"]["scores_sha256"],
            "metrics_sha256": sha256_file(metrics_path),
            "config_sha256": config_sha,
            "seal_sha256": seal_sha,
            "created_at": iso_now(),
            "wiring": "none",
            "training": "none",
        }
        (out / "verdict.json").write_text(
            json.dumps(verdict_out, indent=2) + "\n")
        log(f"status={status} verdict={verdict_out['verdict']} "
            f"n_scored={len(rows)} errors={len(err_rows)}")
        log(f"metrics -> {metrics_path}")
        print(json.dumps(verdict_out, indent=2), flush=True)

        if dry:
            return 0
        return 0 if complete else 2
    except Abort as e:
        log(f"ABORT({e.code}): {e}")
        print(f"ABORT({e.code}): {e}", file=sys.stderr)
        return e.code
    except Exception as e:
        import traceback

        log(f"FATAL: {e!r}")
        log(traceback.format_exc())
        print(f"FATAL: {e!r}", file=sys.stderr)
        return 2
    finally:
        log.close()


def summary(out_dir: str) -> int:
    out = Path(out_dir)
    root = out / "guards" if (out / "guards").is_dir() else out
    exams, guards = {}, {}
    for name in EXAMS:
        d = root / name
        vp, mp = d / "verdict.json", d / "metrics.json"
        if not vp.exists() or not mp.exists():
            exams[name] = {"status": "UNRUN", "verdict": "UNRUN",
                           "dir": str(d)}
            continue
        v = json.loads(vp.read_text())
        m = json.loads(mp.read_text())
        exams[name] = {
            "status": v.get("status"), "verdict": v.get("verdict"),
            "guard": v.get("guard"), "n_scored": v.get("n_scored"),
            "n_manifest": v.get("n_manifest"), "n_errors": v.get("n_errors"),
            "manifest_sha256": v.get("manifest_sha256"),
            "scores_sha256": v.get("scores_sha256"),
            "metrics_sha256": v.get("metrics_sha256"),
            "verdict_sha256": sha256_file(vp),
            "dir": str(d), "detail": v.get("detail"),
            "metrics": m.get("metrics"),
        }
    def gv(name):
        return exams[name].get("verdict")

    guards["G1"] = {"exam": "asv21la", "verdict": gv("asv21la"),
                    "detail": exams["asv21la"].get("detail")}
    guards["G2"] = {"exam": "itw",
                    "verdict": (exams["itw"].get("detail") or {}).get("g2", {}).get("verdict", gv("itw")),
                    "detail": (exams["itw"].get("detail") or {}).get("g2")}
    guards["G3"] = {"exam": "itw",
                    "verdict": (exams["itw"].get("detail") or {}).get("g3", {}).get("verdict", gv("itw")),
                    "detail": (exams["itw"].get("detail") or {}).get("g3")}
    guards["G4"] = {"exam": "codec_real", "verdict": gv("codec_real"),
                    "detail": exams["codec_real"].get("detail")}
    guards["G5"] = {"exam": "dfeval_codec", "verdict": gv("dfeval_codec"),
                    "detail": exams["dfeval_codec"].get("detail")}

    k1 = None
    if READOUT_JSON.exists():
        k1 = json.loads(READOUT_JSON.read_text())
    k1_pass = bool(k1 and k1.get("verdict")
                   and k1["verdict"].get("overall") == "K1R_PASS")
    all_complete = all(e["status"] == "COMPLETE" for e in exams.values())
    unrun = [k for k, g in guards.items() if g["verdict"] == "UNRUN"]
    incomplete = [k for k, g in guards.items() if g["verdict"] == "INCOMPLETE"]
    fails = [k for k, g in guards.items() if g["verdict"] == "FAIL"]
    holds = [k for k, g in guards.items() if g["verdict"] == "HOLD"]

    if not k1_pass:
        status, booked = "A0Z_CLOSED_K1R_FAIL", 0.0
    elif unrun or incomplete or not all_complete:
        status, booked = "A0Z_NOT_CERTIFIED_INCOMPLETE", 0.0
    elif fails:
        status, booked = "A0Z_NOT_CERTIFIED_GUARD_FAIL", 0.0
    elif holds:
        status, booked = "A0Z_NOT_CERTIFIED_GUARD_HOLD", 0.0
    else:
        status, booked = "A0Z_CERTIFIED_NARROW", 0.9

    out_summary = {
        "created_at": iso_now(),
        "spec_sha256": sha256_file(SPEC) if SPEC.exists() else None,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "readout_sha256": sha256_file(READOUT) if READOUT.exists() else None,
        "seal_sha256": sha256_file(SEAL) if SEAL.exists() else None,
        "git_head": git_head(),
        "k1_readout": None if k1 is None else {
            "status": k1.get("status"),
            "confusion": k1.get("confusion"),
            "FPR": k1.get("metrics", {}).get("FPR"),
            "R_all": k1.get("metrics", {}).get("R_all"),
            "coverage": k1.get("metrics", {}).get("coverage"),
            "verdict": (k1.get("verdict") or {}).get("overall"),
            "fragility_statement": (k1.get("verdict") or {}).get(
                "fragility_statement")},
        "exams": exams,
        "guards": guards,
        "section9": {
            "status": status, "booked": booked,
            "court_before": 64.8, "court_after": 64.8 + booked,
            "k1r_pass": k1_pass, "all_exams_complete": all_complete,
            "unrun": unrun, "incomplete": incomplete, "fail": fails,
            "hold": holds,
            "amendments_signed": {"A-1": True, "A-2": True,
                                  "A-3": True, "A-4": True},
            "note": "computed from the receipts by the harness; the "
                    "authoritative statement is RESULT.md",
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "guards_summary.json"
    path.write_text(json.dumps(out_summary, indent=2) + "\n")
    print(json.dumps({"section9": out_summary["section9"],
                      "guards": {k: v["verdict"] for k, v in guards.items()},
                      "summary": str(path)}, indent=2), flush=True)
    return 0


def main() -> int:
    for a in sys.argv[1:]:
        if (a == "--threshold" or a.startswith("--threshold=")
                or a == "--model" or a.startswith("--model=")):
            print("ABORT: --threshold and --model are forbidden in this "
                  "harness (SPEC §4: 1B only, CUT fixed at 0.5)",
                  file=sys.stderr)
            return 5
    ap = argparse.ArgumentParser(
        description="W-AUDIO A0z 1B guard harness (inference only)")
    ap.add_argument("--exam", choices=EXAMS)
    ap.add_argument("--out")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--supersede", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="pre-seal validation: manifest + gates + at most 10 "
                         "scored rows; never a receipt")
    args = ap.parse_args()
    if args.summary:
        if not args.out:
            ap.error("--summary requires --out")
        return summary(args.out)
    if not args.exam or not args.out:
        ap.error("--exam and --out are required unless --summary")
    return run_exam(args)


if __name__ == "__main__":
    sys.exit(main())
