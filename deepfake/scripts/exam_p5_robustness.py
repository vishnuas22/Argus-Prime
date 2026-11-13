#!/usr/bin/env python3
"""P5 — codec/laundering robustness kill test (read-only).

Sealed pre-registration: artifacts/p5_robustness/SPEC.md.
No training, wiring, promotion, pointer change, threshold change, or commit.

Usage:
  ./.venv/bin/python deepfake/scripts/exam_p5_robustness.py --seal
  ./.venv/bin/python deepfake/scripts/exam_p5_robustness.py --smoke
  ./.venv/bin/python deepfake/scripts/exam_p5_robustness.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

WS = Path(__file__).resolve().parents[2]
OUT_DEFAULT = WS / "artifacts" / "p5_robustness"
SMOKE_DEFAULT = WS / "artifacts" / "p5_robustness_smoke"
SPEC = OUT_DEFAULT / "SPEC.md"
SEAL = OUT_DEFAULT / "seal.txt"

ITW_DIR = WS / "data" / "raw" / "itw" / "release_in_the_wild"
AUDIO_SCORES = WS / "artifacts" / "exam_audio_itw_p6v11" / "scores.csv"
FFPP = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23"
VIDEO_FEATURES = WS / "artifacts" / "restack_video_v4e" / "gate_ffpp" / "features.csv"
IMAGE_REAL = WS / "data" / "corpora" / "eval_real"
IMAGE_FAKE = WS / "data" / "corpora" / "eval_fake"

IMAGE_STACKER = WS / "artifacts" / "p6v11" / "stacker.joblib"
IMAGE_META = WS / "artifacts" / "p6v11" / "stacker_meta.json"
VIDEO_STACKER = WS / "artifacts" / "p6vid_v4" / "stacker.joblib"
VIDEO_META = WS / "artifacts" / "p6vid_v4" / "stacker_meta.json"
AUDIO_PROBE = WS / "artifacts" / "p4" / "audio_probe.joblib"
VIDEO_DRIFT = WS / "artifacts" / "p6vid_v4" / "drift_calibration.json"

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
AUDIO_ARMS = ["orig", "mp3_128", "opus_64", "resample_8k"]
VIDEO_ARMS = ["orig", "h264_crf28"]
IMAGE_ARMS = ["orig", "jpeg_q35", "resize_50"]
SEED = 20261113
N_BOOT = 2000

INPUT_HASHES = {
    "artifacts/exam_audio_itw_p6v11/scores.csv":
        "40b60cda0ea6ee4fce8c7c18fa25c5ac2ef788e6180982e415794a5590809db8",
    "artifacts/restack_video_v4e/gate_ffpp/features.csv":
        "852f1d69f1ecfc81d313458328e1a075773e57917ed6a2d26f907c900d6a824b",
    "artifacts/p6v11/stacker.joblib":
        "04c849b4efe1f622a4c625f9be31b87b3b05d4425ec5b79fd8b3a77673b3b8e8",
    "artifacts/p6v11/stacker_meta.json":
        "b66c229d0fd80126defe93e0045e0efb3a5462f5cc87fe4f61fefe48fcf21f45",
    "artifacts/p6vid_v4/stacker.joblib":
        "2a601e9021aecd1362cf463e90c13d3b99d5bc059b6beca7a26cff28961f4d54",
    "artifacts/p6vid_v4/stacker_meta.json":
        "107cdd132fadde42f30b914fcfa6f5d7026ad28cbff7a44bab49839456628b10",
    "artifacts/p4/audio_probe.joblib":
        "3c80c6aea4284a9b92a27a90ec7eb9e26bf9ff56a10f42c9d8cd6526502f50eb",
    "artifacts/p6vid_v4/drift_calibration.json":
        "8ce8041cb7569d41f15734c2c654b7a2255cb5f999b8a4eae57559cec7f19beb",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def read_seal() -> dict:
    if not SEAL.exists():
        raise SystemExit(f"seal missing: {SEAL}")
    out = {}
    for line in SEAL.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise SystemExit(f"bad seal line: {line!r}")
        out[parts[0]] = parts[1]
    return out


def verify_seal() -> dict:
    seal = read_seal()
    actual_spec = sha256_file(SPEC)
    actual_script = sha256_file(Path(__file__).resolve())
    if actual_spec != seal.get("spec_sha256"):
        raise SystemExit(f"SPEC hash mismatch: {actual_spec} != {seal.get('spec_sha256')}")
    if actual_script != seal.get("script_sha256"):
        raise SystemExit(
            f"script hash mismatch: {actual_script} != {seal.get('script_sha256')}"
        )
    return {"spec_sha256": actual_spec, "script_sha256": actual_script}


def verify_inputs() -> dict:
    actual = {}
    for rel, expected in INPUT_HASHES.items():
        p = WS / rel
        if not p.exists():
            raise SystemExit(f"missing fixed input: {rel}")
        got = sha256_file(p)
        if got != expected:
            raise SystemExit(f"fixed input hash mismatch: {rel}: {got} != {expected}")
        actual[rel] = got
    return actual


def write_input_hashes(actual: dict, manifests: dict) -> None:
    lines = [f"{rel} {sha}" for rel, sha in actual.items()]
    for name, sha in manifests.items():
        lines.append(f"{name} {sha}")
    (OUT_DEFAULT / "input_hashes.txt").write_text("\n".join(lines) + "\n")


def _manifest_hash(rows: list[dict]) -> str:
    body = "\n".join(f"{r['label']} {r['key']}" for r in rows) + "\n"
    return sha256_bytes(body.encode())


def _sha_name(p: Path) -> str:
    return hashlib.sha256(p.name.encode()).hexdigest()


def _sha_key(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def load_audio_rows() -> list[dict]:
    rows = list(csv.DictReader(open(AUDIO_SCORES, newline="")))
    if len(rows) != 600:
        raise SystemExit(f"audio sample is {len(rows)} rows, expected 600")
    labels = {int(r["label"]) for r in rows}
    if labels != {0, 1}:
        raise SystemExit(f"audio labels are {labels}, expected 0/1")
    out = []
    for r in rows:
        p = ITW_DIR / r["file"]
        if not p.exists():
            raise SystemExit(f"missing ITW audio: {p}")
        out.append({
            "key": r["file"],
            "label": int(r["label"]),
            "path": p,
            "old_p": float(r["p_fake"]) if r.get("p_fake") not in (None, "") else None,
        })
    if sum(r["label"] == 0 for r in out) != 300 or sum(r["label"] == 1 for r in out) != 300:
        raise SystemExit("audio sample is not 300/300")
    return out


def load_video_rows() -> list[dict]:
    rows = list(csv.DictReader(open(VIDEO_FEATURES, newline="")))
    if len(rows) != 700:
        raise SystemExit(f"video features are {len(rows)} rows, expected 700")
    reals = [r for r in rows if r["label"] == "0" and r["method"] == "real"]
    if len(reals) != 140:
        raise SystemExit(f"video reals are {len(reals)}, expected 140")
    selected = sorted(reals, key=lambda r: _sha_key(f'{r["method"]}/{r["file"]}'))[:140]
    for m in METHODS:
        fakes = [r for r in rows if r["label"] == "1" and r["method"] == m]
        if len(fakes) != 140:
            raise SystemExit(f"video method {m} has {len(fakes)} rows, expected 140")
        selected += sorted(fakes, key=lambda r: _sha_key(f'{r["method"]}/{r["file"]}'))[:70]
    out = []
    for r in selected:
        if r["method"] == "real":
            p = FFPP / "real" / r["file"]
        else:
            p = FFPP / "fake" / r["method"] / r["file"]
        if not p.exists():
            raise SystemExit(f"missing FF++ video: {p}")
        if r["stacked"] in (None, ""):
            raise SystemExit(f"missing baseline stacked score: {r['method']}/{r['file']}")
        out.append({
            "key": f'{r["method"]}/{r["file"]}',
            "label": int(r["label"]),
            "method": r["method"],
            "path": p,
            "base_p": float(r["stacked"]),
            "base_route": r["route"],
            "base_features": r,
        })
    if len(out) != 420:
        raise SystemExit(f"video sample is {len(out)} rows, expected 420")
    if sum(r["label"] == 0 for r in out) != 140 or sum(r["label"] == 1 for r in out) != 280:
        raise SystemExit("video sample is not 140/280")
    return out


def load_image_rows() -> list[dict]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    real = sorted([p for p in IMAGE_REAL.iterdir() if p.suffix.lower() in exts],
                  key=_sha_name)[:100]
    fake = sorted([p for p in IMAGE_FAKE.iterdir() if p.suffix.lower() in exts],
                  key=_sha_name)[:100]
    if len(real) != 100 or len(fake) != 100:
        raise SystemExit(f"image sample is {len(real)}/{len(fake)}, expected 100/100")
    out = [{"key": f"eval_real/{p.name}", "label": 0, "path": p} for p in real]
    out += [{"key": f"eval_fake/{p.name}", "label": 1, "path": p} for p in fake]
    return out


def _smoke_subset(rows: list[dict]) -> list[dict]:
    one_real = next((r for r in rows if r["label"] == 0), None)
    one_fake = next((r for r in rows if r["label"] == 1), None)
    return [r for r in (one_real, one_fake) if r is not None]


def _fmt(x, nd: int = 4) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _safe_auc(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) == 0 or len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def _rate(keys, p_map, labels, cls, threshold=0.5):
    ks = [k for k in keys if labels[k] == cls]
    if not ks:
        return None, 0, 0
    k = sum(1 for x in ks if p_map[x] is not None and p_map[x] >= threshold)
    return k / len(ks), k, len(ks)


def _wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return [None, None]
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _bootstrap_delta(y, p_base, p_arm, kind: str, n_boot: int = N_BOOT):
    y = np.asarray(y, dtype=int)
    pb = np.asarray(p_base, dtype=float)
    pa = np.asarray(p_arm, dtype=float)
    if len(y) < 2:
        return [None, None]
    rng = np.random.default_rng(SEED)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if kind == "auc":
            if len(set(yy.tolist())) < 2:
                continue
            from sklearn.metrics import roc_auc_score
            vals.append(float(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx])))
        elif kind == "fpr":
            m = yy == 0
            if not m.any():
                continue
            vals.append(float(np.mean(pa[idx][m] >= 0.5) - np.mean(pb[idx][m] >= 0.5)))
        elif kind == "tpr":
            m = yy == 1
            if not m.any():
                continue
            vals.append(float(np.mean(pa[idx][m] >= 0.5) - np.mean(pb[idx][m] >= 0.5)))
        else:
            raise ValueError(kind)
    if not vals:
        return [None, None]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _valid_rec(rec: dict) -> bool:
    return (rec.get("p") is not None
            and rec.get("route") not in ("image_court", "error"))


def _covered_rec(rec: dict) -> bool:
    # Coverage = issued (non-abstained) for audio/image; video_face for video.
    # image_court rows carry p=None and abstained=True, so both definitions agree.
    return rec.get("p") is not None and not rec.get("abstained", True)


def _arm_summary(rows, arm, base_map, arm_map):
    keys = [r["key"] for r in rows]
    labels = {r["key"]: r["label"] for r in rows}
    base_valid = [k for k in keys if _valid_rec(base_map[k])]
    arm_valid = [k for k in keys if _valid_rec(arm_map[k])]
    base_covered = [k for k in keys if _covered_rec(base_map[k])]
    arm_covered = [k for k in keys if _covered_rec(arm_map[k])]
    paired = [k for k in keys if k in base_valid and k in arm_valid]
    base_p = {k: base_map[k].get("p") for k in keys}
    arm_p = {k: arm_map[k].get("p") for k in keys}

    y_pair = [labels[k] for k in paired]
    auc_base = _safe_auc(y_pair, [base_p[k] for k in paired])
    auc_arm = _safe_auc(y_pair, [arm_p[k] for k in paired])
    delta_auc = None if (auc_base is None or auc_arm is None) else auc_arm - auc_base
    ci_auc = _bootstrap_delta(y_pair, [base_p[k] for k in paired],
                              [arm_p[k] for k in paired], "auc") if paired else [None, None]

    fpr_base, fpr_b_k, fpr_b_n = _rate(base_valid, base_p, labels, 0)
    fpr_arm, fpr_a_k, fpr_a_n = _rate(arm_valid, arm_p, labels, 0)
    tpr_base, tpr_b_k, tpr_b_n = _rate(base_valid, base_p, labels, 1)
    tpr_arm, tpr_a_k, tpr_a_n = _rate(arm_valid, arm_p, labels, 1)
    delta_fpr = None if (fpr_base is None or fpr_arm is None) else fpr_arm - fpr_base
    delta_tpr = None if (tpr_base is None or tpr_arm is None) else tpr_arm - tpr_base

    pair_real = [k for k in paired if labels[k] == 0]
    pair_fake = [k for k in paired if labels[k] == 1]
    ci_fpr = _bootstrap_delta([labels[k] for k in pair_real],
                              [base_p[k] for k in pair_real],
                              [arm_p[k] for k in pair_real], "fpr") if pair_real else [None, None]
    ci_tpr = _bootstrap_delta([labels[k] for k in pair_fake],
                              [base_p[k] for k in pair_fake],
                              [arm_p[k] for k in pair_fake], "tpr") if pair_fake else [None, None]

    coverage_base = len(base_covered) / len(keys) if keys else None
    coverage_arm = len(arm_covered) / len(keys) if keys else None
    delta_coverage = (None if (coverage_base is None or coverage_arm is None)
                      else coverage_arm - coverage_base)

    errors = sum(1 for k in keys if arm_map[k].get("error"))
    missing = 0
    for k in keys:
        rec = arm_map[k]
        if rec.get("p") is None and not (rec.get("route") == "image_court"):
            missing += 1
    error_rate = errors / len(keys) if keys else None
    missing_rate = missing / len(keys) if keys else None

    r1 = None if delta_auc is None else bool(delta_auc >= -0.05)
    r2 = None if delta_fpr is None else bool(delta_fpr <= 0.05)
    r3 = None if delta_tpr is None else bool(delta_tpr >= -0.05)
    r4 = None if delta_coverage is None else bool(delta_coverage >= -0.05)
    r5 = bool((error_rate is not None and error_rate <= 0.01)
              and (missing_rate is not None and missing_rate <= 0.05))
    return {
        "arm": arm,
        "n_keys": len(keys),
        "n_base_valid": len(base_valid),
        "n_arm_valid": len(arm_valid),
        "n_base_covered": len(base_covered),
        "n_arm_covered": len(arm_covered),
        "n_paired": len(paired),
        "coverage_base": coverage_base,
        "coverage_arm": coverage_arm,
        "delta_coverage": delta_coverage,
        "auc_base": auc_base,
        "auc_arm": auc_arm,
        "delta_auc": delta_auc,
        "delta_auc_ci95": ci_auc,
        "fpr_base": fpr_base,
        "fpr_arm": fpr_arm,
        "fpr_base_k": fpr_b_k,
        "fpr_base_n": fpr_b_n,
        "fpr_arm_k": fpr_a_k,
        "fpr_arm_n": fpr_a_n,
        "fpr_base_wilson95": _wilson(fpr_b_k, fpr_b_n),
        "fpr_arm_wilson95": _wilson(fpr_a_k, fpr_a_n),
        "delta_fpr": delta_fpr,
        "delta_fpr_ci95": ci_fpr,
        "tpr_base": tpr_base,
        "tpr_arm": tpr_arm,
        "tpr_base_k": tpr_b_k,
        "tpr_base_n": tpr_b_n,
        "tpr_arm_k": tpr_a_k,
        "tpr_arm_n": tpr_a_n,
        "tpr_base_wilson95": _wilson(tpr_b_k, tpr_b_n),
        "tpr_arm_wilson95": _wilson(tpr_a_k, tpr_a_n),
        "delta_tpr": delta_tpr,
        "delta_tpr_ci95": ci_tpr,
        "n_errors": errors,
        "n_missing": missing,
        "error_rate": error_rate,
        "missing_rate": missing_rate,
        "R1_auc_retention": r1,
        "R2_fpr_inflation": r2,
        "R3_tpr_loss": r3,
        "R4_coverage_retention": r4,
        "R5_integrity": r5,
        "pass": bool(all(v is True for v in (r1, r2, r3, r4, r5))),
    }


def _video_per_method(rows, base_map, arm_map):
    out = {}
    for m in METHODS:
        keys = [r["key"] for r in rows if r["method"] == m]
        y = [1 if r["label"] == 1 else 0 for r in rows if r["method"] == m]
        bp = [base_map[k].get("p") for k in keys]
        ap = [arm_map[k].get("p") for k in keys]
        out[m] = {
            "n": len(keys),
            "auc_base": _safe_auc(y, bp),
            "auc_arm": _safe_auc(y, ap),
        }
    return out


class AudioScorer:
    def __init__(self):
        sys.path.insert(0, str(WS / "deepfake"))
        import argus_detect as A
        self.A = A

    def score(self, path: Path):
        r = self.A.detect_audio(path, recovery_enabled=False)
        return {
            "p": float(r["p_fake_audio_mean"]),
            "abstained": bool(r["abstained"]),
            "route": "audio",
            "ooc": bool(r.get("out_of_coverage", False)),
            "znorm": float(r.get("znorm_mean", 0.0)),
            "segments": int(r.get("segments_analyzed", 0)),
        }


class VideoScorer:
    def __init__(self):
        sys.path.insert(0, str(WS / "deepfake"))
        sys.path.insert(0, str(WS / "deepfake" / "bench"))
        sys.path.insert(0, str(WS / "deepfake" / "scripts"))
        from exam_video_ffpp import load_stacker
        from video_face_track import set_embedder
        set_embedder("arcface")
        self.clf, self.tag, self.names = load_stacker(VIDEO_STACKER.parent, "arcface")
        self.spike = float(json.loads(VIDEO_DRIFT.read_text())["spike_thresh"])

    def score(self, path: Path):
        from video_dump import featurize_video
        from video_features import row_to_X
        res = featurize_video(path, 12, with_audio=False, spike_thresh=self.spike)
        if res is None:
            return {"p": None, "abstained": True, "route": "image_court"}
        flat = res["row"]
        p = float(self.clf.predict_proba(
            row_to_X(flat, self.names).reshape(1, -1))[0, 1])
        return {"p": p, "abstained": False, "route": "video_face",
                "sha256_12": flat.get("sha256_12")}


class ImageScorer:
    def __init__(self):
        sys.path.insert(0, str(WS / "deepfake"))
        sys.path.insert(0, str(WS / "deepfake" / "bench"))
        from argus_detect import build_modules, _image_stacker_clfs
        from triage import DegradationEstimator
        self.modules, self.mids = build_modules()
        self.estimator = DegradationEstimator()
        self.stacker = _image_stacker_clfs()
        self.tmp = Path(tempfile.mkdtemp(prefix="p5_img_tmp_"))
        self.art = Path(tempfile.mkdtemp(prefix="p5_img_art_"))

    def score(self, pil: Image.Image, raw: bytes, fmt: str, key: str):
        from run_benchmark import run_image
        vid = hashlib.sha256((key + "|" + fmt).encode()).hexdigest()[:12]
        row, _ = run_image(self.modules, self.estimator, pil, None, self.art,
                           vid, self.tmp, self.stacker, original_raw=raw,
                           return_extras=True)
        p = None if row.get("p_real") is None else 1.0 - float(row["p_real"])
        return {"p": p, "abstained": bool(row.get("abstained", False)),
                "route": "image", "sha256_12": row.get("sha")}

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        shutil.rmtree(self.art, ignore_errors=True)


def audio_transform(src: Path, arm: str, tmp: Path):
    if arm == "orig":
        raw = src.read_bytes()
        return src, "identity", raw, raw
    tag = hashlib.sha256(str(src).encode()).hexdigest()[:12]
    if arm == "mp3_128":
        dst = tmp / f"{tag}_mp3_128.mp3"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
               "-c:a", "libmp3lame", "-b:a", "128k", str(dst)]
    elif arm == "opus_64":
        dst = tmp / f"{tag}_opus_64.ogg"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
               "-c:a", "libopus", "-b:a", "64k", "-f", "ogg", str(dst)]
    elif arm == "resample_8k":
        dst = tmp / f"{tag}_resample_8k.wav"
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
               "-ar", "8000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)]
    else:
        raise ValueError(arm)
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    raw = dst.read_bytes()
    return dst, " ".join(cmd), src.read_bytes(), raw


def video_transform(src: Path, tmp: Path):
    tag = hashlib.sha256(str(src).encode()).hexdigest()[:12]
    dst = tmp / f"{tag}_h264_crf28.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
           "-c:v", "libx264", "-crf", "28", "-preset", "medium",
           "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
           "-pix_fmt", "yuv420p", "-an", "-map_metadata", "-1", str(dst)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return dst, " ".join(cmd)


def image_transform(pil: Image.Image, raw: bytes, fmt: str, arm: str):
    if arm == "orig":
        return pil, raw, fmt, "identity"
    if arm == "jpeg_q35":
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=35)
        out_raw = buf.getvalue()
        out = Image.open(io.BytesIO(out_raw)).convert("RGB")
        return out, out_raw, "jpeg", "PIL JPEG quality=35"
    if arm == "resize_50":
        w, h = pil.size
        small = pil.resize((max(1, w // 2), max(1, h // 2)),
                           getattr(Image, "Resampling", Image).LANCZOS)
        out = small.resize((w, h), getattr(Image, "Resampling", Image).LANCZOS)
        buf = io.BytesIO()
        out.save(buf, "PNG")
        return out, buf.getvalue(), "png", "PIL LANCZOS 50% down/up PNG"
    raise ValueError(arm)


def run_audio(rows, smoke, receipts):
    scorer = AudioScorer()
    tmp = Path(tempfile.mkdtemp(prefix="p5_audio_"))
    results = {arm: {} for arm in AUDIO_ARMS}
    t0 = time.time()
    try:
        for arm in AUDIO_ARMS:
            for i, r in enumerate(rows):
                rec = {"p": None, "abstained": True, "route": "audio",
                       "error": None}
                try:
                    path, cmd, in_raw, out_raw = audio_transform(r["path"], arm, tmp)
                    if arm == "orig":
                        rec.update(scorer.score(path))
                    else:
                        rec.update(scorer.score(path))
                    rec["transform"] = cmd
                    receipts.append({
                        "modality": "audio", "key": r["key"], "arm": arm,
                        "command": cmd, "input_sha256": sha256_bytes(in_raw),
                        "output_sha256": sha256_bytes(out_raw),
                        "input_bytes": len(in_raw), "output_bytes": len(out_raw),
                    })
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
                    rec["abstained"] = True
                    rec["route"] = "error"
                results[arm][r["key"]] = rec
                if (i + 1) % 100 == 0:
                    print(f"[p5:audio:{arm}] {i+1}/{len(rows)} "
                          f"({time.time()-t0:.1f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def run_video(rows, smoke, receipts, video_scorer):
    tmp = Path(tempfile.mkdtemp(prefix="p5_video_"))
    results = {arm: {} for arm in VIDEO_ARMS}
    t0 = time.time()
    try:
        for arm in VIDEO_ARMS:
            for i, r in enumerate(rows):
                rec = {"p": None, "abstained": True, "route": "image_court",
                       "error": None}
                if arm == "orig":
                    rec.update({
                        "p": r["base_p"],
                        "abstained": r["base_route"] != "video_face",
                        "route": r["base_route"],
                        "transform": "stored features (identity)",
                    })
                    receipts.append({
                        "modality": "video", "key": r["key"], "arm": arm,
                        "command": "stored features (identity)",
                        "input_sha256": r["base_features"].get("sha256_12"),
                        "output_sha256": r["base_features"].get("sha256_12"),
                        "input_bytes": None, "output_bytes": None,
                    })
                else:
                    try:
                        path, cmd = video_transform(r["path"], tmp)
                        rec.update(video_scorer.score(path))
                        rec["transform"] = cmd
                        out_raw = path.read_bytes()
                        receipts.append({
                            "modality": "video", "key": r["key"], "arm": arm,
                            "command": cmd,
                            "input_sha256": r["base_features"].get("sha256_12"),
                            "output_sha256": sha256_bytes(out_raw),
                            "input_bytes": r["path"].stat().st_size,
                            "output_bytes": len(out_raw),
                        })
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"
                        rec["abstained"] = True
                        rec["route"] = "error"
                results[arm][r["key"]] = rec
                if (i + 1) % 50 == 0:
                    print(f"[p5:video:{arm}] {i+1}/{len(rows)} "
                          f"({time.time()-t0:.1f}s)", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


def run_image(rows, smoke, receipts, image_scorer):
    results = {arm: {} for arm in IMAGE_ARMS}
    t0 = time.time()
    for arm in IMAGE_ARMS:
        for i, r in enumerate(rows):
            rec = {"p": None, "abstained": True, "route": "image",
                   "error": None}
            try:
                raw = r["path"].read_bytes()
                fmt = "jpeg" if r["path"].suffix.lower() in {".jpg", ".jpeg"} else "png"
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                out_pil, out_raw, out_fmt, cmd = image_transform(pil, raw, fmt, arm)
                rec.update(image_scorer.score(out_pil, out_raw, out_fmt, r["key"]))
                rec["transform"] = cmd
                receipts.append({
                    "modality": "image", "key": r["key"], "arm": arm,
                    "command": cmd, "input_sha256": sha256_bytes(raw),
                    "output_sha256": sha256_bytes(out_raw),
                    "input_bytes": len(raw), "output_bytes": len(out_raw),
                })
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                rec["abstained"] = True
                rec["route"] = "error"
            results[arm][r["key"]] = rec
            if (i + 1) % 50 == 0:
                print(f"[p5:image:{arm}] {i+1}/{len(rows)} "
                      f"({time.time()-t0:.1f}s)", flush=True)
    return results


def build_metrics(rows_by_mod, results, smoke, seal_info, input_actual,
                  manifests, video_recompute_delta):
    modalities = ["audio", "video", "image"]
    arms_by_mod = {"audio": AUDIO_ARMS, "video": VIDEO_ARMS, "image": IMAGE_ARMS}
    metrics = {
        "exam": "P5 codec/laundering robustness kill test",
        "sealed": True,
        "smoke": bool(smoke),
        "baseline_court": 64.4,
        "spec_sha256": seal_info.get("spec_sha256"),
        "script_sha256": seal_info.get("script_sha256"),
        "seed": SEED,
        "n_boot": N_BOOT,
        "bars": {
            "R1_auc_retention": "AUC_arm - AUC_base >= -0.05 (paired valid rows)",
            "R2_fpr_inflation": "FPR_arm - FPR_base <= +0.05 at 0.5 (all valid rows)",
            "R3_tpr_loss": "TPR_arm - TPR_base >= -0.05 at 0.5 (all valid rows)",
            "R4_coverage_retention": "coverage_arm >= coverage_base - 0.05",
            "R5_integrity": "errors <= 1%, missing <= 5%, hashes/contracts match",
        },
        "inputs": input_actual,
        "manifests": manifests,
        "samples": {m: {
            "n": len(rows_by_mod[m]),
            "n_real": sum(1 for r in rows_by_mod[m] if r["label"] == 0),
            "n_fake": sum(1 for r in rows_by_mod[m] if r["label"] == 1),
        } for m in modalities},
        "video_recompute_max_delta": video_recompute_delta,
        "arms": {},
        "verdict": None,
        "decision": None,
        "failing": [],
    }
    for mod in modalities:
        base_map = results[mod]["orig"]
        metrics["arms"][mod] = {}
        for arm in arms_by_mod[mod]:
            s = _arm_summary(rows_by_mod[mod], arm, base_map, results[mod][arm])
            if mod == "video" and arm == "h264_crf28":
                s["per_method"] = _video_per_method(
                    rows_by_mod[mod], base_map, results[mod][arm])
            metrics["arms"][mod][arm] = s
            if arm != "orig" and not s["pass"]:
                failed = [k for k in ("R1_auc_retention", "R2_fpr_inflation",
                                      "R3_tpr_loss", "R4_coverage_retention",
                                      "R5_integrity") if s.get(k) is False]
                metrics["failing"].append({
                    "modality": mod, "arm": arm, "failed_bars": failed,
                    "delta_auc": s["delta_auc"], "delta_fpr": s["delta_fpr"],
                    "delta_tpr": s["delta_tpr"],
                    "delta_coverage": s["delta_coverage"],
                })
    integrity = {
        "spec_script_ok": True,
        "input_hashes_ok": True,
        "sample_counts_ok": all(
            metrics["samples"][m]["n_real"] > 0 and metrics["samples"][m]["n_fake"] > 0
            for m in modalities),
        "video_stacker_recompute_ok": (video_recompute_delta is not None
                                       and video_recompute_delta < 0.001),
        "all_arms_r5_ok": all(
            metrics["arms"][m][a]["R5_integrity"]
            for m in modalities for a in arms_by_mod[m]),
    }
    metrics["integrity"] = integrity
    if smoke:
        metrics["verdict"] = "SMOKE"
        metrics["decision"] = "Smoke only; no claim."
    elif not all(integrity.values()):
        metrics["verdict"] = "INVALID"
        metrics["decision"] = "Integrity failure; no claim. Fix and rerun."
    elif metrics["failing"]:
        metrics["verdict"] = "GAP"
        metrics["decision"] = (
            "Robustness gap recorded. No score booked. A codec-diverse "
            "retraining proposal is eligible only via a separate sealed spec "
            "and explicit user approval."
        )
    else:
        metrics["verdict"] = "PASS_ALL"
        metrics["decision"] = (
            "P5 measurement closes at 0. No codec-diverse retraining purchase "
            "is justified. No score booked."
        )
    return metrics


def write_result(metrics, out: Path):
    lines = []
    lines.append("# P5 — Codec / Laundering Robustness Kill Test — Result")
    lines.append("")
    lines.append(f"**Verdict: {metrics['verdict']}**  ")
    lines.append(f"**Run:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    lines.append(f"**Spec SHA-256:** `{metrics.get('spec_sha256')}`  ")
    lines.append(f"**Script SHA-256:** `{metrics.get('script_sha256')}`  ")
    lines.append(f"**Baseline court:** {metrics['baseline_court']}/100  ")
    lines.append("**No training, wiring, promotion, pointer change, or commit.**")
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(metrics["decision"])
    lines.append("")
    lines.append("## Samples")
    lines.append("")
    lines.append("| Modality | n | real | fake |")
    lines.append("|---|---:|---:|---:|")
    for m in ("audio", "video", "image"):
        s = metrics["samples"][m]
        lines.append(f"| {m} | {s['n']} | {s['n_real']} | {s['n_fake']} |")
    lines.append("")
    lines.append("## Bars")
    lines.append("")
    for k, v in metrics["bars"].items():
        lines.append(f"- **{k}:** {v}")
    lines.append("")
    for mod in ("audio", "video", "image"):
        lines.append(f"## {mod.title()}")
        lines.append("")
        lines.append("| arm | n | coverage | AUC | ΔAUC [95% CI] | FPR | ΔFPR | TPR | ΔTPR | R1 | R2 | R3 | R4 | R5 |")
        lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|")
        for arm in (AUDIO_ARMS if mod == "audio" else VIDEO_ARMS if mod == "video" else IMAGE_ARMS):
            s = metrics["arms"][mod][arm]
            ci = s["delta_auc_ci95"]
            ci_s = f"[{_fmt(ci[0])}, {_fmt(ci[1])}]" if ci and ci[0] is not None else "—"
            def mark(x):
                return "—" if x is None else ("PASS" if x else "MISS")
            lines.append(
                f"| {arm} | {s['n_keys']} | {_fmt(s['coverage_arm'], 3)} | "
                f"{_fmt(s['auc_arm'])} | {_fmt(s['delta_auc'])} {ci_s} | "
                f"{_fmt(s['fpr_arm'], 3)} | {_fmt(s['delta_fpr'], 3)} | "
                f"{_fmt(s['tpr_arm'], 3)} | {_fmt(s['delta_tpr'], 3)} | "
                f"{mark(s['R1_auc_retention'])} | {mark(s['R2_fpr_inflation'])} | "
                f"{mark(s['R3_tpr_loss'])} | {mark(s['R4_coverage_retention'])} | "
                f"{mark(s['R5_integrity'])} |"
            )
        lines.append("")
    if metrics.get("failing"):
        lines.append("## Exact failures")
        lines.append("")
        for f in metrics["failing"]:
            lines.append(f"- **{f['modality']}/{f['arm']}** failed "
                         f"{', '.join(f['failed_bars'])}; "
                         f"ΔAUC={_fmt(f['delta_auc'])}, "
                         f"ΔFPR={_fmt(f['delta_fpr'])}, "
                         f"ΔTPR={_fmt(f['delta_tpr'])}, "
                         f"Δcoverage={_fmt(f['delta_coverage'])}.")
        lines.append("")
    lines.append("## Integrity")
    lines.append("")
    for k, v in metrics["integrity"].items():
        lines.append(f"- **{k}:** {v}")
    lines.append(f"- **video stacker recompute max |Δ|:** "
                 f"{_fmt(metrics.get('video_recompute_max_delta'), 6)}")
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append("- P5 is a read-only measurement. It books **0** unless a "
                 "separate retraining proposal is approved and passes its own sealed bars.")
    lines.append("- The image arm is a 100/100 held-out eval spot-check; the "
                 "known T11/T9 recapture real-FPR 45% gap is **not** re-measured here.")
    lines.append("- The audio arm is the fixed external ITW 600-row sample; "
                 "it is not the ASVspoof 2019 in-domain codec exam (W2).")
    lines.append("- The video arm is FF++ C23 official test recompressed at "
                 "H.264 CRF28; it is not the P2 T2V re-encode control.")
    lines.append("- Bars are point-estimate bars; paired bootstrap CIs are "
                 "reported but are not a multiplicity-corrected significance test.")
    lines.append("- No transformed media are retained; commands and hashes are "
                 "in `transform_receipts.json`.")
    lines.append("")
    lines.append("## Receipts")
    lines.append("")
    lines.append("- `scores.csv` — per-key/arm scores")
    lines.append("- `metrics.json` — full bars, deltas, CIs, integrity")
    lines.append("- `transform_receipts.json` — transform commands and hashes")
    lines.append("- `input_hashes.txt` — verified fixed-input/model hashes")
    lines.append("- `seal.txt` — spec/script seal")
    (out / "RESULT.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true", help="write seal.txt and exit")
    ap.add_argument("--verify-only", action="store_true", help="verify seal/inputs and exit")
    ap.add_argument("--smoke", action="store_true", help="2 rows per modality; no claim")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-seal-check", action="store_true",
                    help="development only; skip spec/script hash check")
    args = ap.parse_args()

    if args.seal:
        OUT_DEFAULT.mkdir(parents=True, exist_ok=True)
        actual = verify_inputs()
        manifests = {
            "audio_sample_manifest": _manifest_hash(load_audio_rows()),
            "video_sample_manifest": _manifest_hash(load_video_rows()),
            "image_sample_manifest": _manifest_hash(load_image_rows()),
        }
        write_input_hashes(actual, manifests)
        SEAL.write_text(
            f"spec_sha256 {sha256_file(SPEC)}\n"
            f"script_sha256 {sha256_file(Path(__file__).resolve())}\n"
        )
        print(f"[p5:seal] wrote {SEAL}")
        print(f"[p5:seal] spec_sha256 {sha256_file(SPEC)}")
        print(f"[p5:seal] script_sha256 {sha256_file(Path(__file__).resolve())}")
        return

    if args.verify_only:
        seal_info = {} if args.no_seal_check else verify_seal()
        actual = verify_inputs()
        print(json.dumps({"seal": seal_info, "inputs": actual}, indent=2))
        return

    out = Path(args.out) if args.out else (SMOKE_DEFAULT if args.smoke else OUT_DEFAULT)
    out.mkdir(parents=True, exist_ok=True)
    if not args.force and not args.smoke:
        for name in ("scores.csv", "metrics.json", "RESULT.md"):
            if (out / name).exists():
                raise SystemExit(f"{out/name} exists; use --force")
    seal_info = {} if args.no_seal_check else verify_seal()
    actual = verify_inputs()

    audio_rows = load_audio_rows()
    video_rows = load_video_rows()
    image_rows = load_image_rows()
    manifests = {
        "audio_sample_manifest": _manifest_hash(audio_rows),
        "video_sample_manifest": _manifest_hash(video_rows),
        "image_sample_manifest": _manifest_hash(image_rows),
    }
    if args.smoke:
        audio_rows = _smoke_subset(audio_rows)
        video_rows = _smoke_subset(video_rows)
        image_rows = _smoke_subset(image_rows)
    write_input_hashes(actual, manifests)

    print(f"[p5] out={out} smoke={args.smoke}", flush=True)
    print(f"[p5] samples audio={len(audio_rows)} video={len(video_rows)} "
          f"image={len(image_rows)}", flush=True)

    receipts = []
    results = {}
    results["audio"] = run_audio(audio_rows, args.smoke, receipts)
    video_scorer = VideoScorer()
    # Integrity: reproduce stored video scores from stored features.
    from video_features import row_to_X
    deltas = []
    for r in video_rows:
        p = float(video_scorer.clf.predict_proba(
            row_to_X(r["base_features"], video_scorer.names).reshape(1, -1))[0, 1])
        deltas.append(abs(p - r["base_p"]))
    video_recompute_delta = max(deltas) if deltas else None
    print(f"[p5] video stacker recompute max|delta|={video_recompute_delta}", flush=True)
    results["video"] = run_video(video_rows, args.smoke, receipts, video_scorer)
    image_scorer = ImageScorer()
    try:
        results["image"] = run_image(image_rows, args.smoke, receipts, image_scorer)
    finally:
        image_scorer.close()

    rows_by_mod = {"audio": audio_rows, "video": video_rows, "image": image_rows}
    metrics = build_metrics(rows_by_mod, results, args.smoke, seal_info,
                            actual, manifests, video_recompute_delta)
    (out / "transform_receipts.json").write_text(
        json.dumps(receipts, indent=2) + "\n")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    score_rows = []
    for mod in ("audio", "video", "image"):
        for arm in (AUDIO_ARMS if mod == "audio" else VIDEO_ARMS if mod == "video" else IMAGE_ARMS):
            for r in rows_by_mod[mod]:
                rec = results[mod][arm][r["key"]]
                score_rows.append({
                    "modality": mod,
                    "key": r["key"],
                    "label": r["label"],
                    "arm": arm,
                    "p_fake": rec.get("p"),
                    "abstained": rec.get("abstained"),
                    "route": rec.get("route"),
                    "ooc": rec.get("ooc"),
                    "znorm": rec.get("znorm"),
                    "segments": rec.get("segments"),
                    "error": rec.get("error"),
                })
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
        w.writeheader()
        w.writerows(score_rows)
    write_result(metrics, out)
    print(f"[p5] verdict={metrics['verdict']} failing={len(metrics['failing'])}", flush=True)
    print(f"[p5] wrote {out/'scores.csv'}", flush=True)
    print(f"[p5] wrote {out/'metrics.json'}", flush=True)
    print(f"[p5] wrote {out/'RESULT.md'}", flush=True)


if __name__ == "__main__":
    main()
