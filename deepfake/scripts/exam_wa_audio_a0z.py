#!/usr/bin/env python
"""W-AUDIO A0z 1B DF-Eval exam (inference only; no training, no wiring).

Scores all 703 existing DF-Eval-2024 audio mp3s with the released
nii-yamagishilab/xls-r-1b-anti-deepfake checkpoint and its official
adaptive-average-pool + 2-class FC head. The model card's class order is
index 0 = fake, index 1 = real; the exam uses p_fake = softmax(logits)[0].

This is the primary DF-Eval bar only. It does not run the ASVspoof 2021 LA,
ITW, codec, or 13 s arms, and it does not fit A0p. Guard arms require separate
explicit approval. No threshold is selected; AUC and 10-bin ECE are reported.

Usage:
  .venv/bin/python deepfake/scripts/exam_wa_audio_a0z.py --labels-only
  .venv/bin/python deepfake/scripts/exam_wa_audio_a0z.py --model 1b
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

WS = Path(__file__).resolve().parents[2]
AUDIO_DIR = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data"
STEMS_CSV = WS / "data" / "dfeval" / "meta" / "matched_stems.csv"
PILOT_DIR = WS / "artifacts" / "wa_audio_pilot"
PILOT = PILOT_DIR / "timing_pilot.py"
SPEC = WS / "artifacts" / "wa_audio_prereg" / "SPEC.md"
DEFAULT_OUT = WS / "artifacts" / "wa_audio_a0z_1b"
REPO = "nii-yamagishilab/xls-r-1b-anti-deepfake"
WEIGHTS_SHA_EXPECTED = (
    "57da6a47234bb8aad0c61b7f575378a12c2734a30f1f395b95bbf97ed4ba5373"
)
FAKE_INDEX = 0
REAL_INDEX = 1
SECONDS = 4.0
SAMPLE_RATE = 16000
N_BINS = 10
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20261111
AUC_PASS = 0.650
AUC_HOLD = 0.610
ECE_PASS = 0.05
ECE_HOLD = 0.10

for p in (PILOT_DIR,):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from timing_pilot import map_fairseq_key  # noqa: E402


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


def load_labels() -> tuple[list[dict], list[dict], dict]:
    rows = list(csv.DictReader(open(STEMS_CSV, newline="")))
    if not rows:
        raise RuntimeError(f"empty labels: {STEMS_CSV}")
    if list(rows[0]) != ["stem", "label"]:
        raise RuntimeError(f"unexpected label columns: {list(rows[0])}")
    stems = [r["stem"] for r in rows]
    if len(stems) != len(set(stems)):
        raise RuntimeError("duplicate stems in matched_stems.csv")
    present, missing = [], []
    for r in rows:
        label = r["label"]
        if label not in ("Real", "Fake"):
            raise RuntimeError(f"bad label {label!r} for {r['stem']}")
        p = AUDIO_DIR / f"{r['stem']}.mp3"
        rec = {
            "stem": r["stem"],
            "label": 0 if label == "Real" else 1,
            "label_name": label,
            "path": str(p),
        }
        (present if p.exists() else missing).append(rec)
    mp3s = {p.stem for p in AUDIO_DIR.glob("*.mp3")}
    listed = {r["stem"] for r in rows}
    unlisted = sorted(mp3s - listed)
    if unlisted:
        raise RuntimeError(f"{len(unlisted)} mp3s lack labels, e.g. {unlisted[:5]}")
    present.sort(key=lambda r: r["stem"])
    missing.sort(key=lambda r: r["stem"])
    counts = {
        "csv_rows": len(rows),
        "csv_real": sum(r["label"] == "Real" for r in rows),
        "csv_fake": sum(r["label"] == "Fake" for r in rows),
        "mp3s_on_disk": len(mp3s),
        "existing": len(present),
        "existing_real": sum(r["label"] == 0 for r in present),
        "existing_fake": sum(r["label"] == 1 for r in present),
        "missing": len(missing),
        "missing_real": sum(r["label"] == 0 for r in missing),
        "missing_fake": sum(r["label"] == 1 for r in missing),
        "unlisted_mp3s": 0,
    }
    if counts["existing"] != 703:
        raise RuntimeError(f"expected 703 existing DF-Eval mp3s, got {counts['existing']}")
    if (counts["existing_real"], counts["existing_fake"]) != (395, 308):
        raise RuntimeError(
            f"expected 395 real / 308 fake, got {counts['existing_real']} / "
            f"{counts['existing_fake']}")
    if counts["missing"] != 74:
        raise RuntimeError(f"expected 74 missing stems, got {counts['missing']}")
    return present, missing, counts


def load_audio_meta(path: Path, seconds: float = SECONDS) -> tuple[torch.Tensor, dict]:
    """Exact frozen loader, with metadata for the receipt."""
    import librosa

    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True, duration=seconds)
    if wav.size == 0:
        raise ValueError(f"empty audio: {path}")
    target = int(SAMPLE_RATE * seconds)
    n_loaded = int(wav.size)
    padded = n_loaded < target
    if padded:
        wav = np.pad(wav, (0, target - n_loaded))
    wav = wav[:target]
    x = torch.from_numpy(wav.astype(np.float32))
    x = F.layer_norm(x, x.shape)
    meta = {
        "n_loaded": n_loaded,
        "n_samples": target,
        "padded": bool(padded),
        "orig_sr": None,
        "orig_frames": None,
        "orig_duration_s": None,
    }
    try:
        import soundfile as sf

        info = sf.info(str(path))
        meta["orig_sr"] = int(info.samplerate)
        meta["orig_frames"] = int(info.frames)
        meta["orig_duration_s"] = round(float(info.duration), 6)
    except Exception as e:  # metadata only; scoring does not depend on it
        meta["orig_error"] = repr(e)
    return x.unsqueeze(0), meta


def verify_loader(present: list[dict], n: int = 3) -> dict:
    """Prove the exam loader is bit-identical to the frozen pilot loader."""
    from timing_pilot import load_audio as pilot_load

    checked = []
    for r in present[:n]:
        p = Path(r["path"])
        a = pilot_load(p, SECONDS)
        b, _ = load_audio_meta(p, SECONDS)
        same = bool(torch.equal(a, b))
        checked.append({"stem": r["stem"], "bit_identical": same,
                        "shape": list(b.shape)})
        if not same:
            raise RuntimeError(f"exam loader differs from pilot on {p}")
    return {"checked": len(checked), "all_bit_identical": True, "items": checked}


def load_model(threads: int) -> dict:
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import Wav2Vec2Config, Wav2Vec2Model

    if threads:
        torch.set_num_threads(threads)
    weights_path = Path(hf_hub_download(
        repo_id=REPO, filename="model.safetensors", local_files_only=True))
    weights_sha = sha256_file(weights_path)
    if weights_sha != WEIGHTS_SHA_EXPECTED:
        raise RuntimeError(
            f"checkpoint SHA mismatch: {weights_sha} != {WEIGHTS_SHA_EXPECTED}")
    cfg = Wav2Vec2Config.from_pretrained(REPO, local_files_only=True)
    model = Wav2Vec2Model(cfg)
    state = load_file(str(weights_path))
    mapped, skipped = {}, []
    for k, v in state.items():
        nk = map_fairseq_key(k)
        if nk is None:
            skipped.append(k)
        else:
            mapped[nk] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    encoder_missing = [
        k for k in missing
        if k.startswith(("encoder.", "feature_projection.", "feature_extractor.",
                         "masked_spec_embed"))
    ]
    if encoder_missing or unexpected:
        raise RuntimeError(
            f"mapping fail: missing={encoder_missing[:20]} "
            f"unexpected={unexpected[:20]}")
    model.eval()
    w_fc = state["proj_fc.weight"].float()
    b_fc = state["proj_fc.bias"].float()
    if tuple(w_fc.shape) != (2, cfg.hidden_size):
        raise RuntimeError(f"bad proj_fc shape {tuple(w_fc.shape)}")
    return {
        "model": model,
        "cfg": cfg,
        "state": state,
        "w_fc": w_fc,
        "b_fc": b_fc,
        "weights_path": weights_path,
        "weights_sha256": weights_sha,
        "weights_bytes": weights_path.stat().st_size,
        "repo_revision": weights_path.parent.name,
        "mapped": len(mapped),
        "skipped": len(skipped),
        "skipped_keys": skipped,
        "hidden_size": int(cfg.hidden_size),
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "fc_shape": list(w_fc.shape),
    }


def ece_maxconf(y: np.ndarray, p: np.ndarray,
                n_bins: int = N_BINS) -> tuple:
    """Standard binary confidence ECE: max(p, 1-p), correctness."""
    conf = np.maximum(p, 1.0 - p)
    correct = ((p >= 0.5).astype(np.int64) == y).astype(np.float64)
    return ece_equal_width(correct, conf, n_bins)


def ece_equal_width(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # p=1.0 belongs to the last bin.
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            bins.append({"bin": b, "lo": round(float(edges[b]), 6),
                         "hi": round(float(edges[b + 1]), 6), "n": 0,
                         "confidence": None, "accuracy": None})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        gap = abs(acc - conf)
        ece += (n / len(y)) * gap
        bins.append({"bin": b, "lo": round(float(edges[b]), 6),
                     "hi": round(float(edges[b + 1]), 6), "n": n,
                     "confidence": round(conf, 6),
                     "accuracy": round(acc, 6),
                     "gap": round(gap, 6)})
    return float(ece), bins


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    real = np.where(y == 0)[0]
    fake = np.where(y == 1)[0]
    vals = np.empty(n, dtype=np.float64)
    for i in range(n):
        ri = rng.choice(real, size=len(real), replace=True)
        fi = rng.choice(fake, size=len(fake), replace=True)
        idx = np.concatenate([ri, fi])
        vals[i] = roc_auc_score(y[idx], p[idx])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {
        "n": n,
        "seed": seed,
        "method": "stratified percentile bootstrap",
        "auc_ci95": [round(float(lo), 6), round(float(hi), 6)],
        "auc_boot_mean": round(float(vals.mean()), 6),
        "auc_boot_sd": round(float(vals.std(ddof=1)), 6),
    }


def score_stats(y: np.ndarray, p: np.ndarray) -> dict:
    out = {}
    for name, val in (("real", 0), ("fake", 1)):
        a = p[y == val]
        out[name] = {
            "n": int(len(a)),
            "mean": round(float(a.mean()), 6),
            "median": round(float(np.median(a)), 6),
            "q1": round(float(np.percentile(a, 25)), 6),
            "q3": round(float(np.percentile(a, 75)), 6),
            "min": round(float(a.min()), 6),
            "max": round(float(a.max()), 6),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="W-AUDIO A0z 1B DF-Eval exam")
    ap.add_argument("--model", choices=("1b",), default="1b")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    present, missing, counts = load_labels()
    missing_csv = out / "missing_stems.csv"
    with open(missing_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "label", "label_name", "path"])
        w.writeheader()
        for r in missing:
            w.writerow({k: r[k] for k in ("stem", "label", "label_name", "path")})
    label_summary = {
        "counts": counts,
        "labels_csv": str(STEMS_CSV),
        "labels_sha256": sha256_file(STEMS_CSV),
        "audio_dir": str(AUDIO_DIR),
        "missing_stems_csv": str(missing_csv),
        "missing_stems_sha256": sha256_file(missing_csv),
        "missing_policy": "excluded and named; no substitution, no invented stems",
    }
    if args.labels_only:
        print(json.dumps(label_summary, indent=2))
        return 0

    scores_path = out / "scores.jsonl"
    if scores_path.exists() and not args.force:
        print(f"REFUSING: {scores_path} exists; pass --force to overwrite",
              file=sys.stderr)
        return 1

    print(f"labels: existing={counts['existing']} "
          f"({counts['existing_real']} real / {counts['existing_fake']} fake); "
          f"missing={counts['missing']} "
          f"({counts['missing_real']} real / {counts['missing_fake']} fake)",
          flush=True)
    print(f"loader check: {verify_loader(present, 3)}", flush=True)

    t_start = time.perf_counter()
    m = load_model(args.threads)
    print(f"model loaded: repo={REPO} revision={m['repo_revision']} "
          f"weights={m['weights_bytes']} sha={m['weights_sha256']} "
          f"hidden={m['hidden_size']} layers={m['num_hidden_layers']} "
          f"fc={m['fc_shape']} mapped={m['mapped']} skipped={m['skipped']}",
          flush=True)
    print(f"class order: index {FAKE_INDEX}=fake, index {REAL_INDEX}=real; "
          f"score = p_fake", flush=True)

    rows = []
    failures = []
    model = m["model"]
    w_fc, b_fc = m["w_fc"], m["b_fc"]
    with torch.inference_mode():
        for i, r in enumerate(present):
            p = Path(r["path"])
            t0 = time.perf_counter()
            try:
                x, meta = load_audio_meta(p, SECONDS)
                t_load = time.perf_counter() - t0
                t1 = time.perf_counter()
                emb = model(x).last_hidden_state
                pooled = F.adaptive_avg_pool1d(
                    emb.transpose(1, 2), 1).squeeze(-1)
                logits = F.linear(pooled, w_fc, b_fc)
                t_fwd = time.perf_counter() - t1
                if tuple(logits.shape) != (1, 2) or not torch.isfinite(logits).all():
                    raise RuntimeError(f"bad logits {tuple(logits.shape)}")
                probs = F.softmax(logits.float(), dim=1)
                if not torch.isfinite(probs).all():
                    raise RuntimeError("non-finite softmax")
                row = {
                    **r,
                    "sha256": sha256_file(p),
                    "bytes": p.stat().st_size,
                    **meta,
                    "fake_logit": round(float(logits[0, FAKE_INDEX]), 8),
                    "real_logit": round(float(logits[0, REAL_INDEX]), 8),
                    "p_fake": round(float(probs[0, FAKE_INDEX]), 10),
                    "p_real": round(float(probs[0, REAL_INDEX]), 10),
                    "load_s": round(t_load, 6),
                    "forward_s": round(t_fwd, 6),
                    "latency_s": round(t_load + t_fwd, 6),
                    "error": None,
                }
                rows.append(row)
            except Exception as e:
                failures.append({"stem": r["stem"], "label": r["label"],
                                 "path": str(p), "error": repr(e)})
                rows.append({**r, "sha256": None, "bytes": None,
                             "fake_logit": None, "real_logit": None,
                             "p_fake": None, "p_real": None,
                             "load_s": None, "forward_s": None,
                             "latency_s": None, "error": repr(e)})
            if (i + 1) % 50 == 0:
                print(f"{i+1}/{len(present)} scored; failures={len(failures)}",
                      flush=True)

    with open(scores_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    scores_sha = sha256_file(scores_path)
    ok = [r for r in rows if r["error"] is None and r["p_fake"] is not None]
    y = np.array([r["label"] for r in ok], dtype=np.int64)
    p = np.array([r["p_fake"] for r in ok], dtype=np.float64)
    complete = len(ok) == len(present) and not failures
    if not complete:
        print(f"INCOMPLETE: {len(ok)}/{len(present)} scored; "
              f"{len(failures)} failures", file=sys.stderr)

    metrics = {
        "exam": "W-AUDIO A0z 1B DF-Eval primary exam",
        "created_at": iso_now(),
        "status": "measurement-only; no training, no wiring, no promotion",
        "model": {
            "arm": "A0z",
            "repo": REPO,
            "revision": m["repo_revision"],
            "weights_path": str(m["weights_path"]),
            "weights_bytes": m["weights_bytes"],
            "weights_sha256": m["weights_sha256"],
            "hidden_size": m["hidden_size"],
            "num_hidden_layers": m["num_hidden_layers"],
            "fc_shape": m["fc_shape"],
            "mapped_keys": m["mapped"],
            "skipped_keys": m["skipped"],
            "class_order": {str(FAKE_INDEX): "fake", str(REAL_INDEX): "real"},
            "class_order_source": "model card: prob[0] = fake, prob[1] = real",
            "fake_index": FAKE_INDEX,
            "score": "p_fake = softmax(proj_fc(adaptive_avg_pool1d(last_hidden_state)))[0]",
            "head": "official adaptive-average-pool + 2-class FC",
        },
        "preprocessing": {
            "loader": "librosa.load(sr=16000, mono=True, duration=4.0)",
            "pad": "zero-pad to 64000 samples if shorter; truncate to 64000",
            "normalization": "F.layer_norm(waveform, waveform.shape)",
            "window_s": SECONDS,
            "sample_rate": SAMPLE_RATE,
            "n_samples": int(SAMPLE_RATE * SECONDS),
            "source": "artifacts/wa_audio_pilot/timing_pilot.py (frozen pilot loader)",
        },
        "labels": label_summary,
        "scored": {
            "n": len(ok),
            "real": int((y == 0).sum()),
            "fake": int((y == 1).sum()),
            "failures": failures,
            "complete": bool(complete),
            "scores_jsonl": str(scores_path),
            "scores_sha256": scores_sha,
        },
        "metrics": {
            "auc": None if len(ok) == 0 else round(float(
                roc_auc_score(y, p)), 6),
            "auc_logit_diff": None if len(ok) == 0 else round(float(
                roc_auc_score(y, np.array(
                    [r["fake_logit"] - r["real_logit"] for r in ok]))), 6),
            "ece_10bin_equal_width": None,
            "ece_bins": [],
            "ece_maxconf_10bin": None,
            "ece_maxconf_bins": [],
            "ece_definition": (
                "primary: equal-width 10-bin ECE on p_fake; sensitivity: "
                "equal-width 10-bin binary-confidence ECE on max(p,1-p)"),

            "score_stats": {} if len(ok) == 0 else score_stats(y, p),
        },
        "bars": {
            "dfeval_auc_pass": AUC_PASS,
            "dfeval_auc_hold": AUC_HOLD,
            "ece_pass": ECE_PASS,
            "ece_hold": ECE_HOLD,
            "guard_arms": {
                "asvspoof2021_la_auc_pass": 0.900,
                "itw_auc_pass": 0.900,
                "itw_bona_fpr_0_of_300": True,
                "codec_pooled_fpr_pass": 0.015,
                "codec_auc_drop_pass": 0.05,
            },
        },
        "guard_arms": {
            "run": False,
            "status": "not run",
            "reason": "DF-Eval primary only; guard arms require separate explicit approval",
        },
        "secondary_13s": {
            "run": False,
            "reason": "4 s primary bar only in this approved run",
        },
        "a0p": {
            "run": False,
            "reason": "no training; separate approval required",
        },
        "wiring": "none",
        "runtime": {
            "total_s": None,
            "max_rss_bytes": None,
            "platform": platform.platform(),
            "threads": int(torch.get_num_threads()),
        },
        "hashes": {
            "spec_sha256": sha256_file(SPEC),
            "pilot_sha256": sha256_file(PILOT),
            "exam_code_sha256": sha256_file(Path(__file__)),
            "labels_sha256": label_summary["labels_sha256"],
            "missing_stems_sha256": label_summary["missing_stems_sha256"],
        },
    }
    if len(ok) > 0:
        ece, bins = ece_equal_width(y, p, N_BINS)
        ece_mc, bins_mc = ece_maxconf(y, p, N_BINS)
        metrics["metrics"]["ece_10bin_equal_width"] = round(ece, 6)
        metrics["metrics"]["ece_bins"] = bins
        metrics["metrics"]["ece_maxconf_10bin"] = round(ece_mc, 6)
        metrics["metrics"]["ece_maxconf_bins"] = bins_mc
        metrics["metrics"]["bootstrap"] = bootstrap_auc(
            y, p, BOOTSTRAP_N, BOOTSTRAP_SEED)
    metrics["runtime"]["total_s"] = round(time.perf_counter() - t_start, 3)
    metrics["runtime"]["max_rss_bytes"] = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    metrics["runtime"]["clip_latency"] = {
        "sum_s": round(sum(r["latency_s"] for r in ok), 3) if ok else 0.0,
        "median_s": round(statistics.median(
            [r["latency_s"] for r in ok]), 6) if ok else None,
        "p90_s": round(float(np.percentile(
            [r["latency_s"] for r in ok], 90)), 6) if ok else None,
    }

    auc = metrics["metrics"]["auc"]
    ece = metrics["metrics"]["ece_10bin_equal_width"]
    if not complete or auc is None:
        auc_verdict = "INCOMPLETE"
    elif auc >= AUC_PASS:
        auc_verdict = "PASS"
    elif auc >= AUC_HOLD:
        auc_verdict = "HOLD"
    else:
        auc_verdict = "FAIL"
    if ece is None:
        ece_verdict = "INCOMPLETE"
    elif ece <= ECE_PASS:
        ece_verdict = "PASS"
    elif ece <= ECE_HOLD:
        ece_verdict = "HOLD"
    else:
        ece_verdict = "FAIL"
    if auc_verdict == "PASS":
        decision = "A0Z_DFEVAL_AUC_PASS"
        if ece_verdict == "FAIL":
            next_action = (
                "DF-Eval AUC passes, but ECE fails the <=0.05 bar; no A0z "
                "pass. Guard arms and 13 s were not run; no A0p. Stop and "
                "report the calibration failure.")
        elif ece_verdict == "HOLD":
            next_action = (
                "DF-Eval AUC passes, but ECE is HOLD; no A0z pass. Guard arms "
                "and 13 s were not run; no A0p. Stop and report.")
        else:
            next_action = (
                "DF-Eval AUC and ECE pass; guard arms and 13 s are not run. "
                "Request separate approval before guards; no A0p.")
    elif auc_verdict == "HOLD":
        decision = "A0Z_DFEVAL_HOLD"
        next_action = ("DF-Eval AUC is HOLD; no candidate and no bar change. "
                       "A0p would need separate training approval.")
    elif auc_verdict == "FAIL":
        decision = "A0Z_DFEVAL_FAIL"
        next_action = ("DF-Eval AUC <0.610; stop A0z. A0p is a separate "
                       "explicitly approved fallback.")
    else:
        decision = "A0Z_DFEVAL_INCOMPLETE"
        next_action = "Resolve decode failures; do not silently drop clips."

    verdict = {
        "decision": decision,
        "auc_verdict": auc_verdict,
        "ece_verdict": ece_verdict,
        "auc": auc,
        "auc_ci95": metrics["metrics"].get("bootstrap", {}).get("auc_ci95"),
        "ece_10bin_equal_width": ece,
        "ece_maxconf_10bin": metrics["metrics"].get("ece_maxconf_10bin"),
        "bars": {"auc_pass": AUC_PASS, "auc_hold": AUC_HOLD,
                 "ece_pass": ECE_PASS, "ece_hold": ECE_HOLD},
        "n_scored": len(ok),
        "n_real": int((y == 0).sum()),
        "n_fake": int((y == 1).sum()),
        "n_failures": len(failures),
        "complete": bool(complete),
        "guard_arms": "not run",
        "secondary_13s": "not run",
        "a0p": "not run; separate training approval required",
        "blocking_bars": (
            ([] if ece_verdict == "PASS" else [f"ECE:{ece_verdict}"]) +
            (["guard_arms:not_run"] if auc_verdict == "PASS" else [])),
        "a0z_overall": (
            "NOT_CERTIFIED_ECE_FAIL" if ece_verdict == "FAIL"
            else "NOT_CERTIFIED_ECE_HOLD" if ece_verdict == "HOLD"
            else "NOT_CERTIFIED_GUARD_ARMS_PENDING"
            if auc_verdict == "PASS" else "NOT_CERTIFIED"),
        "next_action": next_action,
        "wiring": "none",
        "created_at": iso_now(),
    }
    metrics["verdict"] = verdict
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"metrics -> {out / 'metrics.json'}", flush=True)
    print(f"scores  -> {scores_path} sha256={scores_sha}", flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
