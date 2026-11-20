"""W9 STEP 6 (D-079 SEALED): FULL ASVspoof 2021 LA eval — all 14 shards.

Sealed rulings:
- Score IN FULL. If the measured row count exceeds 150k, STOP before scoring
  and the user rules on subsampling (pre-registered rule, not outcome-tuned).
- Extract-once-then-score: per utterance, XLS-R + WavLM embeddings + spectral
  features computed ONCE and cached per shard (data/caches/w9_2021/), resumable
  (existing keys skipped); scoring reads the cache. Exam rows are CLEAN-view
  only (serve semantics; the 3-view scheme is a training device).
- Both arms scored from the SAME rows (selection identity for NR2): p4 raw
  (frozen A) and the fused stacker, plus per-channel diagnostics.

Out-dir guard verbatim from exam_audio_2021.py (patterns:32).

Usage:
  ./.venv/bin/python deepfake/scripts/exam_audio_2021_full.py [--force]
"""
import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
SHARD_DIR = WS / "data" / "raw"
CACHE_DIR = WS / "data" / "caches" / "w9_2021"
URL = ("https://huggingface.co/datasets/MoaazTalab/ASVspoof_2021_LA_Balanced_"
       "Normalized/resolve/main/data/test-{0:05d}-of-00014.parquet")
N_SHARDS = 14
SR = 16000
SEG = SR * 4
MAX_WINDOWS = 8
N_STOP = 150_000
SP_BINS = 64
SP_NAMES = ([f"sp_ltas_{i}" for i in range(SP_BINS)] +
            ["sp_flat_mean", "sp_flat_std", "sp_centroid_var", "sp_hf_ratio",
             "sp_edge_freq", "sp_frame_corr", "sp_phase_smooth"])


def _ffmpeg_wav_bytes(b: bytes):
    """Fallback decoder for VALID files libsndfile rejects (2021-eval FLAC class;
    measured: 14,024/73,944, ffprobe/ffmpeg rc=0 where libsndfile errors).
    Mirrors load_wav semantics: native decode -> mono mean -> np.interp to SR."""
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(b)
            tmp = f.name
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=sample_rate,channels", "-of", "json", tmp],
            capture_output=True, text=True, timeout=30)
        info = json.loads(probe.stdout or "{}")
        st = (info.get("streams") or [{}])[0]
        sr = int(st.get("sample_rate") or SR)
        ch = int(st.get("channels") or 1)
        r = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp, "-f", "f32le",
             "-acodec", "pcm_f32le", "-"], capture_output=True, timeout=120)
        if r.returncode != 0 or not r.stdout:
            return None
        a = np.frombuffer(r.stdout, dtype=np.float32)
        if ch < 1 or a.size % ch:
            return None
        mono = a.reshape(-1, ch).mean(axis=1)
        if sr != SR:
            g = np.arange(len(mono)) * (SR / sr)
            mono = np.interp(g, np.arange(len(mono)), mono).astype(np.float32)
        return mono
    except Exception:
        return None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def load_wav_bytes(b: bytes):
    import soundfile as sf
    try:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != SR:
            g = np.arange(len(mono)) * (SR / sr)
            mono = np.interp(g, np.arange(len(mono)), mono)
        return mono
    except Exception:
        return _ffmpeg_wav_bytes(b)


def serve_windows(wav):
    segs = [wav[i:i + SEG] for i in range(0, max(1, len(wav) - SEG + 1), SEG)][:MAX_WINDOWS] or \
           [np.pad(wav[:SEG], (0, max(0, SEG - len(wav))))]
    return [np.pad(s, (0, SEG - len(s))) if len(s) < SEG else s for s in segs]


def spectral_feats(seg):
    from scipy.signal import stft
    f, t, Z = stft(seg, fs=SR, nperseg=1024, noverlap=512, window="hann")
    P = np.abs(Z) ** 2 + 1e-12
    ltas = P.mean(axis=1)
    edges = np.geomspace(50.0, SR / 2 - 1, SP_BINS + 1)
    prof = np.zeros(SP_BINS)
    for i in range(SP_BINS):
        m = (f >= edges[i]) & (f < edges[i + 1])
        prof[i] = 10 * np.log10(ltas[m].mean() + 1e-12) if m.any() else -120.0
    flat = np.exp(np.log(P).mean(0)) / P.mean(0)
    centroid = (P * f[:, None]).sum(0) / P.sum(0)
    hf = f > 5500
    hf_ratio = P[hf].sum() / P.sum()
    peak = P.max()
    edge_mask = P.max(axis=1) > peak * 1e-4
    edge_freq = f[np.where(edge_mask)[0][-1]] if edge_mask.any() else 0.0
    fn = P / P.sum(0, keepdims=True)
    fr_corr = float(np.mean(np.sum(fn[:, 1:] * fn[:, :-1], 0)))
    top = np.argsort(P.mean(1))[-16:]
    ph = np.unwrap(np.angle(Z[top]), axis=1)
    phase_smooth = float(np.mean(np.std(np.diff(ph, axis=1), axis=1)))
    return np.concatenate([prof, np.array([flat.mean(), flat.std(),
                                           centroid.var() / (SR / 2) ** 2,
                                           hf_ratio, edge_freq / (SR / 2),
                                           fr_corr, phase_smooth], dtype=np.float64)])


def shard_path(n) -> Path:
    return SHARD_DIR / f"asvspoof21_la_test-{n:05d}.parquet"


def ensure_shard(n, receipts) -> bool:
    import requests
    p = shard_path(n)
    if p.exists() and p.stat().st_size > 10_000_000:
        receipts[str(n)] = {"size": p.stat().st_size, "downloaded": False}
        return True
    p.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading shard {n} (~445MB)...", flush=True)
    with requests.get(URL.format(n), stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(p.with_suffix(".part"), "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    p.with_suffix(".part").rename(p)
    receipts[str(n)] = {"size": p.stat().st_size, "downloaded": True}
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cand", default="p13_fused",
                    help="candidate dir under artifacts/ (W9.1: p13b_fused)")
    ap.add_argument("--reals-complete", action="store_true",
                    help="D-081 resolution design: extract ALL bona-fide rows")
    ap.add_argument("--max-fakes", type=int, default=None,
                    help="D-081 resolution design: seeded fake budget "
                         "(pre-registered analysis value: 50000)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else WS / "artifacts" / (
        f"exam_audio_2021_full_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force "
              f"to overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    # count phase (labels only — cheap parquet column reads)
    import pyarrow.parquet as pq
    counts, n_total = {}, 0
    labs = {}
    for n in range(N_SHARDS):
        tab = pq.read_table(shard_path(n), columns=["label"]) \
            if shard_path(n).exists() else None
        if tab is None:
            raise SystemExit(f"shard {n} missing — run ensure_shard first")
        lab = tab["label"].to_pylist()
        labs[n] = lab
        counts[n] = {"n": len(lab), "fake": sum(1 for v in lab if v == 0),
                     "real": sum(1 for v in lab if v == 1)}
        n_total += len(lab)
    print(f"measured n_total={n_total} | per-shard: "
          f"{json.dumps({k: v['n'] for k, v in counts.items()})}", flush=True)

    # D-081 resolution (2026-11-07): the sealed rule requires the user to rule on
    # subsampling before scoring. Resolution on user word ("proceed ... without
    # wasting computes"): reals-complete + seeded fake budget — PRE-REGISTERED in
    # D-081 BEFORE any scoring (SE(AUC) 0.00103 vs full 0.00071, both far inside
    # the D6/NR2 tolerances; reals are the constraint). Full scoring remains
    # available on explicit user word.
    subsample = None
    if n_total > N_STOP:
        if not (args.reals_complete and args.max_fakes):
            (out / "STOP_subsampling_rule.json").write_text(json.dumps(
                {"n_total": n_total, "rule": f"n > {N_STOP} — sealed rule: user rules "
                 "on subsampling BEFORE scoring"}, indent=1))
            print(f"STOP: n_total={n_total} > {N_STOP} — sealed rule requires the "
                  f"user to rule on subsampling BEFORE scoring", file=sys.stderr)
            return 2
        seed_base = 20261107
        total_fakes = sum(c["fake"] for c in counts.values())
        n_reals = sum(c["real"] for c in counts.values())
        alloc = [int(round(args.max_fakes * counts[n]["fake"] / total_fakes))
                 for n in range(N_SHARDS)]
        drift = args.max_fakes - sum(alloc)
        big = max(range(N_SHARDS), key=lambda k: counts[k]["fake"])
        alloc[big] += drift
        selected, per_shard = {}, []
        for n in range(N_SHARDS):
            fake_idx = np.array([i for i, v in enumerate(labs[n]) if v == 0])
            k = min(alloc[n], len(fake_idx))
            rng = np.random.default_rng(seed_base + n)
            perm = rng.permutation(len(fake_idx))[:k]
            selected[n] = {int(fake_idx[j]) for j in perm}
            per_shard.append({"shard": n, "fake_avail": int(len(fake_idx)),
                              "selected": int(k)})
        n_selected = int(sum(p["selected"] for p in per_shard))
        n_design = n_reals + n_selected
        subsample = {"design": "reals-complete + seeded fake budget",
                     "decision_ref": "D-081 resolution, user word 2026-11-07",
                     "seed_base": seed_base, "max_fakes": args.max_fakes,
                     "n_reals_full": n_reals, "n_fakes_selected": n_selected,
                     "n_design": n_design, "n_full": n_total,
                     "per_shard": per_shard,
                     "se_full_auc": 0.00071, "se_subsample_auc": 0.00103,
                     "tolerance_ref": "D6 |AUC-0.912|<=0.05; NR2 -0.005",
                     "rationale": "scoring extra fakes buys ~zero precision; "
                                  "reals are the constraint (D-081 analysis)"}
        (out / "RULING_subsampling_resolved.json").write_text(
            json.dumps(subsample, indent=1))
        print(f"D-081 resolution: reals-complete ({n_reals}) + {n_selected} seeded "
              f"fakes -> n_design={n_design} (full {n_total}) — receipt written",
              flush=True)

    # extraction phase (resumable per shard)
    import joblib
    import soundfile as sf
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_grad_enabled(False)
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    fe_x = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    m_x = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval()
    fe_w = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    m_w = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()

    def embed(model, fe, seg):
        inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        return model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)  # np.savez does not create parents

    dl_receipts = {}
    skipped = 0
    for n in range(N_SHARDS):
        ensure_shard(n, dl_receipts)
        tab = pq.read_table(shard_path(n))
        acol, lcol = tab["audio"], tab["label"]
        cx = CACHE_DIR / f"xlsr_shard{n:03d}.npz"
        cw = CACHE_DIR / f"wavlm_shard{n:03d}.npz"
        cs = CACHE_DIR / f"spectral_shard{n:03d}.npz"
        cl = CACHE_DIR / f"labels_shard{n:03d}.json"
        cache_x = dict(np.load(cx)) if cx.exists() else {}
        cache_w = dict(np.load(cw)) if cw.exists() else {}
        cache_s = dict(np.load(cs)) if cs.exists() else {}
        labels = json.loads(cl.read_text()) if cl.exists() else {}
        durs_path = CACHE_DIR / f"durs_shard{n:03d}.json"
        durs = json.loads(durs_path.read_text()) if durs_path.exists() else {}
        t0 = time.time()
        done = 0
        # resume law: cache keys are `uid:view:sN`; membership must be by uid
        done_uids = {k.rsplit(":", 2)[0] for k in cache_x}
        for i in range(len(tab)):
            uid = f"shard{n:03d}_row{i:06d}"
            if uid in done_uids:
                continue
            if subsample is not None and lcol[i].as_py() == 0 \
                    and i not in selected[n]:
                skipped += 1
                continue
            rec = acol[i].as_py()
            data = rec["bytes"] if isinstance(rec, dict) else rec
            wav = load_wav_bytes(data) if data else None
            if wav is None or len(wav) < SR // 2:
                labels[uid] = None
                continue
            segs = serve_windows(wav)
            try:
                info = sf.info(io.BytesIO(data))
                durs[uid] = round(info.frames / info.samplerate, 3)
            except Exception:
                durs[uid] = round(len(wav) / SR, 3)
            for si, seg in enumerate(segs):
                key = f"{uid}:clean:s{si}"
                cache_x[key] = embed(m_x, fe_x, seg).astype(np.float16)
                cache_w[key] = embed(m_w, fe_w, seg).astype(np.float16)
                cache_s[key] = spectral_feats(seg).astype(np.float32)
                labels[key] = 1 - int(lcol[i].as_py())  # dataset 0=fake -> court 1=fake
            done_uids.add(uid)
            done += 1
            if done % 500 == 0:
                np.savez(cx, **cache_x)
                np.savez(cw, **cache_w)
                np.savez(cs, **cache_s)
                cl.write_text(json.dumps(labels))
                durs_path.write_text(json.dumps(durs))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"shard {n}: [{done}] {rate:.2f} utt/s", flush=True)
        np.savez(cx, **cache_x)
        np.savez(cw, **cache_w)
        np.savez(cs, **cache_s)
        cl.write_text(json.dumps(labels))
        durs_path.write_text(json.dumps(durs))
        print(f"shard {n}: extracted {done} new ({len(cache_x)//max(1, 1)} vec keys)", flush=True)

    # scoring phase: both arms from the same rows
    p4 = joblib.load(WS / "artifacts/p4/audio_probe.joblib")
    cand_dir = WS / f"artifacts/{args.cand}"  # W9.1: --cand p13b_fused
    court_obj = joblib.load(cand_dir / "audio_court.joblib")
    if not (isinstance(court_obj, dict) and court_obj.get("fusion")
            and "stacker" in court_obj):
        raise SystemExit("audio_court.joblib is not the fused wrapper dict "
                         "(fusion/stacker keys) — rebuild via "
                         "train_audio_w9_stacker.py — STOP")
    court = court_obj["stacker"]
    meta = json.loads((cand_dir / "stacker_meta.json").read_text())
    wl = joblib.load(cand_dir / "wavlm_probe.joblib")
    env = np.load(cand_dir / "spectral_envelope.npz")
    feat_names = meta["feature_names"]
    thr_conf = float(meta["conformal_p10_top_prob"])

    rows = []
    for n in range(N_SHARDS):
        cache_x = dict(np.load(CACHE_DIR / f"xlsr_shard{n:03d}.npz"))
        cache_w = dict(np.load(CACHE_DIR / f"wavlm_shard{n:03d}.npz"))
        cache_s = dict(np.load(CACHE_DIR / f"spectral_shard{n:03d}.npz"))
        labels = json.loads((CACHE_DIR / f"labels_shard{n:03d}.json").read_text())
        durs = json.loads((CACHE_DIR / f"durs_shard{n:03d}.json").read_text())
        per = {}
        for k in cache_x:
            uid, view, si = k.rsplit(":", 2)
            if view != "clean":
                continue
            per.setdefault(uid, []).append((int(si[1:]), k))
        for uid, lst in per.items():
            y = labels.get(lst[0][1])
            if y is None:
                continue
            ks = [k for _, k in sorted(lst)]
            X = np.stack([cache_x[k].astype(np.float32) for k in ks])
            W = np.stack([cache_w[k].astype(np.float32) for k in ks])
            S = np.stack([cache_s[k].astype(np.float64) for k in ks]).mean(0)
            Xz = (X - p4["mu"]) / p4["sd"]
            Wz = (W - wl["mu"]) / wl["sd"]
            f = {"p4_pfake_mean": float(p4["clf"].predict_proba(Xz)[:, 1].mean()),
                 "p4_znorm_mean": float(np.linalg.norm(Xz, axis=1).mean()),
                 "wavlm_pfake_mean": float(wl["clf"].predict_proba(Wz)[:, 1].mean()),
                 "wavlm_znorm_mean": float(np.linalg.norm(Wz, axis=1).mean()),
                 "n_windows": len(ks),
                 "duration_s": durs[uid]}
            f["p4_ooc"] = int(f["p4_znorm_mean"] > p4["coverage_threshold"])
            f["wavlm_ooc"] = int(f["wavlm_znorm_mean"] > wl["coverage_threshold"])
            f["spectral_score"] = float(np.abs((S - env["mean"]) / env["std"]).mean())
            f.update(zip(SP_NAMES, S.tolist()))
            pf = float(court.predict_proba(
                np.array([[f[k] for k in feat_names]], dtype=float))[0, 1])
            abstain = bool((f["p4_ooc"] and f["wavlm_ooc"])
                           or (max(pf, 1.0 - pf) <= thr_conf))
            rows.append({"uid": uid, "shard": n, "label": int(y),
                         "p_fake_audio_mean": round(f["p4_pfake_mean"], 6),
                         "p_fused": round(pf, 6),
                         "wavlm_pfake_mean": round(f["wavlm_pfake_mean"], 6),
                         "spectral_score": round(f["spectral_score"], 6),
                         "p4_ooc": f["p4_ooc"], "wavlm_ooc": f["wavlm_ooc"],
                         "abstained": abstain})
        print(f"shard {n}: scored {len(rows)} cumulative", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows_2021_full.csv", index=False)
    from sklearn.metrics import roc_auc_score, roc_curve

    def eer(y, s):
        fpr, tpr, _ = roc_curve(y, s)
        d = fpr - (1 - tpr)
        i = int(np.argmin(np.abs(d)))
        return float((fpr[i] + (1 - tpr[i])) / 2)

    y, p4s, pfu = df["label"].values, df["p_fake_audio_mean"].values, df["p_fused"].values
    metrics = {
        "n_total_measured": n_total, "n_scored": len(df),
        "n_bad_decode": int((subsample["n_design"] if subsample else n_total)
                            - len(df)),
        "n_skipped_subsample": int(skipped),
        "per_shard": {str(k): v for k, v in counts.items()},
        "shard_download_receipts": dl_receipts,
        "p4_raw_auc_all_rows": round(float(roc_auc_score(y, p4s)), 4),
        "p4_raw_eer_all_rows": round(eer(y, p4s), 4),
        "fused_auc_all_rows": round(float(roc_auc_score(y, pfu)), 4),
        "fused_eer_all_rows": round(eer(y, pfu), 4),
        "wavlm_auc_all_rows": round(float(roc_auc_score(y, df["wavlm_pfake_mean"])), 4),
        "abstain_rate": round(float(df["abstained"].mean()), 4),
        "p4_ooc_rate": round(float(df["p4_ooc"].mean()), 4),
        "wavlm_ooc_rate": round(float(df["wavlm_ooc"].mean()), 4),
        "selection_identity": "both arms scored from the same rows (NR2)",
    }
    if subsample is not None:
        metrics["subsample"] = {**subsample, "n_scored": len(df)}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: metrics[k] for k in
                      ("n_scored", "p4_raw_auc_all_rows", "fused_auc_all_rows",
                       "abstain_rate")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
