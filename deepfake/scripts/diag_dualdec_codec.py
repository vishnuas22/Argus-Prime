"""Dual-decode probe on codec-exam fakes (D-083 follow-up).

The fused court scores ~0.2 (real-ward) on codec-exam fakes that come from the
SAME 2019-train parquet the W9 diet trained on, where B's val reads 0.97
fake-ward. p4 alone recalls 0.85 on the same files. This probe takes 6 of the
exam's exact stems and computes, per file:
  path A (diet/training decode): parquet bytes -> sf.read(BytesIO) -> np.interp
         -> serve_windows -> embeds -> build_audio_w9_rows-style row -> stacker
  path B (exam/serve decode):    parquet bytes -> ffmpeg pcm_s16le wav (the
         exam's exact wav arm) -> detect_audio (fused serve path)

If p_fused(A) >> p_fused(B) the decode path is the culprit; identical per-channel
values then localize the differing feature. If A == B and both real-ward, the
flip lives in the stacker's non-p4 features on this file class.
"""
import csv
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

WS = Path("/Users/Dataghost/Argus_Prime")
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

from audio_fusion import SP_NAMES
from extract_audio_w9 import SR, load_wav_from_bytes, serve_windows, spectral_feats
from transformers import AutoModel, Wav2Vec2FeatureExtractor

meta = json.loads((WS / "artifacts/p13_fused/stacker_meta.json").read_text())
feat_names = meta["feature_names"]
p4 = joblib.load(WS / "artifacts/p13_fused/p4_snapshot/audio_probe.joblib")
wl = joblib.load(WS / "artifacts/p13_fused/wavlm_probe.joblib")
env = np.load(WS / "artifacts/p13_fused/spectral_envelope.npz")
stacker = joblib.load(WS / "artifacts/p13_fused/audio_court.joblib")["stacker"]

device = "cpu"  # the 2021 exam owns MPS
torch.set_grad_enabled(False)
fe_x = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
m_x = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval()
fe_w = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
m_w = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()


def embed(model, fe, seg):
    inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
    return model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()


def row_from_wav(wav, dur):
    segs = serve_windows(wav)
    hx, hw, hs = [], [], []
    for seg in segs:
        hx.append(embed(m_x, fe_x, seg))
        hw.append(embed(m_w, fe_w, seg))
        hs.append(spectral_feats(seg))
    Xz = (np.stack(hx) - p4["mu"]) / p4["sd"]
    Wz = (np.stack(hw) - wl["mu"]) / wl["sd"]
    S = np.stack(hs).mean(0)
    f = {"p4_pfake_mean": float(p4["clf"].predict_proba(Xz)[:, 1].mean()),
         "p4_znorm_mean": float(np.linalg.norm(Xz, axis=1).mean()),
         "wavlm_pfake_mean": float(wl["clf"].predict_proba(Wz)[:, 1].mean()),
         "wavlm_znorm_mean": float(np.linalg.norm(Wz, axis=1).mean()),
         "n_windows": len(segs), "duration_s": dur}
    f["p4_ooc"] = int(f["p4_znorm_mean"] > p4["coverage_threshold"])
    f["wavlm_ooc"] = int(f["wavlm_znorm_mean"] > wl["coverage_threshold"])
    f["spectral_score"] = float(np.abs((S - env["mean"]) / env["std"]).mean())
    f.update(zip(SP_NAMES, S.astype(float).tolist()))
    x = np.array([[f[n] for n in feat_names]], dtype=float)
    return f, float(stacker.predict_proba(x)[0, 1])


rows = list(csv.DictReader(open(WS / "artifacts/exam_audio_codeccal_p13fused/scores.csv")))
idxs = sorted({int(r["shard_idx"]) for r in rows
               if r["stratum"] == "fake_asv19" and r["arm"] == "wav"})[:6]

tab = pq.read_table(WS / "data/raw/asvspoof/data/train-00000-of-00001.parquet",
                    columns=["audio"])

import argus_detect
argus_detect.AUDIO_PROBE = WS / "artifacts/p13_fused/audio_court.joblib"

tmp = Path(tempfile.mkdtemp(prefix="dualdec_"))
hdr = (f"{'idx':>7s} {'pF_byt':>7s} {'pF_ffm':>7s} | {'p4_byt':>7s} {'p4_ffm':>7s} | "
       f"{'wav_byt':>7s} {'wav_ffm':>7s} | {'spc_byt':>7s} {'spc_ffm':>7s} | {'nw_b':>4s} {'nw_f':>4s} {'dur_b':>6s} {'dur_f':>6s}")
print(hdr)
for idx in idxs:
    rec = tab["audio"][idx].as_py()
    b = rec["bytes"] if isinstance(rec, dict) else rec
    wav_a = load_wav_from_bytes(b)
    info = sf.info(io.BytesIO(b))
    fA, pA = row_from_wav(wav_a, info.frames / info.samplerate)
    src = tmp / f"{idx}_src.flac"
    src.write_bytes(b)
    out = tmp / f"{idx}_wav.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-c:a", "pcm_s16le", "-f", "wav", "-ar", "16000", "-ac", "1",
                    str(out)], check=True)
    res = argus_detect.detect_audio(out)
    print(f"{idx:>7d} {pA:7.4f} {res['p_fake_audio_mean']:7.4f} | "
          f"{fA['p4_pfake_mean']:7.4f} {res['p4_pfake_mean']:7.4f} | "
          f"{fA['wavlm_pfake_mean']:7.4f} {res['wavlm_pfake_mean']:7.4f} | "
          f"{fA['spectral_score']:7.3f} {res['spectral_score']:7.3f} | "
          f"{fA['n_windows']:4d} {res['segments_analyzed']:4d} "
          f"{fA['duration_s']:6.2f} {res['duration_s']:6.2f}")
