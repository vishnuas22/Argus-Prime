"""Instrument-integrity diagnostic for the W9 fused court battery collapse (D-083).

Question: are the battery's wild-exam numbers (ITW fused raw AUC 0.7377 vs p4
receipt 0.9153) a REAL measurement of the fused court, or a serve-path
integration artifact (the D-060 class of bug)?

Method: 8 bona + 8 spoof ITW files taken from the battery's own sampled list.
  1. Fused serve path   -> per-channel values (p4_pfake_mean, wavlm_pfake_mean,
                           spectral_score, p_fused, abstained)
  2. Legacy p4 serve    -> legacy p_fake (the incumbent receipt path)
  3. Training-path wavlm channel (extract_audio_w9.serve_windows + wavlm embed
     + z-norm + predict_proba mean) -> compared against serve's wavlm_pfake_mean

Verdict rule:
  legacy_p4 == fused.p4_pfake_mean  AND  wav_train == wav_serve  (per file, +-1e-3)
  => instrument intact => the collapse is a GENUINE OOD property of the fusion.
  Any divergence => integration bug => fix before any escalation.
"""
import csv
import sys
from pathlib import Path

import numpy as np

WS = Path("/Users/Dataghost/Argus_Prime")
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

ITW = WS / "data/raw/itw/release_in_the_wild"
rows = list(csv.DictReader(open(WS / "artifacts/exam_audio_itw_p13fused/scores.csv")))
bona = [r for r in rows if r["label"] == "0"][:8]
spoof = [r for r in rows if r["label"] == "1"][:8]
sample = bona + spoof
print(f"sample: {len(bona)} bona + {len(spoof)} spoof from the battery's own list")

import argus_detect

FUSED = WS / "artifacts/p13_fused/audio_court.joblib"
LEGACY = WS / "artifacts/p4/audio_probe.joblib"

argus_detect.AUDIO_PROBE = FUSED
fused = {r["file"]: argus_detect.detect_audio(ITW / r["file"]) for r in sample}
argus_detect.AUDIO_PROBE = LEGACY
legacy = {r["file"]: argus_detect.detect_audio(ITW / r["file"]) for r in sample}

# training-path wavlm channel (CPU — the 2021 exam owns MPS right now)
import joblib
import soundfile as sf
import torch
from extract_audio_w9 import SEG, SR, serve_windows
from transformers import AutoModel, Wav2Vec2FeatureExtractor

torch.set_grad_enabled(False)
fe_w = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
m_w = AutoModel.from_pretrained("microsoft/wavlm-base-plus").eval()
wl = joblib.load(WS / "artifacts/p13_fused/wavlm_probe.joblib")

tp = {}
for r in sample:
    wav, sr = sf.read(ITW / r["file"])
    assert sr == SR, f"{r['file']}: sr={sr}"
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = wav.astype(np.float32)
    ps = []
    with torch.no_grad():
        for seg in serve_windows(wav):
            inp = fe_w(seg, sampling_rate=SR, return_tensors="pt").input_values
            h = m_w(inp).last_hidden_state.mean(dim=1).squeeze(0).float().numpy()
            z = ((h - wl["mu"]) / wl["sd"]).reshape(1, -1)
            ps.append(float(wl["clf"].predict_proba(z)[0, 1]))
    tp[r["file"]] = float(np.mean(ps))

print(f"\n{'file':>14s} {'lab':>3s} {'fused_p':>8s} {'p4@fuse':>8s} {'legacy':>8s} "
      f"{'wav_srv':>8s} {'wav_trn':>8s} {'d_wav':>7s} {'spec':>7s} {'abst':>5s} {'ooc4':>5s} {'oocw':>5s}")
d_p4 = d_wav = 0.0
for r in sample:
    f = r["file"]
    fu, lg = fused[f], legacy[f]
    dw = abs(fu["wavlm_pfake_mean"] - tp[f])
    dp = abs(fu["p4_pfake_mean"] - lg["p_fake_audio_mean"])
    d_wav = max(d_wav, dw)
    d_p4 = max(d_p4, dp)
    print(f"{f:>14.14s} {r['label']:>3s} {fu['p_fake_audio_mean']:8.4f} "
          f"{fu['p4_pfake_mean']:8.4f} {lg['p_fake_audio_mean']:8.4f} "
          f"{fu['wavlm_pfake_mean']:8.4f} {tp[f]:8.4f} {dw:7.4f} "
          f"{fu['spectral_score']:7.3f} {str(fu['abstained'])[:5]:>5s} "
          f"{str(bool(fu['out_of_coverage']))[:5]:>5s} {str(bool(fu['wavlm_ooc']))[:5]:>5s}")

print(f"\nmax |p4_in_fused - legacy_p4| = {d_p4:.5f}")
print(f"max |wavlm_serve - wavlm_train| = {d_wav:.5f}")
INSTRUMENT_OK = d_p4 < 1e-3 and d_wav < 1e-3
print("INSTRUMENT:", "INTACT — collapse is a genuine OOD property of the fusion"
      if INSTRUMENT_OK else "BROKEN — integration bug, fix before escalating")
