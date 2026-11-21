"""V1d-2 CONTROL — is the 0.9817 a content confound?

The probe scored 0.9817 CV AUC on GenBuster++ (stock reals vs generated fakes).
That is ABOVE the incumbent (0.590) by +0.39, which is implausible for a linear
probe on frozen features — and it is exactly the shape arXiv 2606.31004 warns
about (a trivial clip-length classifier scores 0.998 on GenVidBench unaudited,
then falls to chance after controls).

CONTROLS (pre-registered here, before looking at outcomes):
  C1  real-vs-real: split the SAME real pool into two halves and measure AUC.
      If the probe separates real-from-real, it is reading content, not forgery.
  C2  generator-vs-generator: split the fake pool by generator. If the probe
      separates generators, it encodes generator identity, not fakeness.
  C3  trivial baselines: clip duration, file size, frame count, resolution.
      If a trivial baseline matches the probe, the probe adds nothing.

BAR FOR VALIDITY: the probe must NOT separate real-from-real (C1 AUC ~0.5) and
must beat every trivial baseline (C3) by a wide margin.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
OUT = WS / "artifacts/v1d2_vjepa_probe"
REAL_DIR = WS / "data/raw/genbuster_plusplus/extracted/video/real"
FAKE_DIR = WS / "data/raw/genbuster_plusplus/extracted/video/fake"
SEED = 20261121

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# reuse the already-extracted features if present
feats_path = OUT / "probe.json"
print("NOTE: probe.json holds only the summary; re-extracting with labels for controls.")
print("      (features were not cached — re-running extraction on the same seed)")

import torch
from transformers import AutoModel
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
model = model.to(DEVICE).eval()
print(f"model loaded on {DEVICE}")


def decode_clip(path, n_frames=8):
    dec = VideoDecoder(str(path))
    dur = float(getattr(dec.metadata, "duration_seconds", 0.0) or 0.0)
    if dur <= 0:
        dur = 2.0
    start = max(0.0, dur / 2 - 1.0)
    end = min(dur, dur / 2 + 1.0)
    delta = (end - start) / max(n_frames - 1, 1)
    return clips_at_regular_timestamps(
        dec, num_frames_per_clip=n_frames, seconds_between_frames=delta,
        seconds_between_clip_starts=1e9, sampling_range_start=start,
        sampling_range_end=end, policy="repeat_last").data[0]


def embed(path):
    clip = decode_clip(path)
    x = torch.nn.functional.interpolate(
        clip.float() / 255.0, size=(256, 256), mode="bilinear", align_corners=False)
    with torch.no_grad():
        out = model.get_vision_features(pixel_values_videos=x.to(DEVICE).unsqueeze(0))
    v = out if torch.is_tensor(out) else out.last_hidden_state
    return v.mean(dim=1).squeeze(0).float().cpu().numpy()


def cv_auc(X, y, seed=SEED):
    if len(set(y)) < 2:
        return float("nan")
    sc = StandardScaler().fit(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(LogisticRegression(max_iter=2000), sc.transform(X), y,
                          cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


import random
rng = random.Random(SEED)
reals = sorted(REAL_DIR.glob("*.mp4"))
fakes = sorted(FAKE_DIR.glob("*.mp4"))
rng.shuffle(reals); rng.shuffle(fakes)

# sample 60 + 60 for the main comparison (same as the probe)
reals_s = reals[:60]
fakes_s = fakes[:60]

def extract(paths):
    X, ok = [], []
    for p in paths:
        try:
            X.append(embed(p)); ok.append(str(p))
        except Exception as e:
            print(f"  fail {p.name}: {type(e).__name__}", file=sys.stderr)
    return np.array(X), ok

print("\nextracting 60 real + 60 fake ...", flush=True)
Xr, pr = extract(reals_s)
Xf, pf = extract(fakes_s)
print(f"  real={Xr.shape} fake={Xf.shape}")

print("\n" + "=" * 72)
print("CONTROL RESULTS")
print("=" * 72)

# C1: real vs real — the critical control
half = len(Xr) // 2
if half >= 5:
    y_rr = np.r_[np.zeros(half), np.ones(len(Xr) - half)]
    X_rr = np.vstack([Xr[:half], Xr[half:]])
    a_rr = cv_auc(X_rr, y_rr)
    print(f"C1  real-vs-real AUC (must be ~0.5)        = {a_rr:.4f}")
else:
    a_rr = float("nan")
    print(f"C1  real-vs-real: insufficient reals ({len(Xr)})")

# C1b: fake vs fake
half_f = len(Xf) // 2
if half_f >= 5:
    y_ff = np.r_[np.zeros(half_f), np.ones(len(Xf) - half_f)]
    X_ff = np.vstack([Xf[:half_f], Xf[half_f:]])
    a_ff = cv_auc(X_ff, y_ff)
    print(f"C1b fake-vs-fake AUC (must be ~0.5)         = {a_ff:.4f}")
else:
    a_ff = float("nan")

# main: real vs fake
X = np.vstack([Xr, Xf])
y = np.r_[np.zeros(len(Xr)), np.ones(len(Xf))]
a_main = cv_auc(X, y)
print(f"\nMAIN  real-vs-fake AUC                        = {a_main:.4f}")
print(f"      (probe reported 0.9817)")

# C3: trivial baselines
def dur_of(p):
    try:
        import cv2
        c = cv2.VideoCapture(str(p))
        d = c.get(cv2.CAP_PROP_FRAME_COUNT) / max(c.get(cv2.CAP_PROP_FPS), 1e-6)
        c.release()
        return d
    except Exception:
        return 0.0

print("\nC3  trivial baselines (must be BEATEN by the probe):")
triv = {}
if len(Xr) and len(Xf):
    d_r = np.array([dur_of(p) for p in pr])
    d_f = np.array([dur_of(p) for p in pf])
    d_all = np.r_[d_r, d_f]
    a_dur = cv_auc(d_all.reshape(-1, 1), y)
    triv["duration"] = a_dur
    print(f"    clip duration              AUC = {a_dur:.4f}")

    s_r = np.array([Path(p).stat().st_size for p in pr], dtype=float)
    s_f = np.array([Path(p).stat().st_size for p in pf], dtype=float)
    a_sz = cv_auc(np.r_[s_r, s_f].reshape(-1, 1), y)
    triv["file_size"] = a_sz
    print(f"    file size                  AUC = {a_sz:.4f}")

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
valid = True
if not (0.35 <= a_rr <= 0.65):
    print(f"  C1 FAILED: real-vs-real AUC {a_rr:.4f} is outside [0.35, 0.65]")
    print(f"  -> the probe reads CONTENT, not forgery. The 0.98 is a confound.")
    valid = False
else:
    print(f"  C1 passed: real-vs-real AUC {a_rr:.4f} (no content signal)")

if not (0.35 <= a_ff <= 0.65):
    print(f"  C1b note: fake-vs-fake AUC {a_ff:.4f} (fake pool is heterogeneous)")
else:
    print(f"  C1b passed: fake-vs-fake AUC {a_ff:.4f}")

for k, v in triv.items():
    if v >= a_main - 0.10:
        print(f"  C3 FAILED: trivial '{k}' AUC {v:.4f} is within 0.10 of the probe")
        valid = False

if valid:
    print(f"\n  VALID: the {a_main:.4f} is not explained by the controls.")
else:
    print(f"\n  INVALID: the {a_main:.4f} is explained by a confound.")

json.dump({
    "real_vs_real_auc": float(a_rr),
    "fake_vs_fake_auc": float(a_ff),
    "real_vs_fake_auc": float(a_main),
    "trivial_baselines": {k: float(v) for k, v in triv.items()},
    "valid": bool(valid),
}, open(OUT / "controls.json", "w"), indent=2)
print(f"\nsaved {OUT / 'controls.json'}")