"""V1d-2 feasibility probe — frozen V-JEPA2 + linear probe on T2V.

The gap: p6vid_v4 scores AUC 0.590 on GenBuster++ T2V (stock vs generated).
The question: do FROZEN video-SSL features (V-JEPA2 ViT-L/16) separate T2V fakes
from reals better than that, with only a linear probe on top?

Design (cheap, falsifiable):
  - 120 GenBuster++ clips (60 real + 60 fake), 8 frames each
  - frozen V-JEPA2 ViT-L/16 -> mean-pooled patch tokens -> 1024-d per clip
  - 5-fold stratified CV logistic regression
  - BAR: CV AUC >= 0.75 on GenBuster T2V (vs p6vid_v4's 0.590)
  - If PASS -> scale to FF++/CDF + a real restack spec
  - If FAIL -> V1d-2 closed at 0, cheaply

Also reports: per-frame feature extraction timing, and whether MPS works.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

WS = Path("/Users/Dataghost/Argus_Prime")
OUT = WS / "artifacts/v1d2_vjepa_probe"
OUT.mkdir(parents=True, exist_ok=True)

FRAMES = 8
N_EACH = 60
SEED = 20261121

REAL_DIR = WS / "data/raw/genbuster_plusplus/extracted/video/real"
FAKE_DIR = WS / "data/raw/genbuster_plusplus/extracted/video/fake"

print("loading model (facebook/vjepa2-vitl-fpc64-256, 300M) ...", flush=True)
import torch
from transformers import AutoModel

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {DEVICE}")

t0 = time.time()
model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float32)
model = model.to(DEVICE).eval()
print(f"loaded in {time.time()-t0:.1f}s  params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

import cv2
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps


def decode_clip(path, n_frames=FRAMES):
    dec = VideoDecoder(str(path))
    dur = float(getattr(dec.metadata, "duration_seconds", 0.0) or 0.0)
    if dur <= 0:
        dur = 2.0
    center = dur / 2.0
    half = 1.0  # 2 s window
    start = max(0.0, center - half)
    end = min(dur, center + half)
    delta = (end - start) / max(n_frames - 1, 1)
    clip = clips_at_regular_timestamps(
        dec, num_frames_per_clip=n_frames, seconds_between_frames=delta,
        seconds_between_clip_starts=1e9, sampling_range_start=start,
        sampling_range_end=end, policy="repeat_last").data[0]
    return clip  # T x C x H x W


def embed(path):
    clip = decode_clip(path)
    # normalise to 0-1 float, resize to 256
    x = clip.float() / 255.0
    x = torch.nn.functional.interpolate(
        x, size=(256, 256), mode="bilinear", align_corners=False)
    x = x.to(DEVICE).unsqueeze(0)  # 1 x T x C x H x W
    with torch.no_grad():
        out = model.get_vision_features(pixel_values_videos=x)
    v = out if torch.is_tensor(out) else out.last_hidden_state
    # mean over temporal tokens -> [1, D]
    return v.mean(dim=1).squeeze(0).float().cpu().numpy()


import random
rng = random.Random(SEED)
reals = sorted(REAL_DIR.glob("*.mp4"))[:N_EACH]
fakes = sorted(FAKE_DIR.glob("*.mp4"))[:N_EACH]
rng.shuffle(reals); rng.shuffle(fakes)
clips = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
print(f"clips: {len(clips)} ({len(reals)} real, {len(fakes)} fake)")

feats, labels, kept = [], [], []
t0 = time.time()
for i, (p, y) in enumerate(clips):
    try:
        f = embed(p)
        feats.append(f); labels.append(y); kept.append(str(p))
    except Exception as e:
        print(f"  fail {p.name}: {type(e).__name__}: {e}", file=sys.stderr)
    if (i + 1) % 20 == 0:
        el = time.time() - t0
        print(f"  [{i+1}/{len(clips)}] {el:.0f}s  {el/(i+1):.2f}s/clip", flush=True)

X = np.array(feats); y = np.array(labels)
print(f"\nextracted {X.shape} in {time.time()-t0:.0f}s")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

sc = StandardScaler().fit(X)
Xs = sc.transform(X)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0), Xs, y,
                      cv=skf, method="predict_proba")[:, 1]
auc = roc_auc_score(y, p)
print(f"\n=== 5-fold CV AUC (frozen V-JEPA2 + LR) = {auc:.4f} ===")
print(f"    incumbent p6vid_v4 on GenBuster T2V = 0.590")
print(f"    BAR (>= 0.75)                       = {'PASS' if auc >= 0.75 else 'FAIL'}")

json.dump({
    "model": "facebook/vjepa2-vitl-fpc64-256",
    "device": DEVICE,
    "frames": FRAMES,
    "n_real": len(reals), "n_fake": len(fakes),
    "n_extracted": int(len(y)),
    "cv_auc": float(auc),
    "incumbent_auc": 0.590,
    "bar": 0.75,
    "pass": bool(auc >= 0.75),
}, open(OUT / "probe.json", "w"), indent=2)
print(f"saved {OUT / 'probe.json'}")