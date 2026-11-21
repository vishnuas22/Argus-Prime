"""V1d-2 decisive wave — does frozen V-JEPA2 + linear probe transfer to face-swap?

The GenBuster probe scored 0.9458 (T2V: stock reals vs generated fakes), passing
all three confound controls. But face-swap (FF++/CDF) is a DIFFERENT task:
subtle local artifacts, not semantic generation.

The question that decides whether V-JEPA2 is a real replacement or a T2V-only
specialist:
  Q1  FF++ c23: probe trained on GenBuster, tested on FF++ (cross-TASK transfer)
  Q2  CDF:      probe trained on GenBuster, tested on CDF
  Q3  FF++ c23: probe trained AND tested on FF++ (in-task, 5-fold CV)
  Q4  CDF:      probe trained AND tested on CDF (in-task, 5-fold CV)

BAR (pre-registered):
  - Q3/Q4 in-task AUC >= 0.90 (p6vid_v4 gets 0.912 / 0.948)
  - Q1/Q2 cross-task AUC >= 0.60 (any positive transfer is valuable; the
    literature says frozen video-SSL generalises worse than audio-informed)
  - Real-vs-real control on FF++ must be ~0.5

Read-only. No training of the backbone, no promotion.
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
OUT = WS / "artifacts/v1d2_vjepa_probe"
OUT.mkdir(parents=True, exist_ok=True)

FFPP = WS / "data/raw/ffpp_mirror/FaceForensics++_C23"
CDF = WS / "data/raw/celebdf/Celeb-DF-v2"
GB_REAL = WS / "data/raw/genbuster_plusplus/extracted/video/real"
GB_FAKE = WS / "data/raw/genbuster_plusplus/extracted/video/fake"

SEED = 20261121
FRAMES = 8
N_PER_CLASS = 150   # 300 per domain

import torch
from transformers import AutoModel
from torchcodec.decoders import VideoDecoder
from torchcodec.samplers import clips_at_regular_timestamps

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"loading V-JEPA2 ViT-L/16 on {DEVICE} ...", flush=True)
t0 = time.time()
model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", dtype=torch.float32)
model = model.to(DEVICE).eval()
print(f"loaded in {time.time()-t0:.0f}s")


def decode_clip(path, n_frames=FRAMES):
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


def sample(paths, n, rng):
    p = list(paths)
    rng.shuffle(p)
    return p[:n]


def extract(paths, tag):
    X, keep = [], []
    t0 = time.time()
    for i, p in enumerate(paths):
        try:
            X.append(embed(p)); keep.append(str(p))
        except Exception as e:
            print(f"  fail {p.name}: {type(e).__name__}", file=sys.stderr)
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{tag} {i+1}/{len(paths)}] {el:.0f}s {el/(i+1):.2f}s/clip",
                  flush=True)
    return np.array(X), keep


rng = random.Random(SEED)

# build samples
gb_r = sample(GB_REAL.glob("*.mp4"), N_PER_CLASS, rng)
gb_f = sample(GB_FAKE.glob("*.mp4"), N_PER_CLASS, rng)
ffpp_r = sample((FFPP / "real").glob("*.mp4"), N_PER_CLASS, rng)
ffpp_f = sample((FFPP / "fake" / "Deepfakes").glob("*.mp4"), N_PER_CLASS, rng)
cdf_r = sample((CDF / "YouTube-real").glob("*.mp4"), N_PER_CLASS, rng)
cdf_f = sample((CDF / "Celeb-synthesis").glob("*.mp4"), N_PER_CLASS, rng)

print(f"samples: gb={len(gb_r)}+{len(gb_f)} ffpp={len(ffpp_r)}+{len(ffpp_f)} "
      f"cdf={len(cdf_r)}+{len(cdf_f)}", flush=True)

data = {}
for tag, paths in (("gb_r", gb_r), ("gb_f", gb_f), ("ffpp_r", ffpp_r),
                   ("ffpp_f", ffpp_f), ("cdf_r", cdf_r), ("cdf_f", cdf_f)):
    print(f"\n=== {tag} ({len(paths)} clips) ===", flush=True)
    data[tag] = extract(paths, tag)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def cv_auc(X, y, seed=SEED):
    if len(set(y)) < 2:
        return float("nan")
    sc = StandardScaler().fit(X)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    p = cross_val_predict(LogisticRegression(max_iter=2000), sc.transform(X), y,
                          cv=skf, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def fit_apply(Xtr, ytr, Xte, yte):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ytr)
    p = lr.predict_proba(sc.transform(Xte))[:, 1]
    return float(roc_auc_score(yte, p))


print("\n" + "=" * 72)
print("RESULTS")
print("=" * 72)

res = {}

# Q3: FF++ in-task
X = np.vstack([data["ffpp_r"][0], data["ffpp_f"][0]])
y = np.r_[np.zeros(len(data["ffpp_r"][0])), np.ones(len(data["ffpp_f"][0]))]
res["ffpp_intask"] = cv_auc(X, y)
print(f"Q3  FF++ in-task (5-fold CV)          = {res['ffpp_intask']:.4f}  "
      f"(p6vid_v4: 0.9121)  bar 0.90 -> {'PASS' if res['ffpp_intask']>=0.90 else 'FAIL'}")

# Q4: CDF in-task
X = np.vstack([data["cdf_r"][0], data["cdf_f"][0]])
y = np.r_[np.zeros(len(data["cdf_r"][0])), np.ones(len(data["cdf_f"][0]))]
res["cdf_intask"] = cv_auc(X, y)
print(f"Q4  CDF in-task (5-fold CV)           = {res['cdf_intask']:.4f}  "
      f"(p6vid_v4: 0.9482)  bar 0.90 -> {'PASS' if res['cdf_intask']>=0.90 else 'FAIL'}")

# Q1: GenBuster -> FF++ cross-task
Xtr = np.vstack([data["gb_r"][0], data["gb_f"][0]])
ytr = np.r_[np.zeros(len(data["gb_r"][0])), np.ones(len(data["gb_f"][0]))]
Xte = np.vstack([data["ffpp_r"][0], data["ffpp_f"][0]])
yte = np.r_[np.zeros(len(data["ffpp_r"][0])), np.ones(len(data["ffpp_f"][0]))]
res["gb_to_ffpp"] = fit_apply(Xtr, ytr, Xte, yte)
print(f"Q1  GenBuster -> FF++ (cross-task)    = {res['gb_to_ffpp']:.4f}  "
      f"bar 0.60 -> {'PASS' if res['gb_to_ffpp']>=0.60 else 'FAIL'}")

# Q2: GenBuster -> CDF cross-task
Xte = np.vstack([data["cdf_r"][0], data["cdf_f"][0]])
yte = np.r_[np.zeros(len(data["cdf_r"][0])), np.ones(len(data["cdf_f"][0]))]
res["gb_to_cdf"] = fit_apply(Xtr, ytr, Xte, yte)
print(f"Q2  GenBuster -> CDF (cross-task)     = {res['gb_to_cdf']:.4f}  "
      f"bar 0.60 -> {'PASS' if res['gb_to_cdf']>=0.60 else 'FAIL'}")

# control: FF++ real vs real
n = len(data["ffpp_r"][0])
if n >= 10:
    h = n // 2
    Xrr = np.vstack([data["ffpp_r"][0][:h], data["ffpp_r"][0][h:]])
    yrr = np.r_[np.zeros(h), np.ones(n - h)]
    res["ffpp_real_vs_real"] = cv_auc(Xrr, yrr)
    print(f"\nCTRL FF++ real-vs-real                = {res['ffpp_real_vs_real']:.4f}  "
          f"(must be ~0.5)")

print("\n" + "=" * 72)
print("VERDICT")
print("=" * 72)
ok_q3 = res["ffpp_intask"] >= 0.90
ok_q4 = res["cdf_intask"] >= 0.90
ok_q1 = res["gb_to_ffpp"] >= 0.60
ok_q2 = res["gb_to_cdf"] >= 0.60
print(f"  Q3 FF++ in-task >= 0.90: {'PASS' if ok_q3 else 'FAIL'}")
print(f"  Q4 CDF  in-task >= 0.90: {'PASS' if ok_q4 else 'FAIL'}")
print(f"  Q1 GB->FF++    >= 0.60: {'PASS' if ok_q1 else 'FAIL'}")
print(f"  Q2 GB->CDF     >= 0.60: {'PASS' if ok_q2 else 'FAIL'}")
print()
if ok_q3 and ok_q4:
    print("  IN-TASK bars cleared: V-JEPA2+LR is competitive on face-swap.")
    print("  Cross-task transfer is a separate (harder) question.")
else:
    print("  IN-TASK bars FAILED: frozen V-JEPA2 ViT-L mean-pool does NOT match")
    print("  p6vid_v4 on face-swap. V1d-2 as a wholesale replacement is dead.")
    print("  Possible salvage: T2V specialist channel (GenBuster 0.9458).")

json.dump({**res, "n_per_class": N_PER_CLASS, "frames": FRAMES, "seed": SEED,
           "device": DEVICE}, open(OUT / "face_swap_wave.json", "w"), indent=2)
print(f"\nsaved {OUT / 'face_swap_wave.json'}")