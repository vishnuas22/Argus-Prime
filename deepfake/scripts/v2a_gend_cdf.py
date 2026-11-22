"""V2a leg 2 — GenD on Celeb-DF-v2 (the real prize: cross-dataset).

Gate (pre-registered in memory/spec_v1d_feasibility_v1 lineage, restated here):
  BAR: CDF AUC >= 0.950 on OUR YuNet/ArcFace crops  -> PASS, proceed to restack spec
       CDF AUC <  0.950                             -> FAIL

Real arm: YouTube-real (300) + Celeb-real (590). Use YouTube-real only, matching
the p6vid_v4 benchmark protocol (benchmarks.json celecdf_auc uses YouTube-real).
Fake arm: Celeb-synthesis (5639).

Read-only. No training, no promotion.
"""
import json
import random
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import cv2

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
OUT = WS / "artifacts/v1g_probe"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = WS / "artifacts/gend_probe/model"
sys.path.insert(0, str(MODEL_DIR))
sys.path[:0] = [str(WS / "deepfake"), str(WS / "deepfake" / "bench")]

from modeling_gend import GenD, GenDConfig
from safetensors.torch import load_file
from video_face_track import _detect_faces, get_rec

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = GenD(GenDConfig(backbone="openai/clip-vit-large-patch14", head="LinearNorm"))
model.load_state_dict(load_file(str(MODEL_DIR / "model.safetensors")), strict=False)
model = model.to(DEVICE).eval()
print(f"GenD ready on {DEVICE}")

CDF = WS / "data/raw/celebdf/Celeb-DF-v2"
N_EACH = 100
rng = random.Random(20261121)

reals = sorted((CDF / "YouTube-real").glob("*.mp4"))[:N_EACH]
fakes = sorted((CDF / "Celeb-synthesis").glob("*.mp4"))[:N_EACH]
rng.shuffle(reals); rng.shuffle(fakes)
clips = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
print(f"clips: {len(clips)} ({len(reals)} real + {len(fakes)} fake)")


def frames_of(path, n=4):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return []
    idxs = np.linspace(0, max(total - 1, 0), n, dtype=int)
    out = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            out.append(fr)
    cap.release()
    return out


def face_crop(frame):
    faces = _detect_faces(frame)
    if len(faces) == 0:
        return None
    h, w = frame.shape[:2]
    f = max(faces, key=lambda x: float(x[2]) * float(x[3]))
    x, y, fw, fh = f[:4]
    x1, y1 = max(0, int(x)), max(0, int(y))
    x2, y2 = min(w, int(x + fw)), min(h, int(y + fh))
    if x2 - x1 < 20 or y2 - y1 < 20:
        return None
    crop = frame[y1:y2, x1:x2]
    try:
        return get_rec().alignCrop(crop)
    except Exception:
        return cv2.resize(crop, (112, 112))


def score(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t = model.feature_extractor.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return float(model(t).softmax(dim=-1)[0][1])


results = []
t0 = time.time()
for i, (p, y) in enumerate(clips):
    ps = []
    for fr in frames_of(p, n=4):
        c = face_crop(fr)
        if c is not None:
            try:
                ps.append(score(c))
            except Exception:
                pass
    if ps:
        results.append({"path": str(p), "y": y,
                        "p_fake_mean": float(np.mean(ps)),
                        "p_fake_max": float(np.max(ps))})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  [{i+1}/{len(clips)}] {el:.0f}s {el/(i+1):.2f}s/clip", flush=True)

from sklearn.metrics import roc_auc_score
y = np.array([r["y"] for r in results])
auc_mean = float(roc_auc_score(y, [r["p_fake_mean"] for r in results]))
auc_max = float(roc_auc_score(y, [r["p_fake_max"] for r in results]))

print(f"\nscored {len(results)} clips in {time.time()-t0:.0f}s")
print(f"  CDF AUC (mean): {auc_mean:.4f}   (n={len(y)})")
print(f"  CDF AUC (max):  {auc_max:.4f}")
print(f"  BAR >= 0.950 -> {'PASS' if auc_mean >= 0.950 else 'FAIL'}")

json.dump({
    "model": "yermandy/GenD_CLIP_L_14",
    "corpus": "Celeb-DF-v2 (YouTube-real vs Celeb-synthesis)",
    "n": len(results),
    "n_real": int((y == 0).sum()),
    "n_fake": int((y == 1).sum()),
    "auc_mean": auc_mean,
    "auc_max": auc_max,
    "bar": 0.950,
    "pass": bool(auc_mean >= 0.950),
    "incumbent_p6vid_v4": 0.9482,
}, open(OUT / "cdf_gate.json", "w"), indent=2)
json.dump(results, open(OUT / "cdf_scores.json", "w"), indent=2)
print(f"saved {OUT / 'cdf_gate.json'}")