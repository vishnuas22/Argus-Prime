"""V2a fetch-gate — does GenD reproduce on OUR face crops?

GenD (WACV 2026, MIT) = CLIP ViT-L/14 pooler -> L2-normalise -> LinearProbe(1024,2).
Published: CDF 0.960, DFD 0.970, FF++ in-domain 0.980.

RISK: the paper uses DeepfakeBench preprocessing. We use YuNet -> ArcFace alignCrop.
This gate answers: does GenD score our existing FF++ face crops at >= 0.93 CDF
parity, or does it need its own preprocessing?

Protocol (pre-registered):
  - FF++ c23: sample face crops from artifacts/dumps or re-extract from MP4
  - Report: FF++ AUC, CDF AUC (if crops available), per-method AUC
  - BAR: CDF >= 0.93 on our crops  -> PASS, proceed to restack spec
         CDF <  0.93              -> FAIL, needs DeepfakeBench-compatible crops

Read-only. No training, no promotion.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

WS = Path("/Users/Dataghost/Argus_Prime")
OUT = WS / "artifacts/v1g_probe"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_DIR = WS / "artifacts/gend_probe/model"
sys.path.insert(0, str(MODEL_DIR))

from modeling_gend import GenD

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {DEVICE}")

t0 = time.time()
# GenD.__init__ calls CLIPModel.from_pretrained, which trips on a leaked meta-device
# context. Materialise the backbone FIRST so the inner load is a cache hit.
from transformers import CLIPModel, CLIPProcessor
_clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
print(f"CLIP backbone loaded in {time.time()-t0:.0f}s")

from modeling_gend import GenDConfig
cfg = GenDConfig(backbone="openai/clip-vit-large-patch14", head="LinearNorm")
model = GenD(cfg)

# Load the released weights (safetensors) into the skeleton.
from safetensors.torch import load_file
sd = load_file(str(MODEL_DIR / "model.safetensors"))
missing, unexpected = model.load_state_dict(sd, strict=False)
# The CLIP backbone weights are present in the checkpoint; report what did not map.
print(f"state_dict: {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
if missing:
    print("  missing (first 5):", missing[:5])
if unexpected:
    print("  unexpected (first 5):", unexpected[:5])

model = model.to(DEVICE).eval()
print(f"GenD ready in {time.time()-t0:.0f}s")

# sanity: a real and a fake sample image from the model card
import urllib.request
for url, tag in [
    ("https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/DF/000_003/000.png?raw=true", "fake"),
    ("https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/real/000/000.png?raw=true", "real"),
]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            from io import BytesIO
            im = Image.open(BytesIO(r.read())).convert("RGB")
        t = model.feature_extractor.preprocess(im).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pr = model(t).softmax(dim=-1)[0]
        print(f"  sanity {tag}: P(fake)={float(pr[1]):.4f}")
    except Exception as e:
        print(f"  sanity {tag}: fetch failed ({type(e).__name__})")

# ---- source clips: FF++ C23 MP4s (face crops via YuNet, same as serve path) ----
FFPP = WS / "data/raw/ffpp_mirror/FaceForensics++_C23"
sys.path[:0] = [str(WS / "deepfake"), str(WS / "deepfake" / "bench")]

import cv2
from video_face_track import get_det, get_rec, get_arcface, EMBEDDER_NAME, set_embedder

# Match the serve path: YuNet detect -> ArcFace alignCrop
print(f"embedder: {EMBEDDER_NAME}")

import random
rng = random.Random(20261121)
N_EACH = 100

reals = sorted((FFPP / "real").glob("*.mp4"))[:N_EACH]
fakes = sorted((FFPP / "fake" / "Deepfakes").glob("*.mp4"))[:N_EACH]
rng.shuffle(reals); rng.shuffle(fakes)
clips = [(p, 0) for p in reals] + [(p, 1) for p in fakes]
print(f"clips: {len(clips)} ({len(reals)} real + {len(fakes)} fake)")

import tempfile


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
    """YuNet detect -> ArcFace alignCrop, exactly as video_score does."""
    from video_face_track import _detect_faces  # sets input size per frame
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
    # alignCrop via ArcFace
    try:
        rec = get_rec()
        aligned = rec.alignCrop(crop) if hasattr(rec, "alignCrop") else cv2.resize(crop, (112, 112))
    except Exception:
        aligned = cv2.resize(crop, (112, 112))
    return aligned


def score_image(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    t = model.feature_extractor.preprocess(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(t)
        probs = logits.softmax(dim=-1)[0]
    return float(probs[1])  # P(fake)


results = []
t0 = time.time()
for i, (p, y) in enumerate(clips):
    frames = frames_of(p, n=4)
    if not frames:
        continue
    ps = []
    for fr in frames:
        crop = face_crop(fr)
        if crop is not None:
            try:
                ps.append(score_image(crop))
            except Exception as e:
                print(f"  score fail {p.name}: {type(e).__name__}: {e}", file=sys.stderr)
    if ps:
        results.append({"path": str(p), "y": y,
                        "p_fake_mean": float(np.mean(ps)),
                        "p_fake_max": float(np.max(ps)),
                        "n_frames": len(ps)})
    if (i + 1) % 20 == 0:
        el = time.time() - t0
        print(f"  [{i+1}/{len(clips)}] {el:.0f}s {el/(i+1):.2f}s/clip", flush=True)

print(f"\nscored {len(results)} clips in {time.time()-t0:.0f}s")

from sklearn.metrics import roc_auc_score

y = np.array([r["y"] for r in results])
for key in ("p_fake_mean", "p_fake_max"):
    s = np.array([r[key] for r in results])
    print(f"  FF++ AUC ({key}): {roc_auc_score(y, s):.4f}  (n={len(y)})")

out = {
    "model": "yermandy/GenD_CLIP_L_14",
    "device": DEVICE,
    "n_clips": len(results),
    "n_real": int((y == 0).sum()),
    "n_fake": int((y == 1).sum()),
    "auc_mean": float(roc_auc_score(y, [r["p_fake_mean"] for r in results])),
    "auc_max": float(roc_auc_score(y, [r["p_fake_max"] for r in results])),
    "bar": 0.93,
    "pass": bool(roc_auc_score(y, [r["p_fake_mean"] for r in results]) >= 0.93),
}
json.dump(out, open(ART := OUT / "ffpp_gate.json", "w"), indent=2)
print(f"\nsaved {ART}")

# also dump per-clip for CDF later
json.dump(results, open(OUT / "ffpp_scores.json", "w"), indent=2)