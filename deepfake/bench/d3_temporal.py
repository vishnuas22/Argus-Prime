"""D3 training-free second-order temporal channel (audit 2026-11-06 fix; SOTA_SCORECARD solution #1).

Recipe extracted from arXiv 2508.00701v2 (ICCV 2025, "D3: Training-Free AI-Generated
Video Detection Using Second-Order Features") + the official MIT-licensed code
(github.com/Zig-HS/D3, transformers==4.57.0 pinned), by the architect spec 2026-11-07:

  - 3 s segment at 8 fps -> up to 16 frames (>=8 required; fail-closed otherwise)
  - per frame: center-crop 10% off each end of the LONGER edge, resize 224x224
    (aspect squash), ImageNet norm (mean .485/.456/.406, std .229/.224/.225),
    JPEG-q80 in-memory round-trip (the paper stores JPEG frames on disk)
  - encoder: XCLIP-B/16 pooler_output per frame -> F0 [T,768]
    (vendored transformers 4.57.0 x_clip — D3's pinned version; 5.15.1 dropped it)
  - F1(k) = L2(F0[k+1] - F0[k]);  F2(k) = F1[k] - F1[k-1];  score = std(F2)
    (Bessel / ddof=1, matching torch.std and paper Eq. 8)
  - polarity: LOW std = fake (real footage carries high second-order volatility;
    generated video is temporally flat). Documented paper numbers: GenVideo AUC
    97.72 / Sora AP 99.91 / OpenSora AP 98.55; JPEG-q60 retention 94.5 mAP.

Documented deviations from the paper (reproducibility over eval-variance fidelity):
  - deterministic t=0 window (paper: seeded random start for clips >3 s)
  - in-memory JPEG-q80 round-trip (paper: on-disk JPEG frames)

No court imports — standalone module; exam_d3_t2v.py calls d3_std(). detect_video
does NOT (D-082: the channel missed its pre-registered bar and was never wired).
Threshold calibration lives in artifacts/d3_t2v/calibration.json (D-035
law: fit on train-split reals only, never on eval data).
"""
import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
FPS_TARGET = 8
WINDOW_S = 3
MAX_FRAMES = 16
MIN_FRAMES = 8
JPEG_Q = 80
CROP_FRAC = 0.1
RESIZE = 224

_ENC_CACHE: dict = {}


def _device():
    # CPU FORCED (2026-11-07): MPS is cross-process NON-deterministic for this
    # model (measured: same file std 0.634/0.779 sdpa, 0.607/0.720 eager —
    # within-process MPS is deterministic but every exam/court process is
    # fresh, and the second-order std metric amplifies tiny feature noise).
    # CPU float32 is deterministic across processes; ~6 s/clip, ~1.8 h for the
    # full eval — inside the spec budget. Revisit only with a determinism
    # receipt in hand.
    return "cpu"


def _sanitize_position_ids(model):
    """transformers-5.x loader intermittently materializes the vendored model's
    non-persistent position_ids buffers as GARBAGE (observed 2026-11-07: max
    5707025168 vs expected 196 -> IndexError inside nn.Embedding, intermittent
    across processes). Recompute both deterministically after every load and
    assert before any forward. The vendored file itself stays verbatim."""
    import torch
    for mod_name in ("vision_model", "text_model"):
        mod = getattr(model, mod_name, None)
        if mod is None or not hasattr(mod, "embeddings"):
            continue
        emb = mod.embeddings
        if not hasattr(emb, "position_embedding"):
            continue
        n = emb.position_embedding.weight.shape[0]
        emb.register_buffer(
            "position_ids",
            torch.arange(n, dtype=torch.long).expand((1, -1)).contiguous(),
            persistent=False)
        assert int(emb.position_ids.max()) == n - 1, "position_ids sanitize failed"


def _get_encoder(name: str):
    """Load once per process (patterns: model loads are 7s+ cold — never per call)."""
    if name in _ENC_CACHE:
        return _ENC_CACHE[name]
    import torch
    if name == "xclip-b16":
        from vendor_xclip.modeling_xclip import XCLIPVisionModel
        model = XCLIPVisionModel.from_pretrained("microsoft/xclip-base-patch16")
        dim = 768
    elif name == "clip-b16":
        from transformers import CLIPVisionModel
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        dim = 768
    else:
        raise ValueError(f"unknown encoder {name!r} (supported: xclip-b16, clip-b16)")
    model.eval()
    if name == "xclip-b16":
        _sanitize_position_ids(model)
    model.to(_device())
    if name == "xclip-b16":
        # assert AFTER the device move too — the buffer must survive it intact
        emb = model.vision_model.embeddings
        assert int(emb.position_ids.max()) == emb.position_embedding.weight.shape[0] - 1
    _ENC_CACHE[name] = (model, dim)
    return _ENC_CACHE[name]


def decode_segment(path: Path) -> list:
    """Deterministic t=0 window: up to WINDOW_S seconds at FPS_TARGET, max 16 frames."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(fps / FPS_TARGET)))
    scan_limit = int(fps * WINDOW_S)
    frames: list = []
    i = 0
    while i < scan_limit and len(frames) < MAX_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(frame)
        i += 1
    cap.release()
    return frames


def _preprocess(frame_bgr: np.ndarray) -> np.ndarray:
    """crop-10% (longer edge, both ends) -> resize 224 (aspect squash) ->
    JPEG-q80 round-trip -> ImageNet norm. Returns float32 [3,224,224]."""
    from PIL import Image
    h, w = frame_bgr.shape[:2]
    if h >= w:
        y0, y1 = int(h * CROP_FRAC / 2), int(h * (1 - CROP_FRAC / 2))
        frame = frame_bgr[y0:y1, :]
    else:
        x0, x1 = int(w * CROP_FRAC / 2), int(w * (1 - CROP_FRAC / 2))
        frame = frame_bgr[:, x0:x1]
    rgb = cv2_cvt(frame)
    pil = Image.fromarray(rgb).resize((RESIZE, RESIZE), Image.BILINEAR)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=JPEG_Q)
    buf.seek(0)
    pil = Image.open(buf)
    arr = np.asarray(pil, dtype=np.float32) / 255.0        # [H,W,3] RGB
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))     # [3,H,W]


def cv2_cvt(frame_bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def frame_features(path: Path, encoder: str = "xclip-b16"):
    """Returns (features [T,768] float32 numpy, input_mode str)."""
    import torch
    model, dim = _get_encoder(encoder)
    frames = decode_segment(path)
    if len(frames) < MIN_FRAMES:
        return None, "too_few_frames"
    batch = np.stack([_preprocess(f) for f in frames])       # [T,3,224,224]
    x = torch.from_numpy(batch)
    dev = _device()
    input_mode = "4d"
    try:
        with torch.no_grad():
            out = model(pixel_values=x.to(dev))
        po = out.pooler_output
        if po.ndim == 3 and po.shape[0] == 1:                # 5-D collapsed batch
            po = po[0]
            input_mode = "5d_collapsed"
    except Exception:
        # 5-D native path (XCLIP expects [B,T,C,H,W] in some versions)
        with torch.no_grad():
            out = model(pixel_values=x.unsqueeze(0).to(dev))
        po = out.pooler_output
        if po.ndim == 3 and po.shape[0] == 1:
            po = po[0]
        input_mode = "5d_fallback"
    feats = po.float().cpu().numpy()
    if feats.ndim != 2 or feats.shape[0] != len(frames) or feats.shape[1] != dim:
        raise ValueError(
            f"encoder output shape {feats.shape} != ({len(frames)}, {dim}) — "
            f"input_mode={input_mode}; refusing to score (assert-at-boundary law)")
    return feats, input_mode


def d3_std(path: Path, encoder: str = "xclip-b16") -> dict:
    """Public entry. Returns {std, n_frames, encoder, input_mode,
    unavailable_reason, latency_s}. std=None on fail-closed (court convention)."""
    t0 = time.time()
    path = Path(path)
    try:
        feats, input_mode = frame_features(path, encoder)
    except Exception as e:
        return {"std": None, "n_frames": None, "encoder": encoder,
                "input_mode": None, "unavailable_reason": f"decode_error:{type(e).__name__}",
                "latency_s": round(time.time() - t0, 3)}
    if feats is None:
        return {"std": None, "n_frames": None, "encoder": encoder,
                "input_mode": None, "unavailable_reason": input_mode,
                "latency_s": round(time.time() - t0, 3)}
    d1 = np.linalg.norm(feats[1:] - feats[:-1], axis=1)      # first-order L2, [T-1]
    d2 = np.diff(d1)                                          # second-order, [T-2]
    std = float(np.std(d2, ddof=1))                           # Bessel == torch.std
    return {"std": round(std, 6), "n_frames": int(feats.shape[0]),
            "encoder": encoder, "input_mode": input_mode,
            "unavailable_reason": None,
            "latency_s": round(time.time() - t0, 3)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="D3 second-order temporal std for one video")
    ap.add_argument("video", type=str)
    ap.add_argument("--encoder", type=str, default="xclip-b16",
                    choices=["xclip-b16", "clip-b16"])
    args = ap.parse_args()
    print(json.dumps(d3_std(Path(args.video), args.encoder)))
