"""SPLIT training-free temporal score (pre-registration 2026-11-10).

Binding spec: artifacts/split_prereg/SPEC.md. Standalone, CPU-only, fp32,
deterministic measurement module; never imported by serving code.

Frozen recipe:
  - native decode, deterministic t=0, 3 s window, exactly T=8 frames
  - t_i = i*3.0/8; frame index floor(t_i*fps + 0.5), capped at the last
    decoded frame; fail closed when fewer than 8 frames are decoded
  - crop 10% off each side of the longer edge (20% total), resize 224x224,
    ImageNet norm, no JPEG round-trip
  - vendored XCLIP-B/16 vision tower, 4-D [8,3,224,224] input, final-layer
    patch tokens (CLS excluded) -> [8,196,768]
  - TTR(n) = log2(L1(n)+eps) - log2(L2(n)+eps) with
    L1(n) = sum_t ||p_{t+1,n}-p_{t,n}||2 and
    L2(n) = (sum_t ||p_{t+2,n}-p_{t,n}||2 / 2) * (T-1)/(T-2)
  - LSMI = 1/2 (mean||dx(M)||2 + mean||dy(M)||2), M_t = P_{t+1} - P_t
  - score = mean_patch(TTR)^8 * LSMI; higher = fake

Only _get_encoder, _device and _sanitize_position_ids are reused from
d3_temporal.py. D3's decode_segment and _preprocess (8/16 frames, JPEG q80)
are deliberately not imported or called.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from d3_temporal import _device, _get_encoder, _sanitize_position_ids  # noqa: E402

T = 8
WINDOW_S = 3.0
CROP_FRAC = 0.1
RESIZE = 224
GAMMA = 8
EPS = 1e-8
ENCODER = "xclip-b16"
ENCODER_HF = "microsoft/xclip-base-patch16"
PATCHES = 196
GRID = 14
DIM = 768
MOTION_SIZE = 112
JPEG_Q = 80
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

POLICY = {
    "name": "SPLIT",
    "prereg": "artifacts/split_prereg/SPEC.md",
    "encoder": ENCODER,
    "encoder_hf": ENCODER_HF,
    "frames": T,
    "window_s": WINDOW_S,
    "sampler": ("t_i=i*3.0/8; idx=floor(t_i*fps+0.5); "
                "cap=last_decoded; start_t=0"),
    "crop_frac_per_side": CROP_FRAC,
    "crop": "longer_edge_10pct_each_side",
    "resize": RESIZE,
    "resize_interp": "cv2.INTER_LINEAR",
    "norm": "imagenet",
    "jpeg": "none",
    "gamma": GAMMA,
    "eps": EPS,
    "ttr": ("log2(L1+eps)-log2(L2+eps); "
            "L2=(sum||p_{t+2}-p_t||/2)*(T-1)/(T-2)"),
    "lsmi": "0.5*(mean||dx(M)||+mean||dy(M)||); M_t=P_{t+1}-P_t",
    "score": "mean_ttr**gamma * lsmi",
    "polarity": "higher_fake",
    "dtype": "float32",
    "device": "cpu",
}


class SplitUnavailable(RuntimeError):
    """Expected per-clip fail-closed condition (decode, fps, non-finite)."""


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


CODE_SHA = _sha_file(Path(__file__).resolve())
ENCODER_CODE_SHA = _sha_file(_HERE / "vendor_xclip" / "modeling_xclip.py")
LOADER_CODE_SHA = _sha_file(_HERE / "d3_temporal.py")


def split_recipe_hash(policy: dict = None) -> str:
    """SHA-256 over this module's code SHA, encoder/loader code SHAs, and
    every frozen policy field. Any policy or code change changes the hash."""
    payload = {
        "code_sha256": CODE_SHA,
        "encoder_code_sha256": ENCODER_CODE_SHA,
        "loader_code_sha256": LOADER_CODE_SHA,
        "policy": POLICY if policy is None else policy,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def sample_indices(fps: float, n_decoded: int,
                   t_frames: int = T, window_s: float = WINDOW_S) -> list:
    """Native even sampler from the spec. Fails closed below T frames."""
    if t_frames != T:
        raise ValueError(f"SPLIT is frozen at T={T}; got {t_frames}")
    if n_decoded < T:
        raise SplitUnavailable(f"too_few_frames:{n_decoded}<{T}")
    if not np.isfinite(fps) or fps <= 0:
        raise SplitUnavailable(f"bad_fps:{fps}")
    idx = []
    for i in range(T):
        t_i = i * window_s / t_frames
        j = int(math.floor(t_i * fps + 0.5))
        idx.append(min(j, n_decoded - 1))
    return idx


def decode_even_frames(path: Path, t_frames: int = T,
                       window_s: float = WINDOW_S) -> tuple:
    """Decode the deterministic t=0 3 s window and return exactly T native
    BGR frames plus metadata. No padding, no label-dependent frame count."""
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SplitUnavailable("cannot_open")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise SplitUnavailable(f"bad_fps:{fps}")
        scan_limit = max(1, int(fps * window_s))
        frames = []
        for _ in range(scan_limit):
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        if len(frames) < T:
            raise SplitUnavailable(f"too_few_frames:{len(frames)}<{T}")
        indices = sample_indices(fps, len(frames), t_frames, window_s)
        selected = [frames[j] for j in indices]
        h, w = selected[0].shape[:2]
        meta = {
            "fps": fps,
            "n_decoded": len(frames),
            "scan_limit": scan_limit,
            "frame_indices": indices,
            "width": int(w),
            "height": int(h),
        }
        return selected, meta
    finally:
        cap.release()


def jpeg_roundtrip(frame_bgr: np.ndarray, quality: int = JPEG_Q) -> np.ndarray:
    """Frame-level JPEG q80 laundering, used ONLY by the Stage 1 S6
    compressed-real arm. The primary SPLIT path never applies JPEG."""
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise SplitUnavailable("jpeg_encode_failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise SplitUnavailable("jpeg_decode_failed")
    return out


def preprocess_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop 10% off each side of the longer edge, resize 224x224, ImageNet
    norm. Returns float32 [3,224,224]. No JPEG round-trip."""
    h, w = frame_bgr.shape[:2]
    if w > h:
        cut = int(w * CROP_FRAC)
        frame = frame_bgr[:, cut:w - cut]
    else:
        cut = int(h * CROP_FRAC)
        frame = frame_bgr[cut:h - cut, :]
    if frame.shape[0] < 1 or frame.shape[1] < 1:
        raise SplitUnavailable("empty_crop")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RESIZE, RESIZE),
                     interpolation=cv2.INTER_LINEAR)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def preprocess_frames(frames: list) -> np.ndarray:
    if len(frames) != T:
        raise ValueError(f"SPLIT is frozen at T={T}; got {len(frames)}")
    batch = np.stack([preprocess_frame(f) for f in frames]).astype(np.float32)
    if batch.shape != (T, 3, RESIZE, RESIZE):
        raise ValueError(f"preprocess shape {batch.shape} != "
                         f"({T},3,{RESIZE},{RESIZE})")
    if not np.isfinite(batch).all():
        raise SplitUnavailable("nonfinite_input")
    return batch


def _features_from_frames(frames: list, encoder: str = ENCODER) -> np.ndarray:
    import torch
    if encoder != ENCODER:
        raise ValueError(f"SPLIT encoder is frozen at {ENCODER!r}; "
                         f"got {encoder!r}")
    if _device() != "cpu":
        raise RuntimeError("SPLIT is CPU-only")
    batch = preprocess_frames(frames)
    model, dim = _get_encoder(encoder)
    _sanitize_position_ids(model)
    x = torch.from_numpy(batch)
    with torch.no_grad():
        out = model(pixel_values=x.to(_device()))
    hidden = out.last_hidden_state
    if hidden is None:
        raise ValueError("encoder returned no last_hidden_state")
    feats = hidden[:, 1:, :].float().cpu().numpy()
    if feats.shape != (T, PATCHES, DIM):
        raise ValueError(f"patch feature shape {feats.shape} != "
                         f"({T},{PATCHES},{DIM})")
    if not np.isfinite(feats).all():
        raise SplitUnavailable("nonfinite_features")
    return feats


def patch_features(path: Path, encoder: str = ENCODER,
                   jpeg_q80: bool = False) -> np.ndarray:
    """Return final-layer patch tokens [8,196,768] float32."""
    frames, _meta = decode_even_frames(path)
    if jpeg_q80:
        frames = [jpeg_roundtrip(f) for f in frames]
    return _features_from_frames(frames, encoder)


def ttr(features: np.ndarray) -> np.ndarray:
    """Per-patch TTR, shape [num_patches]. T != 8 fails closed."""
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 3:
        raise ValueError(f"features must be [T,N,D]; got {f.shape}")
    if f.shape[0] != T:
        raise ValueError(f"SPLIT is frozen at T={T}; got {f.shape[0]}")
    diffs = f[1:] - f[:-1]
    diffs_k2 = f[2:] - f[:-2]
    l1 = np.sum(np.linalg.norm(diffs, axis=-1), axis=0)
    l2 = (np.sum(np.linalg.norm(diffs_k2, axis=-1), axis=0) / 2.0) \
        * ((T - 1) / (T - 2))
    out = np.log2(l1 + EPS) - np.log2(l2 + EPS)
    if not np.isfinite(out).all():
        raise SplitUnavailable("nonfinite_ttr")
    return out


def lsmi(features: np.ndarray) -> float:
    """Mean patch-motion spatial-gradient magnitude. T != 8 fails closed."""
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 3:
        raise ValueError(f"features must be [T,N,D]; got {f.shape}")
    if f.shape[0] != T:
        raise ValueError(f"SPLIT is frozen at T={T}; got {f.shape[0]}")
    n_patches = f.shape[1]
    grid = int(round(math.sqrt(n_patches)))
    if grid * grid != n_patches:
        raise ValueError(f"patch count {n_patches} is not a square grid")
    p = f.reshape(T, grid, grid, f.shape[2])
    motion = p[1:] - p[:-1]
    grad_x = motion[:, :, :-1, :] - motion[:, :, 1:, :]
    grad_y = motion[:, :-1, :, :] - motion[:, 1:, :, :]
    mag_x = np.linalg.norm(grad_x, axis=-1)
    mag_y = np.linalg.norm(grad_y, axis=-1)
    value = float((mag_x.mean() + mag_y.mean()) / 2.0)
    if not np.isfinite(value):
        raise SplitUnavailable("nonfinite_lsmi")
    return value


def split_score(ttr_mean: float, lsmi_value: float,
                gamma: int = GAMMA) -> float:
    """score = mean_ttr ** gamma * lsmi; higher = fake."""
    score = float((float(ttr_mean) ** gamma) * float(lsmi_value))
    if not np.isfinite(score):
        raise SplitUnavailable("nonfinite_score")
    return score


def motion_metric(frames: list) -> float:
    """Audit motion control: grayscale 112x112, d_i = mean|f_{i+1}-f_i| over
    the 7 consecutive pairs, motion = std(d_i). Not a classifier."""
    if len(frames) != T:
        raise ValueError(f"motion_metric needs T={T} frames; got {len(frames)}")
    grays = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (MOTION_SIZE, MOTION_SIZE),
                          interpolation=cv2.INTER_LINEAR)
        grays.append(gray.astype(np.float32))
    d = [float(np.mean(np.abs(grays[i + 1] - grays[i])))
         for i in range(T - 1)]
    return float(np.std(np.asarray(d, dtype=np.float64), ddof=0))


def score_clip(path: Path, encoder: str = ENCODER,
               jpeg_q80: bool = False) -> dict:
    """Public entry. score=None on fail-closed (court convention)."""
    t0 = time.time()
    try:
        frames, meta = decode_even_frames(path)
        if jpeg_q80:
            frames = [jpeg_roundtrip(f) for f in frames]
        feats = _features_from_frames(frames, encoder)
        ttr_patch = ttr(feats)
        ttr_mean = float(ttr_patch.mean())
        lsmi_value = lsmi(feats)
        score = split_score(ttr_mean, lsmi_value)
        motion = motion_metric(frames)
    except SplitUnavailable as e:
        return {"score": None, "ttr": None, "lsmi": None,
                "n_frames": None, "feature_shape": None, "motion": None,
                "fps": None, "n_decoded": None, "frame_indices": None,
                "width": None, "height": None, "jpeg_q80": bool(jpeg_q80),
                "unavailable_reason": str(e),
                "latency_s": round(time.time() - t0, 3)}
    except (OSError, cv2.error) as e:
        return {"score": None, "ttr": None, "lsmi": None,
                "n_frames": None, "feature_shape": None, "motion": None,
                "fps": None, "n_decoded": None, "frame_indices": None,
                "width": None, "height": None, "jpeg_q80": bool(jpeg_q80),
                "unavailable_reason": f"decode_error:{type(e).__name__}",
                "latency_s": round(time.time() - t0, 3)}
    return {"score": score, "ttr": ttr_mean, "lsmi": lsmi_value,
            "n_frames": T, "feature_shape": [T, PATCHES, DIM],
            "motion": motion, "fps": meta["fps"],
            "n_decoded": meta["n_decoded"],
            "frame_indices": meta["frame_indices"],
            "width": meta["width"], "height": meta["height"],
            "jpeg_q80": bool(jpeg_q80), "unavailable_reason": None,
            "latency_s": round(time.time() - t0, 3)}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="SPLIT temporal score for one video (CPU-only)")
    ap.add_argument("video", type=str)
    ap.add_argument("--jpeg-q80", action="store_true",
                    help="S6 laundering arm only; primary path has no JPEG")
    args = ap.parse_args()
    print(json.dumps(score_clip(Path(args.video),
                                jpeg_q80=args.jpeg_q80)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
