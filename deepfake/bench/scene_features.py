"""SCENE feature family (W4c §2) — full-frame/context evidence + conditioning.

Eight features computed from the SAME 12-frame decode budget and the existing
FaceObs boxes: zero new models, zero new decode passes, zero new dependencies
(cv2/numpy only). Two features are CONDITIONING (face-area share — make the
full-scene actor channel explicit so the stacker stops confounding it with
class, the D4 lever); five are EVIDENCE grounded in the blending-boundary
literature (seam/color/frequency/flicker/sharpness — the D1 lever); one flag
carries missingness.

All coordinates live in the PREPARED frame (video_face_track.prepare_frame may
upscale min(h,w) < 200 videos by a deterministic factor; boxes and frame dims
are consistent within that space, and the ratio-based features are
scale-invariant). Aggregation law: every per-frame/per-pair value is SKIPPED
(not zeroed) when its region is invalid; a feature whose valid-set is empty is
0.0; scene_present stays 1 (the channel ran). Rounding is 4 dp (house style).
"""
import cv2
import numpy as np

SCENE_BAND_REL = 0.25     # boundary-band half-width = 0.25 * max(box_w, box_h)
SCENE_MIN_BW = 2          # minimum band half-width (prepared-frame px)
SCENE_RING_REL = 0.50     # context-ring outer radius beyond the box
SCENE_SIGMA_HP = 2.0      # Gaussian high-pass sigma (prepared-frame px)
SCENE_MIN_INNER = 4       # a frame is inner-valid only if eroded sides >= this
SCENE_EPS = 1e-6          # ratio denominator guard

_SCENE_FLOATS = ("scene_face_area_mean", "scene_face_area_p90",
                 "scene_seam_delta", "scene_color_delta", "scene_freq_ratio",
                 "scene_flicker_delta", "scene_context_sharp")


def _rect_mask(shape, x, y, w, h) -> np.ndarray:
    """Boolean rect mask clipped to the frame (empty when fully outside)."""
    m = np.zeros(shape[:2], dtype=bool)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(int(shape[1]), int(x + w)), min(int(shape[0]), int(y + h))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def scene_block(frames: dict, obs: list) -> dict:
    """The 8 SCENE features for one video — pure function of (frames, obs).

    frames: {frame_idx: prepared BGR frame} for every successfully decoded
    sampled frame, deduped by index (observe_video(..., keep_frames=True));
    obs: list[FaceObs]. Scored frames = sampled frames with >=1 obs; the
    largest-area box of a frame scores it (tie -> first in detection order).
    """
    by_frame = {}
    for o in obs:
        by_frame.setdefault(int(o.frame_idx), []).append(o)
    keys = sorted(frames)
    scored = [t for t in keys if t in by_frame]
    out = {k: 0.0 for k in _SCENE_FLOATS}
    out["scene_present"] = 0
    if not scored:
        return out

    grays = {t: cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY) for t in keys}
    areas, seams, colors, freqs, flickers, sharps = [], [], [], [], [], []
    geom = {}
    for t in scored:
        img = frames[t]
        H, W = img.shape[:2]
        best, best_area = None, -1.0
        for o in by_frame[t]:
            x, y, w, h = [float(v) for v in o.box[:4]]
            if w * h > best_area:
                best, best_area = (x, y, w, h), w * h
        x, y, w, h = best
        if w <= 0 or h <= 0:
            continue
        areas.append((w * h) / (H * W))
        xi, yi, wi, hi = int(x), int(y), int(w), int(h)
        bw = max(SCENE_MIN_BW, int(round(SCENE_BAND_REL * max(wi, hi))))
        face = _rect_mask(img.shape, xi, yi, wi, hi)
        inner = _rect_mask(img.shape, xi + bw, yi + bw, wi - 2 * bw, hi - 2 * bw)
        outer = _rect_mask(img.shape, xi - bw, yi - bw, wi + 2 * bw, hi + 2 * bw)
        band = outer & ~inner
        d = int(round(SCENE_RING_REL * max(wi, hi)))
        ring = _rect_mask(img.shape, xi - d, yi - d, wi + 2 * d, hi + 2 * d) \
            & ~face
        bg = ~outer
        gray = grays[t]
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        grad = np.abs(lap)
        hp = gray.astype(np.float64) - cv2.GaussianBlur(
            gray, (0, 0), SCENE_SIGMA_HP).astype(np.float64)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        inner_ok = (wi - 2 * bw >= SCENE_MIN_INNER
                    and hi - 2 * bw >= SCENE_MIN_INNER and inner.any())
        geom[t] = {"face": face, "bg": bg, "gray": gray}
        if inner_ok:
            seams.append(float(grad[band].mean()) - float(grad[inner].mean()))
        if ring.any() and face.any():
            colors.append(sum(abs(float(lab[..., c][face].mean())
                                  - float(lab[..., c][ring].mean()))
                              for c in range(3)))
            freqs.append(float((hp[face] ** 2).mean())
                         / (float((hp[ring] ** 2).mean()) + SCENE_EPS))
        if bg.any():
            sharps.append(float(lap[bg].var()))

    for i, t in enumerate(keys[:-1]):
        g = geom.get(t)
        if g is None:
            continue
        diff = grays[keys[i + 1]].astype(np.float64) \
            - g["gray"].astype(np.float64)
        d2 = diff * diff
        flickers.append(float(d2[g["face"]].mean())
                        - float(d2[g["bg"]].mean()))

    for name, vals in (("scene_face_area_mean", areas),
                       ("scene_face_area_p90", areas),
                       ("scene_seam_delta", seams),
                       ("scene_color_delta", colors),
                       ("scene_freq_ratio", freqs),
                       ("scene_flicker_delta", flickers),
                       ("scene_context_sharp", sharps)):
        if not vals:
            continue
        if name == "scene_face_area_p90":
            out[name] = round(float(np.percentile(vals, 90)), 4)
        else:
            out[name] = round(float(np.mean(vals)), 4)
    out["scene_present"] = 1
    return out
