"""Video stacker feature row — the SINGLE source of train/serve parity.

One compute_row() is called by BOTH the training dump (video_dump.py) and the
serving path (argus_detect.detect_video). The image stacker learned this
lesson the hard way (D-040 train/serve skew) — never let the two paths
featurize differently.

FEATURE_NAMES is the contract (stacker_meta.feature_names must equal it);
row_to_X() orders columns by it exactly.
"""
import sys
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))

# SCENE family (W4c §2) — full-frame/context features computed from the SAME
# 12-frame decode + existing FaceObs boxes (deepfake/bench/scene_features.py).
SCENE_FEATURES = ("scene_face_area_mean", "scene_face_area_p90",
                  "scene_seam_delta", "scene_color_delta", "scene_freq_ratio",
                  "scene_flicker_delta", "scene_context_sharp",
                  "scene_present")

# W4e promotion flip (2026-11-03): FEATURE_NAMES is now the 32-list — the 24
# deployed features + the 8 SCENE features — matching the promoted p6vid_v4
# stacker's stacker_meta.feature_names (the serve path's fail-closed contract
# requires the equality). FEATURE_NAMES_FULL aliases it (historical pre-flip
# name; all dump/train/exam machinery keeps working unchanged).
FEATURE_NAMES: list = [
    # GEN — court-generalist verdict space (12)
    "vid_video_score", "vid_mean_all", "vid_p90_all", "vid_frac_gt_half",
    "vid_largest_face_mean", "n_identities", "faces_detected",
    "id1_topk_mean", "id1_mean", "id1_p90", "id2_topk_mean", "id2_mean",
    # SPEC — FF++-specialist verdict space (8, same tracks/second head; 5b.2)
    "vid_spec_video_score", "vid_spec_mean_all", "vid_spec_p90_all",
    "vid_spec_frac_gt_half", "id1_spec_topk_mean", "id1_spec_mean",
    "id1_spec_p90", "vid_spec_minus_gen_mean",
    # identity-drift (1)
    "drift_p95_max",
    # advisory audio (D-035-consistent missingness modeling)
    "audio_pfake", "audio_present", "audio_in_coverage",
    # SCENE (W4c §2) — promoted with p6vid_v4
    *SCENE_FEATURES,
]

AUDIO_FEATURES = ("audio_pfake", "audio_present", "audio_in_coverage")

FEATURE_NAMES_FULL: list = FEATURE_NAMES  # alias post-flip


def _g(d, k, default=0.0):
    v = (d or {}).get(k, default)
    return default if v is None else v


def compute_row(visual: dict, drift: dict, audio: dict | None) -> dict:
    """Flatten one video's channel outputs into the stacker feature row.

    The row carries the 24 deployed features plus the 8 SCENE features
    (FEATURE_NAMES_FULL); row_to_X orders whichever space the caller names.
    audio=None (or a dict without a scored probe) => audio_present=0,
    audio_pfake=0.0, audio_in_coverage=0 — the flag carries the missingness,
    an explicit train/serve contract with no silent skew. A missing
    visual["spec"] block (legacy rows) zero-fills the 8 spec features; a
    missing visual["scene"] block zero-fills the 8 scene features the same
    way (scene_present=0 documents the legacy/absent path)."""
    v, d = visual or {}, drift or {}
    ids = list(v.get("identities") or [])
    id1 = ids[0] if len(ids) > 0 else {}
    id2 = ids[1] if len(ids) > 1 else {}
    spec = v.get("spec") or {}
    sids = list(spec.get("identities") or [])
    sid1 = sids[0] if len(sids) > 0 else {}
    scene = v.get("scene") or {}

    present = 1 if (isinstance(audio, dict) and "p_fake_audio_mean" in audio) else 0
    in_cov = 1 if (present and not audio.get("out_of_coverage", False)) else 0
    row = {
        "vid_video_score": _g(v, "video_score"),
        "vid_mean_all": _g(v, "mean"),
        "vid_p90_all": _g(v, "p90"),
        "vid_frac_gt_half": _g(v, "frac_gt_half"),
        "vid_largest_face_mean": _g(v, "largest_face_mean"),
        "n_identities": _g(v, "n_identities"),
        "faces_detected": _g(v, "faces_detected"),
        "id1_topk_mean": _g(id1, "topk_mean"),
        "id1_mean": _g(id1, "mean"),
        "id1_p90": _g(id1, "p90"),
        "id2_topk_mean": _g(id2, "topk_mean"),
        "id2_mean": _g(id2, "mean"),
        "vid_spec_video_score": _g(spec, "video_score"),
        "vid_spec_mean_all": _g(spec, "mean"),
        "vid_spec_p90_all": _g(spec, "p90"),
        "vid_spec_frac_gt_half": _g(spec, "frac_gt_half"),
        "id1_spec_topk_mean": _g(sid1, "topk_mean"),
        "id1_spec_mean": _g(sid1, "mean"),
        "id1_spec_p90": _g(sid1, "p90"),
        # zero-fill on a MISSING spec block (never -gen.mean)
        "vid_spec_minus_gen_mean": (round(_g(spec, "mean") - _g(v, "mean"), 4)
                                    if v.get("spec") else 0.0),
        "drift_p95_max": _g(d, "drift_p95", _g(d, "drift_p95_max")),
        "audio_pfake": float(audio["p_fake_audio_mean"]) if present else 0.0,
        "audio_present": present,
        "audio_in_coverage": in_cov,
        # SCENE missingness contract mirrors AUDIO_FEATURES: block absent =>
        # all 7 floats 0.0 + scene_present=0 (the spec zero-fill precedent).
        "scene_face_area_mean": _g(scene, "scene_face_area_mean"),
        "scene_face_area_p90": _g(scene, "scene_face_area_p90"),
        "scene_seam_delta": _g(scene, "scene_seam_delta"),
        "scene_color_delta": _g(scene, "scene_color_delta"),
        "scene_freq_ratio": _g(scene, "scene_freq_ratio"),
        "scene_flicker_delta": _g(scene, "scene_flicker_delta"),
        "scene_context_sharp": _g(scene, "scene_context_sharp"),
        "scene_present": int(scene.get("scene_present", 0) or 0),
    }
    return row


def row_to_X(row: dict, names=None) -> np.ndarray:
    """Row dict -> feature vector, order = the given names exactly.

    Default names=None keeps the deployed FEATURE_NAMES order (the serve path
    is untouched by construction); the W4c dump/train/exam machinery passes
    FEATURE_NAMES_FULL or the candidate's meta names explicitly."""
    ns = FEATURE_NAMES if names is None else names
    return np.asarray([float(row.get(name, 0.0) or 0.0) for name in ns],
                      dtype=np.float64)
