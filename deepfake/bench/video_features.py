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
]

AUDIO_FEATURES = ("audio_pfake", "audio_present", "audio_in_coverage")


def _g(d, k, default=0.0):
    v = (d or {}).get(k, default)
    return default if v is None else v


def compute_row(visual: dict, drift: dict, audio: dict | None) -> dict:
    """Flatten one video's channel outputs into the 24-feature stacker row.

    audio=None (or a dict without a scored probe) => audio_present=0,
    audio_pfake=0.0, audio_in_coverage=0 — the flag carries the missingness,
    an explicit train/serve contract with no silent skew. A missing
    visual["spec"] block (legacy rows) zero-fills the 8 spec features."""
    v, d = visual or {}, drift or {}
    ids = list(v.get("identities") or [])
    id1 = ids[0] if len(ids) > 0 else {}
    id2 = ids[1] if len(ids) > 1 else {}
    spec = v.get("spec") or {}
    sids = list(spec.get("identities") or [])
    sid1 = sids[0] if len(sids) > 0 else {}

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
    }
    return row


def row_to_X(row: dict) -> np.ndarray:
    """Row dict -> feature vector, order = FEATURE_NAMES exactly."""
    return np.asarray([float(row.get(name, 0.0) or 0.0) for name in FEATURE_NAMES],
                      dtype=np.float64)
