"""Video evidence-module contract + the identity-drift temporal module.

Why a parallel base class (D-020): the vendored EvidenceModule is image-native
(assess(ctx: ImageContext, d, base_reliability)) and vendored reliability()
has no curve for video module ids (returns _DEFAULT=0.5). Vendored files stay
pristine (D-009), so VideoEvidenceModule MIRROS the contract:
  - same vendored ModuleOutput/Artifact/DegradationState schema (schemas.py)
  - same polarity law (negative = synthetic)
  - same unavailable_reason fail-closed semantics (base.py:55)
  - same crash isolation + invariant enforcement (base.py:67-86)

Identity drift: unit-norm SFace embeddings give consecutive-observation L2
steps along an identity track; fakes drift more than reals. Thresholds are
CALIBRATED on train-split reals only (D-035 calibration law — no exam
peeking). Absolute literature thresholds (PIA arXiv 2510.14241 real L2 2-6 vs
swap 8-12, TI2Net WACV'23) are ArcFace-512 numbers and are NOT hardcoded —
SFace-128 has its own scale.
"""
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
BACKEND = WS / "integrations" / "Argus_Master" / "backend"
for p in (str(BACKEND), str(Path(__file__).parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from schemas import Artifact, DegradationState, ModuleOutput  # noqa: E402
from video_face_track import EMBEDDER_NAME, IdentityTrack  # noqa: E402
import video_face_track as _vft  # noqa: E402

logger = logging.getLogger("argus.video_module")

DRIFT_CALIB = WS / "artifacts" / "p6vid_v2" / "drift_calibration.json"
MIN_TRACK_OBS = 3                  # a drift-capable track needs >=3 observations


@dataclass
class VideoContext:
    """Everything a video module may read (mirrors ImageContext for video)."""
    path: Path
    frames_sampled: int
    tracks: list = field(default_factory=list)          # list[IdentityTrack]
    track_scores: dict = field(default_factory=dict)    # track_id -> per-obs CLIP p_fake
    degradation: DegradationState = field(default_factory=DegradationState)


def track_l2(track: IdentityTrack) -> np.ndarray:
    """Consecutive-observation embedding steps ||e[i+1]-e[i]||_2 (unit-norm)."""
    if len(track.obs) < 2:
        return np.zeros(0, dtype=np.float32)
    e = np.stack([o.emb for o in track.obs])
    return np.linalg.norm(np.diff(e, axis=0), axis=1).astype(np.float32)


def drift_features(tracks: list, spike_thresh: float | None = None) -> dict:
    """Raw drift features for one video (shared by dump rows and the module).

      drift_p95  : max over drift-capable tracks of p95(l2)
      drift_mean : mean over the pooled l2 of drift-capable tracks
      spikes     : count(l2 > spike_thresh) over the pooled l2 (0 when no
                   threshold is available yet — dump backfills; see video_dump)
      n_tracks   : number of drift-capable tracks (>= MIN_TRACK_OBS obs)
    """
    pools, p95s = [], []
    for t in tracks:
        if t.n >= MIN_TRACK_OBS:
            l2 = track_l2(t)
            if len(l2):
                pools.append(l2)
                p95s.append(float(np.percentile(l2, 95)))
    if not pools:
        return {"drift_p95": 0.0, "drift_mean": 0.0, "drift_spikes": 0,
                "n_drift_tracks": 0, "unavailable_reason": "too_few_frames"}
    pooled = np.concatenate(pools)
    spikes = (int((pooled > spike_thresh).sum()) if spike_thresh is not None else 0)
    return {"drift_p95": round(max(p95s), 6),
            "drift_mean": round(float(pooled.mean()), 6),
            "drift_spikes": spikes,
            "n_drift_tracks": len(pools),
            "unavailable_reason": None}


def fit_drift_calibration(rows) -> dict:
    """Drift calibration from TRAIN-SPLIT REAL videos only (D-035 law).

    rows: iterable of per-video dicts with keys drift_p95, n_drift_tracks
    (or drift_p95_max / n_drift_tracks as written by the dump CSV).
    p50/p90 calibrate the module's evidence_score; spike_thresh is the train-
    real p95 of per-video drift (video_dump uses the same definition for its
    feature backfill, so the stacker's spike feature and the module agree)."""
    vals = []
    for r in rows:
        n = r.get("n_drift_tracks", 0)
        v = r.get("drift_p95", r.get("drift_p95_max", 0.0))
        if n and v:
            vals.append(float(v))
    if len(vals) < 2:
        raise SystemExit(f"drift calibration needs >=2 usable train reals, got {len(vals)}")
    a = np.asarray(vals, dtype=np.float64)
    return {"p50": round(float(np.quantile(a, 0.50)), 6),
            "p90": round(float(np.quantile(a, 0.90)), 6),
            "spike_thresh": round(float(np.quantile(a, 0.95)), 6),
            "n_videos": int(len(vals)),
            "fit_on": "ffpp train-split real videos only",
            "embedder": _vft.EMBEDDER_NAME}


class VideoEvidenceModule(ABC):
    """Mirror of vendored base.EvidenceModule for the video branch (see D-020)."""
    module_id: str = "abstract"
    version: str = "0.0.0"

    @abstractmethod
    def assess(self, ctx: VideoContext, base_reliability: float) -> ModuleOutput:
        """Module forensic logic. May raise — run() captures everything."""

    def _unavailable(self, reason: str) -> ModuleOutput:
        return ModuleOutput(
            module_id=self.module_id,
            version=self.version,
            evidence_score=0.0,
            reliability_score=0.0,
            confidence_score=0.0,
            verdict_direction="neutral",
            artifacts=[],
            unavailable_reason=reason,
        )

    def run(self, ctx: VideoContext, base_reliability: float = 0.6) -> ModuleOutput:
        t0 = time.perf_counter()
        try:
            out = self.assess(ctx, base_reliability)
        except Exception:
            logger.exception("video module %s crashed; isolated per contract",
                             self.module_id)
            out = self._unavailable("internal_error")
        out.module_id = self.module_id
        out.version = self.version
        if out.unavailable_reason is not None:
            out.evidence_score = 0.0
            out.reliability_score = 0.0
            out.confidence_score = 0.0
            out.verdict_direction = "neutral"
        else:
            out.reliability_score = min(out.reliability_score,
                                        max(0.0, min(1.0, float(base_reliability))))
        out.compute_ms = int((time.perf_counter() - t0) * 1000)
        return out


class IdentityDriftProbe(VideoEvidenceModule):
    """SFace identity-drift temporal module (replaces the unsupported
    `std < 0.08` pseudo-temporal threshold)."""
    module_id = "identity_drift"
    version = "0.1.0"

    def __init__(self, calib_path: Path | None = None):
        self.calib_path = Path(calib_path) if calib_path else DRIFT_CALIB

    def _calib(self) -> dict | None:
        if not self.calib_path.exists():
            return None
        return json.loads(self.calib_path.read_text())

    def assess(self, ctx: VideoContext, base_reliability: float) -> ModuleOutput:
        feats = drift_features(ctx.tracks)
        if feats["unavailable_reason"]:
            return self._unavailable(feats["unavailable_reason"])
        cal = self._calib()
        if cal is None:
            return self._unavailable("calibration_missing")
        p50, p90 = float(cal["p50"]), float(cal["p90"])
        e = float(-np.clip((feats["drift_p95"] - p50) / (p90 - p50 + 1e-6), 0.0, 1.0))
        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        spikes = drift_features(ctx.tracks,
                                spike_thresh=float(cal["spike_thresh"]))["drift_spikes"]
        return ModuleOutput(
            module_id=self.module_id,
            version=self.version,
            evidence_score=round(e, 4),
            reliability_score=max(0.0, min(1.0, float(base_reliability))),
            confidence_score=round(min(0.95, 0.5 + abs(e)), 4),
            verdict_direction=direction,
            artifacts=[Artifact(
                type="identity_drift",
                description=(f"SFace identity drift p95={feats['drift_p95']:.4f} vs "
                             f"train-real p50={p50:.4f}/p90={p90:.4f}; "
                             f"spikes={spikes}"),
                strength=min(1.0, abs(e) + 0.15),
                checkable_claim=(
                    f"SFace identity drift p95={feats['drift_p95']:.4f} vs real-cal "
                    f"p50={p50:.4f}/p90={p90:.4f}; spikes={spikes} "
                    f"(cal n={cal.get('n_videos')}, fit_on={cal.get('fit_on')})"),
            )],
        )


OPTIONAL_VIDEO_CHANNELS = {
    "identity_drift": {
        "factory": lambda: IdentityDriftProbe(),
        "description": "SFace identity-track embedding drift (temporal, D-020 video contract)",
    },
}


def build_video_channels(spec: str = "all") -> tuple:
    """Build video evidence modules. spec: 'all' | comma-separated ids | 'none'."""
    if spec in ("all", ""):
        want = list(OPTIONAL_VIDEO_CHANNELS)
    elif spec == "none":
        want = []
    else:
        want = [s.strip() for s in spec.split(",") if s.strip()]
        unknown = [i for i in want if i not in OPTIONAL_VIDEO_CHANNELS]
        if unknown:
            raise SystemExit(f"unknown video channels: {unknown}")
    return [OPTIONAL_VIDEO_CHANNELS[i]["factory"]() for i in want], want


def run_video_modules(ctx: VideoContext, spec: str = "all") -> tuple:
    """Run every requested video module; returns ({module_id: ModuleOutput}, ids)."""
    mods, ids = build_video_channels(spec)
    return {m.module_id: m.run(ctx) for m in mods}, ids
