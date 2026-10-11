"""Video face-observation layer: sampling, multi-face detect, embed, cluster.

Replaces the old "crop largest face" video path with ALL faces on ALL sampled
frames, clustered into identity tracks by SFace (cv2.FaceRecognizerSF) cosine
similarity. Shared by the identity-aware visual channel (video_score.py) and
the identity-drift temporal module (video_channels.py).

Embedder seam: EMBEDDER_NAME is "sface" (default, zero new deps). The Phase 5
contingency (spec item 2.4) may swap in an ArcFace ONNX behind the same seam;
nothing downstream hardcodes the embedder.

CLIP-scoring crop convention: a face's CLIP crop is the SAME pad-30%/center-
square/256px crop the legacy crop_largest_face pipeline produced, applied to
the face's own YuNet box — so the largest face in a frame is bit-identical to
the legacy crop (exact A/B parity for the naive baseline, single decode pass).
Identity EMBEDDING uses rec.alignCrop (the aligned 112px face), as it must.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[2]
YUNET = WS / "tools" / "models" / "yunet.onnx"
SFACE = WS / "tools" / "models" / "face_recognition_sface_2021dec.onnx"
ARCFACE = WS / "tools" / "models" / "arcface_w600k_r50.onnx"   # contingency 2.4
FRAME_BUDGET_DEFAULT = 12          # 8-16 per user constraint
MAX_FACES_PER_FRAME = 4            # YuNet all detections, score>=0.55 (repo convention)
MAX_TRACKS_SCORED = 2              # top-2 identities by obs count get CLIP scores
COS_THRESH = 0.35                  # SFace same-id cosine threshold (docs ~0.363)
EMBEDDER_DIMS = {"sface": 128, "arcface": 512}
EMBEDDER_NAME = "sface"

_DET = None
_REC = None
_ARC_NET = None


def set_embedder(name: str) -> str:
    """Select the identity embedder behind the seam: 'sface' (default) |
    'arcface' (spec contingency 2.4 — triggered when drift auc_probe < 0.60)."""
    global EMBEDDER_NAME
    if name not in EMBEDDER_DIMS:
        raise SystemExit(f"unknown embedder {name!r}; expected one of "
                         f"{sorted(EMBEDDER_DIMS)}")
    EMBEDDER_NAME = name
    return EMBEDDER_NAME


def get_det() -> cv2.FaceDetectorYN:
    """Process-cached YuNet detector, score_threshold=0.55 (repo convention).

    Input size is (re)set per frame via setInputSize."""
    global _DET
    if _DET is None:
        if not YUNET.exists():
            raise FileNotFoundError(f"YuNet model missing: {YUNET}")
        _DET = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320),
                                         score_threshold=0.55)
    return _DET


def get_rec() -> cv2.FaceRecognizerSF:
    """Process-cached SFace recognizer. Fails loud (clear message) when the
    model is absent — callers must surface this, never silently skip."""
    global _REC
    if _REC is None:
        if not SFACE.exists():
            raise FileNotFoundError(
                f"SFace model missing: {SFACE}\n"
                f"Fetch it (idempotent, sha256-verified): "
                f".venv/bin/python deepfake/scripts/fetch_face_models.py")
        _REC = cv2.FaceRecognizerSF.create(str(SFACE), "")
    return _REC


def get_arcface():
    """Process-cached ArcFace (insightface w600k_r50) via cv2.dnn — zero new
    Python deps; smoke-verified on this env (spec contingency 2.4)."""
    global _ARC_NET
    if _ARC_NET is None:
        if not ARCFACE.exists():
            raise FileNotFoundError(
                f"ArcFace model missing: {ARCFACE}\n"
                f"Fetch it: .venv/bin/python deepfake/scripts/fetch_face_models.py --arcface")
        _ARC_NET = cv2.dnn.readNetFromONNX(str(ARCFACE))
    return _ARC_NET


def sample_frame_indices(total: int, n: int) -> np.ndarray:
    """Even linspace sampling over the frame count (mirrors video_score.py:63)."""
    return np.linspace(0, max(total - 1, 1), n, dtype=int)


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """BGR frame -> the detection image the legacy crop_largest_face used.

    Legacy upscaled frames with min(h,w) < 200 before YuNet; replicate so
    boxes (and hence the legacy largest-face crop) match exactly."""
    if min(frame.shape[:2]) < 200:
        f = max(1.0, 260 / min(frame.shape[:2]))
        return cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return frame


def clip_crop(img: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Legacy crop pipeline applied to a specific face box -> 256px RGB ndarray.

    Identical to learned_face_probe.crop_largest_face for the largest box of
    the same detections (same pad/square/resize math)."""
    x, y, fw, fh = [int(v) for v in box[:4]]
    h, w = img.shape[:2]
    pad = int(0.30 * max(fw, fh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
    face = img[y0:y1, x0:x1]
    side = min(face.shape[:2])
    ys, xs = (face.shape[0] - side) // 2, (face.shape[1] - side) // 2
    return cv2.cvtColor(cv2.resize(face[ys:ys + side, xs:xs + side], (256, 256)),
                        cv2.COLOR_BGR2RGB)


@dataclass(frozen=True)
class FaceObs:
    frame_idx: int
    box: np.ndarray            # (4+..) YuNet row prefix: x,y,w,h,...
    landmarks: np.ndarray      # full 15-row YuNet detection (alignCrop input)
    det_score: float
    emb: np.ndarray            # (128,) float32, L2-normalized SFace feature


@dataclass
class IdentityTrack:
    track_id: int
    obs: list = field(default_factory=list)   # list[FaceObs], sorted by frame_idx

    @property
    def n(self) -> int:
        return len(self.obs)

    @property
    def obs_only(self) -> bool:
        """<2 observations: kept for the record, never votes in verdict/drift."""
        return len(self.obs) < 2

    @property
    def voting(self) -> bool:
        return not self.obs_only


def _embed(row: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Embed one aligned face with the ACTIVE embedder (always unit-norm).

    Both branches share rec.alignCrop (YuNet 5-point alignment); sface uses
    cv2.FaceRecognizerSF.feature (128-d), arcface uses the w600k_r50 ONNX via
    cv2.dnn (512-d, RGB, (x-127.5)/128)."""
    aligned = get_rec().alignCrop(img, row)
    if EMBEDDER_NAME == "arcface":
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = ((rgb - 127.5) / 128.0).transpose(2, 0, 1)[None]
        net = get_arcface()
        net.setInput(blob)
        f = net.forward().reshape(-1).astype(np.float32)
    else:
        f = np.asarray(get_rec().feature(aligned), dtype=np.float32).reshape(-1)
    return f / (np.linalg.norm(f) + 1e-8)


def _detect_faces(img: np.ndarray) -> np.ndarray:
    det = get_det()
    det.setInputSize((img.shape[1], img.shape[0]))
    _, faces = det.detect(img)
    if faces is None or len(faces) == 0:
        return np.zeros((0, 15), dtype=np.float32)
    return faces


def observe_video(path: Path, n_frames: int = FRAME_BUDGET_DEFAULT,
                  with_scores: bool = False):
    """Decode the sampled frames ONCE and observe every face on every frame.

    Returns (obs, stats):
      obs   : list[FaceObs] in (frame_idx, detection order) — <=4 faces/frame
      stats : {"frames_sampled", "frames_decoded", "faces_detected"}
              plus, when with_scores=True:
                "scores"      : np.ndarray (len(obs),) gen-head p_fake per face
                "scores_spec" : np.ndarray (len(obs),) FF++-specialist p_fake
                                per face (same CLIP forward, second head)
                "largest_idx" : list[int] obs index of the largest face per frame
    """
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = sample_frame_indices(total, n_frames)
    obs, scores, scores_spec, largest_idx = [], [], [], []
    frames_decoded = 0
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        frames_decoded += 1
        img = prepare_frame(frame)
        faces = _detect_faces(img)
        if len(faces) == 0:
            continue
        order = np.argsort(-faces[:, 14].astype(np.float32), kind="stable")
        keep = list(order[:MAX_FACES_PER_FRAME])
        big = int(np.argmax(faces[:, 2].astype(np.float32)
                            * faces[:, 3].astype(np.float32)))
        if big not in keep:                      # the legacy-largest face always scores
            keep[-1] = big
        big_pos = keep.index(big)
        for pos in keep:
            row = faces[pos]
            e = _embed(row, img)
            obs.append(FaceObs(frame_idx=int(fi), box=row[:4].copy(),
                               landmarks=row.copy(),
                               det_score=float(row[14]), emb=e))
            if with_scores:
                from video_score import frame_scores
                p_gen, p_spec = frame_scores(clip_crop(img, row[:4]))
                scores.append(p_gen)
                scores_spec.append(p_spec)
        if with_scores:
            largest_idx.append(len(obs) - len(keep) + big_pos)
    cap.release()
    stats = {"frames_sampled": int(len(idxs)), "frames_decoded": frames_decoded,
             "faces_detected": len(obs)}
    if with_scores:
        stats["scores"] = np.asarray(scores, dtype=np.float32)
        stats["scores_spec"] = np.asarray(scores_spec, dtype=np.float32)
        stats["largest_idx"] = [i for i in largest_idx if 0 <= i < len(obs)]
    return obs, stats


def cluster_identities(obs: list) -> list:
    """Deterministic single-pass online clustering in frame order.

    Assign an observation to the existing track with max cosine(emb,
    track mean-emb) >= COS_THRESH, else start a new track. Tracks with <2 obs
    are kept but flagged obs_only (they never vote)."""
    tracks: list = []
    for o in sorted(obs, key=lambda t: t.frame_idx):
        best_id, best_sim = None, -1.0
        for t in tracks:
            mean = np.mean([x.emb for x in t.obs], axis=0)
            sim = float(np.dot(o.emb, mean / (np.linalg.norm(mean) + 1e-8)))
            if sim > best_sim:
                best_id, best_sim = t.track_id, sim
        if best_id is not None and best_sim >= COS_THRESH:
            tracks[best_id].obs.append(o)
        else:
            tracks.append(IdentityTrack(track_id=len(tracks), obs=[o]))
    for t in tracks:
        t.obs.sort(key=lambda x: x.frame_idx)
    return tracks


def _score_obs(path: Path, o) -> float:
    """CLIP face-probe p_fake for one observation (256px legacy-style crop)."""
    from video_score import frame_pfake
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, o.frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"re-decode failed at frame {o.frame_idx}")
    return float(frame_pfake(clip_crop(prepare_frame(frame), o.box)))


def track_scores(track, path: Path) -> np.ndarray:
    """CLIP face-probe p_fake per obs of one track. ONLY call for the top
    MAX_TRACKS_SCORED tracks (each obs costs one CLIP forward + a seek)."""
    return np.asarray([_score_obs(path, o) for o in track.obs], dtype=np.float32)


def track_score_map(tracks: list, path: Path,
                    max_tracks: int = MAX_TRACKS_SCORED) -> dict:
    """{track_id: per-obs CLIP scores} for the top-N tracks by obs count.

    Deterministic tie-break by track_id. Non-scored tracks are absent."""
    ranked = sorted((t for t in tracks if t.voting),
                    key=lambda t: (-t.n, t.track_id))[:max_tracks]
    return {t.track_id: track_scores(t, path) for t in ranked}
