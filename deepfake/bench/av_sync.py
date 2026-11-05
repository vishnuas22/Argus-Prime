"""AV-sync channel (W-AV Tier 1): SyncNet-driven lip-sync features.

Spec: artifacts/generator_intake/wav_lipsync_spec.md §2 (sealed D-076).
Vendor: integrations/syncnet_python (MIT; joonson). The SyncNet MODEL is driven
directly on crops built from OUR YuNet observations — the vendor's SFD detector
and preprocessing pipeline are NOT used (spec: "no new detector").

Feature contract (6 values; video_features.AV_FEATURES appends to
FEATURE_NAMES_FULL pre-flip):
  avs_syncnet_mindist    mean over analysis windows of SyncNet min dist
                         (LOWER = better sync; a desynced video scores higher)
  avs_syncnet_conf       mean over windows of SyncNet confidence
                         (median(mdist) - minval; higher = distinct sync peak)
  avs_syncnet_offset_abs median |AV offset| in frames over windows
                         (offset = vshift - minidx, vendor sign convention)
  avs_lipenv_corr        max over windows of max-lag |Pearson| between
                         mouth-region motion energy and the audio RMS envelope
                         (lag search +/-LAG_SEARCH frames at native fps)
  avs_present            1 when an audio stream AND a drift-capable face track
                         exist (the channel ran)
  avs_in_coverage        1 when additionally the speech gate AND the
                         articulation gate pass (the channel is trustworthy)

Missingness contract mirrors AUDIO_FEATURES exactly: avs=None -> avs_present=0,
floats 0.0, avs_in_coverage=0 (compute_row 4th param; D-040 single-featurizer
law — this module is the ONLY source of these values at train and serve).

Analysis windows: up to K_WINDOWS x WINDOW_SEC at native fps (dense decode cap
WINDOW_FRAMES_CAP), chosen deterministically from the largest-area drift-
capable track's contiguous detection runs (longest first, earliest-first
tiebreak). The dense pass re-runs the SAME YuNet detector per window frame
(same model, more frames — lip motion needs dense sampling; the 12-frame
visual bundle is untouched).

SyncNet input contract (vendor-faithful, verified against the checkpoint):
  lip  (B, 3, 5, 224, 224) float32 BGR, RAW 0-255 values (no /255)
  aud  (B, 1, 13, 20)      MFCC (python_speech_features defaults, 13 coeffs),
                           20 frames per 5-video-frame window at 25 fps
                           (100 mfcc/s / 25 fps = 4; generalized to native fps
                           by mstart = round(f * 100 / fps) — exact at 25 fps,
                           approximation elsewhere; disclosed).
Crop geometry reproduces the vendor run_pipeline crop_video: box side bs =
max(w,h)/2, pad cs=0.40, resize 224x224 — built from OUR YuNet box.

Coverage gates (avs_in_coverage=0 when ANY fails):
  (1) track observations inside the windows >= MIN_TRACK_OBS (video_channels)
  (2) audio stream present
  (3) speech gate: voiced-frame fraction >= tau_voice AND mean RMS >= rms_floor
      (a frame is "voiced" when its RMS >= VOICE_REF, a FIXED absolute
      reference — the diagnostic is calibration-independent; tau_voice and
      rms_floor are the calibrated thresholds)
  (4) articulation gate: mean mouth motion energy >= motion_floor
tau_voice / rms_floor / motion_floor are CALIBRATED ON TRAIN-SPLIT REALS ONLY
(D-035 law) by fit_av_calibration(), stored in av_calibration.json beside the
bundle; the serve extractor reads it from VIDEO_STACKER.parent (derived path —
no hardcoded pointer; kills the stale-default class by construction).

Sanity probe (main): a video scored synced vs +0.5s-desynced MUST separate
(higher mindist, lower conf, lower corr on the desynced arm) or the channel is
NOT deployed — exit 2.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[2]
SYNCNET_MODEL = WS / "integrations" / "syncnet_python" / "data" / "syncnet_v2.model"
if str(WS / "deepfake" / "bench") not in sys.path:
    sys.path.insert(0, str(WS / "deepfake" / "bench"))

K_WINDOWS = 3            # analysis windows per video (spec §2)
WINDOW_SEC = 2.0         # each window's duration at native fps
WINDOW_FRAMES_CAP = 150  # dense-decode cap across all windows (spec)
MIN_WIN_FRAMES = 12      # a usable window needs at least this many frames
VSHIFT = 15              # vendor run_syncnet.py default
CROP_SCALE = 0.40        # vendor run_pipeline.py --crop_scale default
CROP_SIZE = 224          # vendor crop output = lip net's expected input
LIP_WIN = 5              # consecutive frames per lip input
MFCC_WIN = 20            # mfcc frames per lip window (vendor: vframe*4 .. +20)
MFCC_PER_SEC = 100       # python_speech_features winstep=0.01
LAG_SEARCH = 5           # lipenv corr lag search ±5 frames (spec)
MIN_CORR_POINTS = 8      # below this a window's corr is skipped
MOUTH_PAD = 2.5          # mouth ROI side = mouth-corner distance x this
VOICE_REF = 25.0         # fixed voiced-frame RMS reference (s16le scale).
                         # MEASURED 2026-11-04 on RAVDESS Actor_06 (the spec's
                         # ~-40 dBFS/328 estimate sat ABOVE all corpus speech):
                         # silence RMS ~2, speech RMS ~50-155 (peak 901 =
                         # -31 dBFS) -> 25 separates ~10x above silence and
                         # ~2x below the quietest speech. Receipted in
                         # artifacts/restack_video_v5/step1_sanity_probe.json.
DEFAULT_FPS = 25.0

# Bump when the extractor's algorithm changes: cached AV results (dump/exam
# resume) with a different version are recomputed, never reused.
AV_SYNC_VERSION = 1

from video_features import AV_FEATURES  # the contract owner (single source)

_NET = None
_DEVICE = None


def _load_syncnet():
    """Process-cached SyncNet (strict state-dict load — shape-checked)."""
    global _NET, _DEVICE
    if _NET is None:
        if not SYNCNET_MODEL.exists():
            raise FileNotFoundError(
                f"SyncNet weights missing: {SYNCNET_MODEL}\n"
                f"Fetch per integrations/syncnet_python/download_model.sh "
                f"(sha256 receipted at acquisition).")
        import torch
        vsp = str(WS / "integrations" / "syncnet_python")
        if vsp not in sys.path:
            sys.path.insert(0, vsp)
        from SyncNetModel import S  # vendored module
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        net = S(num_layers_in_fc_layers=1024).to(dev)
        sd = torch.load(str(SYNCNET_MODEL), map_location="cpu", weights_only=True)
        net.load_state_dict(sd, strict=True)
        net.eval()
        _NET, _DEVICE = net, dev
    return _NET, _DEVICE


# ---------------------------------------------------------------- audio ----

def extract_audio_wav(video: Path) -> tuple[int, np.ndarray] | None:
    """ffmpeg -> 16 kHz mono s16le wav (video_score.audio_channel pattern).
    Returns (sr, int16 samples) or None when no audio stream exists."""
    tmp = Path(tempfile.mkdtemp(prefix="argus_avsync_")) / "a.wav"
    r = subprocess.run(["ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
                        "-ar", "16000", "-acodec", "pcm_s16le", str(tmp)],
                       capture_output=True, timeout=300)
    if not tmp.exists() or tmp.stat().st_size < 1000:
        import shutil
        shutil.rmtree(tmp.parent, ignore_errors=True)
        return None
    from scipy.io import wavfile
    sr, audio = wavfile.read(str(tmp))
    import shutil
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return sr, audio


def rms_envelope(audio: np.ndarray, sr: int, fps: float, n_frames: int) -> np.ndarray:
    """Per-video-frame RMS of the 16 kHz wav (frame-time windows)."""
    out = np.zeros(n_frames, dtype=np.float64)
    fsize = max(1, int(round(sr / fps)))
    x = audio.astype(np.float64)
    for t in range(n_frames):
        seg = x[t * fsize:(t + 1) * fsize]
        out[t] = float(np.sqrt(np.mean(seg ** 2))) if seg.size else 0.0
    return out


# ---------------------------------------------------------------- tracks ----

def _mean_box(track) -> np.ndarray:
    return np.mean([o.box[:4] for o in track.obs], axis=0)


def _select_windows(tracks, fps: float):
    """Largest-area drift-capable track; windows from its coverage span.

    The visual channel samples a SPARSE frame budget (the same tracks feed
    the visual features — D-040 one-featurizer law), so track obs are NOT
    contiguous: consecutive obs sit ~budget-stride apart. Runs therefore
    break only on gaps clearly larger than the sampling stride (real
    detection dropout), never on the stride itself. Each run is tiled into
    non-overlapping WINDOW_SEC windows (a trailing partial window counts
    when >= MIN_WIN_FRAMES); up to K windows survive (longest first,
    earliest-first tiebreak). Dense per-frame re-detection happens later in
    _dense_window — the sparse track only chooses WHERE to look."""
    from video_channels import MIN_TRACK_OBS
    caps = [t for t in tracks if t.n >= MIN_TRACK_OBS]
    if not caps:
        return None, []
    tr = max(caps, key=lambda t: float(np.mean([o.box[2] * o.box[3] for o in t.obs])))
    fids = [o.frame_idx for o in tr.obs]
    if len(fids) < 2:
        return None, []
    strides = [b - a for a, b in zip(fids, fids[1:])]
    tol = max(2.0, 2.0 * float(np.median(strides)))
    runs, cur = [], [fids[0]]
    for a, b in zip(fids, fids[1:]):
        if b - a <= tol:
            cur.append(b)
        else:
            runs.append(cur)
            cur = [b]
    runs.append(cur)
    wf = int(round(WINDOW_SEC * fps))
    cands = []
    for run in runs:
        r0, r1 = run[0], run[-1]
        f = r0
        while f <= r1:
            span = min(wf, r1 - f + 1)
            if span >= MIN_WIN_FRAMES:
                cands.append((f, f + span - 1, span))
            f += wf
    cands.sort(key=lambda c: (-c[2], c[0]))          # longest first, earliest tiebreak
    return tr, [(f0, f1) for f0, f1, _ in cands[:K_WINDOWS]]


def _vendor_crop(img: np.ndarray, box: np.ndarray) -> np.ndarray | None:
    """run_pipeline.crop_video geometry from OUR box: bs=max(w,h)/2, pad 0.40,
    resize 224x224. BGR in, BGR out (the lip net's expected space)."""
    x, y, w, h = [float(v) for v in box[:4]]
    bs = max(w, h) / 2.0
    if bs < 4:
        return None
    cx, cy = x + w / 2.0, y + h / 2.0
    bsi = int(bs * (1 + 2 * CROP_SCALE))
    padded = cv2.copyMakeBorder(img, bsi, bsi, bsi, bsi, cv2.BORDER_CONSTANT,
                                value=(110, 110, 110))
    my, mx = cy + bsi, cx + bsi
    y0, y1 = int(my - bs), int(my + bs * (1 + 2 * CROP_SCALE))
    x0, x1 = int(mx - bs * (1 + CROP_SCALE)), int(mx + bs * (1 + CROP_SCALE))
    face = padded[max(0, y0):y1, max(0, x0):x1]
    if face.size == 0 or face.shape[0] < 8 or face.shape[1] < 8:
        return None
    return cv2.resize(face, (CROP_SIZE, CROP_SIZE))


def _mouth_roi(img_shape: tuple, row: np.ndarray) -> tuple[int, int, int, int] | None:
    """Mouth ROI (x, y, w, h) from the FULL YuNet row (flat indices 10-13 =
    right/left mouth-corner x,y). Falls back to the lower-center of the face
    box when the corners are degenerate."""
    H, W = img_shape[:2]
    if row is not None and np.size(row) >= 14:
        lm = np.asarray(row, dtype=np.float64)
        cx = (lm[10] + lm[12]) / 2.0
        cy = (lm[11] + lm[13]) / 2.0
        side = max(abs(lm[12] - lm[10]), abs(lm[13] - lm[11])) * MOUTH_PAD
        if side >= 4:
            x0 = int(max(0, cx - side / 2)); y0 = int(max(0, cy - side / 2))
            x1 = int(min(W, cx + side / 2)); y1 = int(min(H, cy + side / 2))
            if x1 - x0 >= 4 and y1 - y0 >= 4:
                return (x0, y0, x1 - x0, y1 - y0)
    x, y, w, h = [float(v) for v in row[:4]]
    my0 = int(y + 0.62 * h)
    mh = max(8, int(0.30 * h))
    mx0 = int(x + 0.25 * w)
    mw = max(8, int(0.50 * w))
    if mx0 + mw > W or my0 + mh > H or mw < 4 or mh < 4:
        return None
    return (mx0, my0, mw, mh)


def _dense_window(video: Path, f0: int, f1: int, ref_box: np.ndarray):
    """Decode [f0, f1] densely; YuNet per frame; match to the track by IoU
    (fallback largest). Returns (crops, motion, fids) where motion maps
    frame_idx -> mean |gray diff| inside the window-median mouth ROI, measured
    on the FULL frames (spec: mouth-region motion energy from landmarks).
    Frames are held per window (~150 x 720p x 3B peak) and freed on return."""
    from video_face_track import get_det, prepare_frame

    def iou(a, b):
        ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
        bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
        ix = max(0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        u = a[2] * a[3] + b[2] * b[3] - inter
        return inter / u if u > 0 else 0.0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [], {}, []
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    frames, rows, fids = [], [], []
    det = get_det()
    for fi in range(f0, f1 + 1):
        ok, frame = cap.read()
        if not ok:
            break
        pf = prepare_frame(frame)
        det.setInputSize((pf.shape[1], pf.shape[0]))
        _ret, dets = det.detect(pf)
        pick = None
        if dets is not None and len(dets):
            dets = np.array(dets)
            for d in dets:
                if iou(d, ref_box) >= 0.30:
                    pick = d
                    break
            if pick is None:
                pick = max(dets, key=lambda d: d[2] * d[3])
        if pick is not None:
            frames.append(pf)
            rows.append(np.array(pick, dtype=np.float64))
            fids.append(fi)
    cap.release()
    if not frames:
        return [], {}, []
    # mouth motion on FULL frames inside the window-median mouth ROI
    rois = [_mouth_roi(frames[0].shape, r) for r in rows]
    rois = [r for r in rois if r is not None]
    motion = {}
    if rois:
        rx = int(np.median([r[0] for r in rois])); ry = int(np.median([r[1] for r in rois]))
        rw = int(np.median([r[2] for r in rois])); rh = int(np.median([r[3] for r in rois]))
        prev_g = None
        for fr, fi in zip(frames, fids):
            g = cv2.cvtColor(fr[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY).astype(np.int16)
            if prev_g is not None and g.shape == prev_g.shape:
                motion[fi] = float(np.mean(np.abs(g - prev_g)))
            prev_g = g
    # vendor-style box smoothing within the window (medfilt, odd kernel <= 13)
    from scipy.signal import medfilt
    boxes = [r[:4] for r in rows]
    k = min(13, len(boxes) if len(boxes) % 2 == 1 else len(boxes) - 1)
    if k >= 3:
        arr = np.stack(boxes)
        arr = np.stack([medfilt(arr[:, c], kernel_size=k) for c in range(4)], axis=1)
        boxes = [arr[i] for i in range(len(arr))]
    crops = [_vendor_crop(fr, bx) for fr, bx in zip(frames, boxes)]
    del frames
    return crops, motion, fids


# ------------------------------------------------------------- syncnet ----

def _syncnet_window(crops: list, mfcc: np.ndarray, fids: list, fps: float, net, dev):
    """One analysis window -> (mindist, conf, offset) via the vendor math:
    5-frame lip windows (stride 1 over consecutive detected frames), 20-mfcc
    audio windows, calc_pdist(vshift), mdist=mean over frames, min/offset."""
    import torch
    lip_idx = [i for i, c in enumerate(crops) if c is not None]
    if len(lip_idx) < LIP_WIN:
        return None
    spans, cur = [], [lip_idx[0]]          # consecutive detected-frame spans only
    for a, b in zip(lip_idx, lip_idx[1:]):
        if b == a + 1:
            cur.append(b)
        else:
            spans.append(cur)
            cur = [b]
    spans.append(cur)
    lip_feats, aud_feats = [], []
    for span in spans:
        for j in range(len(span) - LIP_WIN + 1):
            idx = span[j:j + LIP_WIN]
            f_global = fids[idx[0]]
            mstart = int(round(f_global * MFCC_PER_SEC / fps))
            if mstart + MFCC_WIN > mfcc.shape[1]:
                continue
            batch_lip = np.stack([crops[i] for i in idx])            # 5x224x224x3
            lip_feats.append(torch.from_numpy(
                np.transpose(batch_lip, (3, 0, 1, 2)).astype(np.float32)[None]).to(dev))
            aud_feats.append(torch.from_numpy(
                mfcc[:, mstart:mstart + MFCC_WIN].astype(np.float32)[None, None]).to(dev))
    if not lip_feats:
        return None
    BS = 20
    lf, af = [], []
    with torch.no_grad():
        for i in range(0, len(lip_feats), BS):
            lf.append(net.forward_lip(torch.cat(lip_feats[i:i + BS], 0)))
            af.append(net.forward_aud(torch.cat(aud_feats[i:i + BS], 0)))
    lf, af = torch.cat(lf, 0), torch.cat(af, 0)
    feat2p = torch.nn.functional.pad(af, (0, 0, VSHIFT, VSHIFT))     # calc_pdist
    win = VSHIFT * 2 + 1
    dists = [torch.nn.functional.pairwise_distance(
        lf[[i], :].repeat(win, 1), feat2p[i:i + win, :]) for i in range(len(lf))]
    mdist = torch.mean(torch.stack(dists, 1), 1)
    minval, minidx = torch.min(mdist, 0)
    offset = VSHIFT - minidx
    conf = torch.median(mdist) - minval
    return (float(minval.item()), float(conf.item()), float(offset.item()))


def _lipenv_corr(motion: dict, rms: np.ndarray) -> float | None:
    """Max-lag |Pearson| between mouth motion energy (frame_idx -> energy) and
    the RMS envelope. Lag search ±LAG_SEARCH frames; None when too few points
    or a degenerate (zero-variance) series."""
    if len(motion) < MIN_CORR_POINTS:
        return None
    ts = sorted(motion)
    best = None
    for lag in range(-LAG_SEARCH, LAG_SEARCH + 1):
        xs, ys = [], []
        for t in ts:
            u = t + lag
            if 0 <= u < len(rms):
                xs.append(motion[t]); ys.append(rms[u])
        if len(xs) < MIN_CORR_POINTS:
            continue
        if np.std(xs) < 1e-9 or np.std(ys) < 1e-9:
            continue
        r = float(np.corrcoef(xs, ys)[0, 1])
        if best is None or abs(r) > abs(best):
            best = r
    return abs(best) if best is not None else None


# ---------------------------------------------------------------- main ----

@dataclass
class AVResult:
    features: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)

    def row(self) -> dict:
        """The 6 AV feature values under the missingness contract."""
        f = self.features
        return {
            "avs_syncnet_mindist": float(f.get("avs_syncnet_mindist", 0.0)),
            "avs_syncnet_conf": float(f.get("avs_syncnet_conf", 0.0)),
            "avs_syncnet_offset_abs": float(f.get("avs_syncnet_offset_abs", 0.0)),
            "avs_lipenv_corr": float(f.get("avs_lipenv_corr", 0.0)),
            "avs_present": int(f.get("avs_present", 0)),
            "avs_in_coverage": int(f.get("avs_in_coverage", 0)),
        }


def extract_av(video: Path, tracks: list, calib: dict | None = None) -> AVResult:
    """Compute the 6 AV features for one video. Data conditions (no track, no
    audio, too-short windows) -> honest zeros with a reason; infra failures
    (missing model) raise — fail-closed, never silently zeroed."""
    res = AVResult(features={k: 0.0 for k in AV_FEATURES}, diagnostics={})
    diag = res.diagnostics
    if not tracks:
        diag["reason"] = "no_tracks"
        return res
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if not fps or fps != fps or fps <= 1:
        fps = DEFAULT_FPS
    tr, windows = _select_windows(tracks, fps)
    if tr is None:
        diag["reason"] = "no_drift_capable_track"
        return res
    aw = extract_audio_wav(video)
    if aw is None:
        diag["reason"] = "no_audio_stream"
        return res
    sr, audio = aw
    res.features["avs_present"] = 1

    net, dev = _load_syncnet()
    import python_speech_features as psf
    mfcc = np.stack([np.array(c) for c in zip(*psf.mfcc(audio, sr))])  # (13, T)

    mind, conf, offs, corrs = [], [], [], []
    voiced_fracs, rms_means, motion_means = [], [], []
    n_obs_in_windows = 0
    win_end = max(b for _, b in windows)
    rms = rms_envelope(audio, sr, fps, win_end + 1)
    ref_box = _mean_box(tr)
    for (f0, f1) in windows:
        crops, motion, fids = _dense_window(video, f0, f1, ref_box)
        n_obs_in_windows += len(fids)
        if len(fids) < MIN_WIN_FRAMES:
            continue
        w = _syncnet_window(crops, mfcc, fids, fps, net, dev)
        if w is not None:
            mind.append(w[0]); conf.append(w[1]); offs.append(w[2])
        if motion:
            motion_means.append(float(np.mean(list(motion.values()))))
        seg = rms[f0:f1 + 1]
        if seg.size:
            rms_means.append(float(np.mean(seg)))
            voiced_fracs.append(float(np.mean(seg >= VOICE_REF)))
        c_val = _lipenv_corr(motion, rms)
        if c_val is not None:
            corrs.append(c_val)
    diag.update({"windows": [[int(a), int(b)] for a, b in windows],
                 "n_obs_in_windows": int(n_obs_in_windows),
                 "n_sync_windows": len(mind), "n_corr_windows": len(corrs),
                 "voiced_frac_mean": float(np.mean(voiced_fracs)) if voiced_fracs else 0.0,
                 "rms_mean": float(np.mean(rms_means)) if rms_means else 0.0,
                 "motion_mean": float(np.mean(motion_means)) if motion_means else 0.0})
    if not mind:
        diag["reason"] = "no_usable_window"
        return res
    res.features["avs_syncnet_mindist"] = float(np.mean(mind))
    res.features["avs_syncnet_conf"] = float(np.mean(conf))
    res.features["avs_syncnet_offset_abs"] = float(np.median(np.abs(offs)))
    res.features["avs_lipenv_corr"] = float(np.max(corrs)) if corrs else 0.0

    # coverage gates — calibration is REQUIRED for in_coverage (D-035)
    apply_coverage(res.features, diag, calib)
    return res


def apply_coverage(features: dict, diagnostics: dict, calib: dict | None) -> None:
    """Set avs_in_coverage from the diagnostics + calibration (D-035: the
    thresholds are fit on train-split reals ONLY). Mutates both dicts. With
    calib=None the gate is honestly unevaluated (in_coverage=0, reason
    recorded) — never silently treated as covered."""
    if calib is None:
        diagnostics.setdefault("reason", "no_calibration")
        features["avs_in_coverage"] = 0
        return
    speech_ok = (diagnostics.get("voiced_frac_mean", 0.0) >= float(calib["tau_voice"])
                 and diagnostics.get("rms_mean", 0.0) >= float(calib["rms_floor"]))
    artic_ok = diagnostics.get("motion_mean", 0.0) >= float(calib["motion_floor"])
    from video_channels import MIN_TRACK_OBS
    obs_ok = diagnostics.get("n_obs_in_windows", 0) >= MIN_TRACK_OBS
    diagnostics.update({"speech_ok": bool(speech_ok), "artic_ok": bool(artic_ok),
                        "obs_ok": bool(obs_ok)})
    features["avs_in_coverage"] = 1 if (speech_ok and artic_ok and obs_ok) else 0


def cached_extract(video: Path, tracks: list, calib: dict | None,
                   cache_dir: Path | None) -> "AVResult":
    """extract_av with a per-workdir result cache keyed by sha12 + version.

    The cache stores the calibration-INDEPENDENT part (the 4 floats + present
    + diagnostics); avs_in_coverage is recomputed at use time from the stored
    diagnostics + the current calib, so a refitted calibration never serves
    stale coverage decisions. Cache misses (version bump) recompute."""
    import hashlib
    h = hashlib.sha256()
    with open(video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    key = h.hexdigest()[:12]
    res = AVResult(features={k: 0.0 for k in AV_FEATURES}, diagnostics={})
    cpath = (cache_dir / f"{key}.json") if cache_dir else None
    if cpath is not None and cpath.exists():
        try:
            j = json.loads(cpath.read_text())
        except json.JSONDecodeError:
            j = None
        if j and j.get("av_sync_version") == AV_SYNC_VERSION:
            res.features.update({k: float(j["features"][k]) for k in
                                 ("avs_syncnet_mindist", "avs_syncnet_conf",
                                  "avs_syncnet_offset_abs", "avs_lipenv_corr")})
            res.features["avs_present"] = int(j["features"]["avs_present"])
            res.diagnostics.update(j["diagnostics"])
            apply_coverage(res.features, res.diagnostics, calib)
            return res
    res = extract_av(video, tracks, calib=None)   # features only; gate applied below
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        if res.features.get("avs_present"):
            cpath.write_text(json.dumps({
            "av_sync_version": AV_SYNC_VERSION,
            "features": {k: res.features[k] for k in
                         ("avs_syncnet_mindist", "avs_syncnet_conf",
                          "avs_syncnet_offset_abs", "avs_lipenv_corr",
                          "avs_present")},
            "diagnostics": res.diagnostics}, indent=1))
    apply_coverage(res.features, res.diagnostics, calib)
    return res


# ---------------------------------------------------------- calibration ----

def fit_av_calibration(rows: list[dict], out_path: Path) -> dict:
    """Fit tau_voice / rms_floor / motion_floor on TRAIN-SPLIT REALS ONLY
    (D-035): 10th percentile of each diagnostic across fit rows. Rows are the
    diagnostics dicts emitted by extract_av on train reals."""
    vf = [r["voiced_frac_mean"] for r in rows if r.get("n_sync_windows")]
    rm = [r["rms_mean"] for r in rows if r.get("n_sync_windows")]
    mm = [r["motion_mean"] for r in rows if r.get("n_sync_windows")]
    if len(vf) < 20:
        raise SystemExit(f"av_calibration: only {len(vf)} fit rows — need >=20 "
                         f"train reals with usable windows (fail-closed)")
    cal = {"tau_voice": float(np.quantile(vf, 0.10)),
           "rms_floor": float(np.quantile(rm, 0.10)),
           "motion_floor": float(np.quantile(mm, 0.10)),
           "n_videos": len(vf),
           "fit_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "source": "train_split_reals_only (D-035)"}
    out_path.write_text(json.dumps(cal, indent=2))
    return cal


# ------------------------------------------------------------ probe CLI ----

def _score_path(video: Path, calib: dict | None) -> AVResult:
    from video_face_track import cluster_identities, observe_video
    obs, _stats = observe_video(video)
    tracks = cluster_identities(obs)
    return extract_av(video, tracks, calib=calib)


def _wav_tmp(audio: np.ndarray, sr: int) -> Path:
    from scipy.io import wavfile
    p = Path(tempfile.mkdtemp(prefix="argus_avprobe_")) / "d.wav"
    wavfile.write(str(p), sr, audio)
    return p


def main() -> int:
    """Sanity probe: synced vs +0.5s-desynced MUST separate or STOP (spec Step 1)."""
    if len(sys.argv) < 2:
        print("usage: av_sync.py <video> [--calib av_calibration.json]")
        return 2
    video = Path(sys.argv[1])
    calib = None
    if "--calib" in sys.argv:
        calib = json.loads(Path(sys.argv[sys.argv.index("--calib") + 1]).read_text())
    t0 = time.time()
    sync = _score_path(video, calib)
    print(f"synced  : {sync.row()}  diag={sync.diagnostics.get('n_sync_windows')}w "
          f"({time.time() - t0:.1f}s)")
    aw = extract_audio_wav(video)
    if aw is None:
        print("no audio — probe impossible")
        return 2
    sr, audio = aw
    shift = int(0.5 * sr)
    delayed = np.concatenate([np.zeros(shift, dtype=audio.dtype), audio[:-shift]])
    tmp = Path(tempfile.mkdtemp(prefix="argus_avprobe_")) / "desync.mov"
    subprocess.run(["ffmpeg", "-y", "-i", str(video), "-i", str(_wav_tmp(delayed, sr)),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    "-shortest", str(tmp)], capture_output=True, timeout=300)
    t0 = time.time()
    des = _score_path(tmp, calib)
    print(f"desynced: {des.row()}  diag={des.diagnostics.get('n_sync_windows')}w "
          f"({time.time() - t0:.1f}s)")
    s, d = sync.row(), des.row()
    sep_mind = d["avs_syncnet_mindist"] > s["avs_syncnet_mindist"]
    sep_conf = d["avs_syncnet_conf"] < s["avs_syncnet_conf"]
    sep_corr = d["avs_lipenv_corr"] < s["avs_lipenv_corr"]
    ok = (s["avs_present"] == 1 and sep_mind and sep_conf and sep_corr)
    print(f"separation: mindist {'UP' if sep_mind else 'NOT-UP'} | "
          f"conf {'DOWN' if sep_conf else 'NOT-DOWN'} | "
          f"corr {'DOWN' if sep_corr else 'NOT-DOWN'} -> "
          f"{'SEPARATED' if ok else 'FAILED'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
