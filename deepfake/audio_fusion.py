"""W9 fused audio court — serve-side module (D-079 SEALED). DORMANT until promotion.

The deployed AUDIO_PROBE pointer stays artifacts/p4 (legacy) until STEP 8
promotion moves BOTH pointers (argus_detect.py + video_score.py) to
artifacts/p13_fused/audio_court.joblib on explicit user authorization. With the
p4 pointer active, detect_audio takes its EXACT legacy path — this module is
never imported.

Branch law (spec §6): detect_audio branches on the loaded bundle's meta —
legacy bundles (no "fusion" key) take the exact current path; fused bundles
route here. Fail-closed loading mirrors the _load_video_stacker law
(argus_detect.py:198-234): a fused bundle missing ANY required asset
(stacker_meta.json / wavlm_probe.joblib / spectral_envelope.npz /
p4_snapshot/audio_probe.joblib) raises FusedBundleError — never a silent
fallback to legacy.

FUSION LAW: scores enter the stacker as FEATURES only. No max()/OR/mean
composition anywhere in this module; abstention is the sealed rule
(p4_ooc AND wavlm_ooc) OR conformal band — never a vote.

spectral_feats is the VERBATIM twin of extract_audio_w9.spectral_feats (the
training-cache implementation); STEP 5's identity check asserts numeric
equality on real windows before any promotion.
"""
import json
from pathlib import Path

import numpy as np

SR = 16000
SEG = SR * 4
SP_BINS = 64
SP_NAMES = ([f"sp_ltas_{i}" for i in range(SP_BINS)] +
            ["sp_flat_mean", "sp_flat_std", "sp_centroid_var", "sp_hf_ratio",
             "sp_edge_freq", "sp_frame_corr", "sp_phase_smooth"])
EXPECTED_PREFIX = ["p4_pfake_mean", "p4_znorm_mean", "p4_ooc",
                   "wavlm_pfake_mean", "wavlm_znorm_mean", "wavlm_ooc",
                   "spectral_score"]


class FusedBundleError(RuntimeError):
    """Fail-closed: the fused court bundle is incomplete or inconsistent."""


def spectral_feats(seg: np.ndarray) -> np.ndarray:
    """VERBATIM twin of extract_audio_w9.spectral_feats — do not edit one
    without the other (STEP 5 identity check is the guard)."""
    from scipy.signal import stft
    f, t, Z = stft(seg, fs=SR, nperseg=1024, noverlap=512, window="hann")
    P = np.abs(Z) ** 2 + 1e-12                      # freq x frames
    ltas = P.mean(axis=1)
    edges = np.geomspace(50.0, SR / 2 - 1, SP_BINS + 1)
    prof = np.zeros(SP_BINS)
    for i in range(SP_BINS):
        m = (f >= edges[i]) & (f < edges[i + 1])
        prof[i] = 10 * np.log10(ltas[m].mean() + 1e-12) if m.any() else -120.0
    flat = np.exp(np.log(P).mean(0)) / P.mean(0)    # per-frame spectral flatness
    centroid = (P * f[:, None]).sum(0) / P.sum(0)   # per-frame spectral centroid
    hf = f > 5500
    hf_ratio = P[hf].sum() / P.sum()
    peak = P.max()
    edge_mask = P.max(axis=1) > peak * 1e-4         # -40dB spectral edge
    edge_freq = f[np.where(edge_mask)[0][-1]] if edge_mask.any() else 0.0
    fn = P / P.sum(0, keepdims=True)
    fr_corr = float(np.mean(np.sum(fn[:, 1:] * fn[:, :-1], 0)))
    top = np.argsort(P.mean(1))[-16:]               # top-16 magnitude bins
    ph = np.unwrap(np.angle(Z[top]), axis=1)
    phase_smooth = float(np.mean(np.std(np.diff(ph, axis=1), axis=1)))
    return np.concatenate([
        prof,                                        # sp_ltas_0..63 (dB)
        np.array([flat.mean(), flat.std(),
                  centroid.var() / (SR / 2) ** 2,
                  hf_ratio, edge_freq / (SR / 2),
                  fr_corr, phase_smooth], dtype=np.float64),
    ])


_WAVLM_MODEL = None


def _get_wavlm_model(device):
    """Process-level cache, mirroring argus_detect._get_audio_model."""
    global _WAVLM_MODEL
    if _WAVLM_MODEL is None:
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
        model = AutoModel.from_pretrained("microsoft/wavlm-base-plus").to(device).eval()
        _WAVLM_MODEL = (fe, model)
    return _WAVLM_MODEL


_court_cache = None


def load_court(court_path) -> dict:
    """Fail-closed bundle load: every required asset must exist and agree."""
    global _court_cache
    court_path = str(court_path)
    if _court_cache is not None and _court_cache["path"] == court_path:
        return _court_cache
    d = Path(court_path).parent
    meta_p = d / "stacker_meta.json"
    wav_p = d / "wavlm_probe.joblib"
    env_p = d / "spectral_envelope.npz"
    p4_p = d / "p4_snapshot" / "audio_probe.joblib"
    missing = [p.name for p in (meta_p, wav_p, env_p, p4_p) if not p.exists()]
    if missing:
        raise FusedBundleError(
            f"fused court bundle at {d} is incomplete — missing {missing}; "
            f"refusing to serve (fail closed, _load_video_stacker law)")
    import joblib
    meta = json.loads(meta_p.read_text())
    if meta.get("fusion") != "feature-level-only":
        raise FusedBundleError(
            f"stacker_meta.fusion={meta.get('fusion')!r} — expected "
            f"'feature-level-only'; refusing to serve")
    names = meta.get("feature_names") or []
    if names[:len(EXPECTED_PREFIX)] != EXPECTED_PREFIX or \
            any(n not in names for n in SP_NAMES) or \
            "n_windows" not in names or "duration_s" not in names:
        raise FusedBundleError(
            "stacker_meta.feature_names does not match the serve row "
            "construction contract — refusing to serve (D-040 skew class)")
    if "prosody_" in " ".join(names):
        raise FusedBundleError(
            "prosody features present but Channel D was CUT at sealing (D-079) "
            "— refusing to serve")
    court = {"path": court_path, "meta": meta,
             "stacker": joblib.load(court_path),
             "wavlm": joblib.load(wav_p),
             "env": np.load(env_p),
             "p4": joblib.load(p4_p)}
    _court_cache = court
    return court


def detect_audio_fused(path: Path, wav: np.ndarray, segs: list,
                       duration_s: float, ext: str, court_path,
                       recovery_enabled: bool = True) -> dict:
    """Three-channel fused scoring. Row construction mirrors
    build_audio_w9_rows.py exactly (clean windows, detect_audio pooling)."""
    import torch
    from argus_detect import _get_audio_model
    court = load_court(court_path)
    meta = court["meta"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_grad_enabled(False)
    fe_x, m_x = _get_audio_model(device)
    fe_w, m_w = _get_wavlm_model(device)

    hx, hw, hs = [], [], []
    with torch.no_grad():
        for seg in segs:
            if len(seg) < SEG:
                seg = np.pad(seg, (0, SEG - len(seg)))
            ix = fe_x(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            hx.append(m_x(ix).last_hidden_state.mean(dim=1).squeeze(0)
                      .float().cpu().numpy())
            iw = fe_w(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            hw.append(m_w(iw).last_hidden_state.mean(dim=1).squeeze(0)
                      .float().cpu().numpy())
            hs.append(spectral_feats(seg))
    Xz = (np.stack(hx) - court["p4"]["mu"]) / court["p4"]["sd"]
    Wz = (np.stack(hw) - court["wavlm"]["mu"]) / court["wavlm"]["sd"]
    S = np.stack(hs).mean(0)
    env_mean, env_std = court["env"]["mean"], court["env"]["std"]

    feats = {
        "p4_pfake_mean": float(court["p4"]["clf"].predict_proba(Xz)[:, 1].mean()),
        "p4_znorm_mean": float(np.linalg.norm(Xz, axis=1).mean()),
        "wavlm_pfake_mean": float(court["wavlm"]["clf"].predict_proba(Wz)[:, 1].mean()),
        "wavlm_znorm_mean": float(np.linalg.norm(Wz, axis=1).mean()),
        "n_windows": len(segs),
        "duration_s": duration_s,
    }
    feats["p4_ooc"] = int(feats["p4_znorm_mean"] > court["p4"]["coverage_threshold"])
    feats["wavlm_ooc"] = int(feats["wavlm_znorm_mean"]
                             > court["wavlm"]["coverage_threshold"])
    feats["spectral_score"] = float(
        (np.abs((S - env_mean) / env_std)).mean())
    feats.update(zip(SP_NAMES, (S.astype(float)).tolist()))

    x = np.array([[feats[n] for n in meta["feature_names"]]], dtype=float)
    p_fused = float(court["stacker"].predict_proba(x)[0, 1])
    thr = float(meta["conformal_p10_top_prob"])
    abstain = bool((feats["p4_ooc"] and feats["wavlm_ooc"])
                   or (max(p_fused, 1.0 - p_fused) <= thr))

    result = {
        "p_fake_audio_mean": round(p_fused, 4),
        "fused": True,
        "p4_pfake_mean": round(feats["p4_pfake_mean"], 4),
        "wavlm_pfake_mean": round(feats["wavlm_pfake_mean"], 4),
        "spectral_score": round(feats["spectral_score"], 4),
        "segments_analyzed": len(segs),
        "znorm_mean": round(feats["p4_znorm_mean"], 1),
        "coverage_threshold": court["p4"]["coverage_threshold"],
        "out_of_coverage": bool(feats["p4_ooc"]),
        "wavlm_ooc": bool(feats["wavlm_ooc"]),
        "abstained": abstain,
        "abstention_rule": meta.get("abstention_rule"),
        "duration_s": round(duration_s, 2),
        "short_upload": bool(duration_s < 2.0),
    }
    if ext == ".mp3":
        result["confidence_degraded"] = True
        result["degraded_reason"] = "mp3 source (accept-list)"
    if recovery_enabled:
        import recovery as recovery_mod
        result["recovery"] = recovery_mod.run(
            "audio",
            {"p_fake_fused": result["p_fake_audio_mean"],
             "abstained": result["abstained"],
             "out_of_coverage": result["out_of_coverage"],
             "short_upload": result["short_upload"],
             "duration_s": result["duration_s"],
             "trust": None, "risk": None},
            {"path": path, "audio_result": result})
    return result
