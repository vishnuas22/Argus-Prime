"""W9 STEP 5 verification: spectral twin identity + fused-path serve smoke.

Three checks, all required before STEP 8 promotion:

1. SPECTRAL IDENTITY — extract_audio_w9.spectral_feats (training-cache truth)
   vs audio_fusion.spectral_feats (serve twin) must be BIT-IDENTICAL on real
   windows (the guard audio_fusion's docstring promises).

2. FUSED SMOKE — AUDIO_PROBE pointed at the fused bundle; detect_audio must
   route to the fused branch (result["fused"] is True), the p4 channel must
   reproduce the legacy before-JSON numbers on the same file (same XLS-R math:
   p_fake 0.0, znorm 52.5, threshold 67.4), and the sealed abstention rule must
   evaluate without error.

3. LEGACY ROUTING GUARD — with the pointer at the legacy p4 bundle (dict
   without "fusion"), detect_audio must take the exact legacy path (no
   "fused" key in the result) and reproduce the legacy numbers.

Usage: ./.venv/bin/python deepfake/scripts/smoke_fused_court.py [--cand p13_fused]
Exit 0 = all checks pass; receipt written to artifacts/<cand>/SMOKE_RECEIPT.json
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
TMP = Path(__import__("os").environ.get("ARGUS_TMP") or __import__("tempfile").gettempdir())
sys.path.insert(0, str(WS / "deepfake"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", default="p13_fused",
                    help="candidate dir under artifacts/ (W9.1: p13b_fused)")
    args = ap.parse_args()
    receipt = {}

    # ---------- 1. spectral twin identity ----------
    spec = importlib.util.spec_from_file_location(
        "extract_audio_w9", WS / "deepfake/scripts/extract_audio_w9.py")
    ex = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ex)
    import audio_fusion as af

    wav, sr = sf.read(TMP / "la_before_sample.flac", dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != ex.SR:
        g = np.arange(len(wav)) * (ex.SR / sr)
        wav = np.interp(g, np.arange(len(wav)), wav).astype(np.float32)
    segs = [wav[i:i + ex.SEG]
            for i in range(0, max(1, len(wav) - ex.SEG + 1), ex.SEG)][:8] or \
           [np.pad(wav[:ex.SEG], (0, max(0, ex.SEG - len(wav))))]
    segs = [np.pad(s, (0, ex.SEG - len(s))) if len(s) < ex.SEG else s for s in segs]
    n_id = 0
    for i, seg in enumerate(segs[:3]):
        a = ex.spectral_feats(seg)
        b = af.spectral_feats(seg)
        assert a.shape == b.shape, f"win{i} shape {a.shape} vs {b.shape}"
        assert np.array_equal(a, b), \
            f"win{i} NOT bit-identical: maxdiff={float(np.abs(a - b).max())}"
        n_id += 1
    print(f"1. SPECTRAL IDENTITY: {n_id} windows bit-identical (PASS)", flush=True)
    receipt["spectral_identity_windows"] = n_id

    # ---------- 2. fused-path smoke ----------
    import argus_detect as ad
    legacy_ptr = ad.AUDIO_PROBE
    fused_path = WS / f"artifacts/{args.cand}/audio_court.joblib"
    wrapper = joblib.load(fused_path)
    assert isinstance(wrapper, dict) and wrapper.get("fusion") \
        and "stacker" in wrapper, "bundle is not a fused wrapper dict"
    ad.AUDIO_PROBE = str(fused_path)
    try:
        res = ad.detect_audio(TMP / "la_before_sample.flac")
    finally:
        ad.AUDIO_PROBE = legacy_ptr
    before_text = (TMP / "legacy_before_la.json").read_text()
    before = json.loads(before_text[before_text.index("{"):])
    assert res.get("fused") is True, f"fused branch did not route: {list(res)}"
    assert res["p4_pfake_mean"] == before["p_fake_audio_mean"], \
        f"p4 channel drift: {res['p4_pfake_mean']} vs {before['p_fake_audio_mean']}"
    assert abs(res["znorm_mean"] - before["znorm_mean"]) < 0.05, \
        f"znorm drift: {res['znorm_mean']} vs {before['znorm_mean']}"
    assert res["coverage_threshold"] == before["coverage_threshold"], \
        "p4 coverage threshold mismatch vs snapshot"
    assert "wavlm_pfake_mean" in res and "spectral_score" in res, \
        "fused result missing channel fields"
    print("2. FUSED SMOKE (PASS):", json.dumps(res), flush=True)
    receipt["fused_result"] = res
    receipt["legacy_before"] = {k: before[k] for k in
                                ("p_fake_audio_mean", "znorm_mean",
                                 "coverage_threshold")}

    # ---------- 3. legacy routing guard ----------
    res_leg = ad.detect_audio(TMP / "la_before_sample.flac")
    assert "fused" not in res_leg, "legacy pointer routed to the fused branch!"
    assert res_leg["p_fake_audio_mean"] == before["p_fake_audio_mean"] and \
        res_leg["znorm_mean"] == before["znorm_mean"], \
        "legacy path numbers drifted vs banked before-JSON"
    print("3. LEGACY ROUTING GUARD (PASS): legacy path intact, "
          f"p_fake={res_leg['p_fake_audio_mean']} znorm={res_leg['znorm_mean']}",
          flush=True)
    receipt["legacy_guard"] = {"p_fake_audio_mean": res_leg["p_fake_audio_mean"],
                               "znorm_mean": res_leg["znorm_mean"]}

    out = WS / f"artifacts/{args.cand}/SMOKE_RECEIPT.json"
    out.write_text(json.dumps(receipt, indent=1, default=str))
    print(f"receipt -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
