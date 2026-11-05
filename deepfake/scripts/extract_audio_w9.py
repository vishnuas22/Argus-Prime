"""W9 audio extractor (D-079 SEALED): unified per-utterance extraction for the
three-channel court — XLS-R (A, parity-anchored to p4), WavLM-base-plus (B, new),
training-free spectral/codec DSP (C).

Mirrors extract_audio_w7.py law exactly (D-060):
- Windowing mirrors the court's serve path (argus_detect.py:389-401): non-overlapping
  64000-sample windows from 0, cap 8, short windows zero-padded to 64000.
- Decode/resample mirrors audio_cache.load_wav (sf.read float32 always_2d, mono-mean,
  np.interp grid resample to 16k).
- Augment recipe verbatim from audio_cache.py: exactly 3 draws per call
  [uniform(0.7,1.3), uniform(0.001,0.008), normal(size=64000)], pad-before-augment.
- Views: clean / aug1 / aug2, view-major draw order within each row.

DELIBERATE REFINEMENT of the D-060 draw-order law (recorded in train_meta): RNG is
PER-ROW derived — np.random.default_rng([seed, row_index]) — instead of w7's single
stream. Draw order remains fully determined by (seed, row_index) — dataset provenance
is unchanged — but chunked RESUME is now bit-identical (skipping done rows cannot
shift any later row's draws). Clean views consume no draws, so the p4 parity anchor
is unaffected.

Preflight gates (fail-closed):
1. WavLM dry-run load gate BEFORE any extraction: load + dummy forward on device;
   receipt (weights bytes, params, versions, timing) -> w9_wavlm.loadgate.json.
   Failure = exit non-zero. NEVER silently swap models.
2. Meta-vs-seal assert: manifest provenance must carry counts_binding_match_sealed_table
   with counts == §4 (17160/8580/8580/2580/2580/6000/6000).
3. Resume guard: .runmeta.json stores the extractor script sha; a changed script
   refuses resume (--fresh to override deliberately).

Parity (spec STEP 2 VERIFY): every XLS-R clean-view key shared with
data/caches/asv_w7_la.npz must be BIT-IDENTICAL (decode/resample/window/featurizer
parity with the p4 serve path). Aug-view divergence vs w7 is EXPECTED (different
draw scheme) and reported, not asserted.

Usage:
  ./.venv/bin/python deepfake/scripts/extract_audio_w9.py \
      --manifest artifacts/generator_intake/w9_diet_sealed/manifest.jsonl \
      --prefix data/caches/w9 [--limit N] [--fresh]
"""
import argparse
import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

WS = Path("/Users/Dataghost/Argus_Prime")
SR = 16000
SEG = SR * 4
MAX_WINDOWS = 8
VIEWS = ["clean", "aug1", "aug2"]
XLSR_MODEL = "facebook/wav2vec2-xls-r-300m"
WAVLM_MODEL = "microsoft/wavlm-base-plus"  # spec said facebook/ — that repo is dead
# (HF 401); canonical repo verified 2026-11-05: same WavLMModel arch, hidden 768.
EXPECTED = {"total": 17160, "bona": 8580, "fake": 8580, "clean_bona": 2580,
            "clean_fake": 2580, "trans_bona": 6000, "trans_fake": 6000}
SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def load_wav_from_bytes(b: bytes) -> np.ndarray | None:
    try:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    except Exception:
        return None
    mono = wav.mean(axis=1)
    if sr != SR:
        g = np.arange(len(mono)) * (SR / sr)
        mono = np.interp(g, np.arange(len(mono)), mono)
    return mono


def load_wav_from_file(path: str) -> np.ndarray | None:
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        return None
    mono = wav.mean(axis=1)
    if sr != SR:
        g = np.arange(len(mono)) * (SR / sr)
        mono = np.interp(g, np.arange(len(mono)), mono)
    return mono


def serve_windows(wav: np.ndarray) -> list[np.ndarray]:
    """Mirror argus_detect.py:389-401 exactly."""
    segs = [wav[i:i + SEG] for i in range(0, max(1, len(wav) - SEG + 1), SEG)][:MAX_WINDOWS] or \
           [np.pad(wav[:SEG], (0, max(0, SEG - len(wav))))]
    return [np.pad(s, (0, SEG - len(s))) if len(s) < SEG else s for s in segs]


def augment(wav: np.ndarray, rng) -> np.ndarray:
    """audio_cache.py recipe verbatim; consumes EXACTLY 3 draws (fixed-length noise)."""
    out = wav * rng.uniform(0.7, 1.3)
    out += rng.normal(0, rng.uniform(0.001, 0.008), size=len(out)).astype(np.float32)
    return out.astype(np.float32)


# ---------------- channel C: training-free spectral/codec DSP ----------------
# Fixed functions only; the real-envelope calibration (D-035 law, train reals
# only) is fit downstream at row-build, NOT here. Mirrors the image court's
# spectral module style (radial profile -> envelope z -> deviation stats).
SP_BINS = 64


def spectral_feats(seg: np.ndarray) -> np.ndarray:
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


SP_K = SP_BINS + 7


def embed(model, fe, seg, device):
    inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
    return model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()


def wavlm_load_gate(device) -> dict:
    """BUILD-TIME dry-run load gate — BEFORE any extraction. Fail-closed."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    t0 = time.time()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(WAVLM_MODEL)
    model = AutoModel.from_pretrained(WAVLM_MODEL).to(device).eval()
    load_s = time.time() - t0
    with torch.no_grad():
        out = model(torch.zeros(1, SEG, device=device))
    dim = out.last_hidden_state.shape[-1]
    hf_dir = Path.home() / ".cache/huggingface/hub"
    model_dir = hf_dir / f"models--{WAVLM_MODEL.replace('/', '--')}"
    weights_bytes = sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file()) \
        if model_dir.exists() else -1
    receipt = {
        "model": WAVLM_MODEL, "device": device, "embed_dim": int(dim),
        "load_seconds": round(load_s, 1), "hf_cache_weights_bytes": weights_bytes,
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "dummy_forward_ok": True,
        "gate": "dry-run load + forward PASS — extraction may proceed",
    }
    print(json.dumps(receipt, indent=1), flush=True)
    return {"fe": fe, "model": model, "receipt": receipt}


def df_free() -> str:
    try:
        return subprocess.run(["df", "-h", "/"], capture_output=True, text=True,
                              timeout=15).stdout.strip().splitlines()[-1].split()[3]
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--prefix", default="data/caches/w9")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--skip-parity", action="store_true",
                    help="smoke runs only — the full run MUST parity-check")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    # gate 2: meta-vs-seal assert (skip only under --limit smoke)
    prov_path = Path(args.manifest).with_suffix(".provenance.json")
    if not args.limit:
        prov = json.loads(prov_path.read_text())
        if not prov.get("counts_binding_match_sealed_table") or prov.get("counts") != EXPECTED:
            raise SystemExit("SEAL ASSERT FAILED: manifest provenance does not carry "
                             f"the sealed §4 counts {EXPECTED} — STOP")
        print(f"seal assert OK: {prov['counts']}", flush=True)

    prefix = Path(args.prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    runmeta_path = prefix.with_suffix(".runmeta.json") if prefix.suffix == "" else \
        prefix.parent / (prefix.name + ".runmeta.json")

    caches = {k: {} for k in ("xlsr", "wavlm", "spectral")}
    labels: dict[str, int] = {}
    done_rows: set[int] = set()
    if not args.fresh and runmeta_path.exists():
        prev = json.loads(runmeta_path.read_text())
        if prev.get("script_sha") != SCRIPT_SHA:
            raise SystemExit(f"RESUME GUARD: extractor code changed since prior run "
                             f"({prev.get('script_sha')} != {SCRIPT_SHA}) — pass --fresh "
                             f"to discard and restart deliberately")
        for k, suf in (("xlsr", "xlsr"), ("wavlm", "wavlm"), ("spectral", "spectral")):
            p = Path(f"{args.prefix}_{suf}.npz")
            if p.exists():
                z = np.load(p)
                caches[k] = {kk: z[kk] for kk in z.files}
        labels = json.loads(Path(f"{args.prefix}_labels.json").read_text()) \
            if Path(f"{args.prefix}_labels.json").exists() else {}
        done_rows = set(prev.get("done_rows", []))
        print(f"resume: {len(done_rows)} rows already cached", flush=True)
    elif args.fresh:
        for suf in ("xlsr", "wavlm", "spectral"):
            Path(f"{args.prefix}_{suf}.npz").unlink(missing_ok=True)
        Path(f"{args.prefix}_labels.json").unlink(missing_ok=True)
        runmeta_path.unlink(missing_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_grad_enabled(False)

    # gate 1: WavLM dry-run load gate
    wl = wavlm_load_gate(device)
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    fe_x = Wav2Vec2FeatureExtractor.from_pretrained(XLSR_MODEL)
    m_x = AutoModel.from_pretrained(XLSR_MODEL).to(device).eval()
    print(f"device={device} | XLS-R loaded | disk free at start: {df_free()}", flush=True)

    todo = [(i, r) for i, r in enumerate(rows) if i not in done_rows]
    print(f"manifest: {len(rows)} rows | todo {len(todo)}", flush=True)
    t0, counters = time.time(), {"done": len(done_rows), "bad": 0}
    bad_uids = []

    def save_all():
        np.savez(f"{args.prefix}_xlsr.npz", **caches["xlsr"])
        np.savez(f"{args.prefix}_wavlm.npz", **caches["wavlm"])
        np.savez(f"{args.prefix}_spectral.npz", **caches["spectral"])
        Path(f"{args.prefix}_labels.json").write_text(json.dumps(labels))
        runmeta_path.write_text(json.dumps({
            "script_sha": SCRIPT_SHA, "seed": args.seed,
            "done_rows": sorted(done_rows), "models": [XLSR_MODEL, WAVLM_MODEL]}))

    CH = 500
    for ci in range(0, len(todo), CH):
        for i, r in todo[ci:ci + CH]:
            wav = load_wav_from_file(r["path"]) if "path" in r else None
            if wav is None and "parquet" in r:
                import pandas as pd
                pf = r["parquet"]
                if not hasattr(main, "_pf"):
                    main._pf = {}
                if pf not in main._pf:
                    main._pf[pf] = pd.read_parquet(pf)
                a = main._pf[pf].iloc[r["row"]]["audio"]
                b = a["bytes"] if isinstance(a, dict) else a
                wav = load_wav_from_bytes(b)
            if wav is None or len(wav) < SR // 2:
                counters["bad"] += 1
                bad_uids.append(r["uid"])
                done_rows.add(i)
                continue
            segs = serve_windows(wav)
            rng = np.random.default_rng([args.seed, i])  # per-row derived (D-060 note)
            for view in VIEWS:
                for si, seg in enumerate(segs):
                    if view != "clean":
                        seg = augment(seg, rng)
                    key = f"{r['uid']}:{view}:s{si}"
                    caches["xlsr"][key] = embed(m_x, fe_x, seg, device).astype(np.float16)
                    caches["wavlm"][key] = embed(wl["model"], wl["fe"], seg, device).astype(np.float16)
                    caches["spectral"][key] = spectral_feats(seg).astype(np.float32)
                    labels[key] = int(r["label"])
            done_rows.add(i)
            counters["done"] += 1
        save_all()
        rate = counters["done"] / max(time.time() - t0, 1e-9)
        print(f"[{counters['done']}/{len(rows)}] {rate:.2f} utt/s "
              f"bad={counters['bad']} free={df_free()}", flush=True)

    # final compressed save + meta sidecars
    for suf in ("xlsr", "wavlm", "spectral"):
        np.savez_compressed(f"{args.prefix}_{suf}.npz", **caches[suf])
    meta = {
        "seed": args.seed,
        "draw_order": "PER-ROW DERIVED RNG (recorded refinement of the D-060 single-"
                      "stream law): np.random.default_rng([seed, row_index]); within a "
                      "row, view-major (aug1 all windows then aug2), serve window order; "
                      "each augment consumes exactly [uniform(0.7,1.3), "
                      "uniform(0.001,0.008), normal(size=64000)]. Draw order fully "
                      "determined by (seed, row_index) -> resume bit-identical.",
        "pad_convention": "pad-before-augment (windows padded to 64000, then augmented)",
        "windowing": "argus_detect.py:389-401 mirror: non-overlapping 64000 from 0, cap 8, short padded",
        "models": {"xlsr": XLSR_MODEL, "wavlm": WAVLM_MODEL},
        "wavlm_loadgate": wl["receipt"],
        "spectral_layout": f"sp_ltas_0..{SP_BINS - 1} (log-bin dB LTAS) + "
                           "[flat_mean, flat_std, centroid_var_norm, hf_ratio, "
                           "edge_freq_norm, frame_corr, phase_smooth] — raw only; "
                           "real-envelope calibration (D-035) fit downstream on train reals",
        "n_rows": counters["done"], "n_bad": counters["bad"], "bad_uids": bad_uids,
        "n_vectors": len(caches["xlsr"]), "views": VIEWS,
        "script_sha": SCRIPT_SHA,
        "disk_free_at_end": df_free(),
    }
    for suf in ("xlsr", "wavlm", "spectral"):
        Path(f"{args.prefix}_{suf}.train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(caches['xlsr'])} vectors x3 caches ({counters['bad']} bad) "
          f"-> {args.prefix}_*", flush=True)

    # parity: XLS-R clean views MUST bit-match asv_w7_la on shared keys
    if not args.skip_parity:
        ref = np.load("data/caches/asv_w7_la.npz")
        common = sorted(set(ref.files) & {k for k in caches["xlsr"] if ":clean:" in k})
        if not common:
            print("PARITY: no shared clean keys vs asv_w7_la — FAIL")
            return 1
        exact = sum(1 for k in common
                    if np.array_equal(ref[k].astype(np.float32),
                                      caches["xlsr"][k].astype(np.float32)))
        print(f"PARITY: XLS-R clean views bit-identical {exact}/{len(common)} "
              f"({'PASS' if exact == len(common) else 'FAIL'})", flush=True)
        if exact != len(common):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
