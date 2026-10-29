"""W7 audio extractor: multi-window frozen XLS-R embeddings for the diverse diet.

D-060 lessons are BINDING here (spec artifacts/generator_intake/w7_audio_retrain_spec.md §3):
1. Aug-RNG draw order is dataset provenance. Every augment call consumes EXACTLY 3
   fixed draws (uniform gain, uniform sigma, normal(size=64000)) because augmentation
   is applied to PADDED windows (pad-before-augment, mirroring audio_cache.py's
   augment(segs[0]) on a 64000-sample seg). Stream order: single default_rng(seed),
   consumed view-major within each row: for view in (aug1, aug2): for window in
   row-windows: augment(window). Seed + order recorded in train_meta.json.
2. Cache-parity preflight (--parity-check): clean views MUST bit-match the reference
   cache (data/caches/asv_train.npz) on matched keys -> proves decode/resample/window
   parity. Aug-view divergence vs reference is EXPECTED (different draws) and is
   measured + reported, not asserted.
3. Windowing mirrors the court's serve path exactly (argus_detect.py:389-401):
   non-overlapping 64000-sample windows from 0, capped at 8, short windows zero-padded
   to 64000. Decode/resample mirrors audio_cache.load_wav (sf.read float32 always_2d,
   mono-mean, np.interp grid resample to 16k).

Input: diet manifest JSONL rows {"uid", "source", "label", and either "path"
(file on disk) or "parquet"+"row" (parquet byte row)}.
Usage:
  ./.venv/bin/python deepfake/scripts/extract_audio_w7.py --manifest <jsonl> \
      --out data/caches/asv_w7_train.npz --seed 123 [--limit N] \
      [--parity-check data/caches/asv_train.npz] [--sensitivity-probe K]
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

SR = 16000
SEG = SR * 4
MAX_WINDOWS = 8
VIEWS = ["clean", "aug1", "aug2"]


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


def embed_rows(rows, model, fe, device, rng, cache, labels, counters) -> int:
    bad = 0
    for r in rows:
        if "path" in r:
            wav = load_wav_from_file(r["path"])
        else:
            import pandas as pd
            if not hasattr(embed_rows, "_pf"):
                embed_rows._pf = {}
            pf = r["parquet"]
            if pf not in embed_rows._pf:
                embed_rows._pf[pf] = pd.read_parquet(pf)
            row = embed_rows._pf[pf].iloc[r["row"]]
            a = row["audio"]
            b = a["bytes"] if isinstance(a, dict) else a
            wav = load_wav_from_bytes(b)
        if wav is None or len(wav) < SR // 2:
            bad += 1
            counters["bad"] += 1
            continue
        segs = serve_windows(wav)
        for vi, view in enumerate(VIEWS):
            for si, seg in enumerate(segs):
                if view != "clean":
                    seg = augment(seg, rng)  # pad-before-augment: seg is always 64000
                inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
                h = model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
                key = f"{r['uid']}:{view}:s{si}"
                cache[key] = h.astype(np.float16)
                labels[key] = int(r["label"])
        counters["done"] += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--parity-check", default="", help="reference npz for clean-view bit-match assert")
    ap.add_argument("--sensitivity-probe", type=int, default=0,
                    help="embed K rows 3x (clean, augA, augB) and report aug-noise sensitivity scale")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    if args.limit:
        rows = rows[:args.limit]
    print(f"manifest: {len(rows)} rows", flush=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    torch.set_grad_enabled(False)
    print(f"device={device}", flush=True)

    cache, labels = {}, {}
    counters = {"done": 0, "bad": 0}

    if args.sensitivity_probe:
        rng_p = np.random.default_rng(777)
        rng_q = np.random.default_rng(888)
        d_ab, d_ac = [], []
        for r in rows[:args.sensitivity_probe]:
            wav = load_wav_from_file(r["path"]) if "path" in r else None
            if wav is None or len(wav) < SR // 2:
                continue
            seg = serve_windows(wav)[0]
            e = lambda s: model(fe(s, sampling_rate=SR, return_tensors="pt").input_values.to(device)
                                ).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
            ec, ea, eb = e(seg), e(augment(seg, rng_p)), e(augment(seg, rng_q))
            d_ab.append(float(np.linalg.norm(ea - eb)))
            d_ac.append(float(np.linalg.norm(ea - ec)))
        d_ab, d_ac = np.array(d_ab), np.array(d_ac)
        probe = {"n": len(d_ab), "augA_augB_L2_mean": round(float(d_ab.mean()), 4),
                 "aug_clean_L2_mean": round(float(d_ac.mean()), 4),
                 "note": "aug-noise sensitivity scale: how far two different draw realizations "
                         "move the same window's embedding (D-060 chaos quantified)"}
        print(json.dumps(probe, indent=1))
        Path(args.out).with_suffix(".sensitivity.json").write_text(json.dumps(probe, indent=1))
        return 0

    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    for i in range(0, len(rows), 500):
        chunk = rows[i:i + 500]
        embed_rows(chunk, model, fe, device, rng, cache, labels, counters)
        np.savez_compressed(args.out, **cache)
        Path(args.out).with_suffix(".labels.json").write_text(json.dumps(labels))
        rate = counters["done"] / max(time.time() - t0, 1e-9)
        print(f"[{counters['done']}/{len(rows)}] {rate:.2f} utt/s bad={counters['bad']}", flush=True)

    meta = {
        "seed": args.seed,
        "draw_order": "single default_rng(seed); per row (manifest order), view-major "
                      "(aug1 then aug2), per window in serve order; each augment consumes "
                      "exactly [uniform(0.7,1.3), uniform(0.001,0.008), normal(size=64000)]",
        "pad_convention": "pad-before-augment (windows padded to 64000, then augmented)",
        "windowing": "argus_detect.py:389-401 mirror: non-overlapping 64000 from 0, cap 8, short padded",
        "model": args.model,
        "n_rows": counters["done"], "n_bad": counters["bad"], "n_vectors": len(cache),
        "views": VIEWS,
    }
    Path(args.out).with_suffix(".train_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(cache)} vectors ({counters['bad']} bad) -> {args.out}")

    if args.parity_check:
        ref = np.load(args.parity_check)
        # reference (audio_cache) holds segs[0] only -> compare against :clean:s0
        # (NOT the last s-index: multi-window rows would compare s1+ against segs[0])
        norm = {k.rsplit(":s", 1)[0]: k for k in cache if k.endswith(":clean:s0")}
        common = sorted(set(ref.files) & set(norm.keys()))
        if not common:
            print("PARITY: no matched keys vs reference — check uid/view format")
            return 1
        exact = 0
        for k in common:
            if np.array_equal(ref[k].astype(np.float32), cache[norm[k]].astype(np.float32)):
                exact += 1
        print(f"PARITY: clean views bit-identical {exact}/{len(common)} "
              f"({'PASS' if exact == len(common) else 'FAIL'})")
        if exact != len(common):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
