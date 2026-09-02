"""Cache frozen wav2vec2-XLS-R embeddings from ASVspoof parquet splits.

Reads HF-format parquet (audio bytes + label), segments 4s @16k, mean-pools the
transformer output over time -> 1024-d vectors per view.
Views: clean (+ optional noisy/codec-lite augmentation for train).
Usage:
  .venv/bin/python deepfake/scripts/audio_cache.py --parquet data/raw/asvspoof/data/train-00000-of-00001.parquet \
      --out data/caches/asv_train.npz --max-utterances 6000 [--augment] [--views-per-utt 2]
"""
import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

SR = 16000
SEG = SR * 4


def load_wav(b: bytes) -> np.ndarray | None:
    try:
        wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
        mono = wav.mean(axis=1)
        if sr != SR:
            g = np.arange(len(mono)) * (SR / sr)
            mono = np.interp(g, np.arange(len(mono)), mono)
        return mono
    except Exception:
        return None


def augment(wav: np.ndarray, rng) -> np.ndarray:
    out = wav * rng.uniform(0.7, 1.3)
    out += rng.normal(0, rng.uniform(0.001, 0.008), size=len(out)).astype(np.float32)
    return out.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="facebook/wav2vec2-xls-r-300m")
    ap.add_argument("--max-utterances", type=int, default=0)
    ap.add_argument("--augment", action="store_true")
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()
    torch.set_grad_enabled(False)

    df = pd.read_parquet(args.parquet)
    lab_col = "key" if "key" in df.columns else ("label" if "label" in df.columns else None)
    uid_col = "audio_file_name" if "audio_file_name" in df.columns else "path"
    print(f"{len(df)} utterances in parquet; device={device}", flush=True)
    if args.max_utterances:
        df = df.iloc[:args.max_utterances]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache, labels = {}, {}
    rng = np.random.default_rng(123)
    views = ["clean"] + (["aug1", "aug2"] if args.augment else [])
    n_done = bad = 0
    t0 = sys_time = __import__("time").time()
    for i, row in df.iterrows():
        audio_struct = row["audio"]
        b = audio_struct["bytes"] if isinstance(audio_struct, dict) else audio_struct
        wav = load_wav(b)
        if wav is None or len(wav) < SR // 2:
            bad += 1
            continue
        segs = []
        if len(wav) <= SEG:
            segs.append(np.pad(wav, (0, SEG - len(wav))))
        else:
            for s in range(0, len(wav) - SEG + 1, SEG):
                segs.append(wav[s:s + SEG])
                if len(segs) >= 3:
                    break
        uid = Path(str(row.get(uid_col) or f"utt_{i}")).stem
        for vi, view in enumerate(views):
            seg = segs[0]
            if view != "clean":
                seg = augment(seg, rng)
            inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            h = model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
            key = f"{uid}:{view}"
            cache[key] = h.astype(np.float16)
            if lab_col is not None:
                labels[key] = int(row[lab_col])
        n_done += 1
        if n_done % 200 == 0:
            np.savez_compressed(out, **cache)
            (out.with_suffix(".labels.json")).write_text(json.dumps(labels))
            rate = n_done / (__import__("time").time() - t0)
            print(f"[{n_done}/{len(df)}] {rate:.2f} utt/s", flush=True)
    np.savez_compressed(out, **cache)
    (out.with_suffix(".labels.json")).write_text(json.dumps(labels))
    print(f"saved {len(cache)} vectors ({bad} unreadable) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
