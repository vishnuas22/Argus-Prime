"""Extract XLS-R embeddings for the r8 diet -> train/val npz caches (D-036).

Walks data/corpora/audio_r9/{real,fake}/*.wav, segments 4s @16k, mean-pools
XLS-R embeddings. Splits by uid hash (same seed stays in one split).
Usage: ./.venv/bin/python deepfake/scripts/extract_audio_r9.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
SR = 16000
SEG = SR * 4
SRC = WS / "data" / "corpora" / "audio_r9"
OUTDIR = WS / "data" / "caches"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval()
    torch.set_grad_enabled(False)

    caches = {"train": {}, "val": {}}
    labels = {"train": {}, "val": {}}
    files = sorted((SRC / "real").glob("*.wav")) + sorted((SRC / "fake").glob("*.wav"))
    print(f"{len(files)} wavs; device={device}", flush=True)
    bad = 0
    t0 = __import__("time").time()
    for i, p in enumerate(files):
        label = 0 if p.parent.name == "real" else 1
        stem = p.stem  # r00001_wild_a etc.
        uid = stem.split("_")[0]  # r00001
        num = "".join(c for c in uid[1:] if c.isdigit())
        split = "val" if (int(num) % 5 == 0) else "train"  # 20% held-out
        try:
            wav, sr = sf.read(str(p), dtype="float32")
        except Exception:
            bad += 1
            continue
        wav = wav.mean(axis=1) if wav.ndim > 1 else wav
        if sr != SR:
            g = np.arange(len(wav)) * (SR / sr)
            wav = np.interp(g, np.arange(len(wav)), wav).astype(np.float32)
        if len(wav) < SR // 2:
            bad += 1
            continue
        if len(wav) <= SEG:
            segs = [np.pad(wav, (0, SEG - len(wav)))]
        else:
            segs = [wav[:SEG], wav[len(wav) - SEG:]] if len(wav) > SEG else [wav[:SEG]]
        for si, seg in enumerate(segs):
            if len(seg) < SEG:
                seg = np.pad(seg, (0, SEG - len(seg)))
            inp = fe(seg, sampling_rate=SR, return_tensors="pt").input_values.to(device)
            h = model(inp).last_hidden_state.mean(dim=1).squeeze(0).float().cpu().numpy()
            key = f"{stem}:{si}"
            caches[split][key] = h.astype(np.float16)
            labels[split][key] = label
        if (i + 1) % 400 == 0:
            rate = (i + 1) / (__import__("time").time() - t0)
            print(f"[{i+1}/{len(files)}] {rate:.1f} files/s", flush=True)

    for split in ("train", "val"):
        np.savez_compressed(OUTDIR / f"r9_{split}.npz", **caches[split])
        (OUTDIR / f"r9_{split}.labels.json").write_text(json.dumps(labels[split]))
        n = len(caches[split])
        nf = sum(1 for v in labels[split].values() if v == 1)
        print(f"{split}: {n} vectors ({nf} fake) -> r9_{split}.npz", flush=True)
    print(f"done ({bad} unreadable)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
