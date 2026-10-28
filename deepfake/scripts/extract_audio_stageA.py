"""Stage A (W3) extractor: serve-aligned segment-level XLS-R embeddings.

Re-embeds the EXACT utterance sets of p4's caches (asv_train / asv_val) with
the serve path's segmentation (detect_audio): full-length decode -> chop into
<=8 4s windows (pad last) -> one fp16 vector per window per view. Single
variable vs p4: the representation (audio_cache.py embeds only segs[0], the
first 4s; the court standardizes and scores EVERY window). Aug recipe (gain
0.7-1.3 + noise, rng seed 123), split discipline (train = LA_T train shard,
val = LA_D validation shard clean-only), and labels unchanged from
audio_cache.py (D-019 lineage). ASVspoof LA is 16k native -> the legacy
interp quirk in audio_cache.load_wav never engages here; a correct resample
is used if a non-16k row ever appears.

Usage: ./.venv/bin/python deepfake/scripts/extract_audio_stageA.py [--limit N]
Output: data/caches/asv_stageA_{train,val}.npz (+ .labels.json), keys {uid}:{view}:s{si}
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
SR = 16000
SEG = SR * 4
MAX_SEGS = 8  # serve parity (detect_audio caps at 8 windows)
SHARD_TRAIN = WS / "data" / "raw" / "asvspoof" / "data" / "train-00000-of-00001.parquet"
SHARD_VAL = WS / "data" / "raw" / "asvspoof" / "data" / "validation-00000-of-00001.parquet"
OUTDIR = WS / "data" / "caches"


def augment(wav: np.ndarray, rng) -> np.ndarray:
    # verbatim from audio_cache.py (p4 recipe)
    out = wav * rng.uniform(0.7, 1.3)
    out += rng.normal(0, rng.uniform(0.001, 0.008), size=len(out)).astype(np.float32)
    return out.astype(np.float32)


def serve_segments(wav: np.ndarray) -> list:
    # verbatim shape from detect_audio: <=8 4s windows, pad last
    segs = [wav[i:i + SEG] for i in range(0, max(1, len(wav) - SEG + 1), SEG)][:MAX_SEGS] or \
           [np.pad(wav[:SEG], (0, max(0, SEG - len(wav))))]
    return [np.pad(s, (0, SEG - len(s))) if len(s) < SEG else s for s in segs]


def main() -> int:
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 0

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    from transformers import AutoModel, Wav2Vec2FeatureExtractor
    fe = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = AutoModel.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device).eval()
    torch.set_grad_enabled(False)

    def embed_batch(segs):
        # uniform 4s windows -> one batched forward; numerically identical to
        # per-sample forwards (no cross-batch ops in wav2vec2 inference)
        inp = fe(segs, sampling_rate=SR, return_tensors="pt").input_values.to(device)
        return model(inp).last_hidden_state.mean(dim=1).float().cpu().numpy()

    # p4 utterance sets to mirror exactly (single-variable discipline)
    want = {}
    for split, cache in (("train", "asv_train"), ("val", "asv_val")):
        labs = json.loads((OUTDIR / f"{cache}.labels.json").read_text())
        want[split] = {k.split(":")[0] for k in labs}
        print(f"{split}: mirroring {len(want[split])} p4 utterances", flush=True)

    rng = np.random.default_rng(123)
    caches = {"train": {}, "val": {}}
    labels = {"train": {}, "val": {}}
    seen = {"train": 0, "val": 0}
    bad = 0
    t0 = time.time()

    jobs = [("train", SHARD_TRAIN, ["clean", "aug1", "aug2"]),
            ("val", SHARD_VAL, ["clean"])]
    for split, shard, views in jobs:
        pf = pq.ParquetFile(shard)
        for batch in pf.iter_batches(batch_size=256,
                                     columns=["audio_file_name", "key", "audio"]):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                uid = Path(str(row["audio_file_name"])).stem
                if uid not in want[split]:
                    continue
                if limit and seen[split] >= limit:
                    continue
                b = row["audio"]
                b = b["bytes"] if isinstance(b, dict) else b
                try:
                    wav, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
                except Exception:
                    bad += 1
                    continue
                wav = wav.mean(axis=1)
                if len(wav) < SR // 2:
                    bad += 1
                    continue
                if sr != SR:
                    n_out = int(round(len(wav) * SR / sr))
                    wav = np.interp(np.linspace(0, len(wav) - 1, n_out),
                                    np.arange(len(wav)), wav).astype(np.float32)
                for view in views:
                    w = wav if view == "clean" else augment(wav, rng)
                    segs = serve_segments(w)
                    hs = embed_batch(segs)
                    for si, h in enumerate(hs):
                        key = f"{uid}:{view}:s{si}"
                        caches[split][key] = h.astype(np.float16)
                        labels[split][key] = int(row["key"])
                seen[split] += 1
                if seen[split] % 200 == 0:
                    np.savez_compressed(OUTDIR / f"asv_stageA_{split}.npz", **caches[split])
                    (OUTDIR / f"asv_stageA_{split}.labels.json").write_text(json.dumps(labels[split]))
                    rate = seen[split] / (time.time() - t0)
                    print(f"[{split} {seen[split]}] {rate:.2f} utt/s "
                          f"({len(caches[split])} vectors so far)", flush=True)
        np.savez_compressed(OUTDIR / f"asv_stageA_{split}.npz", **caches[split])
        (OUTDIR / f"asv_stageA_{split}.labels.json").write_text(json.dumps(labels[split]))
        nf = sum(1 for v in labels[split].values() if v == 1)
        print(f"{split} DONE: {len(caches[split])} vectors ({nf} fake) "
              f"from {seen[split]} utterances ({bad} bad)", flush=True)

    print(f"total {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
