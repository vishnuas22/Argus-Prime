"""Build the r9 audio diet (D-038): REAL wild corpora, no synthetic-DSP shortcut.

D-036 lesson: synthetic DSP wildization carries no class information and a
single-engine TTS fake head cannot approximate the wild fake distribution.
r9 therefore builds from REAL varied sources only:

  REAL head: Codecfake 'R' (VCTK speech) + WaveFake 'R' + ASVspoof bonafide
             (2019 train + 2021 test) + Kokoro-diet reals if present
  FAKE head: Codecfake F01-F06 (ALM/codec fakes) + WaveFake WF1-7 (vocoders)
             + ASVspoof multi-attack spoofs + Kokoro TTS clips (r8 diet)

No wildizer. Wildness comes from the sources themselves.
Usage: ./.venv/bin/python deepfake/scripts/build_audio_r9_diet.py
"""
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

WS = Path(__file__).resolve().parents[2]
OUT = WS / "data" / "corpora" / "audio_r9"
SR = 16000

rng = np.random.default_rng(20260915)


def decode(b: bytes) -> np.ndarray | None:
    try:
        w, sr = sf.read(io.BytesIO(b), dtype="float32")
    except Exception:
        return None
    w = w.mean(axis=1) if w.ndim > 1 else w
    if sr != SR:
        g = np.arange(len(w)) * (SR / sr)
        w = np.interp(g, np.arange(len(w)), w).astype(np.float32)
    return w if len(w) >= SR else None


def seg16(wav: np.ndarray) -> np.ndarray:
    if len(wav) < SR:
        return np.pad(wav, (0, SR - len(wav)))
    i = int(rng.integers(0, len(wav) - SR + 1))
    return wav[i:i + SR].astype(np.float32)


def save(wav: np.ndarray, label: str, uid: str, manifest: dict, src: str):
    dst = OUT / label / f"{uid}.wav"
    if dst.exists():
        return False
    sf.write(dst, wav, SR)
    manifest[str(dst.relative_to(OUT))] = {"label": label, "uid": uid, "src": src}
    return True


def shards(df_iter, limit, label, prefix, manifest, src, t0, counters):
    n = 0
    for i, row in df_iter:
        if n >= limit:
            break
        b = row["audio"]
        b = b["bytes"] if isinstance(b, dict) else b
        w = decode(b)
        if w is None:
            continue
        if save(seg16(w), label, f"{prefix}{i:06d}", manifest, src):
            n += 1
            counters[label] = counters.get(label, 0) + 1
        if n % 200 == 0 and n:
            print(f"  {label}/{src}: {n} ({time.time()-t0:.0f}s)", flush=True)
    return n


def main():
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    (OUT / "fake").mkdir(parents=True, exist_ok=True)
    mp = OUT / "manifest.json"
    manifest = json.loads(mp.read_text()) if mp.exists() else {}
    t0 = time.time()
    counters = {}

    # --- Codecfake (ALM codec fakes + matched VCTK reals) ---
    for shard in sorted((WS / "data/raw/r9/codecfake-audio/data").glob("*.parquet")):
        df = pd.read_parquet(shard)
        for lab, lim in (("R", 120), ("F", 450)):
            sub = df[df["real_or_fake"].str.startswith(lab)]
            shards(((i, r) for i, r in sub.iterrows()), lim,
                   "real" if lab == "R" else "fake",
                   f"cf{lab}", manifest, "codecfake", t0, counters)

    # --- WaveFake (vocoder fakes + LJSpeech reals) ---
    for shard in sorted((WS / "data/raw/r9/wavefake-audio/data").glob("*.parquet")):
        df = pd.read_parquet(shard)
        for lab, lim in (("R", 100), ("WF", 500)):
            sub = df[df["real_or_fake"].str.startswith(lab)]
            shards(((i, r) for i, r in sub.iterrows()), lim,
                   "real" if lab == "R" else "fake",
                   f"wf{lab}", manifest, "wavefake", t0, counters)

    # --- ASVspoof 2021 bonafide + 2019 spoofs ---
    df21 = pd.read_parquet(WS / "data/raw/asvspoof21_la_test-00000.parquet")
    shards(((i, r) for i, r in df21.iterrows() if r["label"] == 0), 400,
           "real", "asv21", manifest, "asv21_bonafide", t0, counters)
    df19 = pd.read_parquet(WS / "data/raw/asvspoof/data/train-00000-of-00001.parquet")
    sp = df19[df19["key"] != "bonafide"]
    idx = np.linspace(0, len(sp) - 1, 500).astype(int)
    shards(((i, sp.iloc[i]) for i in idx), 500, "fake", "asv19sp",
           manifest, "asv19_spoof", t0, counters)
    bon = df19[df19["key"] == "bonafide"]
    shards(((i, r) for i, r in bon.head(400).iterrows()), 400,
           "real", "asv19bon", manifest, "asv19_bonafide", t0, counters)

    # --- Kokoro TTS from the r8 diet (multi-voice fakes; wild views only —
    # the r8 builder's leftover cleanup removed the clean views, but the
    # wild views are still fakes regardless of view) ---
    n = 0
    for p in sorted((WS / "data/corpora/audio_r8/fake").glob("t*_wild_*.wav")):
        try:
            w, sr = sf.read(str(p), dtype="float32")
        except Exception:
            continue
        w = w.mean(axis=1) if w.ndim > 1 else w
        if sr != SR:
            g = np.arange(len(w)) * (SR / sr)
            w = np.interp(g, np.arange(len(w)), w).astype(np.float32)
        if save(seg16(w), "fake", f"tts{p.stem.split('_')[0][1:]}", manifest, "kokoro_tts"):
            n += 1
    print(f"kokoro tts: {n}", flush=True)

    mp.write_text(json.dumps(manifest, indent=1))
    n_real = len(list((OUT / "real").glob("*.wav")))
    n_fake = len(list((OUT / "fake").glob("*.wav")))
    print(f"DIET DONE: real={n_real} fake={n_fake} ({time.time()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
