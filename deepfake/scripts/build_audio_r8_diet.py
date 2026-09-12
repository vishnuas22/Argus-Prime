"""Build the r8 wild-style audio diet (D-036).

Root cause (D-033/D-035): the audio probe's training distribution is
lab-clean ASVspoof — real-world audio (codecs, noise, phones, clipping) is
2-3x off-manifold, so the court either abstains (coverage gate, D-035) or
mislabels. r8 closes the gap with a self-generated wild-style diet:

  real head: ASVspoof bonafide (2019 LA train + 2021 test) -> clean + wild views
  fake head: ASVspoof spoofs (multi-attack) + fresh Kokoro-TTS clips -> same views

The wildizer simulates real-world recording/transport: gain, additive noise,
clipping, phone bandpass, resample round-trip, quantization, EQ tilt.
DF-Eval-2024 stays held out — evaluation only, never training.

Usage: ./.venv/bin/python deepfake/scripts/build_audio_r8_diet.py [--limit-reals 1200]
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

WS = Path(__file__).resolve().parents[2]
OUT = WS / "data" / "corpora" / "audio_r8"
SR = 16000

VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_fenrir",
          "bf_emma", "bm_george", "af_sky"]
TEXTS = [
    "The city council will vote on the new transit plan next Tuesday evening.",
    "Researchers measured the temperature at several depths before publishing.",
    "She wrote three letters to the editor and none of them were printed.",
    "A quiet river runs behind the old mill, past the broken fence.",
    "Please confirm your appointment for Thursday at half past nine.",
    "The forecast calls for light rain clearing by early afternoon.",
    "Every recipe in this book begins with a simple butter roux.",
    "The committee rejected the proposal without further discussion.",
    "He parked the van across two spaces and left the engine running.",
    "Sunlight moved slowly across the floor of the empty station.",
    "Our flight was delayed twice, once in Denver and once in Boston.",
    "The museum opens at ten and closes promptly at five on Sundays.",
    "Children ran between the fountains while their parents watched.",
    "This chapter explains how the compressor handles sudden spikes.",
    "Call the office if the shipment has not arrived by Friday noon.",
    "The trail climbs steeply for a mile before leveling off near the ridge.",
]

rng = np.random.default_rng(20260910)


def wildize(wav: np.ndarray) -> np.ndarray:
    out = wav.copy()
    n_ops = int(rng.integers(1, 3))
    ops = list(rng.permutation(6)[:n_ops])
    if 0 in ops:  # gain + noise
        out *= rng.uniform(0.3, 1.2)
        snr = rng.uniform(4, 22)
        p_sig = float(np.mean(out ** 2)) + 1e-9
        noise = rng.normal(0, np.sqrt(p_sig / (10 ** (snr / 10))), size=len(out))
        out = (out + noise).astype(np.float32)
    if 1 in ops:  # clip
        peak = rng.uniform(0.55, 0.92)
        out = np.tanh(out / peak) * peak
    if 2 in ops:  # phone bandpass 300-3400 Hz
        try:
            from scipy.signal import butter, lfilter
            b, a = butter(4, [300 / (SR / 2), 3400 / (SR / 2)], btype="band")
            out = lfilter(b, a, out).astype(np.float32)
        except Exception:
            pass
    if 3 in ops:  # resample round-trip
        k = int(rng.choice([2, 3]))
        small = np.interp(np.arange(0, len(out), k), np.arange(len(out)), out)
        out = np.interp(np.arange(len(out)), np.arange(0, len(out), k), small).astype(np.float32)
    if 4 in ops:  # quantization (8-bit compand-ish)
        q = np.round(np.clip(out, -1, 1) * 127) / 127
        out = (out * 0.4 + q * 0.6).astype(np.float32)
    if 5 in ops:  # EQ tilt
        t = np.linspace(0, 1, len(out)).astype(np.float32)
        out = out * (1 + rng.uniform(-0.4, 0.4) * t)
    peak = np.max(np.abs(out)) + 1e-9
    if peak > 0.99:
        out = out * (0.97 / peak)
    return out.astype(np.float32)


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


def save(wav: np.ndarray, label: str, uid: str, manifest: dict):
    for vi, v in enumerate(("clean", "wild_a", "wild_b")):
        seg = wav if v == "clean" else wildize(wav)
        dst = OUT / label / f"{uid}_{v}.wav"
        if dst.exists():
            continue
        sf.write(dst, seg, SR)
        manifest[str(dst.relative_to(OUT))] = {"label": label, "view": v, "uid": uid}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-reals", type=int, default=900)
    ap.add_argument("--limit-spoofs", type=int, default=700)
    ap.add_argument("--limit-tts", type=int, default=350)
    args = ap.parse_args()
    (OUT / "real").mkdir(parents=True, exist_ok=True)
    (OUT / "fake").mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    t0 = time.time()

    # --- real seeds: 2021 test bonafide (all label=0) + 2019 train bonafide ---
    df21 = pd.read_parquet(WS / "data/raw/asvspoof21_la_test-00000.parquet")
    reals = []
    for i in range(len(df21)):
        b = df21.iloc[i]["audio"]
        b = b["bytes"] if isinstance(b, dict) else b
        w = decode(b)
        if w is not None:
            reals.append(w)
        if len(reals) >= args.limit_reals // 2:
            break
    df19 = pd.read_parquet(WS / "data/raw/asvspoof/data/train-00000-of-00001.parquet")
    bon = df19[df19["key"] == "bonafide"]
    for i in range(min(len(bon), args.limit_reals // 2)):
        b = bon.iloc[i]["audio"]
        b = b["bytes"] if isinstance(b, dict) else b
        w = decode(b)
        if w is not None:
            reals.append(w)
    print(f"real seeds: {len(reals)}", flush=True)
    for i, w in enumerate(reals):
        save(seg16(w), "real", f"r{i:05d}", manifest)
        if i % 200 == 0:
            print(f"  reals {i}/{len(reals)} ({time.time()-t0:.0f}s)", flush=True)

    # --- spoof seeds: 2019 train, spread across attack ids ---
    sp = df19[df19["key"] != "bonafide"]
    idx = np.linspace(0, len(sp) - 1, min(args.limit_spoofs, len(sp))).astype(int)
    spoofs = []
    for i in idx:
        b = sp.iloc[i]["audio"]
        b = b["bytes"] if isinstance(b, dict) else b
        w = decode(b)
        if w is not None:
            spoofs.append(w)
    print(f"spoof seeds: {len(spoofs)}", flush=True)
    for i, w in enumerate(spoofs):
        save(seg16(w), "fake", f"s{i:05d}", manifest)
        if i % 200 == 0:
            print(f"  spoofs {i}/{len(spoofs)} ({time.time()-t0:.0f}s)", flush=True)

    # --- fresh TTS fakes (Kokoro via mlx_audio) ---
    import shutil
    from mlx_audio.tts.generate import generate_audio
    n = 0
    for i in range(args.limit_tts):
        uid = f"t{i:05d}"
        dst = OUT / "fake" / f"{uid}_clean.wav"
        if dst.exists():
            n += 1
            continue
        text = TEXTS[i % len(TEXTS)]
        voice = VOICES[i % len(VOICES)]
        tmp_dir = OUT / "fake" / f"{uid}_gen"
        try:
            # generate_audio treats output_path as a DIRECTORY: writes
            # <output_path>/audio_000.wav — read back from there.
            generate_audio(text, model='mlx-community/Kokoro-82M-8bit', voice=voice,
                           speed=float(rng.uniform(0.9, 1.15)), output_path=str(tmp_dir),
                           verbose=False)
            wavs = list(tmp_dir.glob("*.wav")) if tmp_dir.exists() else []
            w = decode(wavs[0].read_bytes()) if wavs else None
            if w is not None:
                save(w, "fake", uid, manifest)
                n += 1
            shutil.rmtree(tmp_dir, ignore_errors=True)
            dst.unlink(missing_ok=True)
        except Exception as e:
            print(f"  tts skip {i}: {e}", flush=True)
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if n % 50 == 0 and n:
            print(f"  tts {n}/{args.limit_tts} ({time.time()-t0:.0f}s)", flush=True)
    print(f"tts clips: {n}", flush=True)

    manifest_path.write_text(json.dumps(manifest, indent=1))
    n_real = len(list((OUT / 'real').glob('*.wav')))
    n_fake = len(list((OUT / 'fake').glob('*.wav')))
    print(f"DIET DONE: real={n_real} fake={n_fake} ({time.time()-t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())