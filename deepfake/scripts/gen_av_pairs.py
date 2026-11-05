"""W-AV generation driver (D-076 Step 3): matched Wav2Lip pairs + confound strata.

Per source key in the diet split manifest (data/raw/av_diet/split.json):
  fake      = Wav2Lip (official wav2lip_gan.pth) lip-synced to the source's
              OWN native audio — the class-conditional audio distribution is
              identical BY CONSTRUCTION (spec §3). Output through the
              vendor's pinned ffmpeg stage (DIVX avi intermediate, then
              `-strict -2 -q:v 1` mux).
  real      = the SAME source re-encoded through the IDENTICAL output stage
              (`-q:v 1`), audio copied byte-identical (`-c:a copy`).
  narration = Kokoro TTS narration muxed onto the source (`-c:v copy`,
              audio -> aac), train 50 + exam 30x2 variants — real-class
              confound stratum (D4): audio present, lips NOT the audio's.
  music     = deterministic numpy-synthesized music bed muxed the same way,
              train 50 + exam 30x2 — real-class confound stratum.

Laws:
 - smoke first (§5.4): --smoke runs 2 sources end-to-end (fake + real +
   narration + music) before any batch.
 - no-face honesty (§5.5): a source whose Wav2Lip arm finds no face is
   receipted into noface.jsonl + gen_receipt["failed"] and NEVER enters the
   diet; the dump reads the receipt and excludes the pair. Infra failures
   (non-face errors) raise — fail-closed.
 - resume: existing outputs are skipped (idempotent re-runs).
 - D-050: the checkpoint's size + sha256 are asserted against the receipted
   values before any generation; the Kokoro model id + license lineage ride
   gen_receipt.json.

Usage:
  .venv/bin/python deepfake/scripts/gen_av_pairs.py [--smoke]
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
AV = WS / "data" / "raw" / "av_diet"
GEN = AV / "gen"
W2L = WS / "integrations" / "Wav2Lip"
CKPT = W2L / "checkpoints" / "wav2lip_gan.pth"
# D-050 receipt (authors' Drive folder, verified at acquisition 2026-11-03):
CKPT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
CKPT_BYTES = 435801865
KOKORO_MODEL = "mlx-community/Kokoro-82M-8bit"      # Apache-2.0 (D-059 lineage)
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
    "He fixed the bicycle chain and rode up the steep hill road.",
    "The museum opens its new wing to visitors this weekend.",
    "Mail delivery moves to the morning schedule starting in June.",
    "The garden club meets on the first Friday of every month.",
]
NARR_TRAIN = 50
EXAM_VARIANTS = 2
SEED = 11


def run(cmd, **kw):
    print(f"[gen] {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): "
                         f"{' '.join(map(str, cmd))}")
    return r


def _ffprobe_duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration", "-of",
                        "default=noprint_wrappers=1:nokey=1", str(path)],
                       capture_output=True, text=True)
    return float(r.stdout.strip())


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gen_fake(src: Path, out: Path) -> None:
    """Wav2Lip the source to its own audio via the vendor entrypoint."""
    # absolute paths: the vendor runs with cwd=integrations/Wav2Lip, so
    # workspace-relative paths would not resolve inside it
    cmd = [sys.executable, "inference.py",
           "--checkpoint_path", str(CKPT), "--face", str(src.resolve()),
           "--audio", str(src.resolve()), "--outfile", str(out.resolve()),
           "--wav2lip_batch_size", "128"]
    print(f"[gen] (Wav2Lip) {' '.join(cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=W2L, capture_output=True,
                       text=True)
    if r.returncode != 0 or not out.exists():
        err = (r.stderr or r.stdout or "")[-800:]
        if "Face not detected" in err:
            raise FaceError(err)
        raise SystemExit(f"Wav2Lip failed on {src.name} (infra, fail-closed):\n{err}")


class FaceError(RuntimeError):
    pass


def gen_real(src: Path, out: Path) -> None:
    """Matched real: IDENTICAL output stage, audio copied byte-identical."""
    run(["ffmpeg", "-y", "-i", src, "-strict", "-2", "-q:v", "1",
         "-c:a", "copy", out], capture_output=True)


def _stable(key: str) -> int:
    """Process-independent digest (str hash() is salted per-run)."""
    import hashlib
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def _narration_wav(key: str, variant: int, rng: np.random.Generator,
                   tmp_dir: Path) -> Path:
    """Kokoro narration for one (source, variant); returns the wav path."""
    from mlx_audio.tts.generate import generate_audio
    text = TEXTS[(_stable(key) + variant) % len(TEXTS)]
    voice = VOICES[(_stable(key) + variant) % len(VOICES)]
    tmp_dir.mkdir(parents=True, exist_ok=True)
    generate_audio(text, model=KOKORO_MODEL, voice=voice,
                   speed=float(rng.uniform(0.9, 1.15)),
                   output_path=str(tmp_dir), verbose=False)
    wavs = list(tmp_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"Kokoro produced no wav for {key} v{variant}")
    return wavs[0]


def gen_narration(src: Path, out: Path, key: str, variant: int,
                  rng: np.random.Generator) -> dict:
    """Kokoro narration muxed onto the source (-c:v copy, audio -> aac)."""
    import shutil
    tmp = AV / "work_narr" / f"{key}__{variant}"
    shutil.rmtree(tmp, ignore_errors=True)
    w = _narration_wav(key, variant, rng, tmp)
    run(["ffmpeg", "-y", "-i", src, "-i", w, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-strict", "-2", "-shortest", out],
        capture_output=True)
    meta = {"text_idx": (_stable(key) + variant) % len(TEXTS),
            "voice": VOICES[(_stable(key) + variant) % len(VOICES)]}
    shutil.rmtree(tmp, ignore_errors=True)
    return meta


def synth_music(dur: float, seed: int, out_wav: Path) -> None:
    """Deterministic numpy music bed (seed recorded per file): a I-vi-IV-V
    sine chord pad with slow attack/release plus a seeded pentatonic melody,
    44.1 kHz stereo int16, normalized to ~-14 dBFS."""
    import soundfile as sf
    rng = np.random.default_rng(seed)
    sr = 44100
    n = int(dur * sr)
    t = np.arange(n) / sr
    chords = [[261.63, 329.63, 392.00], [220.00, 261.63, 329.63],
              [174.61, 220.00, 261.63], [196.00, 246.94, 293.66]]
    bar = 2.0
    sig = np.zeros(n)
    n_bars = int(np.ceil(dur / bar))
    for b in range(n_bars):
        ch = chords[b % len(chords)]
        s0, s1 = int(b * bar * sr), min(int((b + 1) * bar * sr), n)
        tt = t[s0:s1] - b * bar
        env = np.minimum(1.0, tt / 0.4) * np.minimum(1.0, (bar - tt) / 0.4)
        env = np.clip(env, 0.0, 1.0)
        for f in ch:
            sig[s0:s1] += 0.12 * env * np.sin(2 * np.pi * f * tt + 0.7 * b)
            sig[s0:s1] += 0.04 * env * np.sin(2 * np.pi * 2 * f * tt)
    # pentatonic melody, one seeded note per 0.5 s
    pent = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    step = int(0.5 * sr)
    for k in range(0, n, step):
        f = pent[int(rng.integers(0, len(pent)))]
        seg = slice(k, min(k + step, n))
        tt = t[seg] - k / sr
        sig[seg] += 0.10 * np.sin(2 * np.pi * f * tt) * np.exp(-3.0 * tt)
    sig = sig / (np.max(np.abs(sig)) + 1e-9) * 0.2          # ~-14 dBFS
    st = np.stack([sig, np.roll(sig, 33)], 1)               # slight stereo width
    sf.write(out_wav, (st * 32767).astype(np.int16), sr)


def gen_music(src: Path, out: Path, seed: int, tmp_wav: Path) -> None:
    dur = _ffprobe_duration(src)
    synth_music(dur, seed, tmp_wav)
    run(["ffmpeg", "-y", "-i", src, "-i", tmp_wav, "-map", "0:v:0",
         "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-strict", "-2",
         "-shortest", out], capture_output=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="2 sources end-to-end, mechanics only (§5.4)")
    args = ap.parse_args()

    split = json.loads((AV / "split.json").read_text())
    receipt_path = AV / "gen_receipt.json"
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    failed = dict(receipt.get("failed", {}))
    narration = receipt.get("narration", {"train": [], "exam": []})
    music = receipt.get("music", {"train": [], "exam": []})
    music_seeds = dict(receipt.get("music_seeds", {}))
    narr_meta = dict(receipt.get("narration_meta", {}))

    # D-050: assert the checkpoint against its receipt before ANY generation
    if not CKPT.exists():
        raise SystemExit(f"missing {CKPT} — stage the receipted checkpoint first")
    got_sha, got_bytes = _sha256(CKPT), CKPT.stat().st_size
    if got_sha != CKPT_SHA256 or got_bytes != CKPT_BYTES:
        raise SystemExit(f"checkpoint receipt MISMATCH: sha {got_sha} "
                         f"({got_bytes} bytes) != {CKPT_SHA256} ({CKPT_BYTES}) — STOP")

    for sub in ("fakes", "reals", "narration", "music"):
        (GEN / sub).mkdir(parents=True, exist_ok=True)

    # deterministic strata assignment (drawn once, receipted; resume-stable)
    rng = np.random.default_rng(SEED)
    train_keys = sorted(split["splits"]["train"])
    exam_keys = sorted(split["splits"]["exam"])
    narr_train = [train_keys[i] for i in rng.permutation(len(train_keys))[:NARR_TRAIN]]
    music_train = [train_keys[i] for i in rng.permutation(len(train_keys))[:NARR_TRAIN]]
    narr_exam = [f"{k}__{v}" for k in exam_keys for v in range(EXAM_VARIANTS)]
    music_exam = [f"{k}__{v}" for k in exam_keys for v in range(EXAM_VARIANTS)]

    sources = json.loads((AV / "sources.json").read_text())
    jobs = []                                   # (kind, key, src_path, out_path)
    for split_name, keys in (("train", train_keys), ("val", split["splits"]["val"]),
                             ("exam", exam_keys)):
        use = keys if not args.smoke else keys[:2]
        for k in use:
            src = Path(sources[k]["path"])
            if not src.exists():
                raise SystemExit(f"source missing: {src} — the diet manifest "
                                 f"is law; STOP")
            jobs.append(("pair", k, src, GEN / "fakes" / f"{k}.mp4"))
            jobs.append(("real", k, src, GEN / "reals" / f"{k}.mp4"))
    narr_jobs = ([("narration", f"{k}", _src_of(sources, k),
                   GEN / "narration" / f"{k}.mp4") for k in narr_train]
                 + [("narration", k, _src_of(sources, k.rsplit("__", 1)[0]),
                     GEN / "narration" / f"{k}.mp4") for k in narr_exam])
    music_jobs = ([("music", f"{k}", _src_of(sources, k),
                    GEN / "music" / f"{k}.mp4") for k in music_train]
                  + [("music", k, _src_of(sources, k.rsplit("__", 1)[0]),
                      GEN / "music" / f"{k}.mp4") for k in music_exam])
    if args.smoke:
        narr_jobs, music_jobs = narr_jobs[:2], music_jobs[:2]

    noface_path = AV / "noface.jsonl"
    t0 = time.time()
    n_done = 0
    for kind, key, src, out in jobs + narr_jobs + music_jobs:
        if out.exists():
            n_done += 1
            continue
        try:
            if kind == "pair":
                gen_fake(src, out)
            elif kind == "real":
                gen_real(src, out)
            elif kind == "narration":
                narr_meta[out.name] = gen_narration(src, out, key,
                                                    int(key.rsplit("__", 1)[1])
                                                    if "__" in key else 0, rng)
            elif kind == "music":
                seed = music_seeds.get(out.name)
                if seed is None:
                    seed = int(rng.integers(0, 2**31 - 1))
                    music_seeds[out.name] = seed
                gen_music(src, out, seed, AV / "work_music_tmp.wav")
        except FaceError as e:
            failed[key] = "no_face"
            with open(noface_path, "a") as f:
                f.write(json.dumps({"file": src.name, "method": "Wav2Lip",
                                    "label": 1, "reason": "no_face",
                                    "stage": "generation"}) + "\n")
            print(f"[gen] NOFACE {key}: excluded from the diet (§5.5)", flush=True)
            continue
        n_done += 1
        if n_done % 10 == 0:
            print(f"[gen] {n_done} artifacts ({time.time() - t0:.0f}s)",
                  flush=True)

    receipt.update({
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED, "smoke": bool(args.smoke),
        "checkpoint": {"path": str(CKPT.relative_to(WS)), "sha256": got_sha,
                       "bytes": got_bytes},
        "kokoro": {"model": KOKORO_MODEL, "license": "Apache-2.0",
                   "voices": VOICES},
        "commands": {
            "fake": "vendor inference.py --checkpoint_path wav2lip_gan.pth "
                    "--face src --audio src --outfile out --wav2lip_batch_size 128",
            "real": "ffmpeg -y -i src -strict -2 -q:v 1 -c:a copy out",
            "mux": "ffmpeg -y -i src -i aud -map 0:v:0 -map 1:a:0 -c:v copy "
                   "-c:a aac -strict -2 -shortest out"},
        "narration": {"train": [f"{k}.mp4" for k in narr_train],
                      "exam": [f"{k}.mp4" for k in narr_exam]},
        "music": {"train": [f"{k}.mp4" for k in music_train],
                  "exam": [f"{k}.mp4" for k in music_exam]},
        "music_seeds": music_seeds,
        "narration_meta": narr_meta,
        "failed": failed,
        "n_sources_total": len(train_keys) + len(split["splits"]["val"])
        + len(exam_keys),
        "n_failed": len(failed)})
    receipt_path.write_text(json.dumps(receipt, indent=2))
    print(f"[gen] receipt: {receipt_path} — {n_done} artifacts present, "
          f"{len(failed)} sources failed (no-face honesty)", flush=True)
    return 0


def _src_of(sources: dict, key: str) -> Path:
    return Path(sources[key]["path"])


if __name__ == "__main__":
    sys.exit(main())
