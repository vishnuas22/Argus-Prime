"""W8 specialist diet builder — re-partitions the SEALED W7 manifest (zero new extraction).

Per spec artifacts/generator_intake/w8_specialist_spec.md §2 (PRE-REGISTERED 2026-10-30,
user ruling: holdout rule (a) — alphabetically-first 14 fully-clean MLAAD generators):

  TRAIN diet (diet_w8.jsonl):
    la    label=0 (bona)          all rows from W7 manifest          (~2,580 utts)
    radar label=0 (bona)          all rows from W7 manifest          (6,000 utts)
    mlaad label=1, gen NOT in H   122 contributing generators        (~7,180 utts)
  HOLDOUT exam set (holdout_w8.jsonl, S1 ONLY — never trains):
    mlaad label=1, gen IN H       the 14 sealed holdout generators   (840 utts)

  EXCLUDED entirely: kokoro cache (S3 stays a generalization exam), LA spoof,
  RADAR spoof (p4's territory / manifold-narrowness law), ITW/DF-Eval/2021-eval
  (exam-only, leak law).

The holdout is RE-DERIVED here from the sealed W7 diet + extracted MLAAD cache
(missing-uid audit, the D-063 census method) and ASSERTED equal to the spec's
sealed list — derivation and spec cannot drift apart silently.

Usage: ./.venv/bin/python deepfake/scripts/build_audio_w8_diet.py \
    --out-dir artifacts/generator_intake/w8_diet_sealed
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
W7_MANIFEST = WS / "artifacts/generator_intake/w7_diet_sealed/diet_draft.jsonl"
W7_MLAAD_CACHE = WS / "data/caches/asv_w7_mlaad.npz"
SPEC = WS / "artifacts/generator_intake/w8_specialist_spec.md"

# Sealed at pre-registration (spec §2.2) — the builder ASSERTS its re-derivation
# matches this list exactly before emitting anything.
SPEC_HOLDOUT = [
    "Audio8-TTS-Preview-0.1b", "Audio8-TTS-Preview-0.6b", "Cartesia.ai (Sonic-3)",
    "ChatTTS", "Chatterbox", "Chatterbox-Turbo", "DeepGram", "Dramabox",
    "Echo-TTS", "Edge-TTS", "ElevenLabs-Turbo-v2.5", "ElevenLabs-v2-Multilingual",
    "ElevenLabs-v3", "FireRedTTS-2.0",
]
N_HOLDOUT = 14


def gen_of(row) -> str:
    """Generator name from a W7 mlaad manifest row (relative path segment 2)."""
    rel = str(row["path"]).split("data/raw/mlaad/")[-1]
    return rel.split("/")[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="artifacts/generator_intake/w8_diet_sealed")
    args = ap.parse_args()
    out = WS / args.out_dir

    rows = [json.loads(l) for l in W7_MANIFEST.read_text().splitlines() if l.strip()]
    by_src = Counter(r["source"] for r in rows)
    print(f"W7 sealed manifest: {len(rows)} rows {dict(by_src)}", flush=True)

    # --- re-derive holdout: generators with ZERO missing uids in the W7 extraction ---
    z = np.load(W7_MLAAD_CACHE)
    have = {k.split(":")[0] for k in z.files}
    mlaad = [r for r in rows if r["source"] == "mlaad"]
    total = Counter(gen_of(r) for r in mlaad)
    missing = Counter(gen_of(r) for r in mlaad if r["uid"] not in have)
    clean = sorted(g for g in total if missing.get(g, 0) == 0)
    derived = clean[:N_HOLDOUT]
    if derived != SPEC_HOLDOUT:
        print("ERROR: re-derived holdout differs from the spec-sealed list — "
              "refusing to build (D-046: the seal is the seal).", file=sys.stderr)
        for a, b in zip(derived, SPEC_HOLDOUT):
            if a != b:
                print(f"  derived={a!r}  spec={b!r}", file=sys.stderr)
        return 1
    print(f"holdout re-derived == spec-sealed list ({len(derived)} generators) OK", flush=True)
    H = set(derived)

    # --- partition ---
    diet, holdout = [], []
    counts, holdout_counts = Counter(), Counter()
    for r in rows:
        s = r["source"]
        if s == "kokoro":
            continue                      # S3 stays a generalization exam
        if s == "la" and r["label"] != 0:
            continue                      # LA spoof = p4's territory
        if s == "radar" and r["label"] != 0:
            continue                      # RADAR spoof = manifold-narrowness law
        if s == "mlaad" and gen_of(r) in H:
            holdout.append(r)
            holdout_counts[gen_of(r)] += 1
            continue
        diet.append(r)
        counts[(s, r["label"])] += 1

    print(f"W8 TRAIN diet: {len(diet)} rows — "
          f"{ {f'{s}/label={l}': n for (s, l), n in sorted(counts.items())} }", flush=True)
    print(f"W8 HOLDOUT: {len(holdout)} rows across {len(holdout_counts)} generators", flush=True)

    # --- emit + seal ---
    out.mkdir(parents=True, exist_ok=True)
    diet_p, hold_p = out / "diet_w8.jsonl", out / "holdout_w8.jsonl"
    diet_p.write_text("".join(json.dumps(r) + "\n" for r in diet))
    hold_p.write_text("".join(json.dumps(r) + "\n" for r in holdout))
    w7_sha = hashlib.sha256(W7_MANIFEST.read_bytes()).hexdigest()[:16]
    prov = {
        "spec": "artifacts/generator_intake/w8_specialist_spec.md (PRE-REGISTERED 2026-10-30)",
        "parent_w7_manifest_sha256_16": w7_sha,
        "holdout_rule": "alphabetically-first 14 generators with ZERO missing uids in the "
                        "sealed W7 MLAAD extraction (D-063 census method); re-derived here "
                        "and asserted == spec §2.2 sealed list",
        "holdout_generators": derived,
        "train_rows": len(diet),
        "train_by_source_label": {f"{s}/label={l}": n for (s, l), n in sorted(counts.items())},
        "holdout_rows": len(holdout),
        "holdout_by_generator": dict(sorted(holdout_counts.items())),
        "excluded": ["kokoro cache (S3 generalization exam)", "LA spoof", "RADAR spoof",
                     "ITW/DF-Eval/2021-eval/codec fixtures (exam-only, leak law)"],
        "recipe_delta_declared_pre_training": "LogisticRegression(class_weight='balanced') — "
                                               "D-063 mechanism (a) antidote, declared in spec §2.1",
        "diet_w8_sha256_16": hashlib.sha256(diet_p.read_bytes()).hexdigest()[:16],
        "holdout_w8_sha256_16": hashlib.sha256(hold_p.read_bytes()).hexdigest()[:16],
    }
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"sealed: {diet_p} ({prov['diet_w8_sha256_16']}) + {hold_p} "
          f"({prov['holdout_w8_sha256_16']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
