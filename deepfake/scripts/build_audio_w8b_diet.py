"""W8b diet builder — channel-balanced specialist diet (D-064 user-authorized addendum).

Declared change (spec §2.4, BEFORE retraining): W8's diet + RADAR dev spoof
(transformed commercial TTS) on the fake side. Everything else identical to the
sealed W8 diet: same LA/RADAR bona, same MLAAD train generators, same 14-generator
holdout exclusion, Kokoro excluded, LA spoof excluded.

Emits diet_w8b.jsonl = diet_w8.jsonl rows + W7-manifest radar label=1 rows, with
provenance pinning both parents. The holdout is re-asserted against the sealed
W8 holdout manifest (no re-derivation drift possible).

Usage: ./.venv/bin/python deepfake/scripts/build_audio_w8b_diet.py
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
W7_MANIFEST = WS / "artifacts/generator_intake/w7_diet_sealed/diet_draft.jsonl"
W8_DIR = WS / "artifacts/generator_intake/w8_diet_sealed"
OUT = WS / "artifacts/generator_intake/w8b_diet_sealed"


def main() -> int:
    rows_w8 = [json.loads(l) for l in (W8_DIR / "diet_w8.jsonl").read_text().splitlines() if l.strip()]
    hold_w8 = [json.loads(l) for l in (W8_DIR / "holdout_w8.jsonl").read_text().splitlines() if l.strip()]
    w7 = [json.loads(l) for l in W7_MANIFEST.read_text().splitlines() if l.strip()]

    holdout_uids = {r["uid"] for r in hold_w8}
    radar_spoof = [r for r in w7 if r["source"] == "radar" and r["label"] == 1]
    # safety: no radar-spoof uid may collide with the holdout (different sources — assert anyway)
    if holdout_uids & {r["uid"] for r in radar_spoof}:
        print("ERROR: holdout uid collision with radar spoof", file=sys.stderr)
        return 1

    diet = rows_w8 + radar_spoof
    counts = Counter((r["source"], r["label"]) for r in diet)
    print(f"W8b diet: {len(diet)} rows — "
          f"{ {f'{s}/label={l}': n for (s, l), n in sorted(counts.items())} }", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    diet_p = OUT / "diet_w8b.jsonl"
    diet_p.write_text("".join(json.dumps(r) + "\n" for r in diet))
    prov = {
        "spec": "artifacts/generator_intake/w8_specialist_spec.md §2.4 W8b ADDENDUM "
                "(user-authorized 2026-10-30, D-064)",
        "declared_change": "fake side += RADAR dev spoof (transformed commercial TTS) — "
                           "decouples channel cleanliness from class (D-064 confound antidote)",
        "parent_w8_diet_sha256_16": hashlib.sha256((W8_DIR / "diet_w8.jsonl").read_bytes()).hexdigest()[:16],
        "parent_w7_manifest_sha256_16": hashlib.sha256(W7_MANIFEST.read_bytes()).hexdigest()[:16],
        "added_rows": len(radar_spoof),
        "total_rows": len(diet),
        "by_source_label": {f"{s}/label={l}": n for (s, l), n in sorted(counts.items())},
        "holdout_unchanged": True,
        "holdout_generators": sorted({r["path"].split("data/raw/mlaad/")[-1].split("/")[2]
                                      for r in hold_w8}),
        "bars": "UNCHANGED — same sealed S0-S3 + C1-C5 + C7 bind W8b verbatim",
        "diet_w8b_sha256_16": hashlib.sha256(diet_p.read_bytes()).hexdigest()[:16],
    }
    (OUT / "provenance.json").write_text(json.dumps(prov, indent=1))
    print(f"sealed: {diet_p} ({prov['diet_w8b_sha256_16']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
