"""W7 diet builder: assembles the training manifest JSONL from all sources.

Composition per spec artifacts/generator_intake/w7_audio_retrain_spec.md §2 proposal
(SEALED only at pre-registration — this builder emits the DRAFT manifest):
  LA train (p4's exact utterance set)  ~4,000 utts  (parquet rows)
  RADAR dev bona                        6,000 subsample of 10,573 (seed 42)
  RADAR dev spoof                       6,000 subsample of 33,461 (seed 42)
  MLAAD en fakes                        8,580 = 60 x 143 generators (seed 42)
  Kokoro (r8-era wavs)                  all available (label fake)

Manifest rows: {"uid", "source", "label", "path" | ("parquet"+"row")}
Provenance JSON beside the manifest: seeds, selection rules, per-source counts.

Usage: ./.venv/bin/python deepfake/scripts/build_audio_w7_diet.py --out <manifest.jsonl>
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
PQ = WS / "data/raw/asvspoof/data/train-00000-of-00001.parquet"
P4_LABELS = WS / "data/caches/asv_train.labels.json"
RADAR_PROTO = WS / "data/raw/radar2026/label_RADAR2026-dev.txt"
RADAR_FLAC = WS / "data/raw/radar2026/dev/RADAR2026-dev/flac"
MLAAD_LIST = WS / "data/raw/mlaad/_filelist.jsonl"
KOKORO_GLOB = WS / "data/corpora/audio_r8/fake"
SEED = 42
N_RADAR_BONA = 6000
N_RADAR_SPOOF = 6000
N_MLAAD_PER_GEN = 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mlaad-per-gen", type=int, default=N_MLAAD_PER_GEN)
    ap.add_argument("--radar-bona", type=int, default=N_RADAR_BONA)
    ap.add_argument("--radar-spoof", type=int, default=N_RADAR_SPOOF)
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows, prov = [], {"seed": SEED, "sources": {}}

    # --- LA (p4's exact set; parquet rows) ---
    p4 = json.loads(P4_LABELS.read_text())
    uid_label = {k.split(":")[0]: v for k, v in p4.items() if k.endswith(":clean")}
    import pandas as pd
    df = pd.read_parquet(PQ)
    uid_col = "audio_file_name" if "audio_file_name" in df.columns else "path"
    stems = df[uid_col].map(lambda p: Path(str(p)).stem)
    dup = int(stems.duplicated().sum())
    idx_by_uid = {}
    for i, u in enumerate(stems):  # LAST occurrence wins (mirrors p4 iterrows overwrite)
        if u in uid_label:
            idx_by_uid[u] = i
    n_la = 0
    for u, i in idx_by_uid.items():
        rows.append({"uid": u, "source": "la", "label": uid_label[u], "parquet": str(PQ), "row": i})
        n_la += 1
    prov["sources"]["la"] = {"n": n_la, "rule": "p4's exact utterance set (asv_train.labels.json uids); "
                             "LAST parquet occurrence per uid (mirrors p4)", "parquet_duplicate_uids": dup}

    # --- RADAR dev (stratified subsample, seed 42) ---
    proto = [l.split() for l in RADAR_PROTO.read_text().strip().split("\n")]
    bona = [p for p in proto if p[2] == "bonafide"]
    spoof = [p for p in proto if p[2] == "spoof"]
    # map uid -> file path (flac dir layout: flat or nested; resolve once)
    path_by_uid = {}
    for dp, dn, fn in __import__("os").walk(RADAR_FLAC):
        for f in fn:
            if f.endswith(".flac"):
                path_by_uid[f[:-5]] = f"{dp}/{f}"
    pick_b = rng.sample(bona, min(args.radar_bona, len(bona)))
    pick_s = rng.sample(spoof, min(args.radar_spoof, len(spoof)))
    n_rb = n_rs = 0
    for uid, dur, lab, seg in pick_b + pick_s:
        if uid not in path_by_uid:
            continue
        rows.append({"uid": uid, "source": "radar", "label": 0 if lab == "bonafide" else 1,
                     "path": path_by_uid[uid]})
        n_rb += lab == "bonafide"
        n_rs += lab == "spoof"
    prov["sources"]["radar"] = {"n_bona": n_rb, "n_spoof": n_rs,
                                "rule": f"random.sample seed {SEED}: {args.radar_bona} of {len(bona)} bona, "
                                        f"{args.radar_spoof} of {len(spoof)} spoof"}

    # --- MLAAD en (60 per generator, seed 42) ---
    by_gen = defaultdict(list)
    for l in MLAAD_LIST.read_text().splitlines():
        r = json.loads(l)
        by_gen[r["path"].split("/")[2]].append(r)
    n_ml = 0
    for g in sorted(by_gen):
        for it in rng.sample(by_gen[g], min(args.mlaad_per_gen, len(by_gen[g]))):
            base = it["path"].split("/")[-1][:-4]
            # generator-qualified uid: MLAAD basenames repeat ACROSS generators (same
            # LibriVox segment, different synthesis) — bare basenames collided 555x
            rows.append({"uid": f"{g}__{base}", "source": "mlaad",
                         "label": 1, "path": str(WS / "data/raw/mlaad" / it["path"]),
                         "segment": base})
            n_ml += 1
    prov["sources"]["mlaad"] = {"n": n_ml, "generators": len(by_gen),
                                "rule": f"random.sample seed {SEED}: {args.mlaad_per_gen} per generator x {len(by_gen)} gens"}

    # --- Kokoro (r8-era wavs; all available) ---
    n_ko = 0
    for p in sorted(KOKORO_GLOB.glob("t*_wild_*.wav")):
        rows.append({"uid": f"kokoro_{p.stem}", "source": "kokoro", "label": 1, "path": str(p)})
        n_ko += 1
    prov["sources"]["kokoro"] = {"n": n_ko, "rule": "all r8-era kokoro wavs (data/corpora/audio_r8/fake/t*_wild_*.wav)"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    prov["total"] = len(rows)
    prov["bona"] = sum(1 for r in rows if r["label"] == 0)
    prov["fake"] = sum(1 for r in rows if r["label"] == 1)
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps(prov, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
