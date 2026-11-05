"""W9 diet builder (D-079 SEALED): channel-balanced audio manifest per
artifacts/generator_intake/w9_audio_court_spec.md §4.

Composition (BINDING counts — VERIFY clause: receipt == §4 table exactly):
  clean-studio      reals: LA train bona 2,580 (drawn from p4's exact
                    utterance set — parity anchor) | fakes: MLAAD en 2,580
                    uniform across working generators, seed 42
  media-transformed reals: RADAR dev bona 6,000 | fakes: RADAR dev spoof 6,000
  total             8,580 / 8,580 — 50/50 overall AND per channel

Kokoro standalone slice DROPPED (sealed ruling: MLAAD's kokoro generator covers
the family inside the clean channel). Channel D CUT — no prosody rows.

Rows: {"uid", "source", "label", "split", "path" | ("parquet"+"row")}
(+ "segment" for MLAAD — E1-style grouping: same text segment never straddles
splits). Splits: per-source stratified train/val 85/15, seed 42.

Leak ledger (asserted at build): ITW / DF-Eval / 2021-eval shards / RADAR-eval
/ Kokoro absent by construction; RADAR paths restricted to dev/; the LA parquet
reference is the TRAIN parquet only. Dead-file guard: every selected path is
stat-verified at build; dead picks are replaced from the same generator's
remaining pool (replacement log in provenance) so the sealed counts hold on
disk, not just on paper (the W7 wave disclosed 903 dead files post-hoc — this
builder closes that gap).

Usage: ./.venv/bin/python deepfake/scripts/build_audio_w9_diet.py \
         --out artifacts/generator_intake/w9_diet_sealed/manifest.jsonl
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

WS = Path("/Users/Dataghost/Argus_Prime")
PQ = WS / "data/raw/asvspoof/data/train-00000-of-00001.parquet"
P4_LABELS = WS / "data/caches/asv_train.labels.json"
RADAR_PROTO = WS / "data/raw/radar2026/label_RADAR2026-dev.txt"
RADAR_FLAC = WS / "data/raw/radar2026/dev/RADAR2026-dev/flac"
MLAAD_LIST = WS / "data/raw/mlaad/_filelist.jsonl"
SEED = 42
N_LA_BONA = 2580
N_MLAAD = 2580
N_RADAR_BONA = 6000
N_RADAR_SPOOF = 6000
VAL_FRAC = 0.15

# leak-ledger boundaries: every diet path must live INSIDE these roots.
# Forbidden corpora (checked as explicit roots, not naive substrings — MLAAD's
# kokoro GENERATOR is legitimately in-diet per the sealed ruling; the dropped
# slice is the r8-era Kokoro corpus at data/corpora/audio_r8):
FORBIDDEN_ROOTS = [WS / "data/raw/itw", WS / "data/dfeval",
                   WS / "data/raw/radar2026/eval", WS / "data/corpora/audio_r8"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(SEED)
    rows, prov = [], {"seed": SEED, "sources": {}, "replacements": [],
                      "leak_ledger": {}}

    # --- clean-studio reals: LA train bona from p4's exact set ---
    p4 = json.loads(P4_LABELS.read_text())
    uid_label = {k.split(":")[0]: v for k, v in p4.items() if k.endswith(":clean")}
    bona_uids = sorted(uid_label)
    pick = rng.sample(bona_uids, N_LA_BONA)
    import pandas as pd
    df = pd.read_parquet(PQ)
    uid_col = "audio_file_name" if "audio_file_name" in df.columns else "path"
    stems = df[uid_col].map(lambda p: Path(str(p)).stem)
    idx_by_uid = {}
    for i, u in enumerate(stems):  # LAST occurrence wins (mirrors p4)
        if u in uid_label:
            idx_by_uid[u] = i
    missing = [u for u in pick if u not in idx_by_uid]
    if missing:
        raise SystemExit(f"LA parity anchor broken: {len(missing)} picked uids "
                         f"absent from the train parquet — STOP")
    n_la = 0
    val_uids = set(rng.sample(pick, int(round(N_LA_BONA * VAL_FRAC))))
    for u in pick:
        rows.append({"uid": u, "source": "la", "label": 0,
                     "split": "val" if u in val_uids else "train",
                     "parquet": str(PQ), "row": int(idx_by_uid[u])})
        n_la += 1
    prov["sources"]["la"] = {"n": n_la, "n_val": len(val_uids),
        "rule": f"random.sample seed {SEED}: {N_LA_BONA} of {len(bona_uids)} "
                f"p4-clean bona uids (parity anchor); LAST parquet occurrence "
                f"per uid; split 85/15 seed {SEED}"}

    # --- clean-studio fakes: MLAAD en, uniform across working generators ---
    by_gen = defaultdict(list)
    for l in MLAAD_LIST.read_text().splitlines():
        r = json.loads(l)
        by_gen[r["path"].split("/")[2]].append(r)
    # dead-file guard: keep only generators with >= N_MLAAD-per-gen-pool files
    # that EXIST on disk (stat truth, not filelist hope)
    alive_by_gen = {}
    for g, items in by_gen.items():
        alive = [it for it in items
                 if (WS / "data/raw/mlaad" / it["path"]).exists()]
        if len(alive) >= 19:
            alive_by_gen[g] = alive
    gens = sorted(alive_by_gen)
    spec_expected_gens = 136
    if len(gens) != spec_expected_gens:
        print(f"DISCLOSURE: working generators = {len(gens)} "
              f"(spec expected {spec_expected_gens} = 143 wholly-dead removed); "
              f"recorded in provenance — counts below remain binding", flush=True)
    # even allocation: 2,580 across gens, per-gen cap = allocation
    alloc = {g: N_MLAAD // len(gens) for g in gens}
    for g in sorted(gens)[:N_MLAAD % len(gens)]:
        alloc[g] += 1
    n_ml, dead_replaced = 0, 0
    mlaad_rows = []
    for g in gens:
        pool = list(alive_by_gen[g])
        rng.shuffle(pool)
        take, k = [], 0
        while len(take) < alloc[g] and k < len(pool):
            it = pool[k]; k += 1
            p = WS / "data/raw/mlaad" / it["path"]
            if not p.exists():
                dead_replaced += 1
                continue
            take.append((it, p))
        if len(take) < alloc[g]:
            raise SystemExit(f"MLAAD generator {g}: only {len(take)} alive "
                             f"files < allocation {alloc[g]} — STOP")
        for it, p in take:
            base = it["path"].split("/")[-1][:-4]
            # generator-qualified uid: MLAAD basenames repeat ACROSS generators
            mlaad_rows.append({"uid": f"{g}__{base}", "source": "mlaad",
                               "label": 1, "path": str(p), "segment": base})
            n_ml += 1
    # E1-style segment grouping: whole segments to one side, 85/15 by segment
    segs = defaultdict(list)
    for r in mlaad_rows:
        segs[r["segment"]].append(r)
    seg_keys = sorted(segs)
    rng.shuffle(seg_keys)
    n_val_seg = int(round(len(seg_keys) * VAL_FRAC))
    val_segs = set(seg_keys[:n_val_seg])
    for r in mlaad_rows:
        r["split"] = "val" if r["segment"] in val_segs else "train"
    rows.extend(mlaad_rows)
    prov["sources"]["mlaad"] = {"n": n_ml, "generators": len(gens),
        "generators_list_len_checked": len(by_gen),
        "dead_files_replaced_at_build": dead_replaced,
        "n_val_segments": n_val_seg,
        "rule": f"even allocation {N_MLAAD} across {len(gens)} working gens "
                f"(>=19 alive files each, stat-verified); segment-grouped "
                f"split 85/15 seed {SEED}"}

    # --- media-transformed: RADAR dev 6,000/6,000 ---
    proto = [l.split() for l in RADAR_PROTO.read_text().strip().split("\n")]
    bona = [p for p in proto if p[2] == "bonafide"]
    spoof = [p for p in proto if p[2] == "spoof"]
    path_by_uid = {}
    for dp, dn, fn in __import__("os").walk(RADAR_FLAC):
        for f in fn:
            if f.endswith(".flac"):
                path_by_uid[f[:-5]] = f"{dp}/{f}"
    pick_b = rng.sample(bona, min(N_RADAR_BONA, len(bona)))
    pick_s = rng.sample(spoof, min(N_RADAR_SPOOF, len(spoof)))
    n_rb = n_rs = 0
    radar_rows = []
    for uid, dur, lab, seg in pick_b + pick_s:
        if uid not in path_by_uid:
            continue
        radar_rows.append({"uid": uid, "source": "radar",
                           "label": 0 if lab == "bonafide" else 1,
                           "path": path_by_uid[uid]})
        n_rb += lab == "bonafide"
        n_rs += lab == "spoof"
    # per-label stratified split
    for lab_val, subset in ((0, [r for r in radar_rows if r["label"] == 0]),
                            (1, [r for r in radar_rows if r["label"] == 1])):
        keys = sorted(r["uid"] for r in subset)
        rng.shuffle(keys)
        val_set = set(keys[:int(round(len(keys) * VAL_FRAC))])
        for r in subset:
            r["split"] = "val" if r["uid"] in val_set else "train"
    rows.extend(radar_rows)
    prov["sources"]["radar"] = {"n_bona": n_rb, "n_spoof": n_rs,
        "rule": f"random.sample seed {SEED}: {N_RADAR_BONA} of {len(bona)} "
                f"bona, {N_RADAR_SPOOF} of {len(spoof)} spoof; per-label "
                f"stratified split 85/15 seed {SEED}"}

    # --- leak ledger (assert, then record) ---
    violations = []
    for r in rows:
        p = r.get("path") or r.get("parquet", "")
        if any(str(root) in p for root in FORBIDDEN_ROOTS):
            violations.append({"uid": r["uid"], "path": p, "why": "forbidden root"})
        if r["source"] == "la" and r.get("parquet") != str(PQ):
            violations.append({"uid": r["uid"], "parquet": r.get("parquet"),
                               "why": "la parquet must be the TRAIN parquet"})
        if r["source"] == "radar" and "/radar2026/eval/" in p:
            violations.append({"uid": r["uid"], "path": p, "why": "radar eval path"})
    if violations:
        raise SystemExit(f"LEAK LEDGER VIOLATIONS ({len(violations)}): "
                         f"{violations[:5]} — STOP")
    r8_rows = [r for r in rows if "/corpora/audio_r8/" in (r.get("path") or "")]
    if r8_rows:
        raise SystemExit("r8 Kokoro slice rows present — sealed ruling says "
                         "DROPPED — STOP")
    prov["leak_ledger"] = {
        "forbidden_roots": [str(r) for r in FORBIDDEN_ROOTS],
        "violations": 0,
        "itw_dfeval_2021eval_radareval_r8kokoro_absent": True,
        "mlaad_kokoro_generator_in_diet_by_ruling": True,
        "la_parquet_is_train_split_only": str(PQ),
        "generators_not_in_diet": sorted(set(by_gen) - set(gens)),
    }

    # --- binding counts (§4 table, exact) ---
    n_bona = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)
    clean_bona = sum(1 for r in rows if r["label"] == 0 and r["source"] in ("la",))
    clean_fake = sum(1 for r in rows if r["label"] == 1 and r["source"] == "mlaad")
    trans_bona = sum(1 for r in rows if r["label"] == 0 and r["source"] == "radar")
    trans_fake = sum(1 for r in rows if r["label"] == 1 and r["source"] == "radar")
    counts = {"total": len(rows), "bona": n_bona, "fake": n_fake,
              "clean_bona": clean_bona, "clean_fake": clean_fake,
              "trans_bona": trans_bona, "trans_fake": trans_fake}
    expected = {"total": 17160, "bona": 8580, "fake": 8580,
                "clean_bona": 2580, "clean_fake": 2580,
                "trans_bona": 6000, "trans_fake": 6000}
    if counts != expected:
        raise SystemExit(f"COUNT MISMATCH vs §4: {counts} != {expected} — STOP")
    prov["counts"] = counts
    prov["counts_binding_match_sealed_table"] = True

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    out.with_suffix(".provenance.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps({"counts": counts, "generators": len(gens),
                      "dead_replaced": dead_replaced,
                      "splits": {"train": sum(1 for r in rows if r["split"] == "train"),
                                  "val": sum(1 for r in rows if r["split"] == "val")}}, indent=1))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
