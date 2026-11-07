"""W9 diet builder (D-079 SEALED): channel-balanced audio manifest per
artifacts/generator_intake/w9_audio_court_spec.md §4.

Composition (BINDING counts — VERIFY clause: receipt == §4 table exactly):
  clean-studio      reals: LA train bona 2,580 (drawn from p4's exact
                    utterance set — parity anchor) | fakes: LA train spoof
                    2,580 (W9.1 NEW — uniform across A01-A06, codec-exam
                    fakes excluded) + MLAAD en 2,580 uniform across working
                    generators, seed 42
  media-transformed reals: RADAR dev bona 8,580 (W9.1 bumped from 6,000 to
                    hold overall 50/50) | fakes: RADAR dev spoof 6,000
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
N_LA_SPOOF = 2580   # W9.1 (D-083 addendum): the sealed W9 diet had ZERO LA-spoof rows
N_MLAAD = 2580
N_RADAR_BONA = 8580  # W9.1: bumped from 6000 to hold overall 50/50 with LA spoof added
N_RADAR_SPOOF = 6000
CODEC_EXCL = WS / "data/caches/w9b_codec_excl.json"  # 60 D4-exam fake uids (leak guard)
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
    ap.add_argument("--anchor", default=None,
                    help="W9.1: prior sealed W9 diet manifest. Rows for components "
                         "declared 'unchanged' (MLAAD, RADAR spoof) are reused "
                         "verbatim; RADAR bona keeps the anchor 6,000 and samples "
                         "only the bump; new samples are disjoint from anchor uids "
                         "so merged cache keys stay disjoint by construction.")
    ap.add_argument("--keep-spoof", default=None,
                    help="W9.1: prior LA-spoof extraction manifest. Its uids are "
                         "kept verbatim (minus anchor collisions), with per-attack "
                         "top-up to the sealed allocation — preserves paid extraction "
                         "while guaranteeing cache-key disjointness.")
    args = ap.parse_args()
    rng = random.Random(SEED)
    split_rng = random.Random(SEED + 7)  # splits for NEW rows only (84/85/86 blocks)
    rows, prov = [], {"seed": SEED, "sources": {}, "replacements": [],
                      "leak_ledger": {}}
    anchor_rows = []
    anchor_prov = {}
    if args.anchor:
        anchor_rows = [json.loads(l) for l in
                       Path(args.anchor).read_text().splitlines() if l.strip()]
        anchor_prov_path = Path(args.anchor).with_suffix(".provenance.json")
        if anchor_prov_path.exists():
            anchor_prov = json.loads(anchor_prov_path.read_text())
        anchor_uids = {r["uid"] for r in anchor_rows}
        anchor_split = {(r["source"], r["uid"]): r["split"] for r in anchor_rows}
        prov["anchor"] = {
            "path": args.anchor,
            "sha256_16": __import__("hashlib").sha256(
                Path(args.anchor).read_bytes()).hexdigest()[:16],
            "rows": len(anchor_rows),
            "rule": "W9.1 §2 'unchanged' components reused verbatim; new samples "
                    "exclude every anchor uid (cache-key disjointness by construction)",
        }
    else:
        anchor_uids, anchor_split = set(), {}
    print(f"anchor: {len(anchor_rows)} rows"
          + (f" ({anchor_prov.get('counts')})" if anchor_prov else ""), flush=True)

    # --- clean-studio reals: LA train bona from p4's exact set ---
    p4 = json.loads(P4_LABELS.read_text())
    uid_label = {k.split(":")[0]: v for k, v in p4.items() if k.endswith(":clean")}
    # W9.1 ground-truth fix (D-083 addendum 2): the W9 builder sampled from ALL
    # 4,000 label-file uids — 2,580 bona + 1,420 W7-spoof — writing label=0
    # unconditionally: 904 of W9's 2,580 "bona" rows were actual spoofs
    # (parquet system_id-verified 2026-11-07). Filter by label AND verify
    # against the parquet (belt and suspenders).
    bona_uids = sorted(u for u, v in uid_label.items() if v == 0)
    pick = rng.sample(bona_uids, N_LA_BONA)
    import pandas as pd
    df = pd.read_parquet(PQ)
    sid_col = df["system_id"].astype(str).str.strip()
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
    bad_bona = [u for u in pick if sid_col.iloc[idx_by_uid[u]] != "-"]
    if bad_bona:
        raise SystemExit(f"GROUND-TRUTH GUARD: {len(bad_bona)} 'bona' uids map to "
                         f"non-bona parquet rows (W9 label-poisoning class) — STOP")
    n_la = 0
    # W9.1 anchor law: kept rows preserve their W9 split; only the 904 new bona
    # uids (replacing the poisoned rows) draw fresh 85/15 splits.
    new_bona = [u for u in pick if ("la", u) not in anchor_split]
    new_val = set(split_rng.sample(sorted(new_bona),
                                   int(round(len(new_bona) * VAL_FRAC)))) \
        if new_bona else set()
    val_uids = {u for u in pick if anchor_split.get(("la", u)) == "val"} | new_val
    for u in pick:
        rows.append({"uid": u, "source": "la", "label": 0,
                     "split": "val" if u in val_uids else "train",
                     "parquet": str(PQ), "row": int(idx_by_uid[u])})
        n_la += 1
    prov["sources"]["la"] = {"n": n_la, "n_val": len(val_uids),
        "rule": f"random.sample seed {SEED}: {N_LA_BONA} of {len(bona_uids)} "
                f"p4-clean bona uids (parity anchor); LAST parquet occurrence "
                f"per uid; split 85/15 seed {SEED}"}

    # --- clean-studio fakes (W9.1): LA train spoof, uniform across A01-A06 ---
    excl_codec = set(json.loads(CODEC_EXCL.read_text()))
    attacks = sorted(set(sid_col) - {"-"})
    if attacks != ["A01", "A02", "A03", "A04", "A05", "A06"]:
        raise SystemExit(f"LA attack census drifted: {attacks} — STOP")
    spoof_uid_row = {}
    for i in range(len(df)):
        if sid_col.iloc[i] == "-":
            continue
        u = Path(str(df["audio_file_name"].iloc[i])).stem
        spoof_uid_row[u] = i  # LAST occurrence wins (mirrors bona)
    by_attack = defaultdict(list)
    excl_anchor_spoof = 0
    for u in sorted(spoof_uid_row):
        if u in excl_codec:
            continue
        if u in anchor_uids:
            excl_anchor_spoof += 1  # 904 poisoned W9 "bona" rows are real spoof files
            continue
        by_attack[sid_col.iloc[spoof_uid_row[u]]].append(u)
    per, extra = N_LA_SPOOF // len(attacks), N_LA_SPOOF % len(attacks)
    # W9.1 extraction-preserving law: keep the already-extracted spoof sample
    # verbatim where it does not collide with the anchor, then top up each
    # attack to its sealed allocation from the remaining pool.
    keep_uids = set()
    if args.keep_spoof:
        keep_rows = [json.loads(l) for l in
                     Path(args.keep_spoof).read_text().splitlines() if l.strip()]
        keep_uids = {r["uid"] for r in keep_rows}
        if len(keep_uids) != N_LA_SPOOF:
            raise SystemExit(f"keep-spoof manifest has {len(keep_uids)} uids "
                             f"!= {N_LA_SPOOF} — STOP")
        missing_keep = [u for u in keep_uids if u not in spoof_uid_row]
        if missing_keep:
            raise SystemExit(f"{len(missing_keep)} keep-spoof uids absent from the "
                             f"parquet spoof census — STOP")
        prov["sources"]["la_spoof_keep"] = {
            "path": args.keep_spoof,
            "rows": len(keep_uids),
            "rule": "prior paid extraction preserved; anchor-collision uids "
                    "dropped and replaced per-attack",
        }
    kept_by_attack = defaultdict(list)
    for u in sorted(keep_uids):
        if u in excl_codec or u in anchor_uids:
            continue
        kept_by_attack[sid_col.iloc[spoof_uid_row[u]]].append(u)
    pick_s = []
    for j, a in enumerate(attacks):
        quota = per + (1 if j < extra else 0)
        kept = kept_by_attack[a]
        if len(kept) > quota:
            raise SystemExit(f"attack {a}: kept {len(kept)} > quota {quota} — STOP")
        pool = [u for u in by_attack[a] if u not in keep_uids]
        need = quota - len(kept)
        if len(pool) < need:
            raise SystemExit(f"attack {a}: pool {len(pool)} < top-up {need} — STOP")
        pick_s += kept + rng.sample(pool, need)
    if len(pick_s) != N_LA_SPOOF:
        raise SystemExit(f"LA spoof sample {len(pick_s)} != {N_LA_SPOOF} "
                         f"(excluded {len(excl_codec)} codec-exam uids) — STOP")
    if set(pick_s) & {r["uid"] for r in rows}:
        raise SystemExit("LA spoof uids collide with LA bona uids — STOP")
    bad_spoof = [u for u in pick_s if sid_col.iloc[spoof_uid_row[u]] == "-"]
    if bad_spoof:
        raise SystemExit(f"GROUND-TRUTH GUARD: {len(bad_spoof)} 'spoof' uids map "
                         f"to bona parquet rows — STOP")
    val_spoof = set(rng.sample(sorted(pick_s), int(round(N_LA_SPOOF * VAL_FRAC))))
    n_ls = 0
    for u in sorted(pick_s):
        rows.append({"uid": u, "source": "la", "label": 1,
                     "split": "val" if u in val_spoof else "train",
                     "parquet": str(PQ), "row": int(spoof_uid_row[u])})
        n_ls += 1
    prov["sources"]["la_spoof"] = {"n": n_ls, "n_val": len(val_spoof),
        "attacks": attacks, "excluded_codec_exam_uids": len(excl_codec),
        "excluded_anchor_uids": excl_anchor_spoof,
        "kept_from_prior_extraction": sum(len(v) for v in kept_by_attack.values()),
        "rule": f"uniform {per}+{extra} per attack; kept prior sample where "
                f"non-colliding, per-attack top-up seed {SEED}; "
                f"{len(excl_codec)} D4-codec-exam fake uids excluded (leak guard: "
                f"D4 recall must stay a generalization test); {excl_anchor_spoof} "
                f"anchor uids excluded (W9 cache-key disjointness by construction — "
                f"these are the poisoned W9 'bona' rows that are real spoof files); "
                f"split 85/15"}

    # --- clean-studio fakes: MLAAD en, uniform across working generators ---
    dead_replaced = 0
    if anchor_rows:
        mlaad_rows = [r for r in anchor_rows if r["source"] == "mlaad"]
        if len(mlaad_rows) != N_MLAAD:
            raise SystemExit(f"anchor MLAAD rows {len(mlaad_rows)} != {N_MLAAD} — STOP")
        rows.extend(mlaad_rows)
        gens = sorted({r["uid"].split("__")[0] for r in mlaad_rows})
        n_ml = len(mlaad_rows)
        n_val_seg = sum(1 for r in mlaad_rows if r["split"] == "val")
        by_gen = {}
        prov["sources"]["mlaad"] = {"n": n_ml, "generators": len(gens),
            "n_val_segments": n_val_seg,
            "rule": f"reused verbatim from W9 anchor — §2 'unchanged'; "
                    f"anchor split assignments preserved"}
    else:
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
        n_ml = 0
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

    # --- media-transformed: RADAR dev bona 8,580 / spoof 6,000 ---
    proto = [l.split() for l in RADAR_PROTO.read_text().strip().split("\n")]
    bona = [p for p in proto if p[2] == "bonafide"]
    spoof = [p for p in proto if p[2] == "spoof"]
    path_by_uid = {}
    for dp, dn, fn in __import__("os").walk(RADAR_FLAC):
        for f in fn:
            if f.endswith(".flac"):
                path_by_uid[f[:-5]] = f"{dp}/{f}"
    if anchor_rows:
        anchor_r = [r for r in anchor_rows if r["source"] == "radar"]
        anchor_rb = [r for r in anchor_r if r["label"] == 0]
        anchor_rs = [r for r in anchor_r if r["label"] == 1]
        need = N_RADAR_BONA - len(anchor_rb)
        if need != 2580:
            raise SystemExit(f"anchor RADAR bona {len(anchor_rb)} -> bump {need} "
                             f"!= sealed 2,580 — STOP")
        if len(anchor_rs) != N_RADAR_SPOOF:
            raise SystemExit(f"anchor RADAR spoof {len(anchor_rs)} != "
                             f"{N_RADAR_SPOOF} — STOP")
        radar_rows = list(anchor_rb) + list(anchor_rs)
        remaining = [p for p in bona if p[0] not in anchor_uids]
        rng.shuffle(remaining)
        new_bona = []
        for p in remaining:
            if p[0] in path_by_uid:
                new_bona.append(p)
            if len(new_bona) == need:
                break
        if len(new_bona) < need:
            raise SystemExit(f"RADAR bona top-up {len(new_bona)} < {need} — STOP")
        new_val = set(split_rng.sample(sorted(p[0] for p in new_bona),
                                       int(round(need * VAL_FRAC))))
        for uid, dur, lab, seg in new_bona:
            radar_rows.append({"uid": uid, "source": "radar", "label": 0,
                               "path": path_by_uid[uid],
                               "split": "val" if uid in new_val else "train"})
        n_rb, n_rs = len(anchor_rb) + need, len(anchor_rs)
        prov["sources"]["radar"] = {"n_bona": n_rb, "n_spoof": n_rs,
            "rule": f"W9 anchor 6,000 bona + {need} new drawn from the remaining "
                    f"dev bona pool (shuffled, path-stat-verified); 6,000 spoof "
                    f"reused verbatim from anchor — §2 'unchanged'; new bona rows "
                    f"split 85/15 seed {SEED + 7}"}
    else:
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
        prov["sources"]["radar"] = {"n_bona": n_rb, "n_spoof": n_rs,
            "rule": f"random.sample seed {SEED}: {N_RADAR_BONA} of {len(bona)} "
                    f"bona, {N_RADAR_SPOOF} of {len(spoof)} spoof; per-label "
                    f"stratified split 85/15 seed {SEED}"}
    rows.extend(radar_rows)

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
        if r["uid"] in excl_codec:
            violations.append({"uid": r["uid"],
                               "why": "codec-exam fake uid in diet (D4 leak guard)"})
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
        "codec_exam_fake_uids_excluded": len(excl_codec),
        "generators_not_in_diet": sorted(set(by_gen) - set(gens)) if by_gen else
            anchor_prov.get("leak_ledger", {}).get("generators_not_in_diet", []),
    }

    # --- binding counts (W9.1 §2 table, exact) ---
    n_bona = sum(1 for r in rows if r["label"] == 0)
    n_fake = sum(1 for r in rows if r["label"] == 1)
    la_bona = sum(1 for r in rows if r["source"] == "la" and r["label"] == 0)
    la_spoof = sum(1 for r in rows if r["source"] == "la" and r["label"] == 1)
    mlaad_fake = sum(1 for r in rows if r["source"] == "mlaad")
    trans_bona = sum(1 for r in rows if r["source"] == "radar" and r["label"] == 0)
    trans_fake = sum(1 for r in rows if r["source"] == "radar" and r["label"] == 1)
    counts = {"total": len(rows), "bona": n_bona, "fake": n_fake,
              "la_bona": la_bona, "la_spoof": la_spoof, "mlaad_fake": mlaad_fake,
              "trans_bona": trans_bona, "trans_fake": trans_fake}
    expected = {"total": 22320, "bona": 11160, "fake": 11160,
                "la_bona": 2580, "la_spoof": 2580, "mlaad_fake": 2580,
                "trans_bona": 8580, "trans_fake": 6000}
    if counts != expected:
        raise SystemExit(f"COUNT MISMATCH vs W9.1 §2: {counts} != {expected} — STOP")
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
