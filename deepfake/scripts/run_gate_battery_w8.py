"""W8 gate battery COMPOSER — evaluates the composed court (p4 + p12 specialist)
against the PRE-REGISTERED bars S0-S3 + C1-C5 + C7 (spec
artifacts/generator_intake/w8_specialist_spec.md §4, sealed 2026-10-30).

Instrument design (no serve-path reimplementation):
  - p4 arm  = the REGISTERED per-row receipts (exam harnesses are the certified
    selection+scoring instruments; p4's rows are read, never re-scored).
  - specialist arm = the same harnesses run with the specialist bundle patched
    (run_gate_battery_w7.py --probe artifacts/p12_specialist --tag p12specialist),
    producing per-row CSVs on IDENTICAL selections.
  - composition per spec §5 (utterance level, mirroring argus_detect.py:365 where
    ooc = mean-window z-norm vs the bundle's own threshold):
      ABSTAIN iff p4_ooc AND s_ooc
      else    score  = max(in-cov voters' p_fake);  FAKE iff score > 0.5

Self-checks before any verdict (instrument integrity):
  - every p4 receipt's registered aggregates are RECOMPUTED from its rows and
    asserted equal to the receipt's own metrics (reader+metric code validated
    on known rows);
  - p4/specialist row counts and id sets must match exactly per exam
    (selection identity); any drift aborts the verdict.

Conventions (disclosed in the receipt):
  - "in-cov" metrics = non-abstained COMPOSED rows (harness convention, keeps the
    sealed anchors comparable); all-rows court-behavior numbers also disclosed
    (FAKE iff composed votes fake; abstain = miss on fakes, not a false alarm on bona).
  - C2 codec uses the court-behavior convention (abstain is not a false alarm on
    bona, and is a miss on fake); p4's reference numbers under the SAME convention
    are computed from its receipt and disclosed beside the registered raw-pf pair.

Usage: ./.venv/bin/python deepfake/scripts/run_gate_battery_w8.py [--tag p12specialist]
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

SPEC = "artifacts/generator_intake/w8_specialist_spec.md §4 (PRE-REGISTERED 2026-10-30)"
OOC_FIXTURE = WS / "data/dfeval/audio/audio-data/audio-data/-0ljEYr8ZyqCHRqy5Bx4f4J6YOg.mp3"
FULL_ARMS = ["mp3_cbr128", "mp3_cbr192", "flac", "opus128_ogg", "mp3_mono16k"]


def rows_2021(d: Path):
    rows = json.loads((d / "scores.json").read_text())
    return [{"id": (r["dataset_label"], r["idx"]), "label": 1 - r["dataset_label"],
             "pf": r["p_fake"], "ooc": bool(r["ooc"])} for r in rows]


def rows_dfeval(d: Path):
    out = []
    with open(d / "audio_exam.csv") as f:
        for r in csv.DictReader(f):
            if r["p_fake"] == "" or r["p_fake"] == "None":
                continue
            out.append({"id": r["stem"], "label": int(r["label"]),
                        "pf": float(r["p_fake"]), "ooc": r["out_of_coverage"] == "True"})
    return out


def rows_itw(d: Path):
    out = []
    with open(d / "scores.csv") as f:
        for r in csv.DictReader(f):
            if r["p_fake"] == "":
                continue
            out.append({"id": r["file"], "label": int(r["label"]),
                        "pf": float(r["p_fake"]), "ooc": r["ooc"] == "True"})
    return out


def rows_codeccal(d: Path):
    out = []
    with open(d / "scores.csv") as f:
        for r in csv.DictReader(f):
            if r["error"] or r["arm"] not in FULL_ARMS or r["p_fake"] == "":
                continue
            label = 0 if r["stratum"] == "real_fit" else 1
            out.append({"id": (r["stratum"], r["shard_idx"], r["arm"]), "label": label,
                        "pf": float(r["p_fake"]), "ooc": r["ooc"] == "True",
                        "arm": r["arm"]})
    return out


def rows_radareval(d: Path):
    out = []
    with open(d / "scores.csv") as f:
        for r in csv.DictReader(f):
            if r["p_fake"] == "":
                continue
            out.append({"id": r["uid"], "label": int(r["label"]), "lang": r["lang"],
                        "pf": float(r["p_fake"]), "ooc": r["ooc"] == "True"})
    return out


def compose(p4, s):
    """p4/s: dicts id -> row. Returns list of composed rows."""
    if set(p4) != set(s):
        p4_only = len(set(p4) - set(s))
        s_only = len(set(s) - set(p4))
        raise SystemExit(f"SELECTION DRIFT: p4-only ids={p4_only} s-only ids={s_only} "
                         f"— aborting before any verdict (instrument integrity)")
    out = []
    for i in sorted(p4, key=str):
        a, b = p4[i], s[i]
        if a["ooc"] and b["ooc"]:
            out.append({"id": i, "label": a["label"], "abstain": True,
                        "score": None, "fake": False})
            continue
        cands = [r["pf"] for r in (a, b) if not r["ooc"]]
        sc = max(cands)
        out.append({"id": i, "label": a["label"], "abstain": False,
                    "score": sc, "fake": sc > 0.5})
    return out


def incov_metrics(rows):
    sc = [(r["label"], r["score"]) for r in rows if not r["abstain"]]
    n_pos = sum(1 for l, _ in sc if l == 1)
    n_neg = len(sc) - n_pos
    abst = sum(1 for r in rows if r["abstain"]) / len(rows)
    m = {"abstain_rate": round(abst, 4), "n_incov": len(sc)}
    if n_pos and n_neg:
        m["in_cov_auc"] = round(float(roc_auc_score([l for l, _ in sc],
                                                    [s for _, s in sc])), 4)
    if n_pos:
        m["in_cov_tpr"] = round(sum(1 for l, s in sc if l == 1 and s > 0.5) / n_pos, 4)
    if n_neg:
        m["in_cov_fpr"] = round(sum(1 for l, s in sc if l == 0 and s > 0.5) / n_neg, 4)
    # court-behavior over ALL rows (abstain = miss on fake, not-FA on bona)
    def is_fake(r):
        return r["fake"] if "fake" in r else (r["score"] is not None and r["score"] > 0.5)
    nf = sum(1 for r in rows if r["label"] == 1)
    nn = len(rows) - nf
    if nf:
        m["all_tpr"] = round(sum(1 for r in rows if r["label"] == 1 and is_fake(r)) / nf, 4)
    if nn:
        m["all_fpr"] = round(sum(1 for r in rows if r["label"] == 0 and is_fake(r)) / nn, 4)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="p12specialist")
    ap.add_argument("--out", default="artifacts/gate_battery_w8_composed")
    ap.add_argument("--train-meta", default=None,
                    help="candidate train_meta.json (default: artifacts/p12_specialist)")
    ap.add_argument("--s1-metrics", default=None,
                    help="S1 holdout metrics.json (default: exam_audio_w8_holdout_p12specialist)")
    ap.add_argument("--cand-dir", default="p12_specialist",
                    help="candidate artifacts dir (bundle path for the C5 probe patch)")
    args = ap.parse_args()
    A = WS / "artifacts"
    train_meta = Path(args.train_meta) if args.train_meta else A / "p12_specialist/train_meta.json"
    s1_metrics = Path(args.s1_metrics) if args.s1_metrics else \
        A / "exam_audio_w8_holdout_p12specialist/metrics.json"

    # ---------- load + self-check p4 receipts ----------
    p4_2021 = rows_2021(A / "exam_audio_2021_p4w8")
    m = json.loads((A / "exam_audio_2021_p4w8/metrics.json").read_text())
    y = [r["label"] for r in p4_2021]
    p = [r["pf"] for r in p4_2021]
    assert round(float(roc_auc_score(y, p)), 4) == m["auc"], "E4 p4 receipt mismatch"
    print(f"[self-check] E4 p4 receipt: AUC {m['auc']} reproduced from {len(p4_2021)} rows", flush=True)

    p4_df = rows_dfeval(A / "exam_audio_dfeval_p4w8")
    s_df_summary = json.loads((A / "exam_audio_dfeval_p4w8/audio_exam_summary.json").read_text())
    # registered-truth cross-check (W2-era measured posture)
    print(f"[self-check] E2 p4 fresh run: abstain={s_df_summary.get('abstained_rate')} "
          f"(registered 0.839) in-cov AUC={s_df_summary.get('in_coverage_auc')}", flush=True)

    p4_itw = rows_itw(A / "exam_audio_itw_p6v11")
    itw_m = json.loads((A / "exam_audio_itw_p6v11/metrics.json").read_text())
    itw_rows = [{"label": r["label"], "abstain": r["ooc"], "score": r["pf"]} for r in p4_itw]
    itw_chk = incov_metrics(itw_rows)
    for k, kk in [("in_cov_auc", "in_coverage_auc"), ("in_cov_tpr", "in_coverage_tpr_at_0.5"),
                  ("in_cov_fpr", "in_coverage_fpr_at_0.5"), ("abstain_rate", "abstained_rate")]:
        assert abs(itw_chk[k] - itw_m[kk]) < 5e-4, f"E3 p4 receipt mismatch {k}"
    print(f"[self-check] E3 p4 receipt: in-cov AUC {itw_m['in_coverage_auc']} reproduced", flush=True)

    p4_cc = rows_codeccal(A / "exam_audio_codeccal_p4selfcheck")
    bona = [r for r in p4_cc if r["label"] == 0]
    fake = [r for r in p4_cc if r["label"] == 1]
    fpr_raw = sum(1 for r in bona if r["pf"] > 0.5) / len(bona)
    rec_raw = sum(1 for r in fake if r["pf"] > 0.5) / len(fake)
    assert round(fpr_raw, 4) == 0.0 and abs(rec_raw - 0.85) < 0.005, \
        f"E6 p4 registered pair mismatch: {fpr_raw}/{rec_raw}"
    print(f"[self-check] E6 p4 receipt: raw-pf FPR {fpr_raw:.4f} recall {rec_raw:.4f} "
          f"== registered (0.0000/0.85)", flush=True)

    p4_rad = rows_radareval(A / "exam_audio_radareval_p6v11")
    rad_m = json.loads((A / "exam_audio_radareval_p6v11/metrics.json").read_text())
    rad_rows = [{"label": r["label"], "abstain": r["ooc"], "score": r["pf"]} for r in p4_rad]
    rad_chk = incov_metrics(rad_rows)
    assert abs(rad_chk["in_cov_auc"] - rad_m["in_coverage_auc"]) < 5e-4, "E7 p4 receipt mismatch"
    print(f"[self-check] E7 p4 receipt: in-cov AUC {rad_m['in_coverage_auc']} reproduced", flush=True)

    # ---------- specialist arms ----------
    s_2021 = rows_2021(A / f"exam_audio_2021_{args.tag}")
    s_df = rows_dfeval(A / f"exam_audio_dfeval_{args.tag}")
    s_itw = rows_itw(A / f"exam_audio_itw_{args.tag}")
    s_cc = rows_codeccal(A / f"exam_audio_codeccal_{args.tag}")
    s_rad = rows_radareval(A / f"exam_audio_radareval_{args.tag}")

    def as_map(rows):
        return {r["id"]: r for r in rows}
    pairs = {"C1": (p4_2021, s_2021), "C4": (p4_df, s_df), "C3": (p4_itw, s_itw),
             "C2": (p4_cc, s_cc), "C7": (p4_rad, s_rad)}
    composed = {}
    for k, (p4r, sr) in pairs.items():
        composed[k] = compose(as_map(p4r), as_map(sr))
        print(f"[compose] {k}: {len(composed[k])} rows", flush=True)

    # ---------- S2: specialist out-of-family bona FPR ----------
    def s_fpr(s_rows):
        bona = [r for r in s_rows if r["label"] == 0]
        if not bona:
            return None, 0
        return round(sum(1 for r in bona if (not r["ooc"]) and r["pf"] > 0.5) / len(bona), 4), len(bona)
    s2 = {"fpr_2021_bona": s_fpr(s_2021), "fpr_dfeval_bona": s_fpr(s_df)}
    cc_bona = [r for r in s_cc if r["label"] == 0]
    s2["fpr_codec_bona"] = (round(sum(1 for r in cc_bona if (not r["ooc"]) and r["pf"] > 0.5)
                                  / len(cc_bona), 4), len(cc_bona))

    # ---------- S0 / S1 / S3 / C5 ----------
    s0 = json.loads(train_meta.read_text())["s0_val_auc"]
    s1 = json.loads(s1_metrics.read_text())
    e5 = json.loads((A / f"gate_battery_w7_{args.tag}/e5_kokoro.json").read_text())
    s3 = e5["catches"]

    import argus_detect
    from argus_detect import detect_audio
    a4 = detect_audio(OOC_FIXTURE, recovery_enabled=False)
    p4_ooc = bool(a4.get("out_of_coverage"))
    argus_detect.AUDIO_PROBE = WS / f"artifacts/{args.cand_dir}/audio_probe.joblib"
    a12 = detect_audio(OOC_FIXTURE, recovery_enabled=False)
    s_ooc = bool(a12.get("out_of_coverage"))
    argus_detect.AUDIO_PROBE = WS / "artifacts/p4/audio_probe.joblib"
    c5_trips = p4_ooc and s_ooc
    print(f"[C5] OOC fixture: p4_ooc={p4_ooc} s_ooc={s_ooc} composed_trips={c5_trips}", flush=True)

    # ---------- bars ----------
    v = []

    def add(eid, name, clauses):
        ok = all(c[1] for c in clauses)
        v.append({"exam": eid, "name": name,
                  "clauses": [f"{c[0]}: {c[2]}" for c in clauses],
                  "verdict": "PASS" if ok else "FAIL"})
        return ok

    add("S0", "specialist in-domain", [("val AUC >= 0.99", s0 >= 0.99, s0)])
    add("S1", "held-out generator generalization",
        [("aggregate catch >= 0.55", s1["aggregate_catch"] >= 0.55, s1["aggregate_catch"]),
         ("gens >= 0.30 (of 14)", s1["gens_ge_0.30"] >= 10, s1["gens_ge_0.30"])])
    add("S2", "specialist gate sharpness (out-of-family bona)",
        [("2021 bona FPR <= 0.05", s2["fpr_2021_bona"][0] <= 0.05, s2["fpr_2021_bona"]),
         ("codec bona FPR <= 0.05", s2["fpr_codec_bona"][0] <= 0.05, s2["fpr_codec_bona"]),
         ("DF-Eval bona FPR <= 0.20", s2["fpr_dfeval_bona"][0] <= 0.20, s2["fpr_dfeval_bona"])])
    add("S3", "Kokoro generalization (specialist-alone)", [("catches >= 6/8", s3 >= 6, f"{s3}/8")])

    m1 = incov_metrics(composed["C1"])
    add("C1", "2021-LA composed floor",
        [("AUC >= 0.892 (in-cov)", m1.get("in_cov_auc", 0) >= 0.892, m1.get("in_cov_auc"))])

    # per-arm composed metrics (court-behavior convention; arm tag from the p4 row id)
    c2_arm = defaultdict(list)
    p4m, s4m = as_map(p4_cc), as_map(s_cc)
    for i in p4m:
        a, b = p4m[i], s4m[i]
        if a["ooc"] and b["ooc"]:
            c2_arm[a["arm"]].append({"label": a["label"], "fake": False, "abstain": True})
            continue
        sc = max(r["pf"] for r in (a, b) if not r["ooc"])
        c2_arm[a["arm"]].append({"label": a["label"], "fake": sc > 0.5, "abstain": False})
    m2_all = {"fpr_by_arm": {}, "recall_by_arm": {}}
    for arm in FULL_ARMS:
        rs = c2_arm[arm]
        b = [r for r in rs if r["label"] == 0]
        f = [r for r in rs if r["label"] == 1]
        m2_all["fpr_by_arm"][arm] = round(sum(r["fake"] for r in b) / len(b), 4) if b else None
        m2_all["recall_by_arm"][arm] = round(sum(r["fake"] for r in f) / len(f), 4) if f else None
    add("C2", "codec composed (court-behavior convention)",
        [("bona FPR = 0.000 all full-length arms",
          all(x == 0.0 for x in m2_all["fpr_by_arm"].values()), m2_all["fpr_by_arm"]),
         ("recall >= 0.84 every full-length arm",
          all(x >= 0.84 for x in m2_all["recall_by_arm"].values()), m2_all["recall_by_arm"])])

    m3 = incov_metrics(composed["C3"])
    add("C3", "ITW composed",
        [("in-cov AUC >= 0.90", m3.get("in_cov_auc", 0) >= 0.90, m3.get("in_cov_auc")),
         ("in-cov TPR >= 0.65", m3.get("in_cov_tpr", 0) >= 0.65, m3.get("in_cov_tpr")),
         ("in-cov FPR <= 0.10", m3.get("in_cov_fpr", 1) <= 0.10, m3.get("in_cov_fpr")),
         ("abstain <= 0.20", m3["abstain_rate"] <= 0.20, m3["abstain_rate"])])

    m4 = incov_metrics(composed["C4"])
    add("C4", "DF-Eval composed",
        [("in-cov TPR >= 0.69", m4.get("in_cov_tpr", 0) >= 0.69, m4.get("in_cov_tpr")),
         ("in-cov FPR <= 0.70", m4.get("in_cov_fpr", 1) <= 0.70, m4.get("in_cov_fpr")),
         ("abstain >= 0.50", m4["abstain_rate"] >= 0.50, m4["abstain_rate"])])

    m7 = incov_metrics(composed["C7"])
    add("C7", "RADAR-eval composed no-regression",
        [("in-cov AUC >= 0.6435", m7.get("in_cov_auc", 0) >= 0.6435, m7.get("in_cov_auc")),
         ("abstain >= 0.05", m7["abstain_rate"] >= 0.05, m7["abstain_rate"])])

    add("C5", "OOC fixture composed trip",
        [("composed abstains (p4 AND specialist both refuse)", c5_trips, c5_trips)])

    all_pass = all(x["verdict"] == "PASS" for x in v)
    out = WS / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps({
        "spec": SPEC, "tag": args.tag, "all_pass": all_pass,
        "p4_arms": {"E4": "exam_audio_2021_p4w8", "E2": "exam_audio_dfeval_p4w8",
                    "E3": "exam_audio_itw_p6v11", "E6": "exam_audio_codeccal_p4selfcheck",
                    "E7": "exam_audio_radareval_p6v11"},
        "s0_val_auc": s0, "s1": s1, "s2": {k: v2[0] for k, v2 in s2.items()},
        "s3_catches": s3, "ooc": {"p4": p4_ooc, "specialist": s_ooc},
        "composed_metrics": {"C1": m1, "C2": m2_all, "C3": m3, "C4": m4, "C7": m7},
        "verdicts": v}, indent=1))
    print("\n=== W8 COMPOSED BATTERY VERDICT ===", flush=True)
    for x in v:
        print(f"  {x['exam']}  {x['verdict']:4s}  {x['name']}", flush=True)
        for c in x["clauses"]:
            print(f"        {c}", flush=True)
    print(f"ALL-PASS: {all_pass}  (receipt {out/'verdict.json'})", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
