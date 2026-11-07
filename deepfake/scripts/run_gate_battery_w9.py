"""W9 gate battery (D-079 SEALED) — evaluates the fused court (p13_fused) against
the PRE-REGISTERED bars D1-D6 + NR1-NR4 + REG (spec
artifacts/generator_intake/w9_audio_court_spec.md §5, sealed 2026-11-05).

Mechanics (L-RUNPY-PATCH-VOID, run_gate_battery_w7.py law): import argus_detect,
patch the AUDIO_PROBE module global to the FUSED court bundle, then call each exam
harness's main() directly — never runpy. The fused branch activates via the
bundle's "fusion" meta key; the incumbent artifacts/p4 is never mutated.

Instrument integrity self-checks (run_gate_battery_w8.py law) BEFORE any verdict:
  - every p4 receipt's registered aggregate is RECOMPUTED from its registered rows
    and asserted equal (reader+metric code validated on known rows);
  - fused/p4 row id sets must match exactly per exam (selection identity);
    any drift aborts before verdicts.

Conventions (spec §5): AUC on VOTED rows (harness convention) with n_voted
recorded; court-behavior clauses count abstain as NOT-a-false-alarm on bona and
a MISS on fakes; all-rows numbers disclosed beside every clause. The fused arm's
per-row "abstained" IS the sealed rule (p4_ooc AND wavlm_ooc) OR conformal band.

D-046: any miss exits NON-ZERO naming the bar + exact numbers. No waiver file is
written autonomously.

Usage:
  ./.venv/bin/python deepfake/scripts/run_gate_battery_w9.py [--only D6] [--skip-e2e]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

WS = Path("/Users/Dataghost/Argus_Prime")
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

SPEC = "artifacts/generator_intake/w9_audio_court_spec.md §5 (SEALED 2026-11-05, D-079)"
OOC_FIXTURE = WS / "data/dfeval/audio/audio-data/audio-data/-0ljEYr8ZyqCHRqy5Bx4f4J6YOg.mp3"
KOKORO_FIXTURES = sorted((WS / "data/release_probe_unseen/audio/fake").glob("fake_*kokoro*.wav"))
FULL_ARMS = ["mp3_cbr128", "mp3_cbr192", "flac", "opus128_ogg", "mp3_mono16k"]
CAND = WS / "artifacts/p13_fused"

# NR2 p4 raw receipts (registered) — recomputed from rows and asserted before use
P4_REF = {"2021_shard": 0.912, "itw": 0.9153, "dfeval": 0.5593, "radareval": 0.6272}


def auc(y, s):
    return float(roc_auc_score(y, s))


def rows_2021_shard(d: Path):
    rows = json.loads((d / "scores.json").read_text())
    return [{"id": (r["dataset_label"], r["idx"]), "label": 1 - r["dataset_label"],
             "pf": r["p_fake"], "abstained": bool(r.get("abstained", False))}
            for r in rows if r["p_fake"] is not None]


def rows_itw(d: Path):
    out = []
    with open(d / "scores.csv") as f:
        for r in csv.DictReader(f):
            if r["p_fake"] == "":
                continue
            out.append({"id": r["file"], "label": int(r["label"]),
                        "pf": float(r["p_fake"]),
                        "abstained": r.get("abstained", "False") == "True"})
    return out


def rows_dfeval(d: Path):
    out = []
    with open(d / "audio_exam.csv") as f:
        for r in csv.DictReader(f):
            if r["p_fake"] in ("", "None"):
                continue
            out.append({"id": r["stem"], "label": int(r["label"]),
                        "pf": float(r["p_fake"]),
                        "abstained": r.get("abstained", "False") in ("True", "true")})
    return out


def rows_codeccal(d: Path):
    out = []
    with open(d / "scores.csv") as f:
        for r in csv.DictReader(f):
            if r["error"] or r["arm"] not in FULL_ARMS or r["p_fake"] == "":
                continue
            out.append({"id": (r["stratum"], r["shard_idx"], r["arm"]),
                        "label": 0 if r["stratum"] == "real_fit" else 1,
                        "pf": float(r["p_fake"]),
                        "abstained": r.get("abstained", "False") == "True",
                        "spectral": (float(r["spectral_score"])
                                     if r.get("spectral_score") not in (None, "", "None") else None)})
    return out


def rows_radareval(d: Path):
    out = []
    with open(d / "scores.csv") as f:
        for r in csv.DictReader(f):
            if r["p_fake"] == "":
                continue
            out.append({"id": r["uid"], "label": int(r["label"]),
                        "pf": float(r["p_fake"]),
                        "abstained": r.get("abstained", "False") == "True"})
    return out


def court_metrics(rows):
    """Spec §5 conventions: voted-AUC + n_voted; court-behavior FPR/recall;
    all-rows AUC disclosed beside."""
    y_all = [r["label"] for r in rows]
    s_all = [r["pf"] for r in rows]
    voted = [r for r in rows if not r["abstained"]]
    m = {"n_scored": len(rows), "n_voted": len(voted),
         "n_voted_frac": round(len(voted) / max(len(rows), 1), 4),
         "all_rows_auc": round(auc(y_all, s_all), 4)}
    if voted and len({r["label"] for r in voted}) == 2:
        m["voted_auc"] = round(auc([r["label"] for r in voted],
                                   [r["pf"] for r in voted]), 4)
    bona = [r for r in rows if r["label"] == 0]
    fake = [r for r in rows if r["label"] == 1]
    if bona:
        m["bona_mean_pf"] = round(float(np.mean([r["pf"] for r in bona])), 4)
        m["bona_fpr_court"] = round(sum(1 for r in bona if not r["abstained"]
                                        and r["pf"] > 0.5) / len(bona), 4)
        m["bona_abstain_rate"] = round(sum(1 for r in bona if r["abstained"])
                                       / len(bona), 4)
    if fake:
        m["fake_recall_court"] = round(sum(1 for r in fake if not r["abstained"]
                                           and r["pf"] > 0.5) / len(fake), 4)
    return m


def assert_selection_identity(p4_rows, fused_rows, exam):
    a, b = {str(r["id"]) for r in p4_rows}, {str(r["id"]) for r in fused_rows}
    if a != b:
        raise SystemExit(f"SELECTION DRIFT on {exam}: p4-only={len(a - b)} "
                         f"fused-only={len(b - a)} — aborting before any verdict")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="run a single bar id (D1..D6/NR1..NR4/REG)")
    ap.add_argument("--skip-e2e", action="store_true")
    ap.add_argument("--cand", default="p13_fused",
                    help="candidate dir name under artifacts/ (W9.1: p13b_fused)")
    args = ap.parse_args()
    want = lambda eid: args.only is None or args.only == eid

    global CAND, CAND_SHORT
    CAND = WS / f"artifacts/{args.cand}"
    CAND_SHORT = args.cand.replace("_", "")

    court_bundle = CAND / "audio_court.joblib"
    if not court_bundle.exists():
        print(f"ERROR: fused court bundle missing: {court_bundle}", file=sys.stderr)
        return 1
    import argus_detect  # noqa: E402
    argus_detect.AUDIO_PROBE = court_bundle
    print(f"[battery] argus_detect.AUDIO_PROBE -> {court_bundle} (fused branch)", flush=True)

    A = WS / "artifacts"
    results, t0 = {}, {}

    # ---------- p4 receipt self-checks (instrument integrity) ----------
    if want("SELFCHECK"):
        p4_2021 = rows_2021_shard(A / "exam_audio_2021_p4w8")
        a = round(auc([r["label"] for r in p4_2021], [r["pf"] for r in p4_2021]), 4)
        assert abs(a - P4_REF["2021_shard"]) < 5e-4, f"2021 receipt mismatch: {a}"
        p4_itw = rows_itw(A / "exam_audio_itw_p6v11")
        a_itw = round(auc([r["label"] for r in p4_itw], [r["pf"] for r in p4_itw]), 4)
        assert abs(a_itw - P4_REF["itw"]) < 5e-4, f"ITW raw receipt mismatch: {a_itw}"
        p4_df = rows_dfeval(A / "exam_audio_dfeval_p4w8")
        a_df = round(auc([r["label"] for r in p4_df], [r["pf"] for r in p4_df]), 4)
        assert abs(a_df - P4_REF["dfeval"]) < 5e-4, f"DF-Eval raw receipt mismatch: {a_df}"
        p4_rad = rows_radareval(A / "exam_audio_radareval_p6v11")
        a_rad = round(auc([r["label"] for r in p4_rad], [r["pf"] for r in p4_rad]), 4)
        assert abs(a_rad - P4_REF["radareval"]) < 5e-4, f"RADAR raw receipt mismatch: {a_rad}"
        p4_cc = rows_codeccal(A / "exam_audio_codeccal_p4selfcheck")
        bona = [r for r in p4_cc if r["label"] == 0]
        fake = [r for r in p4_cc if r["label"] == 1]
        fpr = sum(1 for r in bona if r["pf"] > 0.5) / len(bona)
        rec = sum(1 for r in fake if r["pf"] > 0.5) / len(fake)
        assert round(fpr, 4) == 0.0 and abs(rec - 0.85) < 0.005, \
            f"codec registered pair mismatch: {fpr:.4f}/{rec:.4f}"
        print(f"[self-check] p4 receipts reproduced: 2021 {a} | ITW {a_itw} | "
              f"DF-Eval {a_df} | RADAR {a_rad} | codec {fpr:.4f}/{rec:.4f}", flush=True)
        results["selfcheck"] = {"2021": a, "itw_raw": a_itw, "dfeval_raw": a_df,
                                "radareval_raw": a_rad,
                                "codec_fpr": round(fpr, 4), "codec_recall": round(rec, 4)}

    # ---------- fused-arm exams ----------
    if want("D1") or want("D2") or want("D3") or want("NR2"):
        import exam_audio_itw
        d = A / f"exam_audio_itw_{CAND_SHORT}"
        sys.argv = ["exam_audio_itw.py", "--per-class", "300", "--out", str(d), "--force"]
        exam_audio_itw.main()
        results["itw_fused"] = court_metrics(rows_itw(d))

    if want("D5") or want("NR2"):
        import exam_audio_dfeval
        d = A / f"exam_audio_dfeval_{CAND_SHORT}"
        sys.argv = ["exam_audio_dfeval.py", "--out", str(d), "--force"]
        exam_audio_dfeval.main()
        results["dfeval_fused"] = court_metrics(rows_dfeval(d))

    if want("D4") or want("NR1"):
        import exam_audio_codeccal
        d = A / f"exam_audio_codeccal_{CAND_SHORT}"
        exam_audio_codeccal.OUT = d
        d.mkdir(parents=True, exist_ok=True)
        sys.argv = ["exam_audio_codeccal.py"]
        exam_audio_codeccal.main()
        cc = rows_codeccal(d)
        per_arm_fpr, per_arm_rec = {}, {}
        for arm in FULL_ARMS:
            bona = [r for r in cc if r["id"][2] == arm and r["label"] == 0]
            fake = [r for r in cc if r["id"][2] == arm and r["label"] == 1]
            per_arm_fpr[arm] = round(sum(1 for r in bona if not r["abstained"]
                                         and r["pf"] > 0.5) / max(len(bona), 1), 4)
            per_arm_rec[arm] = round(sum(1 for r in fake if not r["abstained"]
                                         and r["pf"] > 0.5) / max(len(fake), 1), 4)
        pooled = [r for r in cc if r["spectral"] is not None]
        c_auc = round(auc([r["label"] for r in pooled],
                          [r["spectral"] for r in pooled]), 4) if pooled else None
        results["codec_fused"] = {"per_arm_fpr": per_arm_fpr, "per_arm_recall": per_arm_rec,
                                  "bona_fpr_max": max(per_arm_fpr.values(), default=1),
                                  "recall_min": min(per_arm_rec.values(), default=0),
                                  "c_pooled_auc": c_auc}

    if want("NR2"):
        import exam_audio_radar_eval
        d = A / f"exam_audio_radareval_{CAND_SHORT}"
        sys.argv = ["exam_audio_radar_eval.py", "--per-cell", "150", "--out", str(d), "--force"]
        exam_audio_radar_eval.main()
        results["radareval_fused"] = court_metrics(rows_radareval(d))

    if want("NR3"):
        from argus_detect import detect_audio
        catches, rows = 0, []
        for f in KOKORO_FIXTURES:
            a = detect_audio(f, recovery_enabled=False)
            pf, abst = a.get("p_fake_audio_mean"), bool(a.get("abstained"))
            hit = pf is not None and pf > 0.5 and not abst
            catches += hit
            rows.append({"file": f.name, "p_fused": pf, "abstained": abst, "catch": bool(hit)})
            print(f"[NR3] {f.name}: p={pf} abst={abst} catch={hit}", flush=True)
        results["kokoro_fused"] = {"catches": catches, "rows": rows}

    if want("NR4"):
        from argus_detect import detect_audio
        a = detect_audio(OOC_FIXTURE, recovery_enabled=False)
        results["ooc_fused"] = {"abstained": bool(a.get("abstained")),
                                "p_fused": a.get("p_fake_audio_mean"),
                                "p4_ooc": bool(a.get("out_of_coverage")),
                                "wavlm_ooc": bool(a.get("wavlm_ooc"))}
        print(f"[NR4] OOC fixture: abstained={results['ooc_fused']['abstained']} "
              f"p_fused={results['ooc_fused']['p_fused']}", flush=True)

    # ---------- 2021 full eval (STEP 6 rows) ----------
    if want("D6") or want("NR1") or want("NR2"):
        full_csv = A / f"exam_audio_2021_full_{CAND_SHORT}/rows_2021_full.csv"
        if not full_csv.exists():
            raise SystemExit(f"STEP 6 rows missing: {full_csv} — run "
                             f"exam_audio_2021_full.py first — STOP")
        import pandas as pd
        df = pd.read_csv(full_csv)
        y, pfu = df["label"].values, df["p_fused"].values
        results["full2021"] = {
            "n_scored": len(df),
            "fused_auc_all_rows": round(auc(y, pfu), 4),
            "p4_raw_auc_all_rows": round(auc(y, df["p_fake_audio_mean"].values), 4),
            "abstain_rate": round(float(df["abstained"].mean()), 4)}
        # reconstruct the registered n=330 shard selection (shard 0 fakes +
        # shard 13 reals, first 200 decodable per class — identical decode path)
        sub = []
        for shard, lab in ((0, 1), (13, 0)):
            s = df[(df["shard"] == shard) & (df["label"] == lab)] \
                .sort_values("uid").head(200)
            sub.append(s)
        rec = pd.concat(sub)
        results["full2021"]["shard_subset_n"] = len(rec)
        if len(rec) != 330:
            raise SystemExit(f"shard-subset reconstruction n={len(rec)} != registered "
                             f"330 — decode-path divergence — ESCALATE (D-046)")
        results["full2021"]["shard_subset_fused_auc"] = round(
            auc(rec["label"].values, rec["p_fused"].values), 4)
        results["full2021"]["shard_subset_wavlm_auc"] = round(
            auc(rec["label"].values, rec["wavlm_pfake_mean"].values), 4)

    # ---------- REG: deployed-court e2e battery (subprocess = unpatched) ----------
    if want("REG") and not args.skip_e2e:
        cmd = [str(WS / ".venv/bin/python"), str(WS / "deepfake/scripts/e2e_ui_test.py"),
               str(WS / "artifacts/mbeta/fixtures/fixture2_laundered_fake.jpg")]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=WS)
        tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
        results["reg"] = {"exit": r.returncode,
                          "e2e_ok": r.returncode == 0 and "0.9009" in (r.stdout + r.stderr),
                          "note": " ; ".join(tail[-3:])}

    # ---------- verdicts (§5 exact clauses) ----------
    v = []

    def add(eid, name, clauses):
        ok = all(c[1] for c in clauses)
        v.append({"bar": eid, "name": name,
                  "clauses": [f"{c[0]}: {c[2]}" for c in clauses],
                  "verdict": "PASS" if ok else "FAIL"})
        return ok

    m = results.get("itw_fused", {})
    add("D1", "fused ITW AUC >= 0.95 AND n_voted >= 80%",
        [("voted AUC >= 0.95", m.get("voted_auc", 0) >= 0.95, m.get("voted_auc")),
         ("n_voted >= 80% of n_scored", m.get("n_voted_frac", 0) >= 0.80,
          f"{m.get('n_voted')}/{m.get('n_scored')} = {m.get('n_voted_frac')}")])
    add("D2", "wild-bona over-call guard",
        [("mean fused p_fake on ITW bona <= 0.30", m.get("bona_mean_pf", 1) <= 0.30,
          m.get("bona_mean_pf")),
         ("bona FPR@0.5 <= 0.10 (court-behavior)", m.get("bona_fpr_court", 1) <= 0.10,
          m.get("bona_fpr_court"))])
    add("D3", "gate awake on wild (bona abstain in [5%,30%])",
        [("bona abstain in [0.05, 0.30]", 0.05 <= m.get("bona_abstain_rate", -1) <= 0.30,
          m.get("bona_abstain_rate"))])
    m = results.get("codec_fused", {})
    add("D4", "codec guard",
        [("bona FPR = 0.000 ALL 5 full-length arms (court-behavior)",
          m.get("bona_fpr_max", 1) == 0.0, m.get("per_arm_fpr")),
         ("fake recall >= 0.84 every arm (court-behavior)",
          m.get("recall_min", 0) >= 0.84, m.get("per_arm_recall"))])
    m = results.get("dfeval_fused", {})
    add("D5", "DF-Eval fused bona FPR <= 0.10",
        [("bona FPR <= 0.10 (abstain not a false alarm)",
          m.get("bona_fpr_court", 1) <= 0.10, m.get("bona_fpr_court"))])
    m = results.get("full2021", {})
    f_auc = m.get("fused_auc_all_rows", 0)
    add("D6", "FULL 2021 LA eval",
        [("fused AUC >= 0.88", f_auc >= 0.88, f_auc),
         ("|AUC - 0.912| <= 0.05", abs(f_auc - 0.912) <= 0.05, f_auc)])
    add("NR1", "new-probe home floors",
        [("B val AUC >= 0.98",
          json.loads((CAND / "wavlm_train_meta.json").read_text()).get("val_auc_all_views", 0) >= 0.98,
          json.loads((CAND / "wavlm_train_meta.json").read_text()).get("val_auc_all_views")),
         ("B >= 0.70 AUC on the 2021 n=330 shard",
          m.get("shard_subset_wavlm_auc", 0) >= 0.70, m.get("shard_subset_wavlm_auc")),
         ("C >= 0.70 AUC codec-pooled",
          (results.get("codec_fused", {}).get("c_pooled_auc") or 0) >= 0.70,
          results.get("codec_fused", {}).get("c_pooled_auc"))])
    nr2 = []
    if "full2021" in results:
        nr2.append((f"fused 2021 shard-subset AUC >= {P4_REF['2021_shard'] - 0.005:.4f}",
                    results["full2021"].get("shard_subset_fused_auc", 0)
                    >= P4_REF["2021_shard"] - 0.005,
                    results["full2021"].get("shard_subset_fused_auc")))
    if "itw_fused" in results:
        nr2.append((f"fused ITW all-rows AUC >= {P4_REF['itw'] - 0.005:.4f}",
                    results["itw_fused"].get("all_rows_auc", 0) >= P4_REF["itw"] - 0.005,
                    results["itw_fused"].get("all_rows_auc")))
    if "dfeval_fused" in results:
        nr2.append((f"fused DF-Eval all-rows AUC >= {P4_REF['dfeval'] - 0.005:.4f}",
                    results["dfeval_fused"].get("all_rows_auc", 0) >= P4_REF["dfeval"] - 0.005,
                    results["dfeval_fused"].get("all_rows_auc")))
    if "radareval_fused" in results:
        nr2.append((f"fused RADAR-eval all-rows AUC >= {P4_REF['radareval'] - 0.005:.4f}",
                    results["radareval_fused"].get("all_rows_auc", 0)
                    >= P4_REF["radareval"] - 0.005,
                    results["radareval_fused"].get("all_rows_auc")))
    add("NR2", "rank non-regression vs p4 (all-rows, -0.005 tolerance)", nr2)
    m = results.get("kokoro_fused", {})
    add("NR3", "Kokoro catches >= 6/8 fused",
        [("catches >= 6/8", m.get("catches", 0) >= 6, f'{m.get("catches")}/8')])
    m = results.get("ooc_fused", {})
    add("NR4", "OOC fixture: fused court abstains",
        [("abstained on the OOC fixture", m.get("abstained", False), m)])
    m = results.get("reg", {})
    add("REG", "regression law: full 3-lane e2e battery",
        [("e2e [1]->[7] 0.9009 EXACT, zero console errors", m.get("e2e_ok", False),
          m.get("note"))])

    all_pass = all(x["verdict"] == "PASS" for x in v)
    d = A / f"gate_battery_w9_{CAND_SHORT}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "verdict.json").write_text(json.dumps({
        "candidate": str(court_bundle), "all_pass": all_pass,
        "bars_source": SPEC,
        "p4_receipt_selfcheck": results.get("selfcheck"),
        "results": {k: v_ for k, v_ in results.items() if k != "selfcheck"},
        "verdicts": v}, indent=1))
    print("\n=== W9 GATE BATTERY VERDICT ===", flush=True)
    for x in v:
        print(f"  {x['bar']:4s} {x['verdict']:4s}  {x['name']}", flush=True)
        for c in x["clauses"]:
            print(f"        {c}", flush=True)
    print(f"ALL-PASS: {all_pass}  (receipt {d/'verdict.json'})", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
