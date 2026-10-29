"""W7 gate battery runner — scores a candidate audio probe against the PRE-REGISTERED
bars E1-E8 (spec artifacts/generator_intake/w7_audio_retrain_spec.md, ruling 2026-10-29).

Mechanics (L-RUNPY-PATCH-VOID): import argus_detect, patch the AUDIO_PROBE module global
to the candidate bundle, then call each exam harness's main() directly — never runpy.
The incumbent p4 bundle is never mutated; all candidate receipts land in
artifacts/exam_*_<tag>/ and artifacts/gate_battery_w7_<tag>/verdict.json.

Bars (BINDING, user-approved 2026-10-29 — any miss escalates per D-046):
  E1 in-domain (W7 split)            val AUC >= 0.995                  (train_meta.json)
  E2 DF-Eval-2024 posture            abstain >= 0.70; in-cov TPR >= 0.55; in-cov FPR <= 0.65
  E3 ITW                             in-cov AUC >= 0.90; TPR >= 0.65; FPR <= 0.15; abstain <= 0.20
  E4 2021-LA floor                   AUC >= 0.892
  E5 Kokoro catch (8 fixtures)       >= 6/8
  E6 codec stratum                   bona FPR = 0.000 every full-length arm; recall >= 0.84 per arm
  E7 RADAR-eval multilingual         in-cov AUC >= 0.72 overall AND >= 0.60 every language;
                                     abstention >= 0.05; in-cov n >= 60%; raw AUC >= 0.6272
  E8 OOC fixture + e2e battery       candidate must trip out_of_coverage on the wild fixture;
                                     e2e [1]->[7] = 0.9009 EXACT, zero console errors

Usage:
  ./.venv/bin/python deepfake/scripts/run_gate_battery_w7.py \
      --probe artifacts/p11_audio/audio_probe.joblib --tag p11audio [--only E7] [--skip-e2e]
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

RAW_AUC_BASELINE_E7 = 0.6272  # p4 receipt artifacts/exam_audio_radareval_p6v11/
KOKORO_FIXTURES = sorted((WS / "data/release_probe_unseen/audio/fake").glob("fake_*kokoro*.wav"))
OOC_FIXTURE = (WS / "data/dfeval/audio/audio-data/audio-data/-0ljEYr8ZyqCHRqy5Bx4f4J6YOg.mp3")
FULL_LENGTH_ARMS = ["mp3_cbr128", "mp3_cbr192", "flac", "opus128_ogg", "mp3_mono16k"]


def load_metrics(p: Path):
    return json.loads(p.read_text())


def evaluate(tag: str, probe: Path, results: dict) -> tuple[list[dict], bool]:
    """Evaluate collected results against the binding bars. Returns (verdicts, all_pass)."""
    v = []

    def add(eid, name, clauses):
        ok = all(c[1] for c in clauses)
        v.append({"exam": eid, "name": name,
                  "clauses": [f"{c[0]}: {c[2]}" for c in clauses],
                  "verdict": "PASS" if ok else "FAIL"})
        return ok

    m = results.get("E1", {})
    add("E1", "in-domain (W7 split)",
        [("val AUC >= 0.995", m.get("e1_val_auc", 0) >= 0.995, m.get("e1_val_auc"))])

    m = results.get("E2", {})
    add("E2", "DF-Eval-2024 posture",
        [("abstain >= 0.70", m.get("abstained_rate", 0) >= 0.70, m.get("abstained_rate")),
         ("in-cov TPR >= 0.55", m.get("in_coverage_tpr_at_0.5", 0) >= 0.55, m.get("in_coverage_tpr_at_0.5")),
         ("in-cov FPR <= 0.65", m.get("in_coverage_fpr_at_0.5", 1) <= 0.65, m.get("in_coverage_fpr_at_0.5"))])

    m = results.get("E3", {})
    add("E3", "ITW non-regression floor",
        [("in-cov AUC >= 0.90", m.get("in_coverage_auc", 0) >= 0.90, m.get("in_coverage_auc")),
         ("TPR >= 0.65", m.get("in_coverage_tpr_at_0.5", 0) >= 0.65, m.get("in_coverage_tpr_at_0.5")),
         ("FPR <= 0.15", m.get("in_coverage_fpr_at_0.5", 1) <= 0.15, m.get("in_coverage_fpr_at_0.5")),
         ("abstain <= 0.20", m.get("abstained_rate", 1) <= 0.20, m.get("abstained_rate"))])

    m = results.get("E4", {})
    add("E4", "2021-LA regression floor",
        [("AUC >= 0.892", m.get("auc", 0) >= 0.892, m.get("auc"))])

    m = results.get("E5", {})
    add("E5", "Kokoro catch",
        [("catches >= 6/8", m.get("catches", 0) >= 6, f'{m.get("catches")}/8')])

    m = results.get("E6", {})
    add("E6", "codec stratum",
        [("bona FPR = 0.000 all full-length arms", m.get("bona_fpr_max", 1) == 0.0, m.get("bona_fpr_by_arm")),
         ("recall >= 0.84 every full-length arm", m.get("recall_min", 0) >= 0.84, m.get("recall_by_arm"))])

    m = results.get("E7", {})
    per_lang = m.get("per_language", {})
    lang_aucs = {k: d.get("in_coverage_auc") for k, d in per_lang.items()}
    lang_ok = all(a is not None and a >= 0.60 for a in lang_aucs.values()) and len(lang_aucs) == 6
    incov_n_frac = (m.get("n_scored", 0) / m.get("n_total", 1))
    add("E7", "RADAR-eval multilingual",
        [("in-cov AUC >= 0.72", m.get("in_coverage_auc", 0) >= 0.72, m.get("in_coverage_auc")),
         ("every language >= 0.60", lang_ok, lang_aucs),
         ("abstention >= 0.05", m.get("abstained_rate", 0) >= 0.05, m.get("abstained_rate")),
         ("in-cov n >= 60%", incov_n_frac >= 0.60, round(incov_n_frac, 3)),
         (f"raw AUC >= {RAW_AUC_BASELINE_E7}", m.get("raw_auc_no_gate", 0) >= RAW_AUC_BASELINE_E7,
          m.get("raw_auc_no_gate"))])

    m = results.get("E8", {})
    add("E8", "OOC fixture + e2e battery",
        [("OOC fixture trips out_of_coverage", m.get("ooc_tripped", False), m.get("ooc_tripped")),
         ("e2e battery 0.9009 EXACT + zero errors", m.get("e2e_ok", False), m.get("e2e_note"))])

    all_pass = all(x["verdict"] == "PASS" for x in v)
    return v, all_pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="artifacts/p11_audio/audio_probe.joblib")
    ap.add_argument("--tag", default="p11audio")
    ap.add_argument("--only", default=None, help="run a single exam id (E1..E8)")
    ap.add_argument("--train-meta", default="artifacts/p11_audio/train_meta.json")
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()
    probe = WS / args.probe if not args.probe.startswith("/") else Path(args.probe)
    if not probe.exists():
        print(f"ERROR: probe bundle missing: {probe}", file=sys.stderr)
        return 1

    import argus_detect  # noqa: E402
    argus_detect.AUDIO_PROBE = probe
    print(f"[battery] argus_detect.AUDIO_PROBE -> {probe}", flush=True)

    out_root = WS / "artifacts"
    results, t0 = {}, time.time()

    def want(eid):
        return args.only is None or args.only == eid

    # ---- E1: read from the training receipt ----
    if want("E1"):
        tm = load_metrics(WS / args.train_meta)
        auc = tm.get("e1_val_auc", tm.get("val_auc"))
        results["E1"] = {"e1_val_auc": auc}
        print(f"[E1] val AUC {auc}", flush=True)

    # ---- E4: 2021-LA floor ----
    if want("E4"):
        import exam_audio_2021
        d = out_root / f"exam_audio_2021_{args.tag}"
        sys.argv = ["exam_audio_2021.py", "--n-per-class", "200", "--out", str(d), "--force"]
        exam_audio_2021.main()
        results["E4"] = load_metrics(d / "metrics.json")

    # ---- E2: DF-Eval posture ----
    if want("E2"):
        import exam_audio_dfeval
        d = out_root / f"exam_audio_dfeval_{args.tag}"
        sys.argv = ["exam_audio_dfeval.py", "--out", str(d), "--force"]
        exam_audio_dfeval.main()
        results["E2"] = load_metrics(d / "audio_exam_summary.json")

    # ---- E3: ITW ----
    if want("E3"):
        import exam_audio_itw
        d = out_root / f"exam_audio_itw_{args.tag}"
        sys.argv = ["exam_audio_itw.py", "--per-class", "300", "--out", str(d), "--force"]
        exam_audio_itw.main()
        results["E3"] = load_metrics(d / "metrics.json")

    # ---- E5: Kokoro catch (8 fixtures, inline) ----
    if want("E5"):
        from argus_detect import detect_audio
        catches, rows = 0, []
        for f in KOKORO_FIXTURES:
            a = detect_audio(f)
            pf = a.get("p_fake_audio_mean")
            hit = pf is not None and pf > 0.5
            catches += hit
            rows.append({"file": f.name, "p_fake": pf, "abstained": a.get("abstained"),
                         "ooc": a.get("out_of_coverage"), "catch": bool(hit)})
            print(f"[E5] {f.name}: p={pf} catch={hit}", flush=True)
        d = out_root / f"gate_battery_w7_{args.tag}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "e5_kokoro.json").write_text(json.dumps({"catches": catches, "rows": rows}, indent=1))
        results["E5"] = {"catches": catches}

    # ---- E6: codec stratum ----
    if want("E6"):
        import exam_audio_codeccal
        d = out_root / f"exam_audio_codeccal_{args.tag}"
        exam_audio_codeccal.OUT = d
        d.mkdir(parents=True, exist_ok=True)
        exam_audio_codeccal.main()
        bona_fpr, recall = {}, {}
        with open(d / "scores.csv") as f:
            for r in csv.DictReader(f):
                if r["error"] or r["arm"] not in FULL_LENGTH_ARMS:
                    continue
                pf = float(r["p_fake"]) if r["p_fake"] != "" else None
                if pf is None:
                    continue
                if r["stratum"] == "real_fit":
                    bona_fpr.setdefault(r["arm"], []).append(pf > 0.5)
                elif r["stratum"] == "fake_asv19":
                    recall.setdefault(r["arm"], []).append(pf > 0.5)
        bona_fpr_by_arm = {a: round(sum(v) / len(v), 4) for a, v in bona_fpr.items()}
        recall_by_arm = {a: round(sum(v) / len(v), 4) for a, v in recall.items()}
        results["E6"] = {"bona_fpr_by_arm": bona_fpr_by_arm, "recall_by_arm": recall_by_arm,
                         "bona_fpr_max": max(bona_fpr_by_arm.values(), default=1),
                         "recall_min": min(recall_by_arm.values(), default=0)}

    # ---- E7: RADAR-eval multilingual ----
    if want("E7"):
        import exam_audio_radar_eval
        d = out_root / f"exam_audio_radareval_{args.tag}"
        sys.argv = ["exam_audio_radar_eval.py", "--per-cell", "150", "--out", str(d), "--force"]
        exam_audio_radar_eval.main()
        results["E7"] = load_metrics(d / "metrics.json")

    # ---- E8: OOC fixture (candidate) + e2e battery (deployed court) ----
    if want("E8"):
        from argus_detect import detect_audio
        a = detect_audio(OOC_FIXTURE, recovery_enabled=False)
        ooc = bool(a.get("out_of_coverage"))
        print(f"[E8] OOC fixture: out_of_coverage={ooc} z={a.get('znorm_mean')}", flush=True)
        e2e_ok, e2e_note = False, "skipped"
        if not args.skip_e2e:
            cmd = [str(WS / ".venv/bin/python"), str(WS / "deepfake/scripts/e2e_ui_test.py"),
                   str(WS / "artifacts/mbeta/fixtures/fixture2_laundered_fake.jpg")]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800, cwd=WS)
            tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
            e2e_note = f"exit={r.returncode} | " + " ; ".join(tail[-3:])
            e2e_ok = r.returncode == 0 and "0.9009" in (r.stdout + r.stderr)
        results["E8"] = {"ooc_tripped": ooc, "e2e_ok": e2e_ok, "e2e_note": e2e_note}

    # ---- verdict ----
    verdicts, all_pass = evaluate(args.tag, probe, results)
    d = out_root / f"gate_battery_w7_{args.tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "verdict.json").write_text(json.dumps({
        "probe": str(probe), "tag": args.tag, "all_pass": all_pass,
        "runtime_min": round((time.time() - t0) / 60, 1),
        "bars_source": "artifacts/generator_intake/w7_audio_retrain_spec.md §4 (PRE-REGISTERED 2026-10-29)",
        "results": results, "verdicts": verdicts}, indent=1))
    print("\n=== GATE BATTERY VERDICT ===", flush=True)
    for x in verdicts:
        print(f"  {x['exam']}  {x['verdict']:4s}  {x['name']}", flush=True)
        for c in x["clauses"]:
            print(f"        {c}", flush=True)
    print(f"ALL-PASS: {all_pass}  (receipt {d/'verdict.json'})", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
