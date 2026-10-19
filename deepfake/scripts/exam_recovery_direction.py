#!/usr/bin/env python3
"""Exam — recovery-direction parity (Phase-R slot 1).

Registered protocol: artifacts/generator_intake/recovery_parity_spec.md
Greenlight: user word 2026-10-19 ("proceed with the priority item next") —
recorded as the pre-registration greenlight; the run is READ-ONLY offline:
recovery.py stays byte-frozen (C-RECOV), no court code is touched, the
shipped ship-labels are not modified by this run.

Registry notes (established during harness bring-up 2026-10-19):
  * COURT REQUIRES exiftool on PATH — the house prefix is:
      PATH="$PWD/tools/bin:$PATH"
    Without it the metadata probe fails closed (exiftool_binary_missing,
    10 survivors) and every pf shifts — produced a 0.063 pf gap on a wild
    real vs the stored p6v11 exam row; WITH the prefix the same file
    reproduces the stored row EXACTLY (p_real 0.3624/trust 0.6347/conflict
    0.0606/11 survivors — "EXACT DETERMINISM: PASS").
  * Determinism guard semantics:
      wild rows      : stored CSV is the SAME deployed stacker p6v11 →
                       |drift| must be ≤ 5e-4 (asserted in the report stage).
      provider rows  : stored CSVs are the p6v10-era intake exams (Oct 15,
                       pre-promotion) → pf drift is EXPECTED (stacker
                       version delta); the row-level cross-check is the
                       recorded outcome (outcome_match vs recovery_outcome).
      recapture rows : specimens re-scored fresh; no stored pf.
  * The provider exams' CSVs already carry recovery_outcome (the ladder ran
    at intake time); this replay re-measures the ladder p6v11-current and
    adds the per-variant Δpf the intake CSVs never recorded.

What it does (mirrors the production seam EXACTLY):
  run_image(modules, estimator, pil, None|spec, art, vid, tmp, stacker_clfs,
            original_raw=raw, return_extras=True)
  recovery.run("image", original_dict, rich extras + 5 keys, as the
  argus_detect/server seam does today — R2 live via {outputs,d,survivors}).

Strata replayed (all pre-existing assets; no new intake, no training):
  wild        : data/corpora/wild_sample real_test rows scored in
                artifacts/exam_wild_real/scores.csv (stratified across the
                exam's file-order blocks; abstained + issued-real
                counterfactual arms).
  provider    : generator-slug exam finals under artifacts/exam_modern/*/
                (real-arm rows; wrong-issue concentrate per D-054).
  recapture   : artifacts/exam_launder2/eval_real_*/ regen + recapture views
                (T11/T9 stored specimens; the 9/20-flip stratum).
  fake-sanity : provider fake-arm rows (the ladder must never undo a
                correct issued-fake).

Bars (pre-registered, verbatim from the spec):
  G1: post-recovery issued real-FPR on the wild stratum ≤ pristine (0/701=0%)
      + 2.0 pp.
  G2: mean Δpf on abstained reals |mean| ≤ 0.10 in the fake direction.
  G3: clamp map intact — recovered trust ≤0.6 / risk ≥0.5, needs_review
      ≤0.2 / ≥0.9 (plus the image-court lower caps 0.4/0.6), mechanism-b
      (original dict never mutated by the ladder).

Any miss → escalate to the user (D-046). This script NEVER weighs/patches —
it measures and reports.
"""
import csv
import copy
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

OUT = WS / "artifacts" / "exam_recovery_direction"
OUT.mkdir(parents=True, exist_ok=True)
WILD_CSV = WS / "artifacts" / "exam_wild_real" / "scores.csv"
WILD_DIR = WS / "data" / "corpora" / "wild_sample" / "real_test"
LAUNDER = WS / "artifacts" / "exam_launder2"
TOL = 5e-4          # determinism guard: fresh recompute vs stored pf
SEED = 31
BOOT = 2000

N_WILD_ABSTAINED = 80
N_WILD_ISSUED = 40
N_PROVIDER = 40
N_RECAPTURE = 20
N_FAKESANITY = 20


def resolve_exams():
    """Provider-slug exam finals (p6v10 intake, D-054-era): per-slug CSVs under
    artifacts/generator_intake/exam_p6v10_20261015/, specimens in
    data/corpora/gen_exam/<slug>/{real,fake}."""
    root = WS / "artifacts" / "generator_intake" / "exam_p6v10_20261015"
    gen = WS / "data" / "corpora" / "gen_exam"
    out = []
    for scsv in sorted(root.glob("*/scores.csv")):
        slug = scsv.parent.name
        if slug in ("parity", "recall_wildaigc", "real_ext_reference"):
            continue
        gdir = gen / slug
        if not gdir.exists():
            continue
        out.append({"slug": slug, "scores": scsv,
                    "real_dir": gdir / "real", "fake_dir": gdir / "fake"})
    return out


def find_image(name_or_sha, dirs):
    for d in dirs:
        if d is None:
            continue
        p = Path(d) / name_or_sha
        if p.exists():
            return p
        for sub in ("real", "real_test", "fake", "fake_test", "images", ""):
            p2 = Path(d) / sub / name_or_sha
            if p2.exists():
                return p2
    return None


def court_init():
    """Build the production court context ONCE (identical to argus_detect /
    run_benchmark seams: modules derived from the deployed stacker's meta)."""
    import argus_detect as A
    from triage import DegradationEstimator
    modules, mids = A.build_modules()
    estimator = DegradationEstimator()
    stacker_clfs = A._image_stacker_clfs() if hasattr(A, "_image_stacker_clfs") else {}
    return A, modules, mids, estimator, stacker_clfs


def replay_row(A, modules, mids, estimator, stacker_clfs, path, stored_pf, state):
    """Exact production seam (argus_detect.detect_image, context hoisted):
    original court pass -> recovery.run with rich extras {outputs,d,survivors,
    core,judge,k} + {pil,modules,estimator,stacker_clfs,original_raw}.
    Records R1 variant deltas, ladder outcome, clamps, mechanism-b."""
    import tempfile, shutil
    from run_benchmark import run_image
    import recovery as Rmod
    from PIL import Image
    import io as _io

    raw = Path(path).read_bytes()
    pil = Image.open(_io.BytesIO(raw))
    pil.load()
    tmp = Path(tempfile.mkdtemp(prefix="rec_ex_tmp_"))
    art = Path(tempfile.mkdtemp(prefix="rec_ex_art_"))
    vid = ("r_" + state) + path.stem[:8]
    try:
        row, rich = run_image(modules, estimator, pil, None, art, vid, tmp,
                              stacker_clfs, original_raw=raw, return_extras=True)
        pf_fresh = None if row.get("p_real") is None else round(1.0 - row["p_real"], 4)
        drift = None if (pf_fresh is None or stored_pf is None) else round(abs(pf_fresh - stored_pf), 6)
        module_scores = {m: row.get(m) for m in mids if row.get(m) is not None}
        original = {
            "p_fake_fused": pf_fresh,
            "abstained": bool(row.get("abstained")),
            "trust": row.get("trust"),
            "risk": row.get("risk"),
            "out_of_conflict": bool(row.get("out_of_conflict", False)),
            "module_evidence": {k: round(-v, 4) for k, v in module_scores.items()},
        }
        extras = dict(rich)
        extras.update({"pil": pil, "modules": modules, "estimator": estimator,
                       "stacker_clfs": stacker_clfs or {}, "original_raw": raw})
        orig_snapshot = copy.deepcopy(original)
        t0 = time.time()
        rec = Rmod.run("image", original, extras)
        wall = round(time.time() - t0, 2)
        mech_ok = original == orig_snapshot  # mechanism-b audit
        steps = []
        for s in rec.get("attempted_steps", []):
            d = s.get("detail") or {}
            if s.get("step") == "R1":
                pf = d.get("p_fake_fused")
                steps.append({"step": s["step"], "name": s.get("name"),
                              "ok": s.get("ok"),
                              "p_recovered_variant": pf,
                              "abstained": d.get("abstained"),
                              "out_of_conflict": d.get("out_of_conflict"),
                              "delta": (None if pf is None or pf_fresh is None
                                        else round(pf - pf_fresh, 4))})
        verdict = rec.get("verdict") or {}
        # issued rows short-circuit (trigger none -> not_attempted, no steps);
        # the spec asks for the counterfactual: force an R1 variant replay and
        # record what each variant WOULD have produced.
        if rec.get("trigger") in (None, "none") or rec.get("outcome") == "not_attempted":
            steps2 = []
            _ = Rmod._r1_rescore(original, extras, steps2)
            forced = [{"name": s.get("name"),
                       "pf": (s.get("detail") or {}).get("p_fake_fused"),
                       "abstained": (s.get("detail") or {}).get("abstained"),
                       "ooc": (s.get("detail") or {}).get("out_of_conflict"),
                       "ok": s.get("ok")} for s in steps2]
            mech_ok = mech_ok and (original == orig_snapshot)  # mech-b incl. forced pass
        else:
            forced = None
        out = {
            "row_state": state, "image": str(path), "stored_pf": stored_pf,
            "pf_court": pf_fresh, "drift": drift,
            "original_abstained": original["abstained"],
            "original_trust": original["trust"], "original_risk": original["risk"],
            "ladder_trigger": rec.get("trigger"), "outcome": rec.get("outcome"),
            "pf_final": verdict.get("p_fake_fused"),
            "final_trust": verdict.get("trust", rec.get("trust")),
            "final_risk": verdict.get("risk", rec.get("risk")),
            "reasons": json.dumps(rec.get("reasons", [])),
            "wall_s": wall, "mechanism_b_ok": mech_ok,
            "variants": json.dumps(steps),
            "forced_r1": None if forced is None else json.dumps(forced),
        }
        # clamp audit (G3)
        outc = rec.get("outcome")
        if outc == "recovered":
            out["clamp_ok"] = ((verdict.get("trust") is None or verdict["trust"] <= Rmod.RECOVERED_TRUST_CAP + 1e-9)
                               and (verdict.get("risk") is None or verdict["risk"] >= Rmod.RECOVERED_RISK_FLOOR - 1e-9))
        elif outc == "needs_review":
            out["clamp_ok"] = ((verdict.get("trust") is None or verdict["trust"] <= Rmod.NEEDS_REVIEW_TRUST_CAP + 1e-9)
                               and (verdict.get("risk") is None or verdict["risk"] >= Rmod.NEEDS_REVIEW_RISK_FLOOR - 1e-9))
        else:
            out["clamp_ok"] = None
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(art, ignore_errors=True)


def pick_wild(rows, state, n, rng):
    """Stratified across the exam's file-order blocks (the report warned the
    corpus is video-adjacent blocks with declining abstention pattern)."""
    target = "True" if state == "abstained" else "False"
    cand = [i for i, r in enumerate(rows)
            if r["abstained_p6v11"] == target and r.get("label") == "0"]
    if len(cand) <= n:
        return [rows[i] for i in cand]
    blocks = 4
    per = n / blocks
    picked = []
    for b in range(blocks):
        seg = cand[b * len(cand) // blocks:(b + 1) * len(cand) // blocks]
        k = round(per) if b < blocks - 1 else (n - len(picked))
        k = min(k, len(seg))
        picked += [rows[i] for i in rng.sample(seg, k)]
    rng.shuffle(picked)
    return picked[:n]


def main():
    smoke = "--smoke" in sys.argv
    rng = random.Random(SEED)
    A, modules, mids, estimator, stacker_clfs = court_init()
    print(f"[init] modules={len(modules)} stacker={bool(stacker_clfs)}", flush=True)

    wild_rows = list(csv.DictReader(open(WILD_CSV)))
    exams = resolve_exams()
    print(f"[census] wild rows={len(wild_rows)} provider exams={[e['slug'] for e in exams]}", flush=True)

    # ---------------- wild stratum ----------------
    def stored(row):
        return None if row.get("p_real_p6v11") in ("", None) else float(row["p_real_p6v11"])

    plan = []
    for r in pick_wild(wild_rows, "abstained", N_WILD_ABSTAINED, rng):
        plan.append({"state": "wild_abstained", "csv": r})
    for r in pick_wild(wild_rows, "issued", N_WILD_ISSUED, rng):
        plan.append({"state": "wild_issued", "csv": r})

    # ---------------- provider stratum ----------------
    prov_used = 0
    per_exam = max(2, N_PROVIDER // max(1, len(exams)))
    for e in exams:
        if prov_used >= N_PROVIDER:
            break
        prows = [r for r in csv.DictReader(open(e["scores"])) if r.get("label") == "0"]
        take_ab = [r for r in prows if r.get("abstained") == "True"]
        take_is = [r for r in prows if r.get("abstained") == "False"]
        take = (take_ab + take_is)[:per_exam]
        for r in rng.sample(take, min(len(take), per_exam)):
            if prov_used >= N_PROVIDER:
                break
            plan.append({"state": f"provider_{e['slug']}", "csv": r,
                         "real_dir": e["real_dir"], "fake_dir": e["fake_dir"],
                         "slug": e["slug"], "csv_outcome": r.get("recovery_outcome")})
            prov_used += 1

    # ---------------- recapture stratum ----------------
    regen_dirs = sorted(glob.glob(str(LAUNDER / "eval_real_*")))
    rec_taken = 0
    for d in regen_dirs:
        if rec_taken >= N_RECAPTURE:
            break
        for view in ("regen_0.5.png", "regen_0.7.png", "recapture.jpg"):
            p = Path(d) / view
            if p.exists() and rec_taken < N_RECAPTURE:
                plan.append({"state": "recapture_sim", "path": p, "stored_pf": None, "slug": "launder2"})
                rec_taken += 1

    # ---------------- fake sanity ----------------
    # Prefer ABSTAINED fakes (the only rows that trigger the ladder — the
    # ladder-undo risk lives there); issued fakes are no-op controls.
    fk_used = 0
    for e in exams:
        if fk_used >= N_FAKESANITY:
            break
        frows = [r for r in csv.DictReader(open(e["scores"])) if r.get("label") == "1"]
        take_abf = [r for r in frows if r.get("abstained") == "True"]
        take_isf = [r for r in frows if r.get("abstained") == "False"]
        take = (take_abf + take_isf)[:10]
        for r in rng.sample(take, min(10, len(take))):
            if fk_used >= N_FAKESANITY:
                break
            plan.append({"state": f"fake_sanity_{e['slug']}", "csv": r,
                         "real_dir": e["real_dir"], "fake_dir": e["fake_dir"],
                         "slug": e["slug"], "csv_outcome": r.get("recovery_outcome")})
            fk_used += 1

    print(f"[plan] rows to replay: {len(plan)} "
          f"(wild_ab={sum(1 for p in plan if p['state']=='wild_abstained')}, "
          f"wild_is={sum(1 for p in plan if p['state']=='wild_issued')}, "
          f"provider={sum(1 for p in plan if p['state'].startswith('provider_'))}, "
          f"recapture={sum(1 for p in plan if p['state']=='recapture_sim')}, "
          f"fakes={sum(1 for p in plan if p['state'].startswith('fake_sanity_'))})", flush=True)

    if smoke:
        # one row per stratum class
        seen, trim = set(), []
        for it in plan:
            key = (it["state"].split("_")[0])
            if key not in seen:
                seen.add(key)
                trim.append(it)
        plan = trim[:6]
        print(f"[smoke] trimmed plan to {len(plan)}", flush=True)

    results = []
    t0 = time.time()
    for i, item in enumerate(plan, 1):
        st = item["state"]
        try:
            if "path" in item:
                path, stored_pf = item["path"], item.get("stored_pf")
            else:
                r = item["csv"]
                name = r.get("image") or r.get("file") or r.get("filename")
                if "real_dir" in item:
                    dirs = [item.get("real_dir"), item.get("fake_dir")]
                    # p6v10 CSV semantics: p_fake_fused is the post-recovery
                    # verdict for recovered rows, the original verdict
                    # otherwise (C-MECHB: top level never overwritten).
                    stored_pf = None if r.get("p_fake_fused") in (None, "") else float(r["p_fake_fused"])
                else:
                    dirs = [WILD_DIR]
                    pr = None if r.get("p_real_p6v11") in (None, "") else float(r["p_real_p6v11"])
                    stored_pf = None if pr is None else round(1.0 - pr, 4)  # pf = 1 − p_real
                p = find_image(name, dirs)
                if p is None:
                    results.append({"row_state": st, "image": name, "error": "path_not_resolved"})
                    continue
                path = p
            out = replay_row(A, modules, mids, estimator, stacker_clfs, path,
                             stored_pf, st)
            if "slug" in item:
                out["slug"] = item["slug"]
            if "csv_outcome" in item:
                out["csv_outcome"] = item["csv_outcome"]
                out["outcome_match"] = (out.get("outcome") == item["csv_outcome"])
            if st.startswith("wild_"):
                out["stack_note"] = "stored p6v11 == deployed (drift must be ~0)"
            elif st.startswith("provider_"):
                out["stack_note"] = "stored p6v10-era intake exam — pf drift expected"
            if "slug" in item:
                out["slug"] = item["slug"]
            results.append(out)
        except Exception as ex:
            results.append({"row_state": st, "image": str(item.get("path") or item.get("csv", {}).get("image")),
                            "error": f"{type(ex).__name__}: {ex}"[:220]})
        if i % 5 == 0 or i == len(plan):
            el = time.time() - t0
            print(f"[{i}/{len(plan)}] {el:.0f}s wall — last={st}", flush=True)

    (OUT / "rows.json").write_text(json.dumps(results, indent=1))
    print(f"[done] {len(results)} rows -> {OUT}/rows.json", flush=True)


if __name__ == "__main__":
    main()
