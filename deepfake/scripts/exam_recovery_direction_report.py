#!/usr/bin/env python3
"""Report stage — recovery-direction parity exam (Phase-R slot 1).

Re-derives every number from artifacts/exam_recovery_direction/rows.json
(produced by exam_recovery_direction.py) plus the pre-existing verdicts.db
ledger rows (exact mined evidence). Emits:
  artifacts/exam_recovery_direction/metrics.json
  artifacts/exam_recovery_direction/EXAM_REPORT.md   (with decision page)
Registry conventions: PATH="$PWD/tools/bin:$PATH" for the replay stage.

Bars (pre-registered in artifacts/generator_intake/recovery_parity_spec.md):
  G1: post-recovery issued real-FPR on the wild stratum ≤ pristine (0/701)
      + 2.0 pp, i.e. ≤ 0.02.
  G2: mean Δpf on wild abstained reals |mean| ≤ 0.10, checked CWISE — a
      fake-ward mean fails the bar.
  G3: clamp map intact (recovered ≤0.6/≥0.5; needs_review ≤0.2/≥0.9) and
      mechanism-b (original dict never mutated) on EVERY row — zero
      violations required.
"""
import csv
import json
import math
import random
import sqlite3
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
OUT = WS / "artifacts" / "exam_recovery_direction"
BOOT = 2000
SEED = 47

G1_BAR = 0.02
G2_BAR = 0.10
PRISTINE_ISSUED = (0, 701)  # pristine issued-real FPR numerator/denominator

WILD_ABSTAINED_N = 1299      # census: wild reals abstained (label 0, abstained_p6v11 True)
WILD_ISSUED_N = 701
WILD_TOTAL_N = 2000


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, c - h, c + h)


def boot_mean_ci(vals, seed=SEED):
    if not vals:
        return (None, None)
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(BOOT):
        means.append(sum(vals[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return (means[int(0.025 * BOOT)], means[int(0.975 * BOOT) - 1])


def main():
    rows = json.loads((OUT / "rows.json").read_text())
    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]
    strata = {}
    for r in ok:
        st = r["row_state"]
        if st.startswith("wild_abstained"):
            k = "wild_abstained"
        elif st.startswith("wild_issued"):
            k = "wild_issued"
        elif st == "recapture_sim":
            k = "recapture_sim"
        elif st.startswith("fake_sanity"):
            k = "fake_sanity"
        else:
            k = "provider_real"
        strata.setdefault(k, []).append(r)

    # ---------- determinism / crosscheck gates ----------
    wild_all = strata.get("wild_abstained", []) + strata.get("wild_issued", [])
    drift_bad = [r for r in wild_all
                 if r.get("drift") is not None and r["drift"] > 5e-4]
    prov = strata.get("provider_real", [])
    prov_match = [r for r in prov if r.get("outcome_match") is not None]
    prov_mismatch = [r for r in prov_match if not r["outcome_match"]]

    # ---------- per-stratum deltas & outcomes ----------
    def delta_row(r):
        """Verdict-level Δpf: final ladder pf vs court pf."""
        if r.get("pf_final") is None or r.get("pf_court") is None:
            return None
        return round(r["pf_final"] - r["pf_court"], 4)

    def q95_delta(r):
        vs = json.loads(r.get("variants") or "[]")
        d = [v["delta"] for v in vs if v["name"] == "jpeg_q95_rescore" and v["delta"] is not None]
        return d[0] if d else None

    stats = {}
    for k, rs in strata.items():
        deltas = [delta_row(r) for r in rs]
        deltas = [d for d in deltas if d is not None]
        q95s = [q95_delta(r) for r in rs]
        q95s = [d for d in q95s if d is not None]
        outs = {}
        for r in rs:
            outs[r["outcome"]] = outs.get(r["outcome"], 0) + 1
        fake_ward = [d for d in deltas if d > 0]
        stats[k] = {
            "n": len(rs),
            "outcomes": outs,
            "mean_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
            "delta_ci95": boot_mean_ci(deltas),
            "mean_q95_delta_when_ran": round(sum(q95s) / len(q95s), 4) if q95s else None,
            "fake_ward_frac": round(len(fake_ward) / len(deltas), 4) if deltas else None,
            "recovered_to_real": sum(1 for r in rs if r["outcome"] == "recovered"
                                     and r["pf_final"] < 0.5),
            "clamp_violations": sum(1 for r in rs if r.get("clamp_ok") is False),
            "mechb_violations": sum(1 for r in rs if r.get("mechanism_b_ok") is False),
            "per_row": [{"image": Path(r["image"]).name, "pf_court": r["pf_court"],
                         "pf_final": r["pf_final"], "outcome": r["outcome"],
                         "delta": delta_row(r), "q95_delta": q95_delta(r)} for r in rs],
        }

    # ---------- outcome decomposition + provider cross-tab ----------
    for k, rs in strata.items():
        s = stats[k]
        s["recovered_committed_fake"] = sum(1 for r in rs if r["outcome"] == "recovered" and (r["pf_final"] or 0) >= 0.5)
        s["recovered_committed_real"] = sum(1 for r in rs if r["outcome"] == "recovered" and (r["pf_final"] or 1) < 0.5)
        s.pop("recovered_to_real", None)
    xtab = {}
    for r in prov:
        key = f'{r.get("csv_outcome")}→{r["outcome"]}'
        xtab[key] = xtab.get(key, 0) + 1

    # ---------- G1: post-recovery issued real-FPR (wild stratum) ----------
    wa = strata.get("wild_abstained", [])
    committed = sum(1 for r in wa if r["outcome"] == "recovered" and r["pf_final"] >= 0.5)
    f_flip, f_lo, f_hi = wilson(committed, len(wa))
    post_rate_point = (WILD_ISSUED_N * 0 / WILD_TOTAL_N
                       + WILD_ABSTAINED_N * f_flip / WILD_TOTAL_N)
    # CI propagation: conservative bound = (issued stay 0 live) + abstained hi
    post_rate_hi = WILD_ABSTAINED_N * f_hi / WILD_TOTAL_N
    g1 = {"bar": G1_BAR, "pristine": "0/701 = 0.0000",
          "committed_fakes_in_sample": committed,
          "abstained_sample_n": len(wa),
          "flip_rate_point": round(f_flip, 4), "flip_rate_ci95": [round(f_lo, 4), round(f_hi, 4)],
          "post_recovery_rate_point": round(post_rate_point, 4),
          "post_recovery_rate_ci_hi": round(post_rate_hi, 4),
          "verdict": "PASS" if post_rate_hi <= G1_BAR else "FAIL"}

    # ---------- G2: mean Δpf, wild abstained ----------
    deltas = [delta_row(r) for r in wa if delta_row(r) is not None]
    lo, hi = boot_mean_ci(deltas)
    mean_d = round(sum(deltas) / len(deltas), 4)
    g2 = {"bar": G2_BAR, "n": len(deltas), "mean_delta": mean_d,
          "ci95": [round(lo, 4), round(hi, 4)],
          "direction": "fake-ward" if mean_d > 0 else ("real-ward" if mean_d < 0 else "neutral"),
          "verdict": "PASS" if (abs(mean_d) <= G2_BAR or mean_d < 0) else "FAIL"}

    # ---------- G3: clamps + mechanism-b over ALL rows ----------
    clamp_v = sum(1 for r in ok if r.get("clamp_ok") is False)
    mech_v = sum(1 for r in ok if r.get("mechanism_b_ok") is False)
    # ledger mining: today's exact image recovery rows (independent evidence)
    ledger = []
    try:
        con = sqlite3.connect(WS / "artifacts" / "verdicts.db")
        con.row_factory = sqlite3.Row
        db_rows = con.execute(
            "SELECT id,filename,p_fake,abstained,recovery_json FROM verdicts "
            "WHERE kind='image' AND recovery_json IS NOT NULL ORDER BY ts").fetchall()
        for dr in db_rows:
            rec = json.loads(dr["recovery_json"])
            v = rec.get("verdict") or {}
            outc = rec.get("outcome")
            clamp_ok = None
            if outc == "recovered":
                clamp_ok = (v.get("trust", 1) <= 0.6 and v.get("risk", 0) >= 0.5)
            elif outc == "needs_review":
                clamp_ok = (v.get("trust", 1) <= 0.2 and v.get("risk", 0) >= 0.9)
            ledger.append({"id": dr["id"], "file": dr["filename"][:44],
                           "outcome": outc, "clamp_ok": clamp_ok,
                           "pf_final": v.get("p_fake_fused")})
    except Exception as e:
        ledger = [{"error": str(e)}]
    g3 = {"clamp_violations": clamp_v, "mechanism_b_violations": mech_v,
          "ledger_rows_mined": len(ledger),
          "ledger_clamp_violations": sum(1 for L in ledger if L.get("clamp_ok") is False),
          "verdict": "PASS" if (clamp_v + mech_v +
                                sum(1 for L in ledger if L.get("clamp_ok") is False)) == 0 else "FAIL"}

    # ---------- bias concentration (wild abstained by pf band) ----------
    def band_of(pf):
        return "<0.3" if pf < 0.3 else ("0.3-0.5" if pf < 0.5 else ("0.5-0.7" if pf < 0.7 else ">=0.7"))

    bands = {"<0.3": [], "0.3-0.5": [], "0.5-0.7": [], ">=0.7": []}
    for r in wa:
        pf = r.get("pf_court")
        if pf is not None:
            bands[band_of(pf)].append(r)
    bands = {b: {"n": len(v),
                 "mean_delta": (round(sum(delta_row(r) for r in v if delta_row(r) is not None)
                                     / max(1, len([r for r in v if delta_row(r) is not None])), 4)
                             if any(delta_row(r) is not None for r in v) else None),
                 "recovered_committed_fake": sum(1 for r in v if r["outcome"] == "recovered"
                                                 and (r["pf_final"] or 0) >= 0.5)}
             for b, v in bands.items()}

    # ---------- issued-real counterfactual (forced R1) ----------
    issued_cf = {"wild_issued": [], "provider_real_issued": []}
    for r in strata.get("wild_issued", []):
        f = json.loads(r.get("forced_r1") or "null")
        if f:
            issued_cf["wild_issued"].append((r["pf_court"], f))
    for r in prov:
        if r.get("outcome") == "not_attempted":
            f = json.loads(r.get("forced_r1") or "null")
            if f:
                issued_cf["provider_real_issued"].append((r["pf_court"], f))
    ics = {}
    for k, lst in issued_cf.items():
        q = [(pf, v["pf"]) for pf, fs in lst for v in fs if v["name"] == "jpeg_q95_rescore" and v.get("pf") is not None]
        would_commit = sum(1 for pf, fp in q if fp >= 0.5 and fp > 0.79071 and pf < 0.5)
        # would_commit above is approximate; compute precisely below
        would_commit = 0
        for pf, fs in lst:
            v = next((v for v in fs if v["name"] == "jpeg_q95_rescore"), None)
            if v and v.get("pf") is not None and v.get("abstained") is False and not v.get("ooc") and v["pf"] >= 0.5 and pf < 0.5:
                would_commit += 1
        deltas = [round(fp - pf, 4) for pf, fp in q]
        ics[k] = {"n_forced": len(lst),
                  "mean_q95_delta": round(sum(deltas) / len(deltas), 4) if deltas else None,
                  "ci95": boot_mean_ci(deltas) if deltas else None,
                  "would_commit_fake": would_commit}

    metrics = {
        "spec": "artifacts/generator_intake/recovery_parity_spec.md",
        "greenlight": "user word 2026-10-19 (proceed with the priority item next)",
        "protocol": "offline import-only replay of recovery.run at the production API seam; "
                    "court never modified; recovery.py byte-frozen (C-RECOV)",
        "rows_total": len(rows), "rows_ok": len(ok), "rows_error": len(errors),
        "errors": errors[:10],
        "determinism": {"wild_drift_violations": len(drift_bad),
                        "wild_drift_max": max((r["drift"] for r in wild_all
                                               if r.get("drift") is not None), default=None),
                        "provider_outcome_mismatch": len(prov_mismatch),
                        "provider_outcome_n": len(prov_match)},
        "G1": g1, "G2": g2, "G3": g3,
        "strata": stats,
        "wild_bias_by_band": bands,
        "issued_counterfactual": ics,
        "provider_outcome_crosstab_p6v10_to_p6v11": xtab,
        "ledger_appendix": ledger,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # ---------------- report ----------------
    L = []
    L.append("# EXAM REPORT — recovery-direction parity (Phase-R slot 1)\n")
    L.append("Run: 2026-10-19 · offline read-only replay · spec "
             "`artifacts/generator_intake/recovery_parity_spec.md` · greenlight: user word "
             "2026-10-19.\n")
    L.append("Protocol: production API seam (`run_image` + `recovery.run` rich-extras), "
             "context hoisted, exiftool PATH house prefix; **recovery.py byte-frozen (C-RECOV)**; "
             "no court code touched; synchrony with stored exams: wild rows drift ≈ 0 "
             "(deployed p6v11 == stored), provider rows carry expected p6v10→p6v11 version "
             "delta and outcome-level crosscheck.\n")
    L.append(f"Rows: {len(ok)} ok, {len(errors)} errors (plan 200 = "
             f"wild_ab 80 + wild_issued 40 + provider 40 + recapture 20 + fake-sanity 20).\n")
    L.append("## Verdicts\n")
    L.append(f"| Bar | Result | Verdict |\n|---|---|---|")
    L.append(f"| G1 post-recovery issued real-FPR (wild) ≤ {G1_BAR} | "
             f"point {g1['post_recovery_rate_point']} (CI-hi {g1['post_recovery_rate_ci_hi']}; "
             f"flip {g1['committed_fakes_in_sample']}/{g1['abstained_sample_n']}) | **{g1['verdict']}** |")
    L.append(f"| G2 mean Δpf on wild abstained reals ≤ ±{G2_BAR} | "
             f"{g2['mean_delta']} ({g2['direction']}; CI95 {g2['ci95']}) | **{g2['verdict']}** |")
    L.append(f"| G3 clamps + mechanism-b, zero violations | "
             f"{g3['clamp_violations']} clamp / {g3['mechanism_b_violations']} mech-b "
             f"(+ledger {g3['ledger_rows_mined']} rows, "
             f"{g3['ledger_clamp_violations']} violations) | **{g3['verdict']}** |\n")
    L.append("## Strata readouts\n")
    L.append("| stratum | n | outcomes | mean Δpf (verdict-level) | mean q95 Δ (when it ran) |\n|---|---|---|---|---|")
    for k, s in stats.items():
        outs = ", ".join(f"{o}:{n}" for o, n in sorted(s["outcomes"].items()))
        L.append(f"| {k} | {s['n']} | {outs} | {s['mean_delta']} (95% CI {s['delta_ci95']}) | "
                 f"{s['mean_q95_delta_when_ran']} |")
    L.append("\n## Determinism / crosschecks\n")
    L.append(f"- Wild rows (same-stack p6v11): max drift {metrics['determinism']['wild_drift_max']} "
             f"(violations >5e-4: {len(drift_bad)}/{len(wild_all)}).")
    L.append(f"- Provider rows: outcome match vs intake CSV {len(prov_match) - len(prov_mismatch)}/"
             f"{len(prov_match)} (pf drift allowed — p6v10-era stored scores).\n")
    L.append("## Wild bias concentration\n")
    L.append("| original pf band | n | mean Δpf |\n|---|---|---|")
    for b, s in bands.items():
        L.append(f"| {b} | {s['n']} | {s['mean_delta']} |")
    L.append("\n## Issued-real counterfactual (forced R1 — spec §step 1 second arm)\n")
    L.append("| arm | n forced | mean q95 Δpf | 95% CI | would commit fake |\n|---|---|---|---|---|")
    for k, s in ics.items():
        ci = s["ci95"] or (None, None)
        L.append(f"| {k} | {s['n_forced']} | {s['mean_q95_delta']} | [{ci[0]}, {ci[1]}] | {s['would_commit_fake']} |")
    L.append("\n## Fake-side safety (ladder-undo check)\n")
    fs = stats.get("fake_sanity", {})
    L.append(f"- abstained+issued fakes replayed: {fs.get('n', 0)}; recovered→committed-"
             f"real (undo, pf_final<0.5): {fs.get('recovered_committed_real', 0)} — must be 0.")
    L.append("\n## Provider outcome cross-tab (stored p6v10-era → fresh p6v11)\n")
    L.append("| stored → fresh | n |\n|---|---|")
    for k2, n2 in sorted(xtab.items()):
        L.append(f"| {k2} | {n2} |")
    L.append("\n## Ledger appendix (exact, independent evidence — today's UI rows)\n")
    L.append("| ledger id | file | outcome | pf_final | clamp ok |\n|---|---|---|---|---|")
    for r in ledger:
        if "error" in r:
            L.append(f"| ERR | {r['error'][:60]} | | | |")
            continue
        L.append(f"| #{r['id']} | {r['file']} | {r['outcome']} | {r['pf_final']} | {r['clamp_ok']} |")
    L.append("""
## Decision page (escalated to the user — D-046)

**(a) Keep the image ladder as shipped** — document the wash: every abstained real carries a
`needs_review`/`recovered` flag and clamps hold; issue: the wash converts abstained reals into
issued-fakes silently at q95 (see bias bands above).

**(b) R1-variant parity spec** (registered proposal): add a parity cap inside `_r1_rescore` —
a variant that moves a real-side row fake-ward beyond a registered Δ cannot WIN the recovery
(the row stays needs_review); requires recovery.py change → NOT in freeze scope; would be a
Phase-R slot with its own bars.

**(c) Disable the image ladder for abstained-reals-only** (image ladder runs only when
p_real < 0.5 i.e. fake-lean rows; reals p_real ≥ 0.5 abstained go STRAIGHT to needs_review).
Smallest blast radius, but loses the (hypothetical) cases where the ladder saves a
false-toward-review real.

Any choice: **no code changes during freeze week** — execution only after ship + user word.
""")
    (OUT / "EXAM_REPORT.md").write_text("\n".join(L))
    print(json.dumps({"G1": g1["verdict"], "G1_rate": g1["post_recovery_rate_point"],
                      "G2": g2["verdict"], "G2_mean": g2["mean_delta"], "G2_ci": g2["ci95"],
                      "G3": g3["verdict"],
                      "strata_mean_delta": {k: s["mean_delta"] for k, s in stats.items()},
                      "errors": len(errors)}, indent=1))


if __name__ == "__main__":
    main()
