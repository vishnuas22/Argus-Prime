"""Phase 4 D-046 ablation comparison — r9 vs AB1 vs AB2 (diagnostic only).

Loads the three gates.json files (identical schema, all 13 gates) and emits
artifacts/restack_r9_ablations_summary.json:

  per_gate.<name> = {r9_delta, ab1_delta, ab2_delta, ab1_pass, ab2_pass,
                     abstained_rate_{r9_old,r9_new,ab1_old,ab1_new,
                                     ab2_old,ab2_new},
                     ooc_rate_{r9_new,ab1_new,ab2_new},
                     auc_{r9_new,ab1_new,ab2_new}}

  questions.q1_flux_recovery      ab1 flux auc_new > r9 flux auc_new + 0.05
  questions.q2_abstention_normalize
        AB2 abstained_rate_new within +-0.05 of its own p6v10 old reference on
        EACH of aigc/cdf/ffpp (all three must hold)

Fail-closed: any missing/unreadable gates.json -> exit non-zero, no summary.
NO promotion, NO r9 workdir/candidate writes, NO memory writes.

Usage:
  .venv/bin/python deepfake/scripts/ablation_compare.py [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
ART = WS / "artifacts"
R9 = ART / "restack_r9" / "gates.json"
AB1 = ART / "restack_r9_ab1" / "gates.json"
AB2 = ART / "restack_r9_ab2" / "gates.json"
DEFAULT_OUT = ART / "restack_r9_ablations_summary.json"

ABSTAIN_EPS = 0.05
FLUX_RECOVERY_EPS = 0.05
Q2_GATES = ("aigc", "cdf", "ffpp")


def load(path: Path, tag: str) -> dict:
    if not path.exists():
        raise SystemExit(f"FAIL: missing {tag} gates.json -> {path}")
    try:
        g = json.loads(path.read_text())
    except Exception as e:
        raise SystemExit(f"FAIL: unreadable {tag} gates.json ({path}): {e}")
    if not isinstance(g, dict) or not g:
        raise SystemExit(f"FAIL: {tag} gates.json empty/not a dict -> {path}")
    return g


def main() -> int:
    ap = argparse.ArgumentParser(description="compare r9 vs AB1 vs AB2 gates")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    r9, ab1, ab2 = load(R9, "r9"), load(AB1, "ab1"), load(AB2, "ab2")

    gate_names = list(r9.keys())  # r9 is the canonical 13-gate reference
    per_gate = {}
    for name in gate_names:
        row = {"r9_delta": r9[name].get("delta"),
               "ab1_delta": (ab1.get(name) or {}).get("delta"),
               "ab2_delta": (ab2.get(name) or {}).get("delta"),
               "ab1_pass": (ab1.get(name) or {}).get("pass"),
               "ab2_pass": (ab2.get(name) or {}).get("pass"),
               "auc_r9_new": r9[name].get("auc_new"),
               "auc_ab1_new": (ab1.get(name) or {}).get("auc_new"),
               "auc_ab2_new": (ab2.get(name) or {}).get("auc_new"),
               "abstained_rate_r9_old": r9[name].get("abstained_rate_old"),
               "abstained_rate_r9_new": r9[name].get("abstained_rate_new"),
               "abstained_rate_ab1_old": (ab1.get(name) or {}).get("abstained_rate_old"),
               "abstained_rate_ab1_new": (ab1.get(name) or {}).get("abstained_rate_new"),
               "abstained_rate_ab2_old": (ab2.get(name) or {}).get("abstained_rate_old"),
               "abstained_rate_ab2_new": (ab2.get(name) or {}).get("abstained_rate_new"),
               "ooc_rate_r9_new": r9[name].get("out_of_conflict_rate_new"),
               "ooc_rate_ab1_new": (ab1.get(name) or {}).get("out_of_conflict_rate_new"),
               "ooc_rate_ab2_new": (ab2.get(name) or {}).get("out_of_conflict_rate_new")}
        per_gate[name] = row

    # q1: did the flux exam AUC recover in AB1 vs what r9 produced?
    flux = per_gate.get("flux") or {}
    if flux.get("auc_r9_new") is None or flux.get("auc_ab1_new") is None:
        raise SystemExit("FAIL: flux gate auc missing in r9/ab1 — cannot compute q1")
    q1 = bool(flux["auc_ab1_new"] > flux["auc_r9_new"] + FLUX_RECOVERY_EPS)

    # q2: does AB2's abstention return to its p6v10 (old) reference level?
    q2_detail = {}
    for g in Q2_GATES:
        r = per_gate.get(g)
        if r is None or r.get("abstained_rate_ab2_new") is None \
                or r.get("abstained_rate_ab2_old") is None:
            raise SystemExit(f"FAIL: abstained_rate missing for {g} in ab2 — q2")
        q2_detail[g] = round(abs(r["abstained_rate_ab2_new"]
                                 - r["abstained_rate_ab2_old"]), 4)
    q2 = all(v <= ABSTAIN_EPS for v in q2_detail.values())

    summary = {
        "generated_from": {"r9": str(R9), "ab1": str(AB1), "ab2": str(AB2)},
        "gate_names": gate_names,
        "n_gates": len(gate_names),
        "per_gate": per_gate,
        "questions": {
            "q1_flux_recovery": {
                "question": "does flux exam recover in AB1? "
                            "(ab1 auc_new > r9 auc_new + 0.05)",
                "auc_r9_new": flux["auc_r9_new"],
                "auc_ab1_new": flux["auc_ab1_new"],
                "threshold": FLUX_RECOVERY_EPS,
                "answer": q1,
            },
            "q2_abstention_normalize": {
                "question": "does AB2 abstention return to p6v10-like levels? "
                            "(each of aigc/cdf/ffpp |new-old| <= 0.05)",
                "abs_diffs": q2_detail,
                "threshold": ABSTAIN_EPS,
                "answer": q2,
            },
        },
        "note": "diagnostic only — no promotion, r9 untouched",
    }
    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2))

    print(f"[compare] gates: {len(gate_names)} "
          f"({', '.join(gate_names)})")
    for g in gate_names:
        r = per_gate[g]
        print(f"  {g:16s} r9_d={r['r9_delta']} ab1_d={r['ab1_delta']} "
              f"ab2_d={r['ab2_delta']} ab1_pass={r['ab1_pass']} "
              f"ab2_pass={r['ab2_pass']}")
    print(f"[compare] q1 flux recovery: {q1} "
          f"(r9={flux['auc_r9_new']} ab1={flux['auc_ab1_new']})")
    print(f"[compare] q2 abstention normalize: {q2} abs_diffs={q2_detail}")
    print(f"[compare] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
