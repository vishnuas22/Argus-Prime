"""AB3 pure-retrain control readout (D-046 HOLD cycle, diagnostic).

Loads artifacts/restack_r9_ab3/gates.json (all 13 gates, identical schema) and
writes artifacts/restack_r9_ab3_readout.json:

  per_gate.<name> = {delta, pass, abstained_rate_old, abstained_rate_new}
  questions.c1_abstention_control  AB3 abstained_rate_new within +-0.05 of the
                                   p6v10 reference on aigc/cdf/ffpp
                                   (expected 0.075/0.045/0.26)
  questions.c2_original_gates_pass AB3 mj/flux/ffpp/aigc/cdf all pass

Fail-closed: missing/unreadable gates.json -> exit non-zero, no readout.
NO promotion, NO r9 writes, NO memory writes.

Usage:
  .venv/bin/python deepfake/scripts/ab3_readout.py [--out PATH]
"""
import argparse
import json
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
AB3 = WS / "artifacts" / "restack_r9_ab3" / "gates.json"
DEFAULT_OUT = WS / "artifacts" / "restack_r9_ab3_readout.json"

EPS = 0.05
C1_GATES = {"aigc": 0.075, "cdf": 0.045, "ffpp": 0.26}  # p6v10 reference rates
C2_GATES = ("mj", "flux", "ffpp", "aigc", "cdf")


def main() -> int:
    ap = argparse.ArgumentParser(description="AB3 control readout")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not AB3.exists():
        raise SystemExit(f"FAIL: missing AB3 gates.json -> {AB3}")
    try:
        gates = json.loads(AB3.read_text())
    except Exception as e:
        raise SystemExit(f"FAIL: unreadable AB3 gates.json: {e}")
    if not isinstance(gates, dict) or not gates:
        raise SystemExit("FAIL: AB3 gates.json empty/not a dict")

    per_gate = {}
    for name, g in gates.items():
        per_gate[name] = {
            "delta": g.get("delta"),
            "pass": g.get("pass"),
            "abstained_rate_old": g.get("abstained_rate_old"),
            "abstained_rate_new": g.get("abstained_rate_new"),
        }

    missing = [g for g in (*C1_GATES, *C2_GATES) if g not in per_gate]
    if missing:
        raise SystemExit(f"FAIL: gates missing from AB3 gates.json: {missing}")

    c1_detail, c1 = {}, True
    for g, ref in C1_GATES.items():
        new = per_gate[g].get("abstained_rate_new")
        old = per_gate[g].get("abstained_rate_old")
        if new is None or old is None:
            raise SystemExit(f"FAIL: abstained_rate missing for {g}")
        d_ref = round(abs(new - ref), 4)
        d_old = round(abs(new - old), 4)
        ok = d_old <= EPS and d_ref <= EPS
        c1 = c1 and ok
        c1_detail[g] = {"ab3_new": new, "p6v10_expected_ref": ref,
                        "p6v10_recorded_old": old,
                        "abs_diff_vs_expected": d_ref,
                        "abs_diff_vs_recorded_old": d_old, "within": ok}

    c2 = all(bool(per_gate[g].get("pass")) for g in C2_GATES)
    c2_detail = {g: per_gate[g].get("pass") for g in C2_GATES}

    readout = {
        "source_gates": str(AB3),
        "diet": "7 original domains, ZERO new (pure retrain control)",
        "n_gates": len(per_gate),
        "per_gate": per_gate,
        "questions": {
            "c1_abstention_control": {
                "question": "AB3 abstention on aigc/cdf/ffpp within +-0.05 of "
                            "p6v10 (0.075/0.045/0.26)?",
                "threshold": EPS,
                "detail": c1_detail,
                "answer": c1,
            },
            "c2_original_gates_pass": {
                "question": "AB3 all-original gates (mj/flux/ffpp/aigc/cdf) pass?",
                "detail": c2_detail,
                "answer": c2,
            },
        },
        "note": "diagnostic only — no promotion, r9 untouched, memory not written",
    }
    out = Path(args.out)
    out.write_text(json.dumps(readout, indent=2))

    print(f"[ab3_readout] gates: {len(per_gate)}")
    for g in sorted(per_gate):
        r = per_gate[g]
        print(f"  {g:16s} delta={r['delta']} pass={r['pass']} "
              f"abst_old={r['abstained_rate_old']} abst_new={r['abstained_rate_new']}")
    print(f"[ab3_readout] c1 abstention control: {c1} {c1_detail}")
    print(f"[ab3_readout] c2 original gates pass: {c2} {c2_detail}")
    print(f"[ab3_readout] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
