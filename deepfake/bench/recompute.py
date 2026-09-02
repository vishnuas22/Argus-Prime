"""Recompute metrics.json + REPORT.md from an existing scores.csv (no rerun).

Usage: .venv/bin/python deepfake/bench/recompute.py artifacts/p0_baseline
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from metrics import auc, eer, brier, abstention_rate, mean_ms  # noqa: E402
from report import render_report  # noqa: E402

MODULE_IDS = ["metadata", "compression_history", "spectral_probe",
              "real_distribution_probe", "perturbation_probe"]


def main() -> int:
    out = Path(sys.argv[1])
    rows = []
    with open(out / "scores.csv") as f:
        for r in csv.DictReader(f):
            r["label"] = int(r["label"])
            for k in ("p_real", "trust", "risk", "conflict", "triage_ms"):
                r[k] = float(r[k]) if r[k] not in ("", None) else None
            for m in MODULE_IDS:
                r[m] = float(r[m]) if r[m] not in ("", None) else None
                r[f"{m}_ms"] = float(r[f"{m}_ms"]) if r.get(f"{m}_ms") else None
            r["abstained"] = r["abstained"] == "True"
            rows.append(r)

    variants = sorted({r["variant"] for r in rows})
    n_real = len({r["image"] for r in rows if r["label"] == 0})
    n_fake = len({r["image"] for r in rows if r["label"] == 1})
    results = {}
    for vname in variants:
        sub = [r for r in rows if r["variant"] == vname]
        labels = np.array([r["label"] for r in sub])
        block = {}
        for c in MODULE_IDS:
            scores = np.array([-r[c] if r[c] is not None else np.nan for r in sub])
            mask = ~np.isnan(scores)
            s, l = scores[mask], labels[mask]
            block[c] = {"auc": None if np.isnan(auc(s, l)) else round(auc(s, l), 4),
                        "eer": round(eer(s, l), 4), "n": int(mask.sum())}
        preal = np.array([r["p_real"] for r in sub])
        block["fusion"] = {
            "auc": round(auc(-preal, labels), 4),
            "eer": round(eer(-preal, labels), 4),
            "n": len(sub),
            "brier_p_real": round(brier(preal, labels), 4),
            "abstention_rate": round(abstention_rate([r["abstained"] for r in sub]), 4),
            "mean_trust": round(float(np.mean([r["trust"] for r in sub])), 4),
            "mean_risk": round(float(np.mean([r["risk"] for r in sub])), 4),
        }
        block["runtime_ms"] = {
            "triage": round(mean_ms([r["triage_ms"] for r in sub]), 1),
            **{c: round(mean_ms([r[f"{c}_ms"] for r in sub]), 1) for c in MODULE_IDS},
        }
        caps = {}
        for r in sub:
            caps[r["capacity"]] = caps.get(r["capacity"], 0) + 1
        block["evidence_capacity_mix"] = caps
        results[vname] = block

    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    (out / "REPORT.md").write_text(render_report(results, n_real, n_fake))
    print(f"recomputed -> {out/'metrics.json'}, {out/'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
