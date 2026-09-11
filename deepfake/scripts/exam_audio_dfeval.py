"""External audio exam on Deepfake-Eval-2024 (official test protocol, D-033).

Independent in-the-wild benchmark (arXiv 2503.02857): 45h video / 56.5h audio /
1975 images circulated in 2024. Open-source SOTA drops ~48% AUC on audio vs
lab benchmarks. Terms: evaluation-only use — this exam never trains on the data.

Runs the deployed audio branch (argus_detect.detect_audio) over matched
official-test files (703 mp3s: 395 Real / 308 Fake) and reports AUC, TPR@0.5,
FPR@0.5, abstention. Usage:
  ./.venv/bin/python deepfake/scripts/exam_audio_dfeval.py
"""
import csv
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

AUDIO_DIR = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data"
STEMS_CSV = WS / "data" / "dfeval" / "meta" / "matched_stems.csv"
OUT = WS / "artifacts" / "exam_dfeval"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    stems = {r["stem"]: r["label"] for r in csv.DictReader(open(STEMS_CSV))}
    files = sorted(AUDIO_DIR.glob("*.mp3"))
    rows, t0 = [], time.time()
    for i, p in enumerate(files):
        stem = p.stem
        lab = stems.get(stem)
        if lab is None:
            continue
        y = 0 if lab == "Real" else 1
        try:
            from argus_detect import detect_audio
            a = detect_audio(p)
            pf = a["p_fake_audio_mean"]
            abst = a.get("abstained", False)
            ooc = a.get("out_of_coverage", False)
            zn = a.get("znorm_mean")
            nseg = a.get("segments_analyzed", a.get("n_segments", 0))
        except Exception as e:
            print(f"ERR {p.name}: {e}", flush=True)
            pf, abst, ooc, zn, nseg = None, False, False, None, 0
        rows.append({"stem": stem, "label": y, "p_fake": pf,
                     "abstained": abst, "out_of_coverage": ooc,
                     "znorm_mean": zn, "n_segments": nseg})
        if (i + 1) % 25 == 0:
            print(f"{i+1}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    out = OUT / "audio_exam.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    valid = [r for r in rows if r["p_fake"] is not None]
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y = np.array([r["label"] for r in valid])
    s = np.array([r["p_fake"] for r in valid])
    auc = roc_auc_score(y, s)
    tpr = float((s[y == 1] >= 0.5).mean())
    fpr = float((s[y == 0] >= 0.5).mean())
    ooc = np.array([bool(r["out_of_coverage"]) for r in valid])
    ab = float(np.mean([bool(r["abstained"]) for r in valid]))
    # among stems the court still votes on (in-coverage), what's the accuracy?
    in_cov = ~ooc
    if in_cov.sum() >= 20 and len(set(y[in_cov])) > 1:
        auc_cov = round(float(roc_auc_score(y[in_cov], s[in_cov])), 4)
        fpr_cov = round(float((s[in_cov][y[in_cov] == 0] >= 0.5).mean()), 4)
        tpr_cov = round(float((s[in_cov][y[in_cov] == 1] >= 0.5).mean()), 4)
    else:
        auc_cov = fpr_cov = tpr_cov = None
    summary = {
        "benchmark": "Deepfake-Eval-2024 (official test protocol, matched stems)",
        "n": len(valid), "real": int((y == 0).sum()), "fake": int((y == 1).sum()),
        "raw_auc_no_gate": round(float(auc), 4),
        "fpr_at_0.5_no_gate": round(fpr, 4),
        "tpr_at_0.5_no_gate": round(tpr, 4),
        "out_of_coverage_rate": round(float(ooc.mean()), 4),
        "abstained_rate": round(ab, 4),
        "in_coverage_n": int(in_cov.sum()),
        "in_coverage_auc": auc_cov,
        "in_coverage_fpr_at_0.5": fpr_cov,
        "in_coverage_tpr_at_0.5": tpr_cov,
        "coverage_gate": "z-norm > train p99.5 (D-035); off-manifold => abstain",
        "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
    }
    (OUT / "audio_exam_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print("field reference: open-source SOTA audio AUC drops ~48% on this benchmark", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
