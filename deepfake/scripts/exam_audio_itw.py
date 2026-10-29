"""External audio exam on In-The-Wild (Müller et al. 2022, arXiv:2203.16263).

The canonical wild-audio generalization benchmark: 31,779 clips, 58 speakers
(celebrities/politicians), real + synthetic speech from online videos — 19,963
bona-fide / 11,816 spoof. The paper's core finding is that lab-trained detectors
collapse here; this exam measures exactly that posture for the deployed stack and
for W7 candidates. License CC BY-SA 4.0. Exam-only: never trains on this data.

Stratified subsample (default 300/class, per-speaker cap, seed recorded) scored
through the SERVE path (argus_detect.detect_audio). Candidate probes injected by
patching argus_detect.AUDIO_PROBE before main() (import+patch+call, never runpy).

Usage:
  ./.venv/bin/python deepfake/scripts/exam_audio_itw.py [--per-class 300] [--seed 42]
Output: artifacts/exam_audio_itw[_pXX]/{scores.csv, metrics.json}
"""
import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

ITW_DIR = WS / "data" / "raw" / "itw" / "release_in_the_wild"
META = ITW_DIR / "meta.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--speaker-cap", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from argus_detect import STACKER
    tag = STACKER.parent.name
    outdir = Path(args.out) if args.out else WS / "artifacts" / f"exam_audio_itw_{tag}"
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        print(f"ERROR: {outdir} already contains results — pass --force", file=sys.stderr)
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    meta = list(csv.DictReader(open(META)))
    rng = random.Random(args.seed)
    by_class = defaultdict(lambda: defaultdict(list))  # label -> speaker -> [file]
    for r in meta:
        by_class[r["label"]][r["speaker"]].append(r["file"])
    picked = []
    for lab in sorted(by_class):
        per_speaker = []
        for spk in sorted(by_class[lab]):
            files = sorted(by_class[lab][spk])
            rng.shuffle(files)
            per_speaker.extend((f, spk) for f in files[:args.speaker_cap])
        rng.shuffle(per_speaker)
        take = per_speaker[:args.per_class]
        picked.extend((f, lab, spk) for f, spk in take)
    rng.shuffle(picked)
    print(f"scoring {len(picked)} utts ({args.per_class}/class, speaker cap {args.speaker_cap})", flush=True)

    rows, n_err = [], 0
    t0 = time.time()
    for i, (fname, lab, spk) in enumerate(picked, 1):
        path = ITW_DIR / fname
        try:
            from argus_detect import detect_audio
            a = detect_audio(path)
            pf = a.get("p_fake_audio_mean")
            abst = bool(a.get("abstained", False))
            ooc = bool(a.get("out_of_coverage", False))
            zn = a.get("znorm_mean")
            nseg = a.get("segments_analyzed", 0)
        except Exception as e:
            pf, abst, ooc, zn, nseg = None, False, False, None, 0
            n_err += 1
            if n_err <= 5:
                print(f"  err {fname}: {type(e).__name__}: {e}", flush=True)
        rows.append({"file": fname, "speaker": spk, "label": 0 if lab == "bona-fide" else 1,
                     "p_fake": pf, "abstained": abst, "ooc": ooc, "znorm": zn, "segments": nseg})
        if i % 100 == 0:
            print(f"[{i}/{len(picked)}] {i / (time.time() - t0):.2f} utt/s err={n_err}", flush=True)

    with open(outdir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def auc(pairs):
        import bisect
        pos = sorted(p for l, p in pairs if l == 1)
        neg = sorted(p for l, p in pairs if l == 0)
        if not pos or not neg:
            return None
        wins = ties = 0
        for p in pos:
            lo, hi = bisect.bisect_left(neg, p), bisect.bisect_right(neg, p)
            wins += lo
            ties += hi - lo
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    scored = [(r["label"], r["p_fake"]) for r in rows if r["p_fake"] is not None and not r["abstained"]]
    allp = [(r["label"], r["p_fake"]) for r in rows if r["p_fake"] is not None]
    a_all, a_sc = auc(allp), auc(scored)
    thr = 0.5
    n_pos = sum(1 for l, _ in scored if l == 1)
    n_neg = sum(1 for l, _ in scored if l == 0)
    metrics = {
        "benchmark": "In-The-Wild (Müller et al. 2022) — wild audio generalization exam",
        "n_total": len(rows), "n_scored": len(scored), "n_err": n_err,
        "abstained_rate": round(1 - len(scored) / max(len(rows), 1), 4),
        "raw_auc_no_gate": round(a_all, 4) if a_all is not None else None,
        "in_coverage_auc": round(a_sc, 4) if a_sc is not None else None,
        "in_coverage_tpr_at_0.5": round(sum(1 for l, p in scored if l == 1 and p > thr) / max(n_pos, 1), 4) if n_pos else None,
        "in_coverage_fpr_at_0.5": round(sum(1 for l, p in scored if l == 0 and p > thr) / max(n_neg, 1), 4) if n_neg else None,
        "per_speaker_top": {},
        "seed": args.seed, "per_class": args.per_class, "speaker_cap": args.speaker_cap,
        "coverage_gate": "z-norm > train p99.5 (D-035); off-manifold => abstain",
        "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
    }
    # per-speaker abstention + AUC for the 8 speakers with most scored rows
    sp_rows = defaultdict(list)
    for r in rows:
        sp_rows[r["speaker"]].append(r)
    top = sorted(sp_rows, key=lambda s: -len(sp_rows[s]))[:8]
    for s in top:
        sub = sp_rows[s]
        sc = [(r["label"], r["p_fake"]) for r in sub if r["p_fake"] is not None and not r["abstained"]]
        metrics["per_speaker_top"][s] = {
            "n": len(sub), "n_scored": len(sc),
            "abstained_rate": round(1 - len(sc) / max(len(sub), 1), 4),
            "in_coverage_auc": round(auc(sc), 4) if sc and len(set(l for l, _ in sc)) == 2 else None,
        }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_speaker_top"}, indent=1), flush=True)
    print(f"[done] {len(rows)} rows ({n_err} errors) -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
