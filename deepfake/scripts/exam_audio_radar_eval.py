"""External audio exam on RADAR 2026 evaluation set (multilingual + unseen transforms).

The RADAR 2026 eval (Zenodo 22110815, CC BY 4.0; paper arXiv:2605.09568v3) is the
challenge's blind condition: 102,725 utterances in EN/en-SG/zh/zh-TW/ja/vi, spoofed
by 10 TTS systems (iFlytek, Houshan, ElevenLabs, Cartesia, OpenAI, Chatterbox,
CosyVoice 3.0, Qwen3-TTS, Fish S2 Pro, Piper), with transforms ABSENT from the dev
set (AAC, 8kHz resample, streaming dropout, sound effects, 300-3400Hz band-limit,
loudness norm, speech perturbation). External yardstick: SSL-AASIST baseline 42.6% EER;
best team 5.10% EER.

Exam-only: never trains on this data. Stratified subsample (default 150/lang/class,
seed recorded) scored through the SERVE path (argus_detect.detect_audio) so gate and
windowing semantics match production exactly. Candidate probes are injected by patching
argus_detect.AUDIO_PROBE before calling main() (L-RUNPY-PATCH-VOID: import+patch+call,
never runpy).

Usage:
  ./.venv/bin/python deepfake/scripts/exam_audio_radar_eval.py [--per-cell 150] [--seed 42]
Output: artifacts/exam_audio_radareval[_pXX]/{scores.csv, metrics.json}
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

EVAL_DIR = WS / "data" / "raw" / "radar2026" / "eval" / "RADAR2026-eval" / "flac"
PROTO = WS / "data" / "raw" / "radar2026" / "label_RADAR2026-eval.txt"
LANGS = WS / "data" / "raw" / "radar2026" / "lang_RADAR2026-eval.txt"


def find_flac(uid: str, index: dict) -> str | None:
    return index.get(uid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--per-cell", type=int, default=150, help="max utts per language x class cell")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from argus_detect import STACKER
    tag = STACKER.parent.name  # e.g. p4 -> exam dir exam_audio_radareval_p4
    outdir = Path(args.out) if args.out else WS / "artifacts" / f"exam_audio_radareval_{tag}"
    if outdir.exists() and any(outdir.iterdir()) and not args.force:
        print(f"ERROR: {outdir} already contains results — pass --force", file=sys.stderr)
        return 1
    outdir.mkdir(parents=True, exist_ok=True)

    lang_of = {}
    for line in LANGS.read_text().strip().split("\n"):
        uid, lg = line.split()
        lang_of[uid] = lg
    proto = []
    for line in PROTO.read_text().strip().split("\n"):
        uid, dur, lab, seg = line.split()
        proto.append((uid, float(dur), lab))

    # file index (flat or nested flac dir)
    index = {}
    for dp, dn, fn in __import__("os").walk(EVAL_DIR):
        for f in fn:
            if f.endswith(".flac"):
                index[f[:-5]] = f"{dp}/{f}"

    # stratified subsample: language x class cells
    rng = random.Random(args.seed)
    cells = defaultdict(list)
    for uid, dur, lab in proto:
        cells[(lang_of.get(uid, "?"), lab)].append(uid)
    picked = []
    for cell in sorted(cells):
        lang_cell, lab_cell = cell
        take = rng.sample(cells[cell], min(args.per_cell, len(cells[cell])))
        picked.extend((uid, lab_cell, lang_cell) for uid in take)
    rng.shuffle(picked)
    print(f"scoring {len(picked)} utts ({args.per_cell}/cell x {len(cells)} cells); device warm-up...", flush=True)

    rows = []
    n_err = 0
    t0 = time.time()
    for i, (uid, lab, lg) in enumerate(picked, 1):
        path = find_flac(uid, index)
        if path is None:
            n_err += 1
            continue
        try:
            from argus_detect import detect_audio
            a = detect_audio(Path(path))
            pf = a.get("p_fake_audio_mean")
            abst = bool(a.get("abstained", False))
            ooc = bool(a.get("out_of_coverage", False))
            zn = a.get("znorm_mean")
            nseg = a.get("segments_analyzed", 0)
        except Exception as e:
            pf, abst, ooc, zn, nseg = None, False, False, None, 0
            n_err += 1
            if n_err <= 5:
                print(f"  err {uid}: {type(e).__name__}: {e}", flush=True)
        rows.append({"uid": uid, "lang": lg, "label": 0 if lab == "bonafide" else 1,
                     "p_fake": pf, "abstained": abst, "ooc": ooc, "znorm": zn,
                     "segments": nseg, "duration_s": dur})
        if i % 100 == 0:
            rate = i / (time.time() - t0)
            print(f"[{i}/{len(picked)}] {rate:.2f} utt/s err={n_err}", flush=True)

    with open(outdir / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # metrics: overall + per-language (scored = non-abstained votes; AUC on all with p)
    def auc(pairs):
        pos = sorted(p for l, p in pairs if l == 1)
        neg = sorted(p for l, p in pairs if l == 0)
        if not pos or not neg:
            return None
        import bisect
        wins = ties = 0
        for p in pos:
            lo = bisect.bisect_left(neg, p)
            hi = bisect.bisect_right(neg, p)
            wins += lo
            ties += hi - lo
        return (wins + 0.5 * ties) / (len(pos) * len(neg))

    scored = [(r["label"], r["p_fake"]) for r in rows if r["p_fake"] is not None and not r["abstained"]]
    allp = [(r["label"], r["p_fake"]) for r in rows if r["p_fake"] is not None]
    a_all, a_sc = auc(allp), auc(scored)
    thr = 0.5
    metrics = {
        "benchmark": "RADAR 2026 evaluation set (multilingual + unseen transforms)",
        "n_total": len(rows), "n_scored": len(scored), "n_err": n_err,
        "abstained_rate": round(1 - len(scored) / max(len(rows), 1), 4),
        "raw_auc_no_gate": round(a_all, 4) if a_all is not None else None,
        "in_coverage_auc": round(a_sc, 4) if a_sc is not None else None,
        "in_coverage_tpr_at_0.5": round(sum(1 for l, p in scored if l == 1 and p > thr) /
                                        max(sum(1 for l, _ in scored if l == 1), 1), 4),
        "in_coverage_fpr_at_0.5": round(sum(1 for l, p in scored if l == 0 and p > thr) /
                                        max(sum(1 for l, _ in scored if l == 0), 1), 4),
        "per_language": {},
        "seed": args.seed, "per_cell": args.per_cell,
        "coverage_gate": "z-norm > train p99.5 (D-035); off-manifold => abstain",
        "date_utc": time.strftime("%Y-%m-%d %H:%M", time.gmtime()),
    }
    for lg in sorted({r["lang"] for r in rows}):
        sub = [r for r in rows if r["lang"] == lg]
        sc = [(r["label"], r["p_fake"]) for r in sub if r["p_fake"] is not None and not r["abstained"]]
        metrics["per_language"][lg] = {
            "n": len(sub), "n_scored": len(sc),
            "abstained_rate": round(1 - len(sc) / max(len(sub), 1), 4),
            "in_coverage_auc": round(auc(sc), 4) if sc else None,
        }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps(metrics, indent=1)[:900], flush=True)
    print(f"[done] {len(rows)} rows ({n_err} errors) -> {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
