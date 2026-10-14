"""Phase 3 recovery exam runner (spec §6.2 metrics, §6.3 smoke fixtures).

Usage:
  ./.venv/bin/python deepfake/scripts/exam_recovery.py --image
  ./.venv/bin/python deepfake/scripts/exam_recovery.py --audio
  ./.venv/bin/python deepfake/scripts/exam_recovery.py --video-spotcheck
  ... flags combine; default out artifacts/recovery_exam (stamped, --force to redo)

Metrics (§6.2): M1 conversion rate, M2 recovered correctness AUC (n>=20),
M3 needs_review rate, M4 latency delta (read from latency_post.json if
present), M5 recovered directional accuracy — ESCALATE flag if < 0.50.
Audits: E1 penalty-clamp audit (mechanical), E2 screenshot-laundered real ends
in {recovered, needs_review}, E3 audio DF-Eval conversion always reported
(honest zero passes).
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))
sys.path.insert(0, str(WS / "integrations" / "Argus_Master" / "backend"))

AB = WS / "artifacts" / "recovery_ab"
DFEVAL_CSV = WS / "artifacts" / "exam_dfeval" / "audio_exam.csv"
DFEVAL_AUDIO = WS / "data" / "dfeval" / "audio" / "audio-data" / "audio-data"

# Amended G-NR4 (user decision, binding — recorded verbatim for the sentinel)
G_NR4_AMENDED = (
    'PASS iff (every one of the 30 seed-5 videos has recovery.outcome ∈ '
    '{"not_attempted", "needs_review"}) AND (every video with outcome=='
    '"not_attempted" has zero attempted_steps AND zero recovery-side effects) '
    'AND (the 2 documented identity_conflict videos — real/022.mp4, '
    'fake/NeuralTextures/420_257.mp4 — end needs_review via V3 with '
    'per_identity populated) AND (every p_fake_video_combined is bit-identical '
    'to video_spot_pre.json on all 30). The p-bit-identity is the '
    'non-regression guarantee; V3 triage on identity-conflict videos is '
    'expected ladder behavior, not interference.'
)
DOCUMENTED_IDENTITY_CONFLICT = {
    "data/raw/ffpp_mirror/FaceForensics++_C23/real/022.mp4",
    "data/raw/ffpp_mirror/FaceForensics++_C23/fake/NeuralTextures/420_257.mp4",
}


def _triggered(row: dict) -> bool:
    rec = row.get("recovery") or {}
    return rec.get("outcome") in ("recovered", "needs_review")


def _rec_of(d: dict) -> dict:
    return d.get("recovery") or {}


def _recovered_p(rec: dict):
    return (rec.get("verdict") or {}).get("p_fake_fused")


def _case(case, label, path, p, d):
    rec = _rec_of(d)
    return {"case": case, "label": label, "path": path, "p": p,
            "triggered": _triggered(d), "outcome": rec.get("outcome"),
            "recovered_p": _recovered_p(rec), "trust": rec.get("trust"),
            "risk": rec.get("risk"), "reasons": rec.get("reasons"),
            "steps": [s["step"] for s in (rec.get("attempted_steps") or [])],
            "recovery": rec}


def _auc(labels, scores):
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return round(float(roc_auc_score(labels, scores)), 4)


def _directional(rows):
    """M5: fraction of recovered verdicts whose p_fake direction matches label."""
    if not rows:
        return None
    hits = sum(1 for r in rows
               if (r["recovered_p"] >= 0.5) == (r["label"] == 1))
    return round(hits / len(rows), 4)


def _aggregate(rows):
    total = len(rows)
    trig = [r for r in rows if r["triggered"]]
    rec = [r for r in trig if r["outcome"] == "recovered"]
    rev = [r for r in trig if r["outcome"] == "needs_review"]
    labeled_rec = [r for r in rec
                   if r["label"] is not None and r["recovered_p"] is not None]
    m2 = _auc([r["label"] for r in labeled_rec],
              [r["recovered_p"] for r in labeled_rec])
    m5 = _directional(labeled_rec)
    return {
        "n": total, "triggered": len(trig), "recovered": len(rec),
        "needs_review": len(rev),
        "not_attempted": total - len(trig),
        "M1_conversion_rate": round(len(rec) / len(trig), 4) if trig else None,
        "M1_n": len(trig),
        "M2_recovered_auc": m2, "M2_n": len(labeled_rec),
        "M2_computable": m2 is not None and len(labeled_rec) >= 20,
        "M3_needs_review_rate_all": round(len(rev) / total, 4) if total else None,
        "M3_needs_review_rate_of_triggered": (
            round(len(rev) / len(trig), 4) if trig else None),
        "M5_directional_accuracy": m5, "M5_n": len(labeled_rec),
        "M5_escalate_below_050": (m5 is not None and m5 < 0.50),
    }


def _e1_audit(all_recs):
    """E1 (mechanical): no recovered trust>0.6 or risk<0.5; no needs_review
    trust>0.2."""
    viol = []
    for r in all_recs:
        o, t, k = r.get("outcome"), r.get("trust"), r.get("risk")
        if o == "recovered":
            if t is not None and t > 0.6 + 1e-9:
                viol.append({"outcome": o, "trust": t, "why": "trust>0.6"})
            if k is not None and k < 0.5 - 1e-9:
                viol.append({"outcome": o, "risk": k, "why": "risk<0.5"})
        elif o == "needs_review":
            if t is not None and t > 0.2 + 1e-9:
                viol.append({"outcome": o, "trust": t, "why": "trust>0.2"})
    return {"pass": not viol, "violations": viol}


def _write_cases(out, name, rows):
    keep = [{k: v for k, v in r.items() if k != "recovery"} for r in rows]
    if not keep:
        return
    fields = sorted({k for r in keep for k in r})
    with open(out / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in keep:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


# ---------------- --image ----------------

def exam_image(out):
    from PIL import Image
    from argus_detect import _image_stacker_clfs, detect_image
    stacker_clfs = _image_stacker_clfs()
    fixtures = json.loads((AB / "smoke_fixtures.json").read_text())["fixtures"]
    rows = []

    f2 = fixtures["2_screenshot_laundered_real"]
    p2 = WS / f2["path"]
    pil = Image.open(p2)
    pil.load()
    d = detect_image(pil.convert("RGB"), stacker_clfs, original_raw=p2.read_bytes())
    rows.append(_case("fixture_2_screenshot_real", 0, f2["path"],
                      d["p_fake_fused"], d))
    e2 = rows[-1]["outcome"] in ("recovered", "needs_review")

    f4 = fixtures["4_no_face_still_candidate"]
    p4 = WS / f4["path"]
    pil = Image.open(p4)
    pil.load()
    d = detect_image(pil.convert("RGB"), stacker_clfs, original_raw=p4.read_bytes())
    rows.append(_case("fixture_4_no_face_still", 0, f4["path"],
                      d["p_fake_fused"], d))

    import exam_launder as EL
    reals = sorted((WS / "data" / "corpora" / "eval_real").glob("*.jpg"))[:15]
    for src in reals:
        base = Image.open(src)
        base.load()
        for vname, vfn in EL.VIEWS:
            img, _fmt = vfn(base)
            d = detect_image(img.convert("RGB"), stacker_clfs,
                             original_raw=src.read_bytes())
            rows.append(_case(f"launder_{vname}", 0,
                              str(src.relative_to(WS)), d["p_fake_fused"], d))

    wild_root = WS / "data" / "corpora" / "wild_sample"
    for lab, sub in ((0, "real_test"), (1, "fake_test")):
        for p in sorted((wild_root / sub).iterdir())[:20]:
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            pil = Image.open(p)
            pil.load()
            d = detect_image(pil.convert("RGB"), stacker_clfs,
                             original_raw=p.read_bytes())
            rows.append(_case(f"wild_{sub}", lab, str(p.relative_to(WS)),
                              d["p_fake_fused"], d))

    _write_cases(out, "image_cases.csv", rows)
    agg = _aggregate(rows)
    agg["E2_screenshot_laundered_real_recovered_or_review"] = e2
    print(f"[image] n={agg['n']} triggered={agg['triggered']} "
          f"recovered={agg['recovered']} needs_review={agg['needs_review']} "
          f"M1={agg['M1_conversion_rate']} M5={agg['M5_directional_accuracy']}",
          flush=True)
    return agg, rows


# ---------------- --audio ----------------

def exam_audio(out):
    from argus_detect import detect_audio
    rows = []
    t0 = time.time()
    with open(DFEVAL_CSV) as f:
        for i, rec in enumerate(csv.DictReader(f)):
            stem, label = rec["stem"], int(rec["label"])
            p = DFEVAL_AUDIO / f"{stem}.mp3"
            if not p.exists():
                continue
            d = detect_audio(p)
            rc = _rec_of(d)
            rows.append({"stem": stem, "label": label,
                         "path": str(p.relative_to(WS)),
                         "p": d["p_fake_audio_mean"], "ooc": d["out_of_coverage"],
                         "case": "dfeval", **{
                             k: v for k, v in _case("dfeval", label,
                                                    str(p.relative_to(WS)),
                                                    d["p_fake_audio_mean"],
                                                    d).items() if k != "case"},
                         "coverage_fraction":
                             (rc.get("verdict") or {}).get("coverage_fraction")})
            if (i + 1) % 50 == 0:
                print(f"[audio] {i + 1} stems, {time.time() - t0:.0f}s",
                      flush=True)
    _write_cases(out, "audio_cases.csv", rows)
    agg = _aggregate(rows)
    agg["E3_conversion_always_reported"] = {
        "conversion_rate": agg["M1_conversion_rate"],
        "note": "reported even when 0 (honest zero passes)"}
    agg["ooc_pre_rate_from_csv"] = round(
        sum(1 for r in rows if r["ooc"]) / max(1, len(rows)), 4)
    print(f"[audio] n={agg['n']} triggered={agg['triggered']} "
          f"recovered={agg['recovered']} needs_review={agg['needs_review']} "
          f"M1={agg['M1_conversion_rate']}", flush=True)
    return agg, rows


# ---------------- --video-spotcheck ----------------

def exam_video_spot(out):
    import numpy as np
    from argus_detect import detect_video
    pre = json.loads((AB / "video_spot_pre.json").read_text())
    pre_rows = {v["file"]: v for v in pre["videos"]}
    real_dir = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / "real"
    fake_root = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / "fake"
    reals = sorted(real_dir.glob("*.mp4"))
    fakes = sorted(p for p in fake_root.rglob("*.mp4"))
    rng = np.random.default_rng(5)
    n_half = 15
    pick_r = [reals[i] for i in rng.choice(len(reals), size=n_half, replace=False)]
    pick_f = [fakes[i] for i in rng.choice(len(fakes), size=n_half, replace=False)]
    rows = []
    for label, paths in ((0, pick_r), (1, pick_f)):
        for p in paths:
            rel = str(p.relative_to(WS))
            d = detect_video(p)          # default n_frames=12, same as Stage-A pre
            rec = _rec_of(d)
            p_pre = (pre_rows.get(rel) or {}).get("p_fake_video_combined")
            rows.append({"case": "gnr4", "file": rel, "label": label,
                         "route": d.get("route"), "abstained": d.get("abstained"),
                         "identity_conflict": d.get("out_of_conflict"),
                         "p_pre": p_pre, "p_post": d.get("p_fake_video_combined"),
                         "p_identical": d.get("p_fake_video_combined") == p_pre,
                         "outcome": rec.get("outcome"),
                         "trigger": rec.get("trigger"),
                         "steps": [s["step"] for s in (rec.get("attempted_steps") or [])],
                         "reasons": rec.get("reasons"),
                         "per_identity_n": len((rec.get("verdict") or {}).get("per_identity") or []),
                         "recovery": rec})
    failures = []
    picked = {r["file"] for r in rows}
    if picked != set(pre_rows):
        failures.append(f"selection drift: {len(picked ^ set(pre_rows))} files "
                        f"differ from video_spot_pre.json")
    for r in rows:
        if r["outcome"] not in ("not_attempted", "needs_review"):
            failures.append(f"{r['file']}: outcome {r['outcome']!r} not in "
                            f'{{"not_attempted","needs_review"}}')
        if not r["p_identical"]:
            failures.append(f"{r['file']}: p {r['p_post']} != pre {r['p_pre']}")
        if r["outcome"] == "not_attempted":
            if r["steps"]:
                failures.append(f"{r['file']}: not_attempted with "
                                f"{len(r['steps'])} attempted_steps")
            if (r["recovery"].get("reasons") or []):
                failures.append(f"{r['file']}: not_attempted with reasons")
            if r["recovery"].get("verdict") is not None:
                failures.append(f"{r['file']}: not_attempted with a verdict "
                                f"(recovery-side effect)")
        if r["file"] in DOCUMENTED_IDENTITY_CONFLICT:
            if r["outcome"] != "needs_review":
                failures.append(f"{r['file']}: expected needs_review (V3), "
                                f"got {r['outcome']}")
            elif "V3" not in r["steps"]:
                failures.append(f"{r['file']}: needs_review without V3 step")
            elif not r["per_identity_n"]:
                failures.append(f"{r['file']}: V3 without per_identity populated")
    gnr4 = {"gate_text": G_NR4_AMENDED, "pass": not failures,
            "failures": failures, "n": len(rows),
            "identity_conflict_expected": sorted(DOCUMENTED_IDENTITY_CONFLICT)}

    f3p = real_dir / "953.mp4"
    f3 = detect_video(f3p)
    fixture_3 = {"path": str(f3p.relative_to(WS)),
                 "outcome": _rec_of(f3).get("outcome"),
                 "expect": "not_attempted",
                 "pass": _rec_of(f3).get("outcome") == "not_attempted"}
    f5p = WS / "artifacts" / "fixtures" / "no_face.mp4"
    f5 = detect_video(f5p)
    fixture_5 = {"path": str(f5p.relative_to(WS)),
                 "route": f5.get("route"),
                 "confidence_degraded": f5.get("confidence_degraded"),
                 "frames_court": f5.get("frames_court"),
                 "frames_scored": f5.get("frames_scored"),
                 "abstained": f5.get("abstained"),
                 "outcome": _rec_of(f5).get("outcome"),
                 "legibility_fields_present": all(
                     k in f5 for k in ("route", "confidence_degraded",
                                       "frames_court", "frames_scored"))}
    _write_cases(out, "video_spot_cases.csv", rows)
    print(f"[video-spotcheck] G-NR4 amended: {'PASS' if gnr4['pass'] else 'FAIL'} "
          f"({len(failures)} failures); fixture3={fixture_3['pass']} "
          f"fixture5 legible={fixture_5['legibility_fields_present']}",
          flush=True)
    return {"gnr4": gnr4, "fixture_3": fixture_3, "fixture_5": fixture_5}, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="store_true")
    ap.add_argument("--audio", action="store_true")
    ap.add_argument("--video-spotcheck", action="store_true")
    ap.add_argument("--out", default=str(WS / "artifacts" / "recovery_exam"))
    ap.add_argument("--force", action="store_true",
                    help="write into an output dir that already contains results")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    # stamped-out-dir guard copied verbatim from patterns.md:32
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force "
              "to overwrite", file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    if not (args.image or args.audio or args.video_spotcheck):
        print("ERROR: pick at least one of --image/--audio/--video-spotcheck",
              file=sys.stderr)
        return 1

    summary = {"schema": "argus.recovery_exam.v1",
               "generated": datetime.now(timezone.utc).isoformat(),
               "out": str(out)}
    all_recs = []
    t0 = time.time()
    if args.image:
        agg, rows = exam_image(out)
        summary["image"] = agg
        all_recs += [r["recovery"] for r in rows]
    if args.audio:
        agg, rows = exam_audio(out)
        summary["audio"] = agg
        all_recs += [r["recovery"] for r in rows]
    if args.video_spotcheck:
        spot, rows = exam_video_spot(out)
        summary["video_spotcheck"] = spot
        all_recs += [r["recovery"] for r in rows]

    summary["E1_penalty_audit"] = _e1_audit(all_recs)
    lat_p = AB / "latency_post.json"
    if lat_p.exists():
        try:
            summary["M4_latency_delta_ms"] = json.loads(lat_p.read_text())
        except Exception as e:
            summary["M4_latency_delta_ms"] = {"error": str(e)[:120]}
    summary["elapsed_s"] = round(time.time() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items()
                      if k in ("image", "audio", "video_spotcheck",
                               "E1_penalty_audit", "elapsed_s")}, indent=2)[:2000])
    print(f"\nsummary: {out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
