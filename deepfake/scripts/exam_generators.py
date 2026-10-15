"""Phase 4 Stage A — generator-exam measurement harness (spec A.4/A.5/A.6).

Scores the DEPLOYED court end-to-end on data/corpora/gen_exam/<slug>/{fake,real}:
loads the stacker bundle via argus_detect._image_stacker_clfs() (patterns
2026-10-11 mandatory) and calls argus_detect.detect_image(pil,
stacker_clfs=..., original_raw=file_bytes) so the full stack (11 channels +
stacker + recovery) runs. AUC/FPR conventions consistent with
restack.auc_from_scores (label 1=fake ranked by p_fake_fused).

Outputs (under --out, stamped default artifacts/generator_intake/):
  <slug>/scores.csv + <slug>/metrics.json   per generator
  parity_mj.json                            parity self-check vs p6v10 mj scoreboard
  recall_wildaigc/…                         P4-C24 recall-only section (never a
                                            decision input)
  per_generator_metrics.csv                 aggregate
  decision.json                             mechanical A.5 application (verbatim rule)

Per-slug metrics: fused AUC, FPR@0.5 on the slug's own real set, TPR@0.5,
abstain%, out_of_conflict%, recovery breakdown (not_attempted/recovered/
needs_review), top-3 module_evidence means, c2pa_present_frac,
provenance_nonzero_frac, and the verbatim P4-C25 note.

Parity self-check: modern_exam/fake_mj n=15 within ±0.02 of the last recorded
p6v10 mj scoreboard number (artifacts/scoreboard/p6v10/exam_mj/limit100).

Usage:
  .venv/bin/python deepfake/scripts/exam_generators.py [--smoke 5] [--limit 100]
  Runtime full batch ~2-3h -> nohup; log /tmp/gen_intake_exam.log.
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

import argus_detect  # noqa: E402  (after path inserts)

EXAM_ROOT = WS / "data" / "corpora" / "gen_exam"
RECALL_ROOT = WS / "data" / "corpora" / "gen_exam_recall"
ART_ROOT = WS / "artifacts" / "generator_intake"
MJ_PARITY_DIR = WS / "data" / "corpora" / "modern_exam" / "fake_mj"
REAL_EXT_DIR = WS / "data" / "corpora" / "modern_exam" / "real_ext"
PARITY_REF_JSON = WS / "artifacts" / "scoreboard" / "p6v10" / "exam_mj" / \
    "limit100" / "metrics.json"
PARITY_REF_SCORES = PARITY_REF_JSON.parent / "scores.csv"
PARITY_N = 15
PARITY_TOL = 0.02
# D-048 (user amendment): like-for-like subset comparison, not full n=100 stat
PARITY_GATE_TEXT = ("parity = harness vs reference-subset on identical images "
                    "at same n")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# P4-C25 verbatim — carried on every per-generator metrics row
PROV_NOTE = ("provenance evidence is +/- independent of the statistical "
             "channels; its contribution is additive when manifests exist "
             "(D-034), inert otherwise")

# A.5 rule text — VERBATIM, embedded into decision.json (no paraphrase)
RULE_TEXT = (
    "> **INTAKE CANDIDATE RULE (Stage B targeting).** A generator domain is an "
    "INTAKE CANDIDATE iff fused AUC < 0.90 on its exam (n≥100) OR FPR@0.5 > 0.10 "
    "on matched reals attributable to that generator family. Rank candidates by "
    "AUC shortfall. Local-gen diet is authorized for the top-K (K=2) candidates "
    "THAT HAVE OPEN WEIGHTS runnable on M1 Max per the research table "
    "(FLUX.2-klein 4B, Z-Image-Turbo primary; Kolors backup). Corpus-sourced "
    "train shards preferred over local-gen when a labeled train shard exists "
    "under an acceptable license for the same generator family. API-only "
    "generators with no train shard (GPT-Image-2, Nano Banana 2, Seedream 4.5, "
    "Grok) → recorded as UNREACHABLE-THIS-PHASE with numbers on record "
    "(D-037-style honest negative), extension slot for user-funded API + Treasure."
)

# A.5 operationalization (3): family mapping hardcoded
FAMILY_MAP = {
    "flux2": "FLUX.2", "zimage": "Z-Image", "kolors": "Kolors",
    "gpt4o": "GPT4O subset", "gptimage": "GPT4O subset",
}
OPEN_WEIGHTS = {"FLUX.2": "flux2-klein-4b", "Z-Image": "z-image-turbo",
                "Kolors": "kolors"}
AUC_BAR = 0.90   # P4-C07
FPR_BAR = 0.10   # P4-C07
K = 2            # P4-C08


def family_of(slug: str) -> str | None:
    s = slug.lower()
    for key, fam in FAMILY_MAP.items():
        if key in s:
            return fam
    if s.startswith("ideogram") or "ideogram" in s:
        return "ideogram-27k subset"
    if s.startswith("aigen26"):
        return "aigenimages2026 shard"
    return None


# --------------------------------------------------------------------------

def load_bundle(stacker_path: Path) -> dict:
    """Stacker bundle for detect_image.

    Default (deployed court): argus_detect._image_stacker_clfs() — the ONE
    construction (patterns 2026-10-11). A non-default --stacker is an A/B
    override and builds the same-shaped dict here (never edits argus_detect).
    """
    if Path(stacker_path) == argus_detect.STACKER:
        return argus_detect._image_stacker_clfs()
    import joblib
    sp = Path(stacker_path)
    if not sp.exists():
        print(f"WARN: --stacker {sp} missing — scoring with empty bundle",
              file=sys.stderr)
        return {}
    meta_p = sp.with_name("stacker_meta.json")
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
    return {sp.parent.name: {"clf": joblib.load(sp),
                             "module_ids": meta.get("module_ids"),
                             "feature_names": meta.get("feature_names"),
                             "conflict_p99": meta.get("conflict_p99")}}


def score_file(path: Path, label: int, slug: str, bundle: dict) -> dict:
    from PIL import Image
    raw = path.read_bytes()
    pil = Image.open(path)
    pil.load()
    pil = pil.convert("RGB")
    t0 = time.time()
    res = argus_detect.detect_image(pil, stacker_clfs=bundle, original_raw=raw)
    ms = (time.time() - t0) * 1000.0
    rec = res.get("recovery") or {}
    return {"slug": slug, "file": path.name, "label": label,
            "p_fake_fused": res["p_fake_fused"], "abstained": bool(res["abstained"]),
            "out_of_conflict": bool(res.get("out_of_conflict", False)),
            "trust": res.get("trust"), "risk": res.get("risk"),
            "recovery_outcome": rec.get("outcome", "missing"),
            "runtime_ms": round(ms, 1),
            **{f"ev_{k}": v for k, v in (res.get("module_evidence") or {}).items()}}


def score_slug(slug: Path, bundle: dict, out: Path, limit: int,
               force: bool, smoke: int | None) -> list[dict]:
    """Score one exam slug; append to scores.csv (resume-safe, dir-qualified)."""
    sname = slug.name
    sdir = out / sname
    sdir.mkdir(parents=True, exist_ok=True)
    csv_p = sdir / "scores.csv"
    done = set()
    rows: list[dict] = []
    if csv_p.exists() and not force:
        rows = list(csv.DictReader(open(csv_p)))
        done = {(r["file"], r["label"]) for r in rows}
    elif csv_p.exists() and force:
        # --force resumes/appends: keep scored rows, re-score only missing
        rows = list(csv.DictReader(open(csv_p)))
        done = {(r["file"], r["label"]) for r in rows}
    fresh = not csv_p.exists() or csv_p.stat().st_size == 0
    fh = open(csv_p, "a" if not fresh else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=None, extrasaction="ignore")
    n_done = 0
    for label, sub in ((1, "fake"), (0, "real")):
        d = slug / sub
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if smoke:
            files = files[:smoke]
        else:
            files = files[:limit]
        for p in files:
            if (p.name, str(label)) in done:
                continue
            rec = score_file(p, label, sname, bundle)
            if writer.fieldnames is None:
                writer.fieldnames = list(rec.keys())
                if fresh:
                    writer.writeheader()
            writer.writerow(rec)
            fh.flush()
            rows.append(rec)
            n_done += 1
            if n_done % 25 == 0:
                print(f"  [{sname}] {n_done} scored (last {p.name} "
                      f"p={rec['p_fake_fused']})", flush=True)
    fh.close()
    return rows


def slug_metrics(rows: list[dict]) -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    fakes = [r for r in rows if int(r["label"]) == 1]
    reals = [r for r in rows if int(r["label"]) == 0]
    m = {"n_fake": len(fakes), "n_real": len(reals)}
    if fakes and reals:
        y = np.array([1] * len(fakes) + [0] * len(reals))
        s = np.array([float(r["p_fake_fused"]) for r in fakes + reals])
        m["auc"] = round(float(roc_auc_score(y, s)), 4)
    else:
        m["auc"] = None
    pf = [float(r["p_fake_fused"]) for r in fakes]
    pr = [float(r["p_fake_fused"]) for r in reals]
    m["tpr_at_0.5"] = round(sum(v >= 0.5 for v in pf) / max(1, len(pf)), 4)
    m["fpr_at_0.5"] = round(sum(v >= 0.5 for v in pr) / max(1, len(pr)), 4)
    allr = fakes + reals
    m["abstain_frac"] = round(sum(str(r["abstained"]).lower() in
                                  ("true", "1") for r in allr) / max(1, len(allr)), 4)
    m["out_of_conflict_frac"] = round(sum(str(r["out_of_conflict"]).lower() in
                                         ("true", "1") for r in allr) /
                                     max(1, len(allr)), 4)
    # recovery breakdown
    for outcome in ("not_attempted", "recovered", "needs_review", "missing"):
        m[f"recovery_{outcome}"] = sum(r.get("recovery_outcome") == outcome
                                       for r in allr)
    # top-3 module evidence means (on fakes)
    ev_cols = [c for c in (fakes[0].keys() if fakes else []) if c.startswith("ev_")]
    means = {}
    for c in ev_cols:
        vals = [float(r[c]) for r in fakes if r.get(c) not in (None, "")]
        if vals:
            means[c[3:]] = round(sum(vals) / len(vals), 4)
    top3 = sorted(means.items(), key=lambda kv: -abs(kv[1]))[:3]
    m["top3_module_evidence_means"] = dict(top3)
    # provenance (P4-C25): probe nonzero <=> manifest present (D-034 design)
    prov = [float(r.get("ev_provenance_probe") or 0.0) for r in allr]
    m["c2pa_present_frac"] = round(sum(v != 0.0 for v in prov) / max(1, len(prov)), 4)
    m["provenance_nonzero_frac"] = m["c2pa_present_frac"]
    m["provenance_note"] = PROV_NOTE  # P4-C25 verbatim
    return m


def parity_selfcheck(bundle: dict, out: Path, force: bool) -> dict:
    """D-048 amended: harness vs reference-subset on identical images at same n.

    Compares the harness n=15 AUC against the reference scoreboard scores.csv
    RESTRICTED to the identical first-15 sorted images (like-for-like subset),
    not the full n=100 statistic. Per-image identity (1e-4) is the primary
    court-reproduction evidence and is recorded alongside.
    """
    ref_p = out / "parity_mj.json"
    if ref_p.exists() and not force:
        return json.loads(ref_p.read_text())
    if not PARITY_REF_SCORES.exists():
        res = {"status": "FAIL",
               "reason": f"reference scores missing: {PARITY_REF_SCORES}",
               "gate_text": PARITY_GATE_TEXT}
        ref_p.write_text(json.dumps(res, indent=2))
        return res
    import numpy as np
    from sklearn.metrics import roc_auc_score

    fakes = sorted(MJ_PARITY_DIR.glob("*.jpg"))[:PARITY_N]
    reals = sorted(REAL_EXT_DIR.glob("*.jpg"))[:PARITY_N]

    # reference rows: clean variant, keyed (label, basename) like the harness
    ref_rows = {}
    for r in csv.DictReader(open(PARITY_REF_SCORES)):
        if r.get("variant") == "clean" and r.get("label") in ("0", "1"):
            ref_rows[(r["label"], Path(r["image"]).name)] = r

    rows, ref_subset = [], []
    per_image_match, per_image_n, per_image_diffs = 0, 0, []
    for p in fakes:
        rec = score_file(p, 1, "parity_mj", bundle)
        rows.append(rec)
        rr = ref_rows.get(("1", p.name))
        if rr is not None:
            per_image_n += 1
            ref_p_f = 1 - float(rr["p_real_p6v10"])
            if abs(ref_p_f - rec["p_fake_fused"]) <= 1e-4:
                per_image_match += 1
            else:
                per_image_diffs.append(
                    {"file": p.name, "harness": rec["p_fake_fused"],
                     "reference": round(ref_p_f, 6)})
            ref_subset.append((1, ref_p_f))
    for p in reals:
        rec = score_file(p, 0, "parity_mj", bundle)
        rows.append(rec)
        rr = ref_rows.get(("0", p.name))
        if rr is not None:
            per_image_n += 1
            ref_p_f = 1 - float(rr["p_real_p6v10"])
            if abs(ref_p_f - rec["p_fake_fused"]) <= 1e-4:
                per_image_match += 1
            else:
                per_image_diffs.append(
                    {"file": p.name, "harness": rec["p_fake_fused"],
                     "reference": round(ref_p_f, 6)})
            ref_subset.append((0, ref_p_f))

    m = slug_metrics(rows)
    if len(ref_subset) == PARITY_N * 2:
        y = np.array([lab for lab, _ in ref_subset])
        s = np.array([p for _, p in ref_subset])
        ref_auc = round(float(roc_auc_score(y, s)), 4)
    else:
        ref_auc = None
    delta = abs(m["auc"] - ref_auc) if (m["auc"] is not None and
                                        ref_auc is not None) else None
    res = {"status": "PASS" if delta is not None and delta <= PARITY_TOL
           else "FAIL",
           "gate_text": PARITY_GATE_TEXT,
           "n_fake": len(fakes), "n_real": len(reals),
           "measured_auc": m["auc"], "reference_subset_auc": ref_auc,
           "abs_delta": round(delta, 4) if delta is not None else None,
           "tolerance": PARITY_TOL,
           # primary court-reproduction evidence (per-image identity)
           "per_image_identity": {
               "matched": per_image_match, "n": per_image_n,
               "tol": 1e-4,
               "diffs": per_image_diffs},
           "reference": str(PARITY_REF_SCORES.relative_to(WS)) +
           f" (clean variant, first-{PARITY_N} sorted images per class, "
           f"p_fake = 1 - p_real_p6v10)"}
    (out / "parity_scores.csv").write_text(
        "file,label,p_fake_fused\n" + "\n".join(
            f"{r['file']},{r['label']},{r['p_fake_fused']}" for r in rows))
    ref_p.write_text(json.dumps(res, indent=2))
    print(f"[parity] {PARITY_GATE_TEXT}: measured {m['auc']} vs "
          f"ref-subset {ref_auc} (|d|={res['abs_delta']}) -> {res['status']}; "
          f"per-image {per_image_match}/{per_image_n} @1e-4", flush=True)
    return res


def score_recall_wildaigc(bundle: dict, out: Path, limit: int,
                          force: bool) -> dict:
    """P4-C24: recall-only probe. Positive-only set — reports recall/TPR,
    abstain, recovery; NEVER an AUC/FPR/decision input."""
    root = RECALL_ROOT / "wildaigc" / "fake"
    if not root.exists():
        return {"status": "ABSENT"}
    sdir = out / "recall_wildaigc"
    sdir.mkdir(parents=True, exist_ok=True)
    csv_p = sdir / "scores.csv"
    done = set()
    rows = []
    if csv_p.exists():
        rows = list(csv.DictReader(open(csv_p)))
        done = {r["file"] for r in rows}
    fresh = not csv_p.exists() or csv_p.stat().st_size == 0
    fh = open(csv_p, "a" if not fresh else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=None, extrasaction="ignore")
    files = sorted(p for p in root.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)[:limit]
    for p in files:
        if p.name in done:
            continue
        rec = score_file(p, 1, "wildaigc_recall", bundle)
        if writer.fieldnames is None:
            writer.fieldnames = list(rec.keys())
            if fresh:
                writer.writeheader()
        writer.writerow(rec)
        fh.flush()
        rows.append(rec)
    fh.close()
    pf = [float(r["p_fake_fused"]) for r in rows]
    res = {"status": "SCORED", "n": len(rows),
           "recall_at_0.5": round(sum(v >= 0.5 for v in pf) / max(1, len(pf)), 4),
           "abstain_frac": round(sum(str(r["abstained"]).lower() in ("true", "1")
                                     for r in rows) / max(1, len(rows)), 4),
           "recovery": {o: sum(r.get("recovery_outcome") == o for r in rows)
                        for o in ("not_attempted", "recovered", "needs_review",
                                  "missing")},
           "note": "P4-C24 recall-only — never enters decision rule/training/"
                   "thresholding; positive-only set, no AUC/FPR by construction"}
    (sdir / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def score_real_ext(bundle: dict, out: Path, limit: int, force: bool) -> dict:
    """Reference-real FPR (real_ext) — reported alongside EVERY slug (A.5 op 1)."""
    p = out / "real_ext_reference.json"
    if p.exists() and not force:
        return json.loads(p.read_text())
    if not REAL_EXT_DIR.exists():
        res = {"status": "ABSENT"}
        p.write_text(json.dumps(res, indent=2))
        return res
    sdir = out / "real_ext_reference"
    sdir.mkdir(parents=True, exist_ok=True)
    csv_p = sdir / "scores.csv"
    done = set()
    rows = []
    if csv_p.exists():
        rows = list(csv.DictReader(open(csv_p)))
        done = {r["file"] for r in rows}
    fresh = not csv_p.exists() or csv_p.stat().st_size == 0
    fh = open(csv_p, "a" if not fresh else "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=None, extrasaction="ignore")
    for fp in sorted(REAL_EXT_DIR.glob("*.jpg"))[:limit]:
        if fp.name in done:
            continue
        rec = score_file(fp, 0, "real_ext_reference", bundle)
        if writer.fieldnames is None:
            writer.fieldnames = list(rec.keys())
            if fresh:
                writer.writeheader()
        writer.writerow(rec)
        fh.flush()
        rows.append(rec)
    fh.close()
    pr = [float(r["p_fake_fused"]) for r in rows]
    res = {"status": "SCORED", "n": len(rows),
           "fpr_at_0.5": round(sum(v >= 0.5 for v in pr) / max(1, len(pr)), 4),
           "abstain_frac": round(sum(str(r["abstained"]).lower() in ("true", "1")
                                     for r in rows) / max(1, len(rows)), 4)}
    p.write_text(json.dumps(res, indent=2))
    print(f"[real_ext] reference FPR@0.5={res['fpr_at_0.5']} (n={res['n']})",
          flush=True)
    return res


def aggregate(out: Path, parity: dict, recall: dict,
              ref: dict | None = None) -> tuple[list[dict], Path]:
    """Per-slug metrics -> per_generator_metrics.csv (+ metrics.json per slug)."""
    agg = []
    for sdir in sorted(out.iterdir()):
        if not sdir.is_dir() or sdir.name.startswith(("recall_", "real_ext")):
            continue
        csv_p = sdir / "scores.csv"
        if not csv_p.exists():
            continue
        rows = list(csv.DictReader(open(csv_p)))
        if not rows:
            continue
        m = slug_metrics(rows)
        m["slug"] = sdir.name
        m["family"] = family_of(sdir.name)
        m["parity_status"] = parity.get("status")
        if ref:
            m["fpr_real_ext_ref"] = ref.get("fpr_at_0.5")
        (sdir / "metrics.json").write_text(json.dumps(m, indent=2))
        agg.append(m)
    cols = ["slug", "family", "n_fake", "n_real", "auc", "fpr_at_0.5",
            "fpr_real_ext_ref", "tpr_at_0.5", "abstain_frac", "out_of_conflict_frac",
            "recovery_not_attempted", "recovery_recovered",
            "recovery_needs_review", "recovery_missing",
            "c2pa_present_frac", "provenance_nonzero_frac",
            "top3_module_evidence_means", "provenance_note"]
    csv_p = out / "per_generator_metrics.csv"
    with open(csv_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for m in sorted(agg, key=lambda x: x["slug"]):
            row = dict(m)
            row["top3_module_evidence_means"] = json.dumps(
                m.get("top3_module_evidence_means", {}))
            w.writerow(row)
    return agg, csv_p


def decide(agg: list[dict], out: Path, parity: dict, recall: dict) -> dict:
    """Mechanical A.5 application. Candidates derivable from
    per_generator_metrics.csv alone. WildAIGC absent from decision inputs."""
    candidates, underpowered, cleared = [], [], []
    for m in sorted(agg, key=lambda x: x["slug"]):
        slug, auc, fpr, n = m["slug"], m.get("auc"), m.get("fpr_at_0.5"), m["n_fake"]
        if n < 100:  # operationalization (2)
            underpowered.append({"slug": slug, "n": n,
                                 "status": "UNDERPOWERED",
                                 "note": "not candidate, not cleared"})
            continue
        fail_auc = auc is not None and auc < AUC_BAR
        fail_fpr = fpr is not None and fpr > FPR_BAR  # operationalization (4)
        shortfall = round(AUC_BAR - auc, 4) if auc is not None else None
        rec = {"slug": slug, "n": n, "auc": auc, "fpr_at_0.5": fpr,
               "family": m.get("family"), "auc_shortfall": shortfall,
               "fail_auc": fail_auc, "fail_fpr": fail_fpr}
        if fail_auc or fail_fpr:
            candidates.append(rec)
        else:
            cleared.append(rec)
    candidates.sort(key=lambda r: (-(r["auc_shortfall"] or 0.0),
                                   -(r["fpr_at_0.5"] or 0.0)))
    for i, c in enumerate(candidates, 1):
        c["rank"] = i

    # top-K open-weights local-gen trigger (K=2, P4-C08; recorded, NOT executed)
    triggered, unreachable_slot = [], []
    for c in candidates[:K]:
        fam = c.get("family")
        if fam and fam in OPEN_WEIGHTS:
            triggered.append({"slug": c["slug"], "family": fam,
                              "model": OPEN_WEIGHTS[fam],
                              "route": "LOCAL-GEN (conditional — Stage B, "
                                       "user checkpoint)"})
        else:
            unreachable_slot.append(
                {"slug": c["slug"], "family": fam,
                 "note": "no open-weights entry in research table; "
                         "corpus shard preference / UNREACHABLE per rule"})

    # shard-preference resolutions (rule sentence 3)
    shard_prefs = []
    for c in candidates:
        fam = c.get("family")
        if fam == "aigenimages2026 shard":
            shard_prefs.append({"slug": c["slug"],
                                "resolution": "CORPUS-SHARD PREFERRED "
                                "(aigenimages2026, CC-BY-4.0, on disk)"})
        elif fam == "ideogram-27k subset":
            shard_prefs.append({"slug": c["slug"],
                                "resolution": "SHARD UNAVAILABLE "
                                "(LICENSE_UNRESOLVED — not downloaded)"})
        elif fam == "GPT4O subset":
            shard_prefs.append({"slug": c["slug"],
                                "resolution": "SHARD UNAVAILABLE "
                                "(LICENSE_UNRESOLVED — not downloaded)"})

    decision = {
        "stage": "A",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule_text": RULE_TEXT,  # VERBATIM per A.5
        "thresholds": {"auc": AUC_BAR, "fpr_at_0.5": FPR_BAR},
        "k": K,
        "operationalizations": [
            "FPR clause = FPR@0.5 on slug's own exam real set; reference-real "
            "FPR (real_ext) reported alongside every slug",
            "n<100 slugs -> UNDERPOWERED (not candidate, not cleared)",
            "family mapping hardcoded in generator field",
            "FPR-only failure still a candidate",
            "ZERO candidates -> NO_CANDIDATES; Stage B is a NO-OP pending user"],
        "family_map": FAMILY_MAP,
        "candidates_ranked": candidates,
        "underpowered": underpowered,
        "cleared": cleared,
        "status": ("CANDIDATES" if candidates else "NO_CANDIDATES"),
        "local_gen_trigger_fired": bool(triggered),
        "local_gen_models": [t["model"] for t in triggered],
        "local_gen_detail": triggered,
        "no_open_weights_topk": unreachable_slot,
        "shard_preference_resolutions": shard_prefs,
        "license_unresolved": [],   # filled from acquisition.json below
        "repo_id_unverified": [],
        "unreachable_this_phase": [
            {"generator": g, "status": "UNREACHABLE-THIS-PHASE",
             "numbers_on_record": None,
             "note": "API-only, no train shard; D-037-style honest negative — "
                     "no exam this stage (no cleared corpus); extension slot "
                     "for user-funded API + Treasure"}
            for g in ("GPT-Image-2", "Nano Banana 2", "Seedream 4.5", "Grok")],
        "dep_pins": {},  # placeholder — Stage B, post user checkpoint
        "p4_c23_status": "EXCLUDE (default) — OmniFake-MJ not in intake set",
        "parity_selfcheck": parity,
        "wildaigc_recall": recall,
        "decision_inputs_note": "candidates derived from "
                                "per_generator_metrics.csv alone; WildAIGC "
                                "recall-only (P4-C24) is NOT a decision input",
        "stage_b_status": "NOT STARTED — user checkpoint required "
                          "(no mflux install, no local-gen, no restack edits, "
                          "no gen_intake_train materialization)",
    }
    # merge acquisition outcomes (LICENSE_UNRESOLVED / REPO_ID_UNVERIFIED)
    acq_p = ART_ROOT / "acquisition.json"
    if acq_p.exists():
        acq = json.loads(acq_p.read_text())
        for cid, c in (acq.get("checks") or {}).items():
            if c.get("status") == "LICENSE_UNRESOLVED":
                decision["license_unresolved"].append(
                    {"corpus_id": cid, "repo": c.get("repo"),
                     "note": c.get("note"), "downloaded": False,
                     "on_disk": False})
            elif c.get("status") in ("REPO_ID_UNVERIFIED", "REPO_ID_GATED"):
                decision["repo_id_unverified"].append(
                    {"corpus_id": cid, "repo": c.get("repo"),
                     "status": c.get("status"), "note": c.get("note"),
                     "downloaded": False})
    (out / "decision.json").write_text(json.dumps(decision, indent=2))
    print(f"[decide] {decision['status']}: {len(candidates)} candidates, "
          f"{len(underpowered)} underpowered, {len(cleared)} cleared; "
          f"local_gen_trigger_fired={decision['local_gen_trigger_fired']} "
          f"models={decision['local_gen_models']}", flush=True)
    return decision


def write_report(out: Path, agg: list[dict], decision: dict, parity: dict,
                 recall: dict, wall_s: float, smoke: bool,
                 ref: dict | None = None) -> Path:
    pi = parity.get("per_image_identity") or {}
    L = ["# Stage A Report — Generator Intake (Phase 4)", "",
         f"- generated: {datetime.now(timezone.utc).isoformat()}",
         f"- stacker: {argus_detect.STACKER.relative_to(WS)}",
         f"- runtime: {wall_s/60:.1f} min" + (" (SMOKE)" if smoke else ""),
         f"- parity gate (D-048): {parity.get('gate_text', PARITY_GATE_TEXT)}",
         f"- parity self-check (mj n={PARITY_N}): {parity.get('status')} "
         f"(measured {parity.get('measured_auc')} vs reference-subset "
         f"{parity.get('reference_subset_auc')}, "
         f"|Δ|={parity.get('abs_delta')}, tol {PARITY_TOL})",
         f"- per-image court-reproduction evidence: "
         f"{pi.get('matched')}/{pi.get('n')} match at {pi.get('tol')}",
         f"- reference-real FPR@0.5 (real_ext): "
         f"{(ref or {}).get('fpr_at_0.5')} (n={(ref or {}).get('n')}) "
         f"— reported alongside every slug", "",
         "## Per-generator metrics", "",
         "| slug | n_fake | n_real | AUC | FPR@0.5 | FPR real_ext | TPR@0.5 | "
         "abstain | ooc | recov n/r/needs | c2pa | prov≠0 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in sorted(agg, key=lambda x: x["slug"]):
        L.append(
            f"| {m['slug']} | {m['n_fake']} | {m['n_real']} | {m.get('auc')} "
            f"| {m.get('fpr_at_0.5')} | {m.get('fpr_real_ext_ref')} "
            f"| {m.get('tpr_at_0.5')} "
            f"| {m.get('abstain_frac')} | {m.get('out_of_conflict_frac')} "
            f"| {m.get('recovery_not_attempted')}/{m.get('recovery_recovered')}"
            f"/{m.get('recovery_needs_review')} | {m.get('c2pa_present_frac')} "
            f"| {m.get('provenance_nonzero_frac')} |")
    L += ["", f"Provenance note (P4-C25, every row): {PROV_NOTE}", "",
          "## Decision (mechanical A.5)", "",
          f"- status: **{decision['status']}**",
          f"- candidates: {[c['slug'] for c in decision['candidates_ranked']]}",
          f"- underpowered: {[u['slug'] for u in decision['underpowered']]}",
          f"- local-gen trigger fired: {decision['local_gen_trigger_fired']} "
          f"({decision['local_gen_models']}) — RECORD ONLY this stage", "",
          "## UNREACHABLE-THIS-PHASE (API-only, honest negatives)", ""]
    for u in decision["unreachable_this_phase"]:
        L.append(f"- {u['generator']}: {u['status']} — {u['note']}")
    L += ["", "## UNDERPOWERED (n<100)", ""]
    for u in decision["underpowered"]:
        L.append(f"- {u['slug']}: n={u['n']}")
    L += ["", "## Recall-only WildAIGC (P4-C24)", "",
          f"- {json.dumps(recall)}",
          "- NEVER enters decision rule/training/thresholding.", "",
          "## Exclusions (recorded, not downloaded)", ""]
    for e in decision.get("license_unresolved", []):
        L.append(f"- LICENSE_UNRESOLVED: {e['corpus_id']} ({e['repo']}) — "
                 f"{e['note']}")
    for e in decision.get("repo_id_unverified", []):
        L.append(f"- {e['status']}: {e['corpus_id']} ({e['repo']}) — "
                 f"{e.get('note')}")
    L += ["", "Full decision: decision.json (rule_text verbatim). "
          "Stage B NOT started — user checkpoint.", ""]
    p = out / "stage_a_report.md"
    p.write_text("\n".join(L))
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacker", default=str(argus_detect.STACKER))
    ap.add_argument("--exam-root", default=str(EXAM_ROOT))
    ap.add_argument("--out", default=None,
                    help="stamped default: artifacts/generator_intake/"
                         "exam_<stacker>_<YYYYMMDD>")
    ap.add_argument("--force", action="store_true",
                    help="resume/append (re-scores only missing rows)")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--smoke", type=int, default=None,
                    help="P4-C13: n imgs per class per slug + parity")
    ap.add_argument("--skip-recall", action="store_true")
    ap.add_argument("--skip-decide", action="store_true")
    args = ap.parse_args()

    stacker_tag = Path(args.stacker).parent.name
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = Path(args.out) if args.out else ART_ROOT / f"exam_{stacker_tag}_{day}"
    # stamped out-dir guard verbatim (patterns 2026-10-10)
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: {out} is non-empty — pass --force to resume/append",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    exam_root = Path(args.exam_root)
    slugs = sorted(p for p in exam_root.iterdir()
                   if p.is_dir() and (p / "fake").exists())
    if not slugs:
        print(f"FAIL: no exam slugs under {exam_root} "
              f"(run build_gen_exam.py first)", file=sys.stderr)
        return 1

    print(f"[harness] stacker={stacker_tag} slugs={len(slugs)} "
          f"smoke={args.smoke} limit={args.limit} out={out}", flush=True)
    bundle = load_bundle(Path(args.stacker))
    print(f"[harness] bundle keys: {list(bundle.keys())}", flush=True)

    t0 = time.time()
    parity = parity_selfcheck(bundle, out, args.force)
    if parity.get("status") == "FAIL" and not args.force:
        # parity miss = escalate (spec: report, don't improvise); still allow
        # smoke to proceed so the user sees the decomposition
        print(f"WARN: parity self-check FAIL: {parity}", file=sys.stderr)

    recall = {"status": "SKIPPED"}
    if not args.skip_recall:
        recall = score_recall_wildaigc(bundle, out,
                                       args.smoke or args.limit, args.force)
        print(f"[recall] wildaigc: {recall.get('status')} "
              f"n={recall.get('n')} recall@0.5={recall.get('recall_at_0.5')}",
              flush=True)

    ref = score_real_ext(bundle, out, args.smoke or args.limit, args.force)

    for slug in slugs:
        rows = score_slug(slug, bundle, out, args.limit, args.force, args.smoke)
        m = slug_metrics(rows)
        print(f"[{slug.name}] n_fake={m['n_fake']} n_real={m['n_real']} "
              f"auc={m['auc']} fpr@0.5={m['fpr_at_0.5']} "
              f"abstain={m['abstain_frac']}", flush=True)

    wall = time.time() - t0
    agg, csv_p = aggregate(out, parity, recall, ref)
    print(f"[aggregate] {len(agg)} slugs -> {csv_p}", flush=True)
    decision = {}
    if not args.skip_decide and not args.smoke:
        decision = decide(agg, out, parity, recall)
        # attach reference-real FPR (op-1) to decision inputs on record
        decision["fpr_real_ext_reference"] = ref
        (out / "decision.json").write_text(json.dumps(decision, indent=2))
    elif args.smoke:
        print("[decide] skipped in --smoke mode", flush=True)
    write_report(out, agg, decision or {"status": "SMOKE", "candidates_ranked": [],
                                        "underpowered": [], "unreachable_this_phase": [],
                                        "license_unresolved": [], "repo_id_unverified": [],
                                        "local_gen_trigger_fired": False,
                                        "local_gen_models": []},
                 parity, recall, wall, bool(args.smoke), ref)
    print(f"[done] wall {wall/60:.1f} min -> {out}", flush=True)
    if parity.get("status") == "FAIL":
        print("PARITY MISS — escalate to user (spec: report, don't improvise)",
              file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
