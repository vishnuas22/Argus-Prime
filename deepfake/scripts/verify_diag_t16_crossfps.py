#!/usr/bin/env python3
"""Hardened verifier for DIAG_T16_CROSSFPS.

The sealed exam (`diag_t16_crossfps.py`) is preserved byte-exact at
artifacts/split_t2v_matched/diag_t16_crossfps/exam_sealed.py.txt and is NOT
edited. Review found four invariants the as-sealed `--verdict` path did not
enforce; this verifier enforces them independently:

  V1  the recorded input hashes (stage0_scores, v1_scores, v1_splits) still match
  V2  exactly 1,000 negatives, arm `stage0_real`, label 0, all scored
  V3  every score row's payload (path, label, sha256) matches rows.json,
      not only its key
  V4  the recipe hash equals the Stage 0 seal's recipe hash

It then recomputes the primary AUC with an independent Mann-Whitney
implementation (ties averaged) and writes verify.json.

Report-only. 0 booked. Never imported by serving code.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "deepfake" / "bench"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

DEFAULT_OUT = ROOT / "artifacts" / "split_t2v_matched" / "diag_t16_crossfps"
STAGE0_DIR = ROOT / "artifacts" / "split_t2v_matched" / "stage0_t16"
N_NEG = 1000
PRIMARY_BAND_CUT = 0.70


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def parse_seal(path: Path) -> dict:
    """Line-based seal: `name value [path]`. Keeps the path when present."""
    out = {}
    for line in path.read_text().splitlines():
        parts = line.split(" ", 2)
        if not parts or not parts[0] or parts[0].startswith("#"):
            continue
        if parts[0] == "DIAG_T16_CROSSFPS":
            continue
        if len(parts) == 3:
            out[parts[0]] = f"{parts[1]} {parts[2]}"
        elif len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def load_seal_any(path: Path) -> dict:
    """Stage 0's seal is JSON; this run's is line-based."""
    text = path.read_text()
    if text.lstrip().startswith("{"):
        return json.loads(text)
    return parse_seal(path)


def mann_whitney_auc(fakes, reals) -> float:
    """AUC = P(fake > real) with ties counted 1/2, via average ranks."""
    from scipy.stats import rankdata
    f = np.asarray(fakes, dtype=np.float64)
    r = np.asarray(reals, dtype=np.float64)
    if f.size == 0 or r.size == 0:
        raise ValueError("empty arm")
    ranks = rankdata(np.concatenate([f, r]))
    r1 = ranks[:f.size].sum()
    return float((r1 - f.size * (f.size + 1) / 2.0) / (f.size * r.size))


def check_input_hashes(seal: dict) -> list:
    """V1: every sealed input path must still hash to the sealed value."""
    problems = []
    for name in ("v1_splits", "v1_scores", "stage0_scores"):
        recorded = seal.get(name)
        if not recorded:
            problems.append(f"{name}: not recorded in seal")
            continue
        sha, path = recorded.split(" ", 1) if " " in recorded else (recorded, "")
        p = Path(path) if path else None
        if p is None or not p.exists():
            problems.append(f"{name}: missing {path}")
            continue
        if _sha256(p) != sha:
            problems.append(f"{name}: hash drift")
    return problems


def check_rows(doc: dict, score_rows: list) -> list:
    """V3 + recipe: exact keys, payload equality, unique keys, frozen recipe."""
    problems = []
    want = {(r["arm"], r["generator"], r["filename"]): r for r in doc["rows"]}
    got = {}
    for r in score_rows:
        k = (r["arm"], r["generator"], r["filename"])
        if k in got:
            problems.append(f"duplicate score key {k}")
        got[k] = r
    if set(want) != set(got):
        problems.append(f"row set mismatch: {len(set(got) - set(want))} foreign, "
                        f"{len(set(want) - set(got))} missing")
    recipe = doc.get("recipe_hash")
    for k in sorted(set(want) & set(got)):
        w, g = want[k], got[k]
        for field in ("sha256", "path", "label"):
            if w.get(field) != g.get(field):
                problems.append(f"payload mismatch {k} field={field}")
        if g.get("recipe_hash") != recipe:
            problems.append(f"score recipe mismatch {k}")
    return problems


def check_negatives(neg_rows: list, recipe: str, min_n: int = N_NEG) -> list:
    """V2: exactly N_NEG sealed negatives, label 0, all scored."""
    problems = []
    if len(neg_rows) != min_n:
        problems.append(f"negatives n={len(neg_rows)} != {min_n}")
    if any(r.get("label") != 0 for r in neg_rows):
        problems.append("negatives contain label != 0")
    if any(r.get("score") is None for r in neg_rows):
        problems.append("negatives contain unavailable rows")
    if any(r.get("recipe_hash") != recipe for r in neg_rows):
        problems.append("negatives scored under a different recipe")
    return problems


def check_stage0_recipe(seal: dict) -> list:
    """V4: this run's recipe must equal the Stage 0 seal's recipe."""
    problems = []
    s0 = load_seal_any(STAGE0_DIR / "seal.txt")
    if s0.get("recipe_hash") != _recipe_of(seal):
        problems.append("recipe_hash != Stage 0 seal recipe_hash")
    return problems


def _recipe_of(seal: dict) -> str:
    """Strip a trailing path from a line-based seal value."""
    v = seal.get("recipe_hash") or ""
    return v.split(" ")[0]


def verify(out: Path, stage0_path: Path | None = None) -> tuple:
    seal = parse_seal(out / "seal.txt")
    doc = _load_json(out / "rows.json")
    scores = _load_jsonl(out / "scores.jsonl")
    problems = []

    # sealed exam bytes must be preserved and match the seal
    preserved = out / "exam_sealed.py.txt"
    if not preserved.exists():
        problems.append("exam_sealed.py.txt missing (sealed harness not preserved)")
    elif _sha256(preserved) != seal.get("exam"):
        problems.append("exam_sealed.py.txt != seal exam hash")

    problems += check_input_hashes(seal)
    problems += check_rows(doc, scores)
    problems += check_stage0_recipe(seal)

    s0 = _load_jsonl(stage0_path or STAGE0_DIR / "scores.jsonl")
    neg = [r for r in s0 if r.get("arm") == "stage0_real"]
    problems += check_negatives(neg, seal.get("recipe_hash"))

    fakes = [r["score"] for r in scores
             if r["arm"] == "genvid_fake" and r.get("score") is not None]
    reals = [r["score"] for r in neg if r.get("score") is not None]
    auc = mann_whitney_auc(fakes, reals) if fakes and reals else None

    verdict = _load_json(out / "verdict.json")
    if auc is not None and abs(auc - float(verdict["primary_auc"])) > 1e-9:
        problems.append(f"recomputed AUC {auc} != verdict {verdict['primary_auc']}")
    if auc is not None and auc < PRIMARY_BAND_CUT and \
            verdict["band"] != "NO_TRANSFER_EVIDENCE":
        problems.append("band inconsistent with recomputed AUC")
    if verdict.get("booked") != 0:
        problems.append("booked != 0")

    seen = set()
    dedup = []
    for r in sorted(scores, key=lambda x: x["filename"]):
        if r["arm"] != "genvid_fake" or r.get("score") is None:
            continue
        if r["sha256"] in seen:
            continue
        seen.add(r["sha256"])
        dedup.append(r["score"])
    auc_dedup = mann_whitney_auc(dedup, reals) if dedup and reals else None

    per_gen = {}
    for gen in sorted({r["generator"] for r in scores
                       if r["arm"] == "genvid_fake"}):
        s = [r["score"] for r in scores
             if r["arm"] == "genvid_fake" and r["generator"] == gen
             and r.get("score") is not None]
        per_gen[gen] = {"n": len(s),
                        "auc": mann_whitney_auc(s, reals) if s else None}

    report = {
        "ok": not problems,
        "problems": problems,
        "checks": {
            "exam_preserved": not any("exam_sealed" in p for p in problems),
            "input_hashes": not any(p.startswith(("v1_", "stage0_")) for p in problems),
            "row_payloads": not any(p.startswith(("payload", "duplicate", "score recipe"))
                                    for p in problems),
            "negatives_n": len(neg),
            "negatives_locked": len(neg) == N_NEG,
            "stage0_recipe": not any("Stage 0 seal" in p for p in problems),
            "booked": verdict.get("booked"),
        },
        "scores_sha256": _sha256(out / "scores.jsonl"),
        "primary_auc_recomputed": auc,
        "primary_auc_dedup": auc_dedup,
        "n_fakes": len(fakes),
        "n_negatives": len(reals),
        "per_generator": per_gen,
        "band": verdict.get("band"),
        "note": "independent Mann-Whitney AUC (ties averaged); report-only",
    }
    return report, problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stage0", default=None,
                    help="override the negatives file (still hash-pinned by seal)")
    a = ap.parse_args()
    out = Path(a.out) if a.out else DEFAULT_OUT
    report, problems = verify(out, Path(a.stage0) if a.stage0 else None)
    (out / "verify.json").write_text(json.dumps(report, indent=1,
                                                sort_keys=True) + "\n")
    for p in problems:
        print(f"FAIL: {p}", file=sys.stderr)
    print(json.dumps({k: report[k] for k in
                      ("ok", "primary_auc_recomputed", "primary_auc_dedup",
                       "n_fakes", "n_negatives", "band")}, indent=1))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
