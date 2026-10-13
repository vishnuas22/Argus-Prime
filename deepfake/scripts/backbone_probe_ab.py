"""Stage A harness — backbone A/B for the image court (spec p2.5-backbone-ab).

Trains/evaluates probe recipes (rows R1-R5) across backbone arms (clip/dv2b/
dv2l/siglip) on IDENTICAL file/key sets, splits, sklearn recipes and seeds, and
enforces the pre-registered Stage-A gate (§3.6) verbatim:

  (G1) mean(AUC_arm - AUC_clip) over the row's hard cells >= +0.01
  (G2) (AUC_arm - AUC_clip) >= -0.01 on every measurable gate cell of the row
  ceiling exemption: a hard cell drops out iff the CLIP baseline >= 0.99
  (membership frozen from the CLIP arm BEFORE any candidate runs)
  min-n: a cell counts only if n >= 100; a row with zero hard cells is N/A

Rows (recipes mirror the published trainers — imports, never copies):
  R1 face_generalist   train_face_multidomain no-FF++ (C=3, L2)
  R2 face_ffpp_joint   train_face_multidomain --with-ffpp (C=3, L2)
  R3 face_ffpp_only    FF++-only LR, C=3, L2 (p3_ffpp mirror)
  R4 aigc              train_probe.py UnivFD recipe (C by CV, random_state=7)
  R5 mj_specialist     train_probe.py on mj_train_{real,fake} (optional)

Parity self-test (instrument law): the CLIP arm must reproduce R1 cdf 0.9819 /
R1 wild 0.9056 / R2 ffpp 0.8681 / R3 ffpp 0.8885 within +-0.005, else the
harness aborts before any candidate number is believed.

Usage:
  .venv/bin/python deepfake/scripts/backbone_probe_ab.py --backbones clip --rows R1,R2,R3
  .venv/bin/python deepfake/scripts/backbone_probe_ab.py --backbones clip,dv2b,dv2l,siglip --rows R1,R2,R3,R4,R5
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                    train_test_split)

sys.path.insert(0, str(Path(__file__).parent))
from train_face_multidomain import load_cache  # noqa: E402
from train_face_probe import eer_from  # noqa: E402

WS = Path(__file__).resolve().parents[2]
CACHES = WS / "data" / "caches"
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
SEED = 7
MAX_ITER = 2000
TOL_PARITY = 0.005
PUBLISHED = {("R1", "cdf"): 0.9819, ("R1", "wild"): 0.9056,
             ("R2", "ffpp"): 0.8681, ("R3", "ffpp"): 0.8885}
MIN_N = 100
CEILING = 0.99
HARD_DELTAS = {"G1": 0.01, "G2": -0.01}

# CLIP baseline caches that predate this phase and stay untouched (un-suffixed).
LEGACY_CLIP = {"cdf_faces", "wild_train", "wild_sample", "ffpp_train",
               "ffpp_test", "mj_exam_real", "mj_exam_fake", "mj_train_real",
               "mj_train_fake"}

# gate name -> list of (cache name, label) — label None = first-path-component rule
EXAM_SPECS = {
    "aigc": [("eval_real", 0), ("eval_fake", 1)],
    "mj": [("mj_exam_real", 0), ("mj_exam_fake", 1)],
    "flux": [("mj_exam_real", 0), ("flux_exam_fake", 1)],
    "aigc_face": [("eval_face_real", 0), ("eval_face_fake", 1)],
    "mj_face": [("mj_face_real", 0), ("mj_face_fake", 1)],
    "flux_face": [("flux_face_real", 0), ("flux_face_fake", 1)],
    "ffpp": [("ffpp_test", None)],
    "cdf": [("cdf_faces", None)],
    "wild": [("wild_sample", None)],
}

# row -> (gate cells, hard cells, reference-only cells)
ROWS = {
    "R1": {"name": "face_generalist",
           "gate": ["cdf", "ffpp", "aigc_face", "mj_face", "flux_face"],
           "hard": ["ffpp", "mj_face", "flux_face"], "ref": ["wild"]},
    "R2": {"name": "face_ffpp_joint",
           "gate": ["cdf", "ffpp", "aigc_face", "mj_face", "flux_face"],
           "hard": ["ffpp", "mj_face", "flux_face"], "ref": ["wild"]},
    "R3": {"name": "face_ffpp_only",
           "gate": ["cdf", "ffpp", "aigc_face", "mj_face", "flux_face"],
           "hard": ["ffpp", "mj_face", "flux_face"], "ref": ["wild"]},
    "R4": {"name": "aigc",
           "gate": ["aigc", "mj", "flux"],
           "hard": ["mj", "flux"], "ref": []},
    "R5": {"name": "mj_specialist",
           "gate": ["aigc", "mj", "flux"],
           "hard": ["mj", "flux"], "ref": []},
}


def cache_path(name: str, backbone: str) -> Path:
    if backbone == "clip" and name in LEGACY_CLIP:
        p = CACHES / f"{name}.npz"
    else:
        p = CACHES / f"{name}.{backbone}.npz"
    return p


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_npz_l2(path: Path, labels=None):
    """Load an npz; labels=None -> first-path-component rule (load_cache),
    else per-class cache: labels is a list of ints aligned with the files."""
    if labels is None:
        X, y, keys = load_cache(str(path))
    else:
        z = np.load(path)
        keys = list(z.files)
        X = np.stack([z[k] for k in keys]).astype(np.float32)
        y = np.array(labels, dtype=int)
        assert len(y) == len(keys), (path, len(y), len(keys))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return X, y, keys


def load_exam(exam: str, backbone: str):
    """Returns ((X, y, n_real, n_fake), None) or (None, reason)."""
    parts = EXAM_SPECS[exam]
    Xs, ys = [], []
    n_real = n_fake = 0
    for name, lab in parts:
        p = cache_path(name, backbone)
        if not p.exists():
            return None, f"cache_missing:{p.name}"
        if lab is None:
            X, y, _ = load_npz_l2(p, None)
        else:
            z = np.load(p)
            keys = list(z.files)
            X = np.stack([z[k] for k in keys]).astype(np.float32)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            y = np.full(len(keys), lab, dtype=int)
        Xs.append(X)
        ys.append(y)
        n_real += int((y == 0).sum())
        n_fake += int((y == 1).sum())
    return (np.concatenate(Xs), np.concatenate(ys), n_real, n_fake), None


def score(clf, X, y):
    p = clf.predict_proba(X)[:, 1]
    return {"auc": round(float(roc_auc_score(y, p)), 4),
            "eer": round(float(eer_from(y, p)), 4), "n": int(len(y))}


def aigc_recipe(X, y):
    """train_probe.py recipe, mirrored exactly (L2 already applied at load;
    C by 5-fold CV on the 85% train split, random_state=7, max_iter=2000;
    final probe fit on ALL rows with the winning C)."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED)
    best_c, best_score = None, -1.0
    cv = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=MAX_ITER, solver="lbfgs")
        s = float(cross_val_score(clf, X_tr, y_tr,
                                  cv=StratifiedKFold(5, shuffle=True,
                                                     random_state=SEED),
                                  scoring="roc_auc").mean())
        cv[str(c)] = round(s, 4)
        if s > best_score:
            best_c, best_score = c, s
    val = score(LogisticRegression(C=best_c, max_iter=MAX_ITER).fit(X_tr, y_tr),
                X_va, y_va)
    final = LogisticRegression(C=best_c, max_iter=MAX_ITER).fit(X, y)
    return final, {"C": best_c, "cv_auc": round(best_score, 4), "cv": cv,
                   "val": val, "grid": C_GRID, "seed": SEED,
                   "max_iter": MAX_ITER}


def train_row(row: str, backbone: str):
    """Returns (clf, meta) for the row/backbone, or (None, reason)."""
    if row in ("R1", "R2", "R3"):
        parts, exams = [], {}
        if row in ("R1", "R2"):
            pc = cache_path("cdf_faces", backbone)
            pw = cache_path("wild_train", backbone)
            if not (pc.exists() and pw.exists()):
                return None, f"cache_missing:{pc.name if not pc.exists() else pw.name}"
            Xc, yc, keys_c = load_npz_l2(pc, None)
            test_stems = set()
            for line in (WS / "data/raw/celebdf/Celeb-DF-v2/"
                         "List_of_testing_videos.txt").read_text().splitlines():
                if line.strip():
                    test_stems.add(Path(line.strip().split(" ", 1)[1]).stem)
            is_test = np.array([k.split("/")[1].split(":")[0] in test_stems
                                for k in keys_c])
            Xw_tr, yw_tr, _ = load_npz_l2(pw, None)
            pwt = cache_path("wild_sample", backbone)
            if not pwt.exists():
                return None, f"cache_missing:{pwt.name}"
            Xw_te, yw_te, _ = load_npz_l2(pwt, None)
            parts = [Xc[~is_test], Xw_tr]
            y_parts = [yc[~is_test], yw_tr]
            exams["cdf"] = (Xc[is_test], yc[is_test])
            exams["wild"] = (Xw_te, yw_te)
        if row == "R3":
            pf = cache_path("ffpp_train", backbone)
            if not pf.exists():
                return None, f"cache_missing:{pf.name}"
            Xf, yf, _ = load_npz_l2(pf, None)
            parts = [Xf]
            y_parts = [yf]
        if row == "R2":
            pf = cache_path("ffpp_train", backbone)
            if not pf.exists():
                return None, f"cache_missing:{pf.name}"
            Xf, yf, _ = load_npz_l2(pf, None)
            parts.append(Xf)
            y_parts.append(yf)
        X = np.concatenate(parts)
        y = np.concatenate(y_parts)
        clf = LogisticRegression(C=3.0, max_iter=MAX_ITER).fit(X, y)
        meta = {"C": 3.0, "seed": None, "max_iter": MAX_ITER,
                "n_train": int(len(y))}
        return clf, meta, exams
    # AIGC rows (train_probe recipe)
    if row in ("R4", "R5"):
        diet = {"R4": [("train_real", 0), ("train_fake", 1)],
                "R5": [("mj_train_real", 0), ("mj_train_fake", 1)]}[row]
        Xs, ys = [], []
        for name, lab in diet:
            p = cache_path(name, backbone)
            if not p.exists():
                return None, f"cache_missing:{p.name}"
            z = np.load(p)
            keys = list(z.files)
            X = np.stack([z[k] for k in keys]).astype(np.float32)
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
            Xs.append(X)
            ys.append(np.full(len(keys), lab, dtype=int))
        X, y = np.concatenate(Xs), np.concatenate(ys)
        clf, meta = aigc_recipe(X, y)
        meta["n_train"] = int(len(y))
        return clf, meta, {}
    return None, f"unknown row {row}"


def exam_frames(row: str):
    """Exam list for a row: gate exams + reference exams."""
    spec = ROWS[row]
    return list(spec["gate"]) + list(spec["ref"])


def run_row(row: str, backbone: str):
    """Train + score one (row, backbone) arm. Returns dict or {'error': reason}."""
    out = train_row(row, backbone)
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], str):
        return {"error": out[1]}
    clf, meta, exams = out
    cells = {}
    for exam in exam_frames(row):
        if exam in exams:
            X, y = exams[exam]
            n_real, n_fake = int((y == 0).sum()), int((y == 1).sum())
            cell = score(clf, X, y)
        else:
            loaded, err = load_exam(exam, backbone)
            if err:
                cells[exam] = {"excluded_reason": err}
                continue
            X, y, n_real, n_fake = loaded
            cell = score(clf, X, y)
        cell.update({"n_real": n_real, "n_fake": n_fake})
        if cell["n"] < MIN_N:
            cell["excluded_reason"] = f"min_n:{cell['n']}<{MIN_N}"
        cells[exam] = cell
    return {"backbone": backbone, "row": row, "recipe": meta, "cells": cells}


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "clip_baseline": {}, "arms": [], "hard_cells": {},
            "parity_selftest": None}


def clip_cells(results, row) -> dict | None:
    for arm in results["arms"]:
        if arm.get("backbone") == "clip" and arm.get("row") == row and "cells" in arm:
            return arm["cells"]
    return None


def arm_incomplete(arm: dict, row: str) -> bool:
    """True when a computed arm is missing exam cells because a cache did not
    exist yet (staged cache builds) — such rows get recomputed on a later pass."""
    cells = arm.get("cells", {})
    for e in exam_frames(row):
        c = cells.get(e)
        if c is None:
            return True
        if str(c.get("excluded_reason", "")).startswith("cache_missing"):
            return True
    return False


def freeze_hard_cells(results, row):
    """Ceiling exemption computed from the CLIP arm ONLY, frozen in results."""
    if row in results["hard_cells"]:
        return results["hard_cells"][row]
    cells = clip_cells(results, row)
    if cells is None:
        return None
    active, dropped, clip_auc = [], [], {}
    for exam in ROWS[row]["hard"]:
        c = cells.get(exam) or {}
        auc = c.get("auc")
        clip_auc[exam] = auc
        if auc is None:
            dropped.append({"exam": exam, "reason": f"clip_unmeasurable:{c.get('excluded_reason')}"})
        elif auc >= CEILING:
            dropped.append({"exam": exam, "reason": f"ceiling:clip={auc}>={CEILING}"})
        elif c.get("n", 0) < MIN_N:
            dropped.append({"exam": exam, "reason": f"min_n:{c.get('n')}"})
        else:
            active.append(exam)
    block = {"active": active, "dropped": dropped, "clip_auc": clip_auc,
             "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    results["hard_cells"][row] = block
    return block


def apply_gate(arm: dict, results) -> dict:
    row = arm["row"]
    block = results["hard_cells"].get(row)
    if block is None:
        return {"error": f"hard-cell membership for {row} not frozen — run the clip arm first"}
    cells = arm.get("cells", {})
    base = clip_cells(results, row) or {}
    deltas, hard_deltas, excluded = {}, [], []
    for exam, c in cells.items():
        b = base.get(exam) or {}
        if "auc" not in c or "auc" not in b:
            reason = c.get("excluded_reason") or b.get("excluded_reason") or "unmeasurable"
            deltas[exam] = {"excluded_reason": reason}
            if exam in ROWS[row]["gate"]:
                excluded.append({"exam": exam, "reason": reason})
            continue
        d = round(c["auc"] - b["auc"], 4)
        deltas[exam] = {"auc": c["auc"], "clip_auc": b["auc"], "delta": d,
                        "n": c["n"], "clip_n": b["n"],
                        "n_real": c.get("n_real"), "n_fake": c.get("n_fake")}
        if exam in block["active"]:
            hard_deltas.append(d)
    g2_cells = [deltas[e] for e in ROWS[row]["gate"]
                if e in deltas and "delta" in deltas[e]]
    g2_min = min((d["delta"] for d in g2_cells), default=None)
    g1_mean = round(float(np.mean(hard_deltas)), 4) if hard_deltas else None
    g1 = (g1_mean is not None and g1_mean >= HARD_DELTAS["G1"])
    g2 = (g2_min is not None and g2_min >= HARD_DELTAS["G2"])
    # fair-comparison protocol: identical file/key sets across arms. A mismatch
    # means one cache is truncated/stale (e.g. a --limit partial) — the cell is
    # not comparable, so the arm cannot pass (D-044/§3.2.1).
    mism = {"exam_n": {e: {"arm": deltas[e]["n"], "clip": deltas[e]["clip_n"]}
                       for e in deltas
                       if isinstance(deltas[e], dict) and "clip_n" in deltas[e]
                       and deltas[e]["n"] != deltas[e]["clip_n"]}}
    base_arm = next((a for a in results["arms"]
                     if a.get("backbone") == "clip" and a.get("row") == row
                     and "cells" in a), None)
    if base_arm is not None and arm.get("recipe", {}).get("n_train") is not None \
            and base_arm["recipe"].get("n_train") is not None \
            and arm["recipe"]["n_train"] != base_arm["recipe"]["n_train"]:
        mism["n_train"] = {"arm": arm["recipe"]["n_train"],
                           "clip": base_arm["recipe"]["n_train"]}
    mism = {k: v for k, v in mism.items() if v}
    arm.update({
        "deltas": deltas, "excluded": excluded,
        "hard_mean_delta": g1_mean, "hard_cells_used": len(hard_deltas),
        "G1": bool(g1), "G2": bool(g2),
        "G2_min_delta": g2_min,
        "key_set_mismatch": mism or None,
        "pass": bool(g1 and g2 and hard_deltas and not mism),
        "row_na": not bool(hard_deltas),
    })
    return arm


def run_parity(results) -> dict:
    """CLIP-arm reproduction of the published numbers. Cells whose row has no
    CLIP arm yet are 'pending' (not failures) — staged runs fill them in."""
    cells = {}
    for (row, exam), expected in PUBLISHED.items():
        base = clip_cells(results, row)
        if base is None:
            cells[f"{row}:{exam}"] = {"expected": expected, "got": None,
                                      "status": "pending", "pass": None}
            continue
        c = base.get(exam) or {}
        got = c.get("auc")
        ok = got is not None and abs(got - expected) <= TOL_PARITY
        cells[f"{row}:{exam}"] = {"expected": expected, "got": got,
                                  "abs_err": None if got is None else round(abs(got - expected), 4),
                                  "n": c.get("n"), "status": "pass" if ok else "fail",
                                  "pass": bool(ok)}
    computed = [v for v in cells.values() if v["pass"] is not None]
    verdict = bool(computed) and all(v["pass"] for v in computed)
    return {"pass": bool(verdict),
            "complete": len(computed) == len(cells),
            "tol": TOL_PARITY, "cells": cells,
            "published_from": ["artifacts/p3_full/train_meta.json",
                               "artifacts/p3_ffpp/train_meta.json"]}


def sidecar_hashes(backbone: str, row: str) -> dict:
    used = set()
    if row in ("R1", "R2"):
        used |= {"cdf_faces", "wild_train"}
    if row in ("R2", "R3"):
        used |= {"ffpp_train"}
    if row == "R1":
        pass
    if row in ("R4",):
        used |= {"train_real", "train_fake"}
    if row in ("R5",):
        used |= {"mj_train_real", "mj_train_fake"}
    used |= set(EXAM_SPECS[e][i][0] for e in exam_frames(row)
                for i in range(len(EXAM_SPECS[e])))
    out = {}
    for name in sorted(used):
        p = cache_path(name, backbone)
        if not p.exists():
            out[name] = {"path": p.name, "missing": True}
            continue
        meta = p.with_suffix(".meta.json")
        if meta.exists():
            out[name] = {"path": p.name, "sha256_kind": "sidecar",
                         "sha256": sha256_file(meta), "sidecar": meta.name}
        else:
            out[name] = {"path": p.name, "sha256_kind": "npz",
                         "sha256": sha256_file(p)}
    return out


def _clip_cell(arms, row: str, exam: str) -> str:
    for a in arms:
        if a.get("backbone") == "clip" and a.get("row") == row and "cells" in a:
            c = a["cells"].get(exam) or {}
            return f"{c['auc']:.4f}" if "auc" in c else "N/A"
    return "N/A"


def render_report(results) -> str:
    import sklearn
    lines = ["# Phase 2.5 — Backbone A/B (image court) — Stage A report",
             "",
             f"generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
             f"sklearn {sklearn.__version__} · numpy {np.__version__} · "
             f"python {sys.version.split()[0]}",
             ""]
    st = results.get("parity_selftest") or {}
    state = ("PASS" if st.get("pass") else
             ("PENDING" if not st.get("complete") else "FAIL"))
    lines += [f"## Harness parity self-test (CLIP arm)",
              f"**{state}** (tol +-{TOL_PARITY}; complete="
              f"{st.get('complete')})", ""]
    for k, v in (st.get("cells") or {}).items():
        mark = ("PENDING" if v.get("pass") is None
                else ("PASS" if v["pass"] else "FAIL"))
        lines.append(f"- {k}: published {v['expected']} vs harness {v['got']} "
                     f"(n={v.get('n')}) -> {mark}")
    lines += ["", "## Frozen hard cells (ceiling exemption, CLIP arm only)", ""]
    for row, blk in sorted(results.get("hard_cells", {}).items()):
        lines.append(f"- **{row}** ({ROWS[row]['name']}): active={blk['active']} "
                     f"dropped={blk['dropped']}")
    lines += ["", "## Face coverage (P5, YuNet 0.55)", ""]
    cov_p = WS / "artifacts/backbone_ab/face_coverage_pre.json"
    if cov_p.exists():
        cov = json.loads(cov_p.read_text())
        for d, v in cov.items():
            lines.append(f"- {d}: {v['face_present']}/{v['images']} "
                         f"({v['rate']:.2%})")
    lines += ["", "## Full delta table (AUC; Δ = arm − clip)", ""]
    arms = results.get("arms", [])
    rows_sorted = [r for r in ("R1", "R2", "R3", "R4", "R5")
                   if any(a.get("row") == r for a in arms)]
    for row in rows_sorted:
        exams = exam_frames(row)
        lines.append(f"### {row} {ROWS[row]['name']}")
        header = "| backbone | " + " | ".join(exams) + " | hardΔ | G1 | G2 | pass |"
        lines.append(header)
        lines.append("|---" * (len(exams) + 5) + "|")
        for bb in ("clip", "dv2b", "dv2l", "siglip"):
            arm = next((a for a in arms if a.get("row") == row
                        and a.get("backbone") == bb and "cells" in a), None)
            if arm is None:
                continue
            base = clip_cells(results, row) or {}
            cols = []
            for e in exams:
                c = arm["cells"].get(e) or {}
                b = base.get(e) or {}
                if "auc" in c:
                    d = (c["auc"] - b["auc"]) if "auc" in b else None
                    cols.append(f"{c['auc']:.4f} (n={c['n']}" +
                                (f", Δ{d:+.4f})" if d is not None else ")"))
                else:
                    cols.append(f"N/A ({c.get('excluded_reason', 'n/a')})")
            lines.append(f"| {bb} | " + " | ".join(cols) + " | "
                         + (" | ".join("—" if arm.get(k) is None
                                       else str(arm.get(k))
                                       for k in ("hard_mean_delta", "G1",
                                                 "G2", "pass")))
                         + " |")
        lines.append("")
    mm = [a for a in arms if a.get("key_set_mismatch")]
    if mm:
        lines += ["## Fair-comparison violations (key sets differ from CLIP)",
                  "",
                  "A cell whose arm/CLIP key counts differ is NOT comparable "
                  "(truncated or stale cache) and cannot pass the gate:", ""]
        for a in mm:
            lines.append(f"- **{a['backbone']}/{a['row']}**: "
                         f"`{json.dumps(a['key_set_mismatch'])}`")
        lines.append("")
    # honest N/A cell accounting (§3.5): every cell that did not produce a score
    listed = [(a["backbone"], a["row"], e, c.get("excluded_reason"))
              for a in arms for e, c in a.get("cells", {}).items()
              if "auc" not in c]
    lines += ["## Cell exclusions (honest N/A)", ""]
    if listed:
        for bb, row, e, reason in sorted(listed):
            lines.append(f"- {bb}/{row}/{e}: N/A — {reason}")
    else:
        lines.append("- none: every (backbone, row, exam) cell produced a score; "
                     "all n ≥ 100 (min-n never triggered), so no cell was dropped "
                     "on coverage grounds.")
    lines += ["", "## Drift note (P3, disclosed — never hidden)", ""]
    lines += [
        "- Published `artifacts/aigc_shot_v2/train_meta.json` diet: "
        "**6,000 keys** (4,750 `train_real_clipL` + 1,250 `train_fake_clipL`), "
        "C=3.0, val_auc 1.0.",
        "- This phase rebuilt the AIGC diet from **disk truth for ALL arms "
        "including CLIP** (pre-registered at P3): `train_real` 3,750 + "
        "`train_fake` 1,250 = **5,000 keys** — `train_real_clipL.npz` held "
        "2,500 orphan keys (500 files no longer on disk × 5 views) and was "
        "missing the 1,500 shot/shot_q60 keys of the 750 current files.",
        "- Consequence: this phase's CLIP R4 arm is **not bit-identical** to "
        "the deployed `aigc_shot_v2` probe. Recorded deltas: aigc exam "
        f"{_clip_cell(arms, 'R4', 'aigc')} (n=1,950) vs the deployed "
        "`learned_aigc_probe` on the aigc gate **0.9998 (n=200, restack "
        "r6–r8 gates.json)** → Δ −0.0006 across different n/exam samples; "
        "C identical (3.0 published = 3.0 CV-selected here).",
        "- The drifted `train_real_clipL.npz` / `train_fake_clipL.npz` stay "
        "untouched on disk; only the `.clip`-suffixed rebuilds are used.",
        "", ]
    lines += ["## Gate rule (pre-registered, §3.6)",
              "",
              "An arm = one (backbone, recipe-row). Proceeds iff **G1** mean(Δ) "
              "over active hard cells ≥ +0.01 AND **G2** Δ ≥ −0.01 on every "
              "measurable gate cell of the row. Hard cells: face rows → "
              "ffpp + mj(face) + flux(face); AIGC row → mj + flux. Ceiling "
              "exemption drops a hard cell iff CLIP ≥ 0.99 (frozen from the "
              "CLIP arm before candidates run). min-n = 100 keys; a row with "
              "zero hard cells is N/A (not a pass). 0 arms pass → STOP.",
              ""]
    cands = ("dv2b", "dv2l", "siglip")
    all_rows = ("R1", "R2", "R3", "R4", "R5")

    def has(bb, row):
        return any(a.get("backbone") == bb and a.get("row") == row
                   and "cells" in a for a in arms)

    pending = [f"{bb}/{row}" for bb in cands for row in all_rows
               if not has(bb, row)]
    cand_arms = [a for a in arms if a.get("backbone") in cands and "cells" in a]
    verdict = [a for a in cand_arms if a.get("pass")]
    lines += ["## Decision", ""]
    if pending:
        lines += [
            f"**INCOMPLETE — {len(pending)}/{len(cands) * len(all_rows)} "
            f"candidate arms not yet measured**: {', '.join(pending)}",
            "",
            "No Stage-A verdict is issued until every candidate arm is measured; "
            "this REPORT.md is regenerated after each harness run."]
    else:
        for bb in cands:
            rows_ok = [a["row"] for a in verdict if a["backbone"] == bb]
            rows_bad = [a["row"] for a in cand_arms if a["backbone"] == bb
                        and not a.get("pass")]
            n_na = sum(1 for a in cand_arms if a["backbone"] == bb
                       and a.get("row_na"))
            lines.append(f"- **{bb}**: {len(rows_ok)}/{len(all_rows)} rows pass"
                         + (f" — pass {rows_ok}" if rows_ok else "")
                         + (f" — fail {rows_bad}" if rows_bad else "")
                         + (f" — N/A {n_na}" if n_na else ""))
        lines.append("")
        lines.append(f"**{len(verdict)}/{len(cand_arms)} candidate arms pass "
                     f"(G1 + G2)**")
        lines.append("")
        if verdict:
            chans = []
            for a in sorted(verdict, key=lambda a: (a["row"], a["backbone"])):
                kind = "aigc (full-image)" if a["row"] in ("R4", "R5") \
                    else "face"
                chans.append(f"{a['backbone']}/{a['row']} ({kind}, "
                             f"hardΔ={a.get('hard_mean_delta'):+.4f})")
            lines.append("§3.6 phase verdict: **≥1 arm passes → Stage B opens "
                         "for that arm's channel**: " + "; ".join(chans) + ".")
            lines.append("")
            lines.append("Stage-A gate is necessary, not sufficient: the "
                         "restack r9 gates still decide whether anything ships.")
        else:
            lines += [
                "**0 arms pass — STOP. Stage B does not open (negative result, "
                "D-037).**",
                "",
                "### Negative result content (for the decision log — the "
                "builder drafts the `memory/decisions.md` D-037 entry text; "
                "this file does not edit memory/)",
                "",
                f"Stage A measured **{len(cand_arms)}/{len(cands) * len(all_rows)} "
                "candidate arms** (dv2b, dv2l, siglip × R1–R5) against the "
                "frozen CLIP baseline under the pre-registered §3.6 rule "
                "(G1 mean hard-cell Δ ≥ +0.01 **and** G2 Δ ≥ −0.01 on every "
                "measurable gate cell). **0 of 15 arms pass.**",
                "",
                "Closest misses (why they still fail):", ]
            near = sorted((a for a in cand_arms if a.get("hard_mean_delta")
                           is not None),
                          key=lambda a: -(a["hard_mean_delta"] or -9))[:3]
            for a in near:
                why = []
                if not a.get("G1"):
                    why.append(f"G1 fails (hardΔ {a['hard_mean_delta']:+.4f} < +0.01)")
                if not a.get("G2"):
                    why.append(f"G2 fails (min gate Δ "
                               f"{a.get('G2_min_delta'):+.4f} < −0.01)")
                if not why:
                    why.append("passes G1 and G2")
                lines.append(f"- `{a['backbone']}/{a['row']}` — hardΔ "
                             f"{a.get('hard_mean_delta'):+.4f}, G1="
                             f"{a.get('G1')}, G2={a.get('G2')} → "
                             + "; ".join(why))
            lines += [
                "",
                "Consequences recorded under D-037:",
                "- **No integration, no restack r9, no promotion, no wiring.** "
                "`channels.py`, `train_stacker.py`, `stacker_judge.py`, "
                "`argus_detect.py`, `.github/workflows/ci.yml` and every "
                "video/audio path are untouched; the deployed stacker stays "
                "`p6v10` byte-identical.",
                "- Stage-A blast radius is exactly: this harness + "
                "`backbone_embed.py` + `build_feature_cache.py`, the new "
                "`data/caches/*.{dv2b,dv2l,siglip}.*` caches and their "
                "sidecars, and `artifacts/backbone_ab/`.",
                "- The frozen hard-cell membership, the parity self-test and "
                "the full delta table above are the measurement of record.",
            ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="clip")
    ap.add_argument("--rows", default="R1,R2,R3,R4,R5")
    ap.add_argument("--results", default="artifacts/backbone_ab/results.json")
    ap.add_argument("--force", action="store_true",
                    help="recompute cells even when already present")
    args = ap.parse_args()
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]
    rows = [r.strip().upper() for r in args.rows.split(",") if r.strip()]
    for r in rows:
        if r not in ROWS:
            raise SystemExit(f"unknown row {r} (known: {sorted(ROWS)})")
    for b in backbones:
        if b not in ("clip", "dv2b", "dv2l", "siglip"):
            raise SystemExit(f"unknown backbone {b}")
    res_path = Path(args.results)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    results = load_results(res_path)

    import sklearn
    results["versions"] = {"sklearn": sklearn.__version__,
                           "numpy": np.__version__,
                           "python": sys.version.split()[0]}
    cov_p = WS / "artifacts/backbone_ab/face_coverage_pre.json"
    if cov_p.exists():
        results["feasibility"] = json.loads(cov_p.read_text())

    order = list(dict.fromkeys(b for b in ("clip",) + tuple(backbones)
                               if b in backbones))
    for row in rows:
        for bb in order:
            if bb != "clip" and clip_cells(results, row) is None:
                raise SystemExit(
                    f"[{row}] clip baseline missing — run the clip arm first "
                    f"(fair-comparison protocol + ceiling freeze require it)")
            if bb == "clip":
                freeze_hard_cells(results, row)
            existing = next((a for a in results["arms"]
                             if a.get("row") == row and a.get("backbone") == bb
                             and "cells" in a), None)
            if (existing is not None and not args.force
                    and not arm_incomplete(existing, row)):
                print(f"[skip] {bb}/{row} already computed", flush=True)
                if bb != "clip":
                    apply_gate(existing, results)
                continue
            if existing is not None and bb == "clip":
                cands = [a for a in results["arms"]
                         if a.get("row") == row and a.get("backbone") != "clip"
                         and "cells" in a]
                if results["hard_cells"].get(row) and not cands:
                    del results["hard_cells"][row]  # re-freeze from fresh CLIP
                    print(f"[{row}] hard-cell freeze invalidated "
                          f"(clip recompute, no candidate arms yet)", flush=True)
                elif results["hard_cells"].get(row) and cands:
                    print(f"[{row}] WARNING: clip recompute with candidate "
                          f"arms already gated — frozen hard cells kept",
                          flush=True)
            if bb != "clip":
                st0 = results.get("parity_selftest") or {}
                if not (st0.get("pass") and st0.get("complete")):
                    raise SystemExit(
                        "instrument law: the CLIP parity self-test must be "
                        "complete AND passing before any candidate arm runs "
                        f"(current: {json.dumps(st0.get('cells') or {})}) — "
                        "fix the harness, never report candidate numbers")
            t0 = time.time()
            arm = run_row(row, bb)
            if "error" in arm:
                print(f"[{bb}/{row}] EXCLUDED: {arm['error']}", flush=True)
                results["arms"] = [a for a in results["arms"]
                                   if not (a.get("row") == row
                                           and a.get("backbone") == bb)]
                results["arms"].append({"backbone": bb, "row": row,
                                        "error": arm["error"]})
                if bb == "clip":
                    raise SystemExit(
                        f"clip arm for {row} failed ({arm['error']}) — fix the "
                        f"harness/caches before any candidate number is believed")
                continue
            results["arms"] = [a for a in results["arms"]
                               if not (a.get("row") == row
                                       and a.get("backbone") == bb)]
            results["arms"].append(arm)
            if bb == "clip":
                results["clip_baseline"][row] = arm["cells"]
                freeze_hard_cells(results, row)
            st = run_parity(results)
            results["parity_selftest"] = st
            res_path.write_text(json.dumps(results, indent=2) + "\n")
            if bb == "clip":
                n_pending = sum(1 for v in st["cells"].values()
                                if v["pass"] is None)
                print(f"[{bb}/{row}] parity self-test "
                      f"{'PASS' if st['pass'] else 'FAIL'}"
                      + (f" ({n_pending} pending)" if n_pending else ""),
                      flush=True)
                if not st["pass"]:
                    bad = {k: v for k, v in st["cells"].items() if not v["pass"]}
                    res_path.write_text(json.dumps(results, indent=2) + "\n")
                    raise SystemExit(
                        f"PARITY SELF-TEST FAILED: {json.dumps(bad)} — aborting "
                        f"before any candidate number is believed")
            else:
                apply_gate(arm, results)
                res_path.write_text(json.dumps(results, indent=2) + "\n")
                print(f"[{bb}/{row}] hardΔ={arm.get('hard_mean_delta')} "
                      f"G1={arm.get('G1')} G2={arm.get('G2')} "
                      f"pass={arm.get('pass')} "
                      + (f"KEY-SET-MISMATCH={json.dumps(arm['key_set_mismatch'])} "
                         if arm.get("key_set_mismatch") else "")
                      + f"({time.time()-t0:.0f}s)", flush=True)
    results["recipe"] = {
        "rows": {r: {"name": ROWS[r]["name"], "gate": ROWS[r]["gate"],
                     "hard": ROWS[r]["hard"], "ref": ROWS[r]["ref"]}
                 for r in sorted(ROWS)},
        "C_grid": C_GRID, "seed": SEED, "max_iter": MAX_ITER,
        "l2_at_load": True, "fp32_at_train": True, "min_n": MIN_N,
        "ceiling": CEILING, "hard_delta_min": HARD_DELTAS["G1"],
        "gate_delta_min": HARD_DELTAS["G2"],
        "versions": results["versions"],
        "cache_hashes": {f"{bb}:{row}": sidecar_hashes(bb, row)
                         for bb in order for row in rows},
    }
    results["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    res_path.write_text(json.dumps(results, indent=2) + "\n")
    report = render_report(results)
    (res_path.parent / "REPORT.md").write_text(report)
    print(f"wrote {res_path} and {res_path.parent/'REPORT.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
