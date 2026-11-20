"""V1d stage 3 — analyse against the sealed bars (v1d-feasibility-v1).

One combiner is fitted on the GenBuster-T2V FIT half only, then evaluated on
every held-out half. That is stricter and cleaner than per-domain fitting: the
same fitted model must generalise, which is what B5 claims.

Bars (memory/spec_v1d_feasibility_v1.md):
  B1  GenBuster T2V   combined >= p6vid_v4 + 0.05
  B2  FF++ and CDF    combined >= p6vid_v4 - 0.02
  B3  |Pearson r| on the T2V fit split <= 0.30
  B4  >= 200 per class per domain after decode filtering
  B5  kling + sora    combined >= p6vid_v4 (cross-domain generalisation)

Usage:
    .venv/bin/python deepfake/scripts/v1d_analyze.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import pearsonr

WS = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(WS / "deepfake"), str(WS / "deepfake" / "bench"),
                str(WS / "deepfake" / "scripts")]

OUT = WS / "artifacts" / "v1d_restrav"
SEED = 20261120


def load_jsonl(p):
    rows = []
    if not p.exists():
        return rows
    for line in open(p):
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def bootstrap_auc(y, s, n=2000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    vals = np.sort(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build_split(clips, frac=0.5, seed=SEED):
    """Deterministic 50/50 split over the given clip list."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clips))
    cut = int(len(clips) * frac)
    return set(clips[i] for i in idx[:cut]), set(clips[i] for i in idx[cut:])


def main():
    manifest = json.load(open(OUT / "manifest.json"))
    scores = {r["path"]: r for r in load_jsonl(OUT / "scores.jsonl")}
    feats = {r["path"]: r for r in load_jsonl(OUT / "features.jsonl")}

    domains = manifest["domains"]

    # ---- assemble per-domain arrays ----
    # A clip is usable when it has a feature vector. Its score may be None
    # (abstained) — that is a legitimate outcome and must be RETAINED so the
    # conservative variant can score it as a missed fake. Dropping abstained
    # clips silently would hide the class-skewed abstention (2.7% real vs 11.0% fake).
    def usable(p):
        return p in feats and bool(feats[p].get("features"))

    data = {}
    for name, d in domains.items():
        rows = []
        for p in d["real"]:
            if usable(p):
                rows.append((p, 0))
        for p in d["fake"]:
            if usable(p):
                rows.append((p, 1))
        data[name] = rows
        n_ab = sum(1 for p, _ in rows if scores.get(p, {}).get("abstained") is True)
        n_sc = sum(1 for p, _ in rows if scores.get(p, {}).get("p_fake") is not None)
        print(f"{name}: rows={len(rows)} "
              f"(real={sum(1 for _, y in rows if y == 0)}, "
              f"fake={sum(1 for _, y in rows if y == 1)}, "
              f"scored={n_sc}, abstained={n_ab})")

    # ---- B4: sample size ----
    print("\n" + "=" * 78)
    print("B4 — sample size (200 per class per domain, after decode filtering)")
    print("=" * 78)
    # Completeness is about STAGE COVERAGE, not usable rows: an abstained clip
    # legitimately has p_fake=None and must not count as "not collected".
    all_clips = []
    for d in domains.values():
        all_clips.extend(d.get("real", []))
        all_clips.extend(d.get("fake", []))
    missing_scores = [p for p in all_clips if p not in scores]
    missing_feats = [p for p in all_clips if p not in feats]
    stage_complete = not missing_scores and not missing_feats
    if not stage_complete:
        print(f"  stage coverage: scores missing {len(missing_scores)}, "
              f"features missing {len(missing_feats)}")

    b4_pass = True
    incomplete = []
    for name, d in domains.items():
        nr, nf = d["n_real"], d["n_fake"]
        rows = data[name]
        ur = sum(1 for _, y in rows if y == 0)
        uf = sum(1 for _, y in rows if y == 1)
        n_abst = sum(1 for pp, _ in rows if scores[pp].get("abstained") is True)
        ok = nr >= 200 and nf >= 200
        if not ok:
            b4_pass = False
        print(f"  {name:<20} sampled real={nr:>4} fake={nf:>4} | "
              f"usable real={ur:>4} fake={uf:>4} | abstained={n_abst:>3} | "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"  -> B4 {'PASS' if b4_pass else 'FAIL'}")
    print("  note: genvid_sora has 51 fakes in the whole corpus (51 is the")
    print("        corpus total, not decode attrition) — corpus-capped, not fixable.")

    # ---- fit on GenBuster T2V fit-half ----
    t2v_rows = data["genbuster_t2v"]
    t2v_clips = [p for p, _ in t2v_rows]
    fit_clips, eval_clips = build_split(t2v_clips)

    def arr(rows, subset):
        return [(p, y) for p, y in rows if p in subset]

    t2v_fit = arr(t2v_rows, fit_clips)
    t2v_eval = arr(t2v_rows, eval_clips)

    # The combiner cannot be fitted on abstained clips (no p6vid score exists).
    # Abstentions are handled in the conservative EVALUATION variant instead.
    t2v_fit_scored = [(pp, yy) for pp, yy in t2v_fit
                      if scores[pp].get("p_fake") is not None]
    y_fit = np.array([y for _, y in t2v_fit_scored])
    p_fit = np.array([scores[p]["p_fake"] for p, _ in t2v_fit_scored])
    X_fit = np.array([feats[p]["features"] for p, _ in t2v_fit_scored])
    print(f"\nfit split n={len(t2v_fit)} (scored={len(t2v_fit_scored)}, "
          f"abstained={len(t2v_fit) - len(t2v_fit_scored)})")

    # ReStraV feature selection on FIT only
    aucs = np.array([roc_auc_score(y_fit, X_fit[:, i]) for i in range(X_fit.shape[1])])
    best_i = int(np.argmax(np.abs(aucs - 0.5)))
    from restrav_feats import FEATURE_NAMES
    best_name = FEATURE_NAMES[best_i]
    best_auc_fit = float(aucs[best_i])
    print(f"\nfit split n={len(y_fit)}  ReStraV best feature={best_name} "
          f"(AUC {best_auc_fit:.4f}, |AUC-0.5|={abs(best_auc_fit-0.5):.4f})")

    # ---- B3: decorrelation on fit split ----
    r = float(pearsonr(X_fit[:, best_i], p_fit).statistic)
    print("\n" + "=" * 78)
    print("B3 — decorrelation on the T2V fit split")
    print("=" * 78)
    b3_pass = abs(r) <= 0.30
    print(f"  Pearson r(ReStraV_best, p6vid_v4 p_fake) = {r:.4f}")
    print(f"  -> B3 {'PASS' if b3_pass else 'FAIL'} (|r| <= 0.30)")

    # ---- combiner fitted on fit half ----
    Z_fit = np.column_stack([X_fit[:, best_i], p_fit])
    comb = LogisticRegression().fit(Z_fit, y_fit)
    print(f"\ncombiner coefs: ReStraV={comb.coef_[0,0]:.4f} p6vid={comb.coef_[0,1]:.4f}")

    # ---- per-domain eval on the held-out half ----
    print("\n" + "=" * 78)
    print("Held-out evaluation (50% split, disjoint by clip, seed 20261120)")
    print("=" * 78)
    print("Two variants are reported because abstention is CLASS-SKEWED")
    print("(measured: 2.7% of reals vs 11.0% of fakes abstain). Dropping abstained")
    print("clips therefore INFLATES fake recall and AUC. The conservative variant")
    print("scores an abstention as p_fake=0 (a missed fake), matching V1f precedent.")
    print()
    hdr = (f"{'domain':<20}{'n':>5}{'abst':>6}{'p6vid':>8}{'comb':>8}{'delta':>8}"
           f"{'p6vid_cons':>12}{'comb_cons':>11}")
    print(hdr)
    print("-" * 78)

    results = {}
    for name, rows in data.items():
        ev = rows if name != "genbuster_t2v" else t2v_eval
        if len(ev) < 10:
            print(f"{name:<20}{len(ev):>5}  too few rows")
            continue

        # ---- issued-only variant: abstained clips dropped ----
        issued = [(pp, yy) for pp, yy in ev
                  if scores.get(pp, {}).get("p_fake") is not None]
        y = np.array([yy for _, yy in issued])
        if len(np.unique(y)) < 2:
            print(f"{name:<20}{len(ev):>5}  single-class issued split")
            continue
        p = np.array([scores[pp]["p_fake"] for pp, _ in issued])
        X = np.array([feats[pp]["features"] for pp, _ in issued])
        z = np.column_stack([X[:, best_i], p])
        comb_p = comb.predict_proba(z)[:, 1]
        auc_p = roc_auc_score(y, p)
        auc_c = roc_auc_score(y, comb_p)
        lo, hi = bootstrap_auc(y, comb_p)

        # ---- conservative variant: abstained clips score p_fake = 0 ----
        cons_all = [(pp, yy) for pp, yy in ev]
        yc = np.array([yy for _, yy in cons_all])
        pc = np.array([0.0 if scores[pp].get("p_fake") is None
                       else scores[pp]["p_fake"] for pp, _ in cons_all])
        Xc = np.array([feats[pp]["features"] for pp, _ in cons_all])
        auc_p_cons = roc_auc_score(yc, pc)
        zc = np.column_stack([Xc[:, best_i], pc])
        auc_c_cons = roc_auc_score(yc, comb.predict_proba(zc)[:, 1])

        results[name] = {
            "n": len(ev), "n_issued": len(issued),
            "n_abstained": len(ev) - len(issued),
            "auc_p6vid": float(auc_p), "auc_comb": float(auc_c),
            "ci": (float(lo), float(hi)),
            "auc_p6vid_conservative": float(auc_p_cons),
            "auc_comb_conservative": float(auc_c_cons),
        }
        print(f"{name:<20}{len(issued):>5}{len(ev)-len(issued):>5}"
              f"{auc_p:>8.4f}{auc_c:>8.4f}{auc_c-auc_p:>+8.4f}"
              f"{auc_p_cons:>12.4f}{auc_c_cons:>11.4f}")

    print("-" * 78)

    # ---- bar evaluation ----
    print("\n" + "=" * 78)
    print("BAR EVALUATION")
    print("=" * 78)
    verdicts = {}

    if "genbuster_t2v" in results:
        b = results["genbuster_t2v"]
        d = b["auc_comb"] - b["auc_p6vid"]
        verdicts["B1"] = (d >= 0.05, f"combined {b['auc_comb']:.4f} vs p6vid "
                          f"{b['auc_p6vid']:.4f} -> delta {d:+.4f} (need >= +0.05)")
    else:
        verdicts["B1"] = (False, "no usable GenBuster T2V rows")

    for key, dom in (("B2a", "ffpp_curated"), ("B2b", "cdf_curated")):
        if dom in results:
            b = results[dom]
            d = b["auc_comb"] - b["auc_p6vid"]
            verdicts[key] = (d >= -0.02, f"{dom}: combined {b['auc_comb']:.4f} vs "
                             f"p6vid {b['auc_p6vid']:.4f} -> {d:+.4f} (need >= -0.02)")
        else:
            verdicts[key] = (False, f"no usable {dom} rows")

    verdicts["B3"] = (b3_pass, f"Pearson r = {r:.4f} (need |r| <= 0.30)")
    verdicts["B4"] = (b4_pass, "see table above (genvid_sora corpus-capped at 51)")

    if "genvid_kling" in results and "genvid_sora" in results:
        k = results["genvid_kling"]
        s = results["genvid_sora"]
        d = (k["auc_comb"] + s["auc_comb"]) / 2 - (k["auc_p6vid"] + s["auc_p6vid"]) / 2
        verdicts["B5"] = (d >= 0.0, f"kling {k['auc_comb']:.4f} vs {k['auc_p6vid']:.4f}; "
                          f"sora {s['auc_comb']:.4f} vs {s['auc_p6vid']:.4f} -> {d:+.4f}")
    else:
        verdicts["B5"] = (False, "missing kling/sora rows")

    all_pass = True
    for k in ("B1", "B2a", "B2b", "B3", "B4", "B5"):
        ok, msg = verdicts.get(k, (False, "not evaluated"))
        if not ok:
            all_pass = False
        print(f"  {k:<4} {'PASS' if ok else 'FAIL'}  {msg}")

    print("\n" + "=" * 78)
    if not stage_complete:
        print(f"INCOMPLETE — stage coverage gaps: "
              f"{len(missing_scores)} missing scores, "
              f"{len(missing_feats)} missing features")
        print("No verdict is recorded. Re-run when scoring + features are complete.")
        print("=" * 78)
        out = {
            "spec": "v1d-feasibility-v1",
            "status": "INCOMPLETE",
            "missing_scores": len(missing_scores),
            "missing_features": len(missing_feats),
            "results": results,
        }
        with open(OUT / "analysis_partial.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"saved {OUT / 'analysis_partial.json'} (not a verdict)")
        return

    print(f"OVERALL: {'PASS — advance to a sealed restack spec' if all_pass else 'FAIL — line closed at 0 with receipt'}")
    print("=" * 78)

    # ---- t2v per-generator comparison to the incumbent baseline ----
    print("\nPer-generator T2V vs incumbent (exam_video_t2v_20261011_arcface):")
    inc = {"genvid_kling": 0.6035, "genvid_sora": 0.6812, "genvid_opentora": 0.7497}
    for g, base in inc.items():
        if g in results:
            print(f"  {g:<20} incumbent {base:.4f} -> p6vid here "
                  f"{results[g]['auc_p6vid']:.4f} | combined {results[g]['auc_comb']:.4f}")

    out = {
        "spec": "v1d-feasibility-v1",
        "seed": SEED,
        "n_features": X_fit.shape[1],
        "best_feature": best_name,
        "best_feature_index": best_i,
        "best_feature_auc_fit": best_auc_fit,
        "pearson_r_fit": r,
        "combiner_coef_restrav": float(comb.coef_[0, 0]),
        "combiner_coef_p6vid": float(comb.coef_[0, 1]),
        "results": results,
        "bars": {k: {"pass": bool(v[0]), "detail": v[1]} for k, v in verdicts.items()},
        "overall_pass": bool(all_pass),
    }
    with open(OUT / "analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {OUT / 'analysis.json'}")


if __name__ == "__main__":
    main()