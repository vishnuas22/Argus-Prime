"""Court scoreboard — one-command regression suite over all standing exams.

Runs the CURRENT court stacker (--stacker, default the deployed p6v3) over the
standing exam sets and diffs each fused AUC against artifacts/golden.json.

  .venv/bin/python deepfake/scripts/scoreboard.py                 # verify vs golden
  .venv/bin/python deepfake/scripts/scoreboard.py --update-golden # (re)baseline
  .venv/bin/python deepfake/scripts/scoreboard.py --limit 50      # fast pass

Exit code 1 if any exam regresses beyond tolerance (CI-ready).
"""
import argparse
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from restack import run_bench, build_ffpp_gate, auc_from_scores  # noqa: E402

# The deployed court stacker — imported from argus_detect so the scoreboard
# always verifies what is actually wired (D-024: no consumer hardcodes a tag).
sys.path.insert(0, str(WS / "deepfake"))
from argus_detect import STACKER  # noqa: E402
GOLDEN = WS / "artifacts" / "golden.json"
WORK = WS / "artifacts" / "scoreboard"
TOL = -0.01  # allowed AUC drop vs golden (sampling noise headroom)

EXAMS = [  # (name, real, fake, default limit)
    ("aigc", "data/corpora/eval_real", "data/corpora/eval_fake", 100),
    ("cdf", "data/corpora/cdf_flat_real", "data/corpora/cdf_flat_fake", 100),
    ("ffpp", None, None, 100),
    ("mj", "data/corpora/modern_exam/real_ext", "data/corpora/modern_exam/fake_mj", 150),
    ("flux", "data/corpora/modern_exam/real_ext", "data/corpora/modern_exam/fake_flux", 150),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stacker", default=str(STACKER))
    ap.add_argument("--limit", type=int, default=0, help="override per-exam limits")
    ap.add_argument("--update-golden", action="store_true")
    args = ap.parse_args()

    stacker = Path(args.stacker)
    tag = stacker.parent.name
    golden = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
    work = WORK / tag
    results, regressions = {}, []
    t0 = time.time()
    for name, real, fake, lim in EXAMS:
        limit = args.limit or lim
        out = work / f"exam_{name}" / (f"limit{limit}")
        if name == "ffpp":
            real, fake = build_ffpp_gate(work, limit)
        if not (out / "scores.csv").exists():
            run_bench(real, fake, out, limit, stackers=[stacker])
        auc, n = auc_from_scores(out / "scores.csv", tag)
        results[name] = {"auc": round(auc, 4), "n": n, "limit": limit}
        line = f"{name:6s} auc={auc:.4f} (n={n})"
        if name in golden and not args.update_golden:
            g = golden[name]["auc"]
            delta = auc - g
            line += f"  golden={g:.4f} delta={delta:+.4f}"
            if delta < TOL:
                line += "  REGRESSION"
                regressions.append(name)
        print(line, flush=True)
        out.mkdir(parents=True, exist_ok=True)
    print(f"\nstacker: {tag} | {time.time()-t0:.0f}s")
    if args.update_golden:
        golden = {k: {"auc": v["auc"], "limit": v["limit"]} for k, v in results.items()}
        GOLDEN.write_text(json.dumps(golden, indent=2))
        print(f"golden baselined -> {GOLDEN}")
        return 0
    if not golden:
        print("no golden file — run with --update-golden to baseline")
        return 0
    print("STATUS:", "FAIL (regression)" if regressions else "PASS")
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
