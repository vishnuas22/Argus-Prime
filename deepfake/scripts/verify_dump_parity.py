"""W4c §3 parity law — the v4 full re-dump vs the v3b incumbent diet.

Compares artifacts/restack_video_v4/{dump_train,dump_val,dump_cdf}/scores.csv
against artifacts/restack_video_v3b's, rows aligned by the (method, file) key:
  (a) row counts 2284/495/1000 and full-population (method, file, sha256_12)
      SET equality per dump;
  (b) noface reconciliation: v4's noface sets == v3b's (dump_train exactly the
      1 receipted DFD fake 06_15__outside_talking_still_laughing__QRCD27P8.mp4;
      dump_val/dump_cdf none);
  (c) sampled value parity: per dump, the first 25 label==0 + first 25
      label==1 rows in CSV order (deterministic sample of 50; sample size is a
      design constant, not a bar) — EVERY legacy column value string-identical
      to v3b's (all 34 columns: file, sha256_12, label, method, split, route,
      naive_combined, identity_video_score, drift_l2, the 24 features,
      n_drift_tracks). SCENE columns are excluded (they are the new content).
Any violation -> exit 2 + artifacts/restack_video_v4/parity_receipt.json
listing every violating (row, column, v4, v3b) + a MANIFEST section. A
violation means featurizer drift -> STOP (D-046; no tolerance re-seal by the
builder — escalation to the user with the receipt).

W4c §3b RE-BASE (user-ruled 2026-11-02, D-073 — supersedes §3's reference
values): the two drift columns (drift_l2, drift_p95_max) are EXCLUDED from
string-identity — the environment-vintage drift (ArcFace embeddings ~1e-4,
invisible at 4dp) is receipted instead with measured delta bounds; the 32
non-drift legacy columns remain string-identical (any NEW violation there ->
STOP + escalate per §3b clause 4).

Usage: ./.venv/bin/python deepfake/scripts/verify_dump_parity.py
       [--work artifacts/restack_video_v4] [--ref artifacts/restack_video_v3b]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))
from video_features import FEATURE_NAMES  # noqa: E402

DUMPS = ["dump_train", "dump_val", "dump_cdf"]
EXPECTED_ROWS = {"dump_train": 2284, "dump_val": 495, "dump_cdf": 1000}
SAMPLE_N = 25   # per label; 50 rows per dump (design constant, spec §3)
NOFACE_EXPECTED = {
    "dump_train": {("DeepFakeDetection",
                    "06_15__outside_talking_still_laughing__QRCD27P8.mp4")},
    "dump_val": set(),
    "dump_cdf": set(),
}
LEGACY_COLS = ["file", "sha256_12", "label", "method", "split", "route",
               "naive_combined", "identity_video_score", "drift_l2"] \
    + FEATURE_NAMES + ["n_drift_tracks"]
# §3b clause 1 (D-073): excluded from string-identity, delta-bounded instead.
DRIFT_EXCLUDED = ("drift_l2", "drift_p95_max")


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def noface_set(dump_dir: Path) -> set:
    p = dump_dir / "noface.jsonl"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out.add((r.get("method", ""), r["file"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(WS / "artifacts" / "restack_video_v4"))
    ap.add_argument("--ref", default=str(WS / "artifacts" / "restack_video_v3b"))
    args = ap.parse_args()
    work, ref = Path(args.work), Path(args.ref)

    violations = []
    receipt = {"created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "work": str(work), "ref": str(ref),
               "law": "W4c §3b re-based parity (D-073) — drift columns "
                      "delta-bounded, 32 non-drift columns string-identical",
               "legacy_cols": len(LEGACY_COLS), "sample_n_per_label": SAMPLE_N,
               "dumps": {}, "violations": []}

    for name in DUMPS:
        v4p, r3p = work / name / "scores.csv", ref / name / "scores.csv"
        if not v4p.exists():
            raise SystemExit(f"missing {v4p} — run the full re-dump first")
        if not r3p.exists():
            raise SystemExit(f"missing reference {r3p}")
        v4_rows, r3_rows = read_rows(v4p), read_rows(r3p)
        expected = EXPECTED_ROWS[name]

        # (a) row counts + full-population (method, file, sha256_12) sets
        count_ok = len(v4_rows) == expected and len(r3_rows) == expected
        v4_keys = [(r["method"], r["file"]) for r in v4_rows]
        r3_keys = [(r["method"], r["file"]) for r in r3_rows]
        v4_set = {(m, f, r["sha256_12"]) for r, (m, f) in zip(v4_rows, v4_keys)}
        r3_set = {(m, f, r["sha256_12"]) for r, (m, f) in zip(r3_rows, r3_keys)}
        set_ok = v4_set == r3_set
        dup_v4 = len(v4_keys) - len(set(v4_keys))
        dup_r3 = len(r3_keys) - len(set(r3_keys))
        if dup_v4 or dup_r3:
            violations.append({"dump": name, "row": "(duplicate keys)",
                               "column": "(method, file)",
                               "v4": f"{dup_v4} duplicate keys",
                               "v3b": f"{dup_r3} duplicate keys"})
        if not count_ok:
            violations.append({"dump": name, "row": "(row count)",
                               "column": "n_rows", "v4": len(v4_rows),
                               "v3b": len(r3_rows), "expected": expected})
        if not set_ok:
            only_v4 = sorted(v4_set - r3_set)
            only_r3 = sorted(r3_set - v4_set)
            for k in only_v4:
                violations.append({"dump": name, "row": str(k),
                                   "column": "(method, file, sha256_12)",
                                   "v4": "present", "v3b": "absent"})
            for k in only_r3:
                violations.append({"dump": name, "row": str(k),
                                   "column": "(method, file, sha256_12)",
                                   "v4": "absent", "v3b": "present"})

        # (b) noface reconciliation
        v4_nf, r3_nf = noface_set(work / name), noface_set(ref / name)
        nf_ok = v4_nf == r3_nf == NOFACE_EXPECTED[name]
        if not nf_ok:
            violations.append({"dump": name, "row": "(noface.jsonl)",
                               "column": "(method, file) set",
                               "v4": sorted(v4_nf), "v3b": sorted(r3_nf),
                               "expected": sorted(NOFACE_EXPECTED[name])})

        # (c) sampled value parity — first SAMPLE_N per label in v4 CSV order,
        # aligned to v3b by (method, file); the 32 non-drift legacy columns as
        # strings (§3b), the 2 drift columns delta-bounded (excluded from
        # string-identity per the D-073 re-base).
        r3_by_key = {}
        for r, k in zip(r3_rows, r3_keys):
            r3_by_key.setdefault(k, r)
        sample, seen = {"0": 0, "1": 0}, set()
        n_compared = 0
        drift_deltas = {c: [] for c in DRIFT_EXCLUDED}
        for r, k in zip(v4_rows, v4_keys):
            lab = r["label"]
            if lab not in sample or sample[lab] >= SAMPLE_N:
                continue
            sample[lab] += 1
            seen.add(k)
            counter = r3_by_key.get(k)
            if counter is None:
                violations.append({"dump": name, "row": str(k),
                                   "column": "(alignment)",
                                   "v4": "sampled row", "v3b": "missing"})
                continue
            for col in LEGACY_COLS:
                if col not in r or col not in counter:
                    violations.append({"dump": name, "row": str(k),
                                       "column": col, "v4": r.get(col, "<absent>"),
                                       "v3b": counter.get(col, "<absent>")})
                    continue
                if col in drift_deltas:
                    if col == "drift_l2":
                        a = [float(x) for x in r[col].split()]
                        b = [float(x) for x in counter[col].split()]
                        if len(a) == len(b):
                            drift_deltas[col].extend(x - y for x, y in zip(a, b))
                        else:
                            violations.append({"dump": name, "row": str(k),
                                               "column": col,
                                               "v4": f"{len(a)} steps",
                                               "v3b": f"{len(b)} steps"})
                    else:
                        drift_deltas[col].append(float(r[col])
                                                 - float(counter[col]))
                    continue
                if r[col] != counter[col]:
                    violations.append({"dump": name, "row": str(k),
                                       "column": col, "v4": r[col],
                                       "v3b": counter[col]})
            n_compared += 1
        undersampled = [lab for lab, n in sample.items() if n < SAMPLE_N]
        if undersampled:
            violations.append({"dump": name, "row": "(sample)",
                               "column": "label", "v4": f"only "
                               f"{[sample[l] for l in undersampled]} rows",
                               "v3b": f"expected {SAMPLE_N} per label"})
        def _bounds(ds):
            if not ds:
                return {"n": 0}
            return {"n": len(ds),
                    "min": round(min(ds), 6), "max": round(max(ds), 6),
                    "abs_mean": round(sum(abs(d) for d in ds) / len(ds), 6)}
        receipt["dumps"][name] = {
            "rows_v4": len(v4_rows), "rows_v3b": len(r3_rows),
            "expected": expected, "count_ok": count_ok,
            "key_set_equal": set_ok, "dup_keys_v4": dup_v4,
            "dup_keys_v3b": dup_r3, "noface_v4": sorted(v4_nf),
            "noface_v3b": sorted(r3_nf), "noface_ok": nf_ok,
            "sampled_rows_compared": n_compared,
            "string_cols_enforced": len(LEGACY_COLS) - len(DRIFT_EXCLUDED),
            "drift_delta_bounds": {c: _bounds(ds)
                                   for c, ds in drift_deltas.items()},
            "sampled_violations": sum(
                1 for v in violations if v["dump"] == name)}

    receipt["violations"] = violations
    receipt["result"] = "VIOLATIONS" if violations else "CLEAN"
    (work / "parity_receipt.json").write_text(json.dumps(receipt, indent=2))

    lines = ["## W4c §3 re-dump parity receipt", ""]
    if violations:
        lines.append(f"- RESULT: VIOLATIONS ({len(violations)}) — featurizer "
                     f"drift suspected; STOP per D-046 (no tolerance re-seal)")
        for v in violations[:50]:
            lines.append(f"  - {v['dump']} row {v['row']} col {v['column']}: "
                         f"v4={v['v4']!r} v3b={v['v3b']!r}")
        if len(violations) > 50:
            lines.append(f"  - ... {len(violations) - 50} more in "
                         f"parity_receipt.json")
    else:
        lines.append(f"- RESULT: CLEAN (§3b re-based, D-073) — row counts "
                     f"{'/'.join(str(EXPECTED_ROWS[d]) for d in DUMPS)}; "
                     f"(method, file, sha256_12) set equality all three dumps; "
                     f"noface sets equal (1 receipted train DFD fake); "
                     f"{SAMPLE_N * 2} sampled rows per dump x "
                     f"{len(LEGACY_COLS) - len(DRIFT_EXCLUDED)} non-drift "
                     f"legacy columns string-identical; drift columns "
                     f"delta-bounded in the receipt")
        for name in DUMPS:
            for c, b in receipt["dumps"][name]["drift_delta_bounds"].items():
                if b.get("n"):
                    lines.append(f"  - {name} {c}: n={b['n']} "
                                 f"min={b['min']:+.6f} max={b['max']:+.6f} "
                                 f"abs_mean={b['abs_mean']:.6f}")
    lines.append(f"- receipt: {work / 'parity_receipt.json'}")
    with open(work / "MANIFEST.md", "a") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[parity] {receipt['result']}: "
          f"{sum(1 for v in violations)} violations across {len(DUMPS)} dumps",
          flush=True)
    for name in DUMPS:
        d = receipt["dumps"][name]
        print(f"[parity] {name}: rows {d['rows_v4']}/{d['rows_v3b']} "
              f"(expected {d['expected']}), set_equal={d['key_set_equal']}, "
              f"noface_ok={d['noface_ok']}, "
              f"sampled={d['sampled_rows_compared']}"
              f"x{d['string_cols_enforced']}+{len(DRIFT_EXCLUDED)}drift-bounded",
              flush=True)
    if violations:
        print(f"[parity] receipt: {work / 'parity_receipt.json'} — STOP "
              f"(D-046 escalation, no tolerance re-seal)", flush=True)
        return 2
    print(f"[parity] receipt: {work / 'parity_receipt.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
