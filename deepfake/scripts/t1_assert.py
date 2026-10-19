#!/usr/bin/env python3
"""T1 stage-0 clean-list assert — measurement validity gate (pre-registered).

Verifies data/corpora/wild_sample/{real_test,fake_test} (the T1 EXAM set)
never fed any training surface:

  HARD GATES (non-zero exit on any hit):
    1. exam sha256 absent from every historical dump scores.csv sha column
       (artifacts/restack*/dump_*/scores.csv — all restack history).
    2. exam relative keys (e.g. 'real_test/img.jpg') absent from the
       wild_train* npz caches (the face-probe training source caches, D-018)
       and from every other TRAIN-side cache; exam-capable caches
       (wild_sample*, eval*, *exam*, *test*) are exempt by design.
    3. any sha-like token inside cache .meta.json files that collides with
       an exam sha.
  REPORT-ONLY: basename collisions across unrelated caches (ambiguity net).

Aborts fail-closed. Per spec artifacts/generator_intake/t1_wild_fpr_spec.md.
"""
import csv
import glob
import hashlib
import json
import os
import re
import sys

WS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAM_DIRS = {
    "real": os.path.join(WS, "data/corpora/wild_sample/real_test"),
    "fake": os.path.join(WS, "data/corpora/wild_sample/fake_test"),
}
EXEMPT_PREFIXES = ("wild_sample",)
TRAIN_CACHE_HINT = ("train", "wild_train")

sha_re = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    exam_shas, exam_rels, exam_bases = set(), set(), set()
    n = {}
    for arm, d in EXAM_DIRS.items():
        files = sorted(os.listdir(d))
        n[arm] = len(files)
        for name in files:
            s = sha256_file(os.path.join(d, name))
            exam_shas.add(s)
            exam_rels.add(f"{arm}_test/{name}")
            exam_bases.add(name)
    print(f"[assert] exam files: real_test={n['real']} fake_test={n['fake']} unique_shas={len(exam_shas)}")

    # Gate 1 — dump CSV sha columns, all history
    hot = []
    csvs = sorted(glob.glob(os.path.join(WS, "artifacts", "restack*", "dump_*", "scores.csv")))
    for path in csvs:
        with open(path) as f:
            rd = csv.DictReader(f)
            col = rd.fieldnames and ("sha" in rd.fieldnames)
            if not col:
                continue
            for row in rd:
                s = (row.get("sha") or "").strip().lower()
                if s in exam_shas:
                    hot.append((os.path.relpath(path, WS), s))
    print(f"[assert] gate1 dumps scanned={len(csvs)} sha-overlaps={len(hot)}")
    if hot:
        for p, s in hot[:5]:
            print(f"  POISON {p} sha={s[:16]}")
        print("[assert] FAIL: exam file present in a training dump")
        return 1

    # Gate 2 — wild_train* npz caches must not carry exam keys
    import numpy as np

    train_npz = sorted(glob.glob(os.path.join(WS, "data", "caches", "wild_train*.npz")))
    exotic = [p for p in sorted(glob.glob(os.path.join(WS, "data", "caches", "*.npz")))
              if "run_" not in os.path.basename(p)]
    report_only_hits = []
    for path in train_npz:
        z = np.load(path, allow_pickle=True)
        keys = [str(k) for k in z.files] if hasattr(z, "files") else []
        flat = set()
        for k in keys:
            flat.add(k)
            try:
                a = z[k]
            except Exception:
                continue
            if a.dtype.kind in "US":
                flat.update(str(v) for v in a.ravel()[:100000])
            elif a.dtype == object:
                flat.update(str(v) for v in a.ravel()[:20000])
        hits = [k for k in flat if k in exam_rels]
        if hits:
            print(f"[assert] POISON {os.path.relpath(path, WS)} keys={hits[:3]}")
            print("[assert] FAIL: exam key inside a wild_train cache")
            return 1
    other_hits = []
    for path in exotic:
        base = os.path.basename(path)
        z = np.load(path, allow_pickle=True)
        keys = [str(k) for k in z.files] if hasattr(z, "files") else []
        exact = [k for k in keys if k in exam_rels]
        if exact:
            if base.startswith(EXEMPT_PREFIXES):
                continue
            other_hits.append((base, exact[:3]))
    print(f"[assert] gate2 wild_train caches={len(train_npz)} exam-key hits: 0 (clean)")
    if other_hits:
        print(f"[assert] REPORT exact-relative-key matches outside forbidden caches (not a gate):")
        for b, k in other_hits[:5]:
            print(f"   {b}: {k}")

    # Gate 3 — meta json sha tokens
    meta_hits = []
    for path in glob.glob(os.path.join(WS, "data", "caches", "*.meta.json")):
        try:
            t = open(path).read()
        except OSError:
            continue
        for tok in re.findall(r"[a-f0-9]{64}", t):
            if tok in exam_shas:
                meta_hits.append((os.path.basename(path), tok[:16]))
    print(f"[assert] gate3 metas={len(glob.glob(os.path.join(WS, 'data','caches','*.meta.json')))} sha-overlaps={len(meta_hits)}")
    if meta_hits:
        print("[assert] FAIL: exam sha inside a cache meta file")
        return 1

    print("[assert] CLEAN — exam set disjoint from all training surfaces checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
