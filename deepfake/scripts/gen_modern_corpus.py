#!/usr/bin/env python3
"""P4-B.1 local generation — mflux-based self-generated corpora.

Spec B.1:
  models: flux2-klein-4b (4 steps) only; HiDream-I1 feasibility NO-GO
          (45.55 GB fp16 > 35 GB, no MLX port) — escalated, not substituted.
  P4-C10 prompt bank  : deepfake/scripts/prompt_bank_p4.json (64 prompts,
                        16/category, ids 0..63), consumed cyclically
                        prompt_id = i % 64.
  P4-C11 seeds        : seed_i = 150_000 + model_index*1_000 + i
                        (model_index 0 = flux2, 1 = hidream).
  resolution          : 1024x1024, steps = 4 (distilled klein).
  n                   : 1100 per model -> 100 SELF-GENERATED exam
                        (sha256-first 100, moved out) + 1000 train.

Outputs:
  data/corpora/gen_intake_train/<slug>/*.jpg  + manifest.jsonl (train rows)
  data/corpora/gen_exam/<slug>/{fake,real}/   + manifest.json  (role tagged)

Resume: manifest-keyed by seed (skips already-generated images).
        --force overwrites the whole staging dir (guarded, non-empty check).
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
PROMPT_BANK = WS / "deepfake" / "scripts" / "prompt_bank_p4.json"
TRAIN_ROOT = WS / "data" / "corpora" / "gen_intake_train"
EXAM_ROOT = WS / "data" / "corpora" / "gen_exam"
GEN_REAL = WS / "data" / "corpora" / "gen_train" / "real"
ART_SMOKE = WS / "artifacts" / "gen_smoke"

# B.1 model registry — HiDream intentionally absent (NO-GO, escalated).
MODELS = {
    "flux2-klein-4b": {"model_index": 0, "steps": 4, "n": 1100,
                       "kind": "flux2"},
    "hidream-i1": {"no_go": "fp16 45.55 GB > 35 GB AND no MLX port "
                            "for HiDream-I1 (feasibility gate NO-GO; "
                            "Kolors substitution NOT approved — spec B.1 "
                            "requires user approval)"},
}
EXAM_HOLDOUT = 100  # P4-C10/C11 split
SEED_BASE = 150_000  # P4-C11
SEED_MODEL_STRIDE = 1_000  # P4-C11
RES = 1024

# Real-side carve-out pool (user ruling, D-048/D-049 close): full ranked
# gen-train-real pool = aigen 0_real, bench-readable exts. Ordering identical
# to build_intake_train.py's gen_train/real materialization.
AIGEN_REAL_SRC = (WS / "data" / "corpora" / "gen_intake" / "aigenimages2026" /
                  "aigenimages2026" / "mnt" / "cephfs" / "home" / "common" /
                  "datasets" / "online_sid_dataset" / "AIGenImages" /
                  "AIGenImages2026" / "train" / "0_real")
POOL_EXPECT = 4749   # bench-readable 0_real count (P4-C22 collisions=0 @ B.2)
TRAIN_REAL_N = 1000  # gen_train/real holds pool ranks 100..1099


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank() -> list[dict]:
    bank = json.loads(PROMPT_BANK.read_text())
    prompts = bank["prompts"]
    assert len(prompts) == 64, f"P4-C10: expected 64 prompts, got {len(prompts)}"
    assert [p["id"] for p in prompts] == list(range(64)), "P4-C10: ids must be 0..63"
    cats = {}
    for p in prompts:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    assert set(cats.values()) == {16}, f"P4-C10: unbalanced categories {cats}"
    return prompts


def image_stem(i: int) -> str:
    return f"gen_{i:05d}.jpg"


def load_manifest(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def build_model(spec: dict):
    from mflux.models.flux2.variants import Flux2Klein
    if spec["kind"] == "flux2":
        return Flux2Klein()
    raise SystemExit(f"unsupported model kind: {spec['kind']}")


def generate(slug: str, spec: dict, out_dir: Path, n: int,
             smoke: bool, force: bool) -> list[dict]:
    if force and out_dir.exists() and any(out_dir.iterdir()) and not smoke:
        print(f"[gen] --force: clearing staging {out_dir}", flush=True)
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mf_p = out_dir / "manifest.jsonl"
    if force and not smoke and mf_p.exists():
        mf_p.unlink()
    rows = load_manifest(mf_p)
    done_seeds = {r["seed"] for r in rows}
    prompts = load_bank()

    if not rows:
        print(f"[gen] loading {slug} (first image downloads weights on "
              f"first run)...", flush=True)
    model = build_model(spec)
    print(f"[gen] model ready; generating {n - len(rows)} new images "
          f"({len(rows)}/{n} already done)", flush=True)

    t0 = time.time()
    for i in range(n):
        seed = SEED_BASE + spec["model_index"] * SEED_MODEL_STRIDE + i
        if seed in done_seeds:
            continue
        prompt = prompts[i % 64]  # P4-C10: deterministic cyclic assignment
        img = model.generate_image(
            seed=seed,
            prompt=prompt["text"],
            num_inference_steps=spec["steps"],
            height=RES,
            width=RES,
        )
        fp = out_dir / image_stem(i)
        img.save(fp)
        rec = {"file": fp.name, "sha256": sha256_file(fp), "model": slug,
               "steps": spec["steps"], "seed": seed,
               "prompt_id": prompt["id"]}
        with open(mf_p, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        rows.append(rec)
        if (len(rows) % 10) == 0 or len(rows) == n:
            el = time.time() - t0
            rate = el / max(1, len(rows) - (n - (n - len(rows))))
            print(f"[gen] {slug} {len(rows)}/{n} "
                  f"({el/60:.1f} min elapsed, last seed={seed} "
                  f"prompt={prompt['id']})", flush=True)
    print(f"[gen] {slug} done: {len(rows)} images in {(time.time()-t0)/60:.1f} "
          f"min -> {out_dir}", flush=True)
    return rows


def real_pool_ranking() -> list[tuple[str, Path]]:
    """Full ranked gen-train-real pool: aigen 0_real, bench-readable exts,
    sorted by (sha256, name). Deterministic and reproducible; 4749 files
    (excluded==0 vs every exam/train corpus at B.2 time, so no exclusion
    filter is re-applied here)."""
    assert AIGEN_REAL_SRC.is_dir(), f"missing {AIGEN_REAL_SRC}"
    rows = [(sha256_file(p), p) for p in AIGEN_REAL_SRC.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    rows.sort(key=lambda t: (t[0], t[1].name))
    assert len(rows) == POOL_EXPECT, \
        f"gen-train-real pool {len(rows)} != {POOL_EXPECT} — source drifted"
    return rows


def carveout_real_pool(exam_real: Path) -> list[dict]:
    """Carve the pool so exam reals are DISJOINT from the training reals
    (user ruling: training-seen reals would leniently inflate gate AUC —
    D-035-class leakage, not acceptable even for a regression gate).

      pool ranks 0..99     -> gen_exam/<slug>/real  (SELFGEN-REFERENCE)
      pool ranks 100..1099 -> gen_train/real        (r0100..r1099)

    Rebuilds both sides from the pool so re-runs are stable and the
    exam/train split is disjoint by construction. The gen_train/real
    manifest.jsonl is rewritten to match the carve-out ranks.
    """
    pool = real_pool_ranking()

    exam_real.mkdir(parents=True, exist_ok=True)
    for p in exam_real.iterdir():
        if p.is_file():
            p.unlink()
    exam_rows = []
    for h, src in pool[:EXAM_HOLDOUT]:
        dst = exam_real / src.name
        shutil.copy2(src, dst)
        exam_rows.append({"file": dst.name, "sha256": h,
                          "role": "SELFGEN-REFERENCE",
                          "source": str(AIGEN_REAL_SRC.relative_to(WS))})

    GEN_REAL.mkdir(parents=True, exist_ok=True)
    for p in GEN_REAL.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png",
                                                ".webp", ".bmp"}:
            p.unlink()
    train_rows = []
    for rank, (h, src) in enumerate(
            pool[EXAM_HOLDOUT:EXAM_HOLDOUT + TRAIN_REAL_N],
            start=EXAM_HOLDOUT):
        dst = GEN_REAL / f"r{rank:04d}_{src.name}"
        shutil.copy2(src, dst)
        train_rows.append({"file": dst.name, "sha256": h, "rank": rank,
                           "source": str(AIGEN_REAL_SRC.relative_to(WS))})
    (GEN_REAL / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train_rows))
    return exam_rows


def _reals_disjoint(exam_real: Path) -> bool:
    if not (exam_real.exists() and GEN_REAL.exists()):
        return False
    eh = {sha256_file(p) for p in exam_real.iterdir() if p.is_file()}
    th = {sha256_file(p) for p in GEN_REAL.iterdir()
          if p.is_file() and p.suffix.lower() != ".jsonl"}
    return bool(eh) and bool(th) and not (eh & th)


def holdout_split(slug: str, train_dir: Path, rows: list[dict],
                  n_train_expect: int = 1000) -> dict:
    """sha256-first 100 -> gen_exam/<slug>/fake (moved), rest stays train.

    Real side (user ruling, D-048/D-049 close): exam reals are carved
    OUT of the full ranked gen-train-real pool so they are DISJOINT from
    gen_train/real — pool ranks 0..99 -> exam (SELFGEN-REFERENCE), ranks
    100..1099 -> gen_train/real. Training-seen reals would leniently
    inflate gate AUC (D-035-class leakage), so the carve-out is
    mandatory even for a regression gate.

    Idempotent + repairable: skips only when the manifest already carries
    the carved-out reals AND exam/train reals verify disjoint on disk;
    a legacy split (mj_train/real reals, written by the pre-carve-out
    in-memory copy of this module) keeps its held-out fakes and rebuilds
    the real side via the carve-out.
    """
    exam_dir = EXAM_ROOT / slug
    exam_fake = exam_dir / "fake"
    exam_real = exam_dir / "real"
    exam_manifest = exam_dir / "manifest.json"

    def _write_real_side(fake_rows: list[dict], n_train: int) -> dict:
        real_rows = carveout_real_pool(exam_real)
        assert _reals_disjoint(exam_real), \
            f"{slug}: exam/gen_train real overlap after carve-out"
        man = {"slug": slug, "role": "SELF-GENERATED",
               "disclosure": "in-domain regression gate, NOT a generalization "
                             "test (self-generated fakes vs reused reals); "
                             "exam reals disjoint from gen_train/real by "
                             "construction (carve-out, ranks 0-99 exam / "
                             "100-1099 train)",
               "generated_at": datetime.now(timezone.utc).isoformat(),
               "n_exam_fake": len(fake_rows), "n_exam_real": len(real_rows),
               "n_train": n_train,
               "fakes": fake_rows, "reals": real_rows}
        exam_dir.mkdir(parents=True, exist_ok=True)
        exam_manifest.write_text(json.dumps(man, indent=2))
        return man

    # already carved out (manifest reals from the pool AND disjoint on disk)
    if (exam_fake.exists() and exam_manifest.exists()
            and len(list(exam_fake.iterdir())) == EXAM_HOLDOUT):
        man = json.loads(exam_manifest.read_text())
        reals = man.get("reals", [])
        carved = bool(reals) and all(
            str(r.get("source", "")) == str(AIGEN_REAL_SRC.relative_to(WS))
            for r in reals)
        if carved and _reals_disjoint(exam_real):
            print(f"[split] {slug}: carve-out exam/train reals disjoint, "
                  f"idempotent skip", flush=True)
            return man
        # not yet carved (legacy mj_train/real reals): repair reals only,
        # keep the held-out fakes the batch already moved
        held = [{"file": p.name, "sha256": sha256_file(p), "model": slug,
                 "role": "SELF-GENERATED"}
                for p in sorted(exam_fake.iterdir()) if p.is_file()]
        assert len(held) == EXAM_HOLDOUT, \
            f"{slug}: exam fake count {len(held)} != {EXAM_HOLDOUT}"
        train_left = [r for r in rows if (train_dir / r["file"]).exists()]
        assert len(train_left) == n_train_expect, \
            f"{slug}: train leftover {len(train_left)} != {n_train_expect}"
        th = {r["sha256"] for r in train_left}
        assert not ({r["sha256"] for r in held} & th), "P4-C22 exam∩train non-empty"
        man = _write_real_side(held, len(train_left))
        print(f"[split] {slug}: real side carved out "
              f"(exam {man['n_exam_real']} / train {TRAIN_REAL_N}, disjoint)",
              flush=True)
        return man

    assert len(rows) == EXAM_HOLDOUT + n_train_expect, \
        f"{slug}: expected {EXAM_HOLDOUT + n_train_expect} generated, got {len(rows)}"
    by_hash = sorted(rows, key=lambda r: (r["sha256"], r["file"]))
    held = by_hash[:EXAM_HOLDOUT]
    train = by_hash[EXAM_HOLDOUT:]
    assert len({r["sha256"] for r in held} &
               {r["sha256"] for r in train}) == 0, "holdout/train hash overlap"

    exam_fake.mkdir(parents=True, exist_ok=True)
    for r in held:  # MOVE: train ends at exactly 1000, disjoint by construction
        src = train_dir / r["file"]
        dst = exam_fake / r["file"]
        shutil.move(str(src), str(dst))

    train_left = [r for r in train if (train_dir / r["file"]).exists()]
    assert len(train_left) == n_train_expect, \
        f"{slug}: train leftover {len(train_left)} != {n_train_expect}"
    # P4-C22: exam fakes disjoint from train leftovers
    th = {r["sha256"] for r in train_left}
    assert not ({r["sha256"] for r in held} & th), "P4-C22 exam∩train non-empty"

    man = _write_real_side([{**r, "role": "SELF-GENERATED"} for r in held],
                           len(train_left))
    print(f"[split] {slug}: 100 held out -> {exam_fake}, "
          f"{len(train_left)} train remain", flush=True)
    return man


def dedup_assert(slug: str, held_hashes: set, train_hashes: set) -> dict:
    """P4-C22: exam fakes vs all named train sets -> 0 collisions."""
    sets = {"gen_intake_train/" + slug: train_hashes}
    for name, path in [("mj_train", WS / "data" / "corpora" / "mj_train" / "fake"),
                       ("flux_train", WS / "data" / "corpora" / "flux_train" / "fake"),
                       ("modern_exam", WS / "data" / "corpora" / "modern_exam" / "fake_mj")]:
        if path.exists():
            sets[name] = {sha256_file(p) for p in path.rglob("*")
                          if p.is_file() and p.suffix.lower() in
                          {".jpg", ".jpeg", ".png", ".webp", ".bmp"}}
    violations = {k: len(held_hashes & v) for k, v in sets.items()}
    assert all(v == 0 for v in violations.values()), \
        f"P4-C22 dedup violations {violations}"
    print(f"[dedup] {slug}: all-zero violations {violations}", flush=True)
    return violations


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--n", type=int, default=None,
                    help="images to generate (default: registry n)")
    ap.add_argument("--smoke", type=int, default=None,
                    help="smoke mode: generate N images into artifacts/gen_smoke")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-split", action="store_true",
                    help="skip holdout split (smoke)")
    ap.add_argument("--split-only", action="store_true",
                    help="skip generation; run/repair the holdout split on "
                         "the existing manifest.jsonl (used by "
                         "stage_b_auto.sh after the batch exits — repairs a "
                         "legacy mj_train/real real-side in place)")
    args = ap.parse_args()

    spec = MODELS[args.model]
    if spec.get("no_go"):
        print(f"NO-GO {args.model}: {spec['no_go']}", file=sys.stderr)
        return 3
    n = args.n or spec["n"]
    if args.smoke:
        n = args.smoke

    bank_hash = hashlib.sha256(PROMPT_BANK.read_bytes()).hexdigest()
    print(f"[p4-c10] prompt bank {PROMPT_BANK.name} sha256={bank_hash}", flush=True)

    if args.smoke:
        out_dir = ART_SMOKE / args.model
        rows = generate(args.model, spec, out_dir, n, smoke=True,
                        force=args.force)
        print(f"[smoke] complete: {len(rows)} images -> {out_dir}", flush=True)
        return 0

    out_dir = TRAIN_ROOT / args.model
    if args.split_only:
        mf_p = out_dir / "manifest.jsonl"
        if not mf_p.exists():
            raise SystemExit(f"missing {mf_p} — generation not complete?")
        rows = load_manifest(mf_p)
        print(f"[split-only] {args.model}: {len(rows)} manifest rows",
              flush=True)
        man = holdout_split(args.model, out_dir, rows)
        train_rows = [r for r in rows if (out_dir / r["file"]).exists()]
        held = {man["fakes"][j]["sha256"] for j in range(man["n_exam_fake"])}
        train_h = {r["sha256"] for r in train_rows}
        dedup_assert(args.model, held, train_h)
        print(f"[split-only] done: exam {man['n_exam_fake']}f/"
              f"{man['n_exam_real']}r, train {man['n_train']}", flush=True)
        return 0
    t0 = time.time()
    rows = generate(args.model, spec, out_dir, n, smoke=False,
                    force=args.force)
    man = None
    if not args.no_split:
        man = holdout_split(args.model, out_dir, rows)
        train_rows = [r for r in rows
                      if (out_dir / r["file"]).exists()]
        held = set(man["fakes"][j]["sha256"] for j in range(man["n_exam_fake"]))
        train_h = {r["sha256"] for r in train_rows}
        dedup_assert(args.model, held, train_h)
    print(f"[done] {args.model} wall {(time.time()-t0)/60:.1f} min "
          f"n={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
