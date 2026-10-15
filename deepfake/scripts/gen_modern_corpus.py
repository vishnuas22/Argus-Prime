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
MJ_REAL = WS / "data" / "corpora" / "mj_train" / "real"
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


def holdout_split(slug: str, train_dir: Path, rows: list[dict],
                  n_train_expect: int = 1000) -> dict:
    """sha256-first 100 -> gen_exam/<slug>/fake (moved), rest stays train."""
    exam_fake = EXAM_ROOT / slug / "fake"
    exam_real = EXAM_ROOT / slug / "real"
    if exam_fake.exists() and len(list(exam_fake.iterdir())) == EXAM_HOLDOUT:
        print(f"[split] {slug}: exam already split, idempotent skip", flush=True)
        return json.loads((EXAM_ROOT / slug / "manifest.json").read_text())

    assert len(rows) == EXAM_HOLDOUT + n_train_expect, \
        f"{slug}: expected {EXAM_HOLDOUT + n_train_expect} generated, got {len(rows)}"
    by_hash = sorted(rows, key=lambda r: (r["sha256"], r["file"]))
    held = by_hash[:EXAM_HOLDOUT]
    train = by_hash[EXAM_HOLDOUT:]
    assert len({r["sha256"] for r in held} &
               {r["sha256"] for r in train}) == 0, "holdout/train hash overlap"

    exam_fake.mkdir(parents=True, exist_ok=True)
    exam_real.mkdir(parents=True, exist_ok=True)
    for r in held:  # MOVE: train ends at exactly 1000, disjoint by construction
        src = train_dir / r["file"]
        dst = exam_fake / r["file"]
        shutil.move(str(src), str(dst))

    # real side: reuse mj_train/real, sha256-first 100, tagged SELFGEN-REFERENCE
    assert MJ_REAL.exists(), f"missing {MJ_REAL}"
    real_files = sorted(MJ_REAL.iterdir(),
                        key=lambda p: (sha256_file(p), p.name))[:EXAM_HOLDOUT]
    exam_real_rows = []
    for p in real_files:
        dst = exam_real / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
        exam_real_rows.append({"file": p.name, "sha256": sha256_file(dst),
                               "model": slug, "role": "SELFGEN-REFERENCE",
                               "source": str(MJ_REAL.relative_to(WS))})

    train_left = [r for r in train if (train_dir / r["file"]).exists()]
    assert len(train_left) == n_train_expect, \
        f"{slug}: train leftover {len(train_left)} != {n_train_expect}"
    # P4-C22: exam fakes disjoint from train leftovers
    th = {r["sha256"] for r in train_left}
    assert not ({r["sha256"] for r in held} & th), "P4-C22 exam∩train non-empty"

    man = {"slug": slug, "role": "SELF-GENERATED",
           "disclosure": "in-domain regression gate, NOT a generalization "
                         "test (self-generated fakes vs reused reals)",
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "n_exam_fake": len(held), "n_exam_real": len(exam_real_rows),
           "n_train": len(train_left),
           "fakes": [{**r, "role": "SELF-GENERATED"} for r in held],
           "reals": exam_real_rows}
    (EXAM_ROOT / slug).mkdir(parents=True, exist_ok=True)
    (EXAM_ROOT / slug / "manifest.json").write_text(json.dumps(man, indent=2))
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
