#!/usr/bin/env python3
"""Phase 4 Stage B.2 — aigenimages2026 train-shard materialization.

Spec B.2 + D-049: corpus shards from aigenimages2026 (on disk, CC-BY-4.0) are
materialized ONLY for generator families that BOTH overlap a failing Stage A
exam slug AND have an exam gate (D-027: no ungated deployed domains).
P4-C04: aigenimages2026 is TRAIN-ONLY — never used to build/refresh exams.

Family mapping (map by family name; raw Omni generator strings in parens):
  fal-ai_flux-pro_v1.1 + fal-ai_flux_dev   -> flux-1-pro    ('Flux.1_pro', FLUX.1 family)
  fal-ai_gpt-image-1 + fal-ai_gpt-image-1.5 -> gpt4o        ('GPT4o'; GPT-4o image gen)
  fal-ai_hidream-i1-dev                    -> hidream       ('Hidream')
  fal-ai_ideogram_v3                       -> ideogram3     ('Ideogram3', version-exact)
  fal-ai_imagen4_preview                   -> imagen4       ('Imagen4', version-exact)
  fal-ai_bytedance_seedream (val seedreamv4.5) -> seedream3-0 ('Seedream3.0'; Seedream family)
  fal-ai_gemini-25-flash-image             -> nanao-banana  ('nanao_banana'; Nano Banana = gemini-2.5-flash)

Slugs with NO version-matched aigen shard (recorded, not materialized):
  ideogram2 (aigen has ideogram-v3 only -> mapped to ideogram3)
  imagen3   (aigen has imagen4 only     -> mapped to imagen4)
  recraftv3 (no aigen recraft family)

Output:
  data/corpora/gen_intake_train/<slug>/f<rank>_<orig>.<ext> + manifest.jsonl
  data/corpora/gen_train/real/r<rank>_<orig>.<ext>           + manifest.jsonl
Files renamed so name order == sha256 order (P4-C19 sha256-first-N is
deterministic end-to-end, incl. the dump's first-100 sampling).

Rules: P4-C19 cap 1000 imgs/domain, sha256-first-N; P4-C22 dedup — any image
whose sha256 appears in gen_exam/**, mj_train/**, flux_train/** or
modern_exam/** is EXCLUDED (counted per source); post-materialization overlap
with those sets asserted 0. Idempotent (skips when manifests + file counts
match); --force re-materializes.
"""

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
AIGEN = (WS / "data" / "corpora" / "gen_intake" / "aigenimages2026" /
         "aigenimages2026" / "mnt" / "cephfs" / "home" / "common" /
         "datasets" / "online_sid_dataset" / "AIGenImages" / "AIGenImages2026")
TRAIN_ROOT = WS / "data" / "corpora" / "gen_intake_train"
GEN_REAL = WS / "data" / "corpora" / "gen_train" / "real"
ART = WS / "artifacts" / "generator_intake"
CAP = 1000  # P4-C19

# slug -> aigen sources: train/1_fake filename prefixes + val/1_fake dir names
SLUG_SOURCES = {
    "flux-1-pro": {"train": ["fal-ai_flux-pro_v1.1_", "fal-ai_flux_dev_"],
                   "val": ["flux-pro-v1.1", "flux-dev"],
                   "note": "FLUX.1 family: exam generator 'Flux.1_pro'; shards "
                           "include FLUX.1 dev (same family, version noted)"},
    "gpt4o": {"train": ["fal-ai_gpt-image-1_", "fal-ai_gpt-image-1.5_"],
              "val": ["gpt-image-1", "gpt-image-1.5"],
              "note": "GPT-4o image generation = gpt-image family; "
                      "gpt-image-1.5 is the newer sibling"},
    "hidream": {"train": ["fal-ai_hidream-i1-dev_"], "val": ["hidream-i1-dev"],
                "note": "HiDream family; B.1 local-gen was feasibility NO-GO, "
                        "shard diet comes from aigen"},
    "ideogram3": {"train": ["fal-ai_ideogram_v3_"], "val": ["ideogram-v3"],
                  "note": "version-exact"},
    "imagen4": {"train": ["fal-ai_imagen4_preview_"], "val": ["imagen4"],
                "note": "version-exact"},
    "seedream3-0": {"train": ["fal-ai_bytedance_seedream_"],
                    "val": ["seedreamv4.5"],
                    "note": "Seedream family; aigen shards are v4.5-era "
                            "(val counterpart labeled seedreamv4.5), exam is "
                            "Seedream 3.0 — family-level transfer"},
    "nanao-banana": {"train": ["fal-ai_gemini-25-flash-image_"],
                     "val": ["gemini-25-flash-image"],
                     "note": "Nano Banana IS gemini-2.5-flash-image"},
}

NO_SHARD_SLUGS = {
    "ideogram2": "aigen has ideogram-v3 only (mapped to ideogram3); intended "
                 "ideogram-27k shard remains LICENSE_UNRESOLVED",
    "imagen3": "aigen has imagen4 only (mapped to imagen4)",
    "recraftv3": "no recraft family in aigenimages2026",
}

# families present in aigen but without a Stage A exam slug — recorded, skipped
NO_FAMILY_OVERLAP = {
    "flux-2": "FLUX.2 — no Stage A exam slug (B.1 self-gen exam is a separate "
              "in-domain gate, not a Stage A slug)",
    "flux-2-max": "same as flux-2",
    "flux-2-pro": "same as flux-2",
    "gemini-3-pro-image-preview": "Gemini 3 Pro Image (Nano Banana Pro) — no "
                                  "exam slug (Stage A recorded the Nano Banana "
                                  "2 family as UNREACHABLE-THIS-PHASE)",
    "midjourney_v7": "no Stage A exam slug (mj gate is the legacy modern_exam)",
    "z-image_turbo": "Z-Image dropped from Stage B targeting (D-049)",
    "reve": "no Stage A exam slug",
    "fast-sdxl": "no Stage A exam slug",
    "stable-diffusion-v35-medium": "no Stage A exam slug",
    "firefly_image5": "no Stage A exam slug",
    "TEST_*": "unclassified test artifact in train/1_fake",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exclusion_hashes() -> set[str]:
    """P4-C22: hashes of ALL exam + legacy train/gate corpora."""
    roots = [WS / "data" / "corpora" / "gen_exam",
             WS / "data" / "corpora" / "mj_train",
             WS / "data" / "corpora" / "flux_train",
             WS / "data" / "corpora" / "modern_exam"]
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    out: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts:
                out.add(sha256_file(p))
    return out


def slug_sources(src: dict) -> list[Path]:
    """Resolve a slug's source image paths (train flat dir + val subdir).
    Restricted to bench-readable extensions (run_benchmark globs
    .jpg/.jpeg/.png — webp would be silently skipped at dump time)."""
    paths: list[Path] = []
    exts = {".jpg", ".jpeg", ".png"}
    train_fake = AIGEN / "train" / "1_fake"
    for p in sorted(train_fake.iterdir()):
        if (p.is_file() and p.suffix.lower() in exts
                and any(p.name.startswith(pref) for pref in src["train"])):
            paths.append(p)
    val_fake = AIGEN / "val" / "1_fake"
    for d in src["val"]:
        dd = val_fake / d
        if dd.is_dir():
            paths.extend(sorted(p for p in dd.iterdir()
                                if p.is_file() and p.suffix.lower() in exts))
    return paths


def materialize(paths: list[Path], out_dir: Path, prefix: str,
                excluded: set[str]) -> dict:
    """sha256-dedupe within sources, exclude vs exam/train sets,
    sha256-first-N (P4-C19), copy with rank-ordered names."""
    seen: dict[str, Path] = {}
    n_dup = 0
    for p in paths:
        h = sha256_file(p)
        if h in seen:
            n_dup += 1
            continue
        seen[h] = p
    hits = sorted(h for h in seen if h in excluded)
    n_excluded = len(hits)
    keep = [h for h in sorted(seen) if h not in excluded][:CAP]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, h in enumerate(keep):
        src = seen[h]
        dst = out_dir / f"{prefix}{rank:04d}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        rows.append({"file": dst.name, "sha256": h, "rank": rank,
                     "source": str(src.relative_to(WS))})
    mf = out_dir / "manifest.jsonl"
    mf.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"n_source": len(paths), "n_unique": len(seen),
            "n_dup_in_sources": n_dup, "n_excluded_vs_exam_train": n_excluded,
            "n_materialized": len(rows), "capped": len(seen) - n_excluded > CAP}


def main() -> int:
    force = "--force" in sys.argv
    assert AIGEN.is_dir(), f"missing aigen corpus: {AIGEN}"
    print(f"[b2] hashing exclusion sets (gen_exam/**, mj_train/**, "
          f"flux_train/**, modern_exam/**)...", flush=True)
    excluded = exclusion_hashes()
    print(f"[b2] exclusion set: {len(excluded)} hashes", flush=True)

    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "corpus": "aigenimages2026 (CC-BY-4.0, TRAIN-ONLY per P4-C04)",
              "cap_per_domain": CAP, "exclusion_hashes": len(excluded),
              "slugs": {}, "no_shard_slugs": NO_SHARD_SLUGS,
              "no_family_overlap": NO_FAMILY_OVERLAP, "violations": {}}

    for slug, src in sorted(SLUG_SOURCES.items()):
        out_dir = TRAIN_ROOT / slug
        paths = slug_sources(src)
        assert paths, f"{slug}: no source images resolved"
        mf = out_dir / "manifest.jsonl"
        if not force and mf.exists():
            n_have = len([l for l in mf.read_text().splitlines() if l.strip()])
            if n_have and len([p for p in out_dir.iterdir()
                               if p.suffix.lower() != ".jsonl"]) == n_have:
                print(f"[b2] {slug}: exists ({n_have}), skipped", flush=True)
                report["slugs"][slug] = {"skipped": True, "n": n_have,
                                         "note": src["note"]}
                continue
        stats = materialize(paths, out_dir, "f", excluded)
        stats["note"] = src["note"]
        report["slugs"][slug] = stats
        print(f"[b2] {slug}: {stats['n_materialized']} imgs "
              f"(src {stats['n_source']}, dup {stats['n_dup_in_sources']}, "
              f"excluded {stats['n_excluded_vs_exam_train']})", flush=True)

    # shared training reals: aigen train/0_real, sha256-first-N
    real_src = AIGEN / "train" / "0_real"
    assert real_src.is_dir(), f"missing {real_src}"
    real_paths = sorted(p for p in real_src.iterdir()
                        if p.is_file() and p.suffix.lower() in
                        {".jpg", ".jpeg", ".png"})
    stats = materialize(real_paths, GEN_REAL, "r", excluded)
    report["gen_train_real"] = stats
    print(f"[b2] gen_train/real: {stats['n_materialized']} imgs "
          f"(src {stats['n_source']}, excluded "
          f"{stats['n_excluded_vs_exam_train']})", flush=True)

    # P4-C22 post-check: materialized sets intersect exclusion sets == 0
    violations = {}
    for slug in SLUG_SOURCES:
        d = TRAIN_ROOT / slug
        bad = sum(1 for p in d.iterdir()
                  if p.suffix.lower() != ".jsonl"
                  and sha256_file(p) in excluded)
        if bad:
            violations[slug] = bad
    bad_real = sum(1 for p in GEN_REAL.iterdir()
                   if p.suffix.lower() != ".jsonl"
                   and sha256_file(p) in excluded)
    if bad_real:
        violations["gen_train/real"] = bad_real
    report["violations"] = violations
    assert not violations, f"P4-C22 post-check violations {violations}"
    print(f"[b2] P4-C22 post-check: all-zero violations {violations}",
          flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    out = ART / "intake_train_manifest.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[b2] manifest -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
