"""Phase 4 Stage A — generator exam construction (spec A.2/A.3, P4-C03..C06/C22/C24).

Reads data/corpora/gen_intake/ (fetched by fetch_generator_intake.py) and builds
deterministic per-generator exams at data/corpora/gen_exam/<slug>/{fake,real}/*.jpg:

  * slug = normalize(corpus generator field) -> lowercase alnum slug
    (Flux.1_pro -> flux-1-pro). Slug = exam name = future gate name = decision key.
  * EXAM-ONLY corpora only (P4-C03): omni-fake-ood, openfaketiny, wildaigc
    (wildaigc is RECALL-ONLY — P4-C24 — and goes to a separate root,
    data/corpora/gen_exam_recall/, never into the decision exam set).
  * TRAIN-ONLY corpora (P4-C04, aigenimages2026, gpt4o/ideogram subsets) NEVER
    contribute exam images; aigenimages2026 images are hashed for dedup only.
  * Merge across EXAM-ONLY corpora per slug (union); deterministic selection:
    sort by sha256(content), first 100 (P4-C05; report actual n if <100).
  * Provider splits honored: only rows with split in {test, eval, heldout} when
    the corpus carries a split field.
  * Reals (P4-C06): provider-matched when the corpus supplies them
    (sha256-first-100, tagged PROVIDER); else data/corpora/modern_exam/real_ext
    tagged REFERENCE.
  * P4-C22 sha256 dedup: every exam image vs every new train image (both
    directions) + vs existing mj_train/flux_train/modern_exam — asserted empty.
  * Stamped out-dir guard verbatim (non-empty + no --force -> error).
  * NO rm -rf.

Usage:
  .venv/bin/python deepfake/scripts/build_gen_exam.py [--per-gen 100] [--force]
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
INTAKE = WS / "data" / "corpora" / "gen_intake"
EXAM_ROOT = WS / "data" / "corpora" / "gen_exam"
RECALL_ROOT = WS / "data" / "corpora" / "gen_exam_recall"
ART_ROOT = WS / "artifacts" / "generator_intake"
REAL_REF = WS / "data" / "corpora" / "modern_exam" / "real_ext"

EXAM_ONLY = {"omni-fake-ood", "openfaketiny"}      # P4-C03 (wildaigc handled apart)
RECALL_ONLY = {"wildaigc"}                          # P4-C24
TRAIN_ONLY = {"aigenimages2026", "gpt4o-t2i-subset", "ideogram-27k-subset"}  # P4-C04
SPLIT_OK = {"test", "eval", "heldout", "held_out", "validation"}
DEDUP_CORPORA = ["mj_train", "flux_train", "modern_exam"]  # existing sets (P4-C22)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(s: str) -> str:
    """lowercase alnum slug: non-alnum runs -> single '-' (Flux.1_pro -> flux-1-pro)."""
    out, prev_dash = [], True
    for ch in (s or "").lower():
        if ch.isalnum():
            out.append(ch)
            prev_dash = False
        elif not prev_dash:
            out.append("-")
            prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "unknown"


def image_column(table):
    for col in ("image", "img"):
        if col in table.column_names:
            return table.column(col)
    raise SystemExit(f"no image column; have {table.column_names}")


def _img_bytes(rec) -> bytes | None:
    if isinstance(rec, dict):
        b = rec.get("bytes")
        return b if b else None
    return rec if isinstance(rec, (bytes, bytearray)) else None


# --------------------------------------------------------------------------
# Omni-Fake-OOD (parquet, image config): label/generator/filename/split fields
# --------------------------------------------------------------------------

def omni_shards() -> list[Path]:
    d = INTAKE / "omni-fake-ood" / "data" / "Image"
    return sorted(p for p in d.glob("test-*.parquet")) if d.exists() else []


def scan_omni() -> tuple[dict, dict, int]:
    """Pass 1 over Omni-Fake-OOD image shards.

    Returns (fake_index, real_index, n_tampered_skipped) where indexes map
    slug -> list of (sha256, shard_path, row_idx) (fakes) or a flat list
    (reals). Only split==test rows (all rows carry split; honored per A.3).
    """
    import pyarrow.parquet as pq
    fake_index: dict[str, list] = {}
    real_index: list = []
    n_tampered = 0
    shards = omni_shards()
    if not shards:
        return fake_index, real_index, n_tampered
    for sp in shards:
        pf = pq.ParquetFile(sp)
        cols = [c for c in ("label", "generator", "filename", "split")
                if c in pf.schema_arrow.names]
        meta = pf.read(columns=cols).to_pydict()
        # stream image column in batches to bound memory
        batch_rows = 256
        n = pf.metadata.num_rows
        reader = pf.iter_batches(batch_size=batch_rows, columns=["image"])
        row0 = 0
        for bi, batch in enumerate(reader):
            col = batch.column(0)
            for j in range(batch.num_rows):
                idx = row0 + j
                label = (meta.get("label") or [""] * n)[idx]
                split = (meta.get("split") or [""] * n)[idx].lower()
                if split and split not in SPLIT_OK:
                    continue
                data = _img_bytes(col[j].as_py())
                if not data:
                    continue
                if label == "real":
                    real_index.append((sha256_bytes(data), sp, idx))
                elif label in ("full_synthetic", "fake", "synthetic", "ai_generated"):
                    gen = (meta.get("generator") or ["unknown"] * n)[idx] or "unknown"
                    fake_index.setdefault(slugify(gen), []).append(
                        (sha256_bytes(data), sp, idx))
                else:
                    n_tampered += 1  # tampered = manipulation, not generator fake
            row0 += batch.num_rows
    return fake_index, real_index, n_tampered


def materialize_selected(selections: dict, out_root: Path) -> int:
    """One streaming pass per shard; write each selected row once.

    selections: shard_path -> list of (row_idx, dest_dir, stem).
    A row may fan out to several destinations (shared provider reals).
    (Single pass over each shard — never re-stream per slug.)
    """
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    written = 0
    for sp, wants in selections.items():
        want: dict[int, list] = {}
        for idx, dst_dir, stem in wants:
            want.setdefault(idx, []).append((dst_dir, stem))
        for d in {d for lst in want.values() for d, _ in lst}:
            d.mkdir(parents=True, exist_ok=True)
        pf = pq.ParquetFile(sp)
        row0 = 0
        reader = pf.iter_batches(batch_size=128, columns=["image"])
        for batch in reader:
            col = batch.column(0)
            for j in range(batch.num_rows):
                idx = row0 + j
                if idx not in want:
                    continue
                data = _img_bytes(col[j].as_py())
                if not data:
                    continue
                decoded = None
                for dst_dir, stem in want[idx]:
                    dst = dst_dir / f"{stem}.jpg"
                    if dst.exists():
                        written += 1
                        continue
                    if decoded is None:
                        try:
                            im = Image.open(io.BytesIO(data))
                            im.load()
                        except Exception:
                            break
                        if min(im.size) < 64:
                            break
                        decoded = im.convert("RGB")
                    decoded.save(dst, quality=95)
                    written += 1
            row0 += batch.num_rows
    return written


# --------------------------------------------------------------------------
# flat-dir corpora (openfaketiny future layout): manifest rows carry generator
# --------------------------------------------------------------------------

def scan_flat_corpus(cid: str) -> dict:
    """Generic fallback: read per-file metadata from corpus MANIFEST/rows.

    OpenFakeTiny is REPO_ID_UNVERIFIED this stage (skipped); this path exists
    so a corrected repo-id in a later stage needs no exam-builder rewrite.
    Expected layout: gen_intake/<cid>/images/<file> + rows.csv with a
    generator/model column. Returns slug -> [(sha, path, None)].
    """
    root = INTAKE / cid
    index: dict[str, list] = {}
    rows_p = root / "rows.csv"
    meta = {}
    if rows_p.exists():
        for r in csv.DictReader(open(rows_p)):
            gen = r.get("generator") or r.get("model") or r.get("model_name") or ""
            meta[r.get("file") or r.get("path") or ""] = gen
    img_root = root / "images"
    if not img_root.exists():
        img_root = root
    for p in sorted(img_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if p.name.startswith("._") or p.name == ".DS_Store":
            continue
        rel = str(p.relative_to(root))
        gen = meta.get(rel) or meta.get(p.name) or "unknown"
        index.setdefault(slugify(gen), []).append((sha256_of(p), p, None))
    return index


# --------------------------------------------------------------------------
# WildAIGC recall-only (P4-C24)
# --------------------------------------------------------------------------

def build_wildaigc_recall(force: bool) -> dict:
    src = INTAKE / "wildaigc" / "raw"
    out = RECALL_ROOT / "wildaigc" / "fake"
    if not src.exists():
        return {"status": "ABSENT", "note": "wildaigc not on disk"}
    if out.exists() and any(out.iterdir()) and not force:
        n = len([p for p in out.iterdir() if p.suffix.lower() in IMG_EXTS])
        return {"status": "EXISTS", "n": n, "dir": str(out.relative_to(WS))}
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS
                   and not p.name.startswith("._") and p.name != ".DS_Store")
    hashes = {}
    for p in files:
        b = p.read_bytes()
        h = sha256_bytes(b)
        hashes[h] = str(p)
        dst = out / f"{h[:16]}.jpg"
        if not dst.exists():
            from PIL import Image
            import io
            im = Image.open(io.BytesIO(b))
            im.load()
            im.convert("RGB").save(dst, quality=95)
    n = len(list(out.iterdir()))
    man = {"corpus": "wildaigc", "role": "RECALL-ONLY", "p4_c24":
           "recall-only probe, NEVER enters decision rule/training/thresholding",
           "n": n, "files": [{"file": p.name, "sha256": sha256_of(p)}
                             for p in sorted(out.iterdir())
                             if p.suffix.lower() in IMG_EXTS]}
    (out.parent / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    return {"status": "BUILT", "n": n, "dir": str(out.relative_to(WS))}


# --------------------------------------------------------------------------
# dedup (P4-C22)
# --------------------------------------------------------------------------

def existing_train_hashes() -> dict[str, set]:
    """sha256 sets for new-train (aigenimages2026) + existing train/exam corpora."""
    out: dict[str, set] = {}
    aig = INTAKE / "aigenimages2026"
    if aig.exists():
        s = set()
        for p in aig.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS \
                    and ".cache" not in p.parts and not p.name.startswith("._"):
                s.add(sha256_of(p))
        if s:
            out["aigenimages2026(new-train)"] = s
    for name in DEDUP_CORPORA:
        root = WS / "data" / "corpora" / name
        if not root.exists():
            continue
        s = set()
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS \
                    and not p.name.startswith("._"):
                s.add(sha256_of(p))
        out[name] = s
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-gen", type=int, default=100, help="P4-C05 n fakes/slug")
    ap.add_argument("--per-real", type=int, default=100)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    # stamped out-dir guard verbatim (patterns 2026-10-10)
    if EXAM_ROOT.exists() and any(EXAM_ROOT.iterdir()) and not args.force:
        print(f"ERROR: {EXAM_ROOT} is non-empty — pass --force to rebuild "
              f"(rebuild overwrites per-slug dirs; recall root untouched "
              f"unless wildaigc changes)", file=sys.stderr)
        return 1

    ART_ROOT.mkdir(parents=True, exist_ok=True)
    EXAM_ROOT.mkdir(parents=True, exist_ok=True)
    report = {"built_at": datetime.now(timezone.utc).isoformat(),
              "per_gen": args.per_gen, "slugs": {}, "skipped_corpora": [],
              "wildaigc_recall": None, "n_tampered_skipped": 0, "dedup": {},
              "n_materialized": 0, "n_flat_copied": 0}

    # ---- which corpora are present
    present = {p.name for p in INTAKE.iterdir() if p.is_dir()} if INTAKE.exists() else set()
    for cid in sorted(TRAIN_ONLY | RECALL_ONLY | EXAM_ONLY - {"wildaigc"}):
        if cid not in present:
            report["skipped_corpora"].append(
                {"corpus_id": cid, "reason": "not_on_disk"})

    # ---- WildAIGC recall-only (separate section, P4-C24)
    report["wildaigc_recall"] = build_wildaigc_recall(args.force)

    # ---- Omni-Fake-OOD exam slugs
    omni_present = "omni-fake-ood" in present and omni_shards()
    all_candidates: dict[str, list] = {}
    real_candidates: list = []
    if omni_present:
        print("[omni] pass 1: scanning image shards (hash all candidates)...",
              flush=True)
        fake_index, real_index, n_tamp = scan_omni()
        report["n_tampered_skipped"] = n_tamp
        print(f"[omni] {len(fake_index)} generator slugs, "
              f"{sum(len(v) for v in fake_index.values())} fakes, "
              f"{len(real_index)} reals, {n_tamp} tampered skipped", flush=True)
        for slug, rows in fake_index.items():
            all_candidates.setdefault(slug, []).extend(
                (h, sp, idx, "omni-fake-ood") for h, sp, idx in rows)
        real_candidates.extend((h, sp, idx, "omni-fake-ood")
                               for h, sp, idx in real_index)

    # ---- future flat corpora (openfaketiny once repo-id is resolved)
    if "openfaketiny" in present:
        idx = scan_flat_corpus("openfaketiny")
        print(f"[openfaketiny] {len(idx)} slugs discovered", flush=True)
        for slug, rows in idx.items():
            all_candidates.setdefault(slug, []).extend(
                (h, p, None, "openfaketiny") for h, p, _ in rows)

    if not all_candidates:
        print("FAIL: no EXAM-ONLY candidates on disk — nothing to build "
              "(openfaketiny REPO_ID_UNVERIFIED; omni-fake-ood missing?)",
              file=sys.stderr)
        return 1

    # ---- deterministic selection per slug: sort by sha256, first N (P4-C05)
    sel: dict[str, list] = {}
    for slug, rows in all_candidates.items():
        rows.sort(key=lambda r: r[0])
        sel[slug] = rows[: args.per_gen]

    # real set: provider-matched first (P4-C06)
    real_candidates.sort(key=lambda r: r[0])
    provider_reals = real_candidates[: args.per_real]
    real_tag = "PROVIDER"
    if not provider_reals and REAL_REF.exists():
        provider_reals = [(sha256_of(p), p, None, "modern_exam/real_ext")
                          for p in sorted(REAL_REF.glob("*.jpg"))[: args.per_real]]
        real_tag = "REFERENCE"
    if not provider_reals:
        print("FAIL: no reals available (provider or real_ext)", file=sys.stderr)
        return 1

    # ---- materialize ALL slugs in ONE streaming pass per shard (P4-C05)
    # build selection plan: shard -> [(row_idx, dest_dir, stem)]
    plan: dict[Path, list] = {}
    flat_fakes: list = []
    for slug in sorted(sel):
        fake_dir = EXAM_ROOT / slug / "fake"
        for h, sp, idx, cid in sel[slug]:
            if cid == "omni-fake-ood":
                plan.setdefault(sp, []).append((idx, fake_dir, h[:16]))
            else:
                flat_fakes.append((h, sp, fake_dir, cid))
    # provider reals fan out to every slug's real/ dir in the same pass
    for h, sp, idx, cid in provider_reals:
        if isinstance(sp, Path) and sp.suffix == ".parquet":
            stem = f"r_{h[:12]}_{idx:06d}"
            for slug in sorted(sel):
                plan.setdefault(sp, []).append(
                    (idx, EXAM_ROOT / slug / "real", stem))
        else:
            flat_fakes.append((h, sp, None, ("real", cid)))
    n_written = materialize_selected(plan, EXAM_ROOT) if plan else 0
    report["n_materialized"] = n_written
    # flat (non-parquet) copies: fakes + provider real files
    n_flat = 0
    for h, sp, dest_dir, meta in flat_fakes:
        if dest_dir is None and isinstance(meta, tuple) and meta[0] == "real":
            for slug in sorted(sel):
                d = EXAM_ROOT / slug / "real"
                d.mkdir(parents=True, exist_ok=True)
                dst = d / f"{h[:16]}.jpg"
                if not dst.exists():
                    shutil.copy(sp, dst)
                n_flat += 1
        elif dest_dir is not None:
            dest_dir.mkdir(parents=True, exist_ok=True)
            dst = dest_dir / f"{h[:16]}.jpg"
            if not dst.exists():
                shutil.copy(sp, dst)
            n_flat += 1
    report["n_flat_copied"] = n_flat

    # ---- per-slug tallies (post-materialization, from disk truth)
    for slug in sorted(sel):
        rows = sel[slug]
        fake_dir = EXAM_ROOT / slug / "fake"
        real_dir = EXAM_ROOT / slug / "real"
        n_f = len([p for p in fake_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS]) \
            if fake_dir.exists() else 0
        n_r = len([p for p in real_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS]) \
            if real_dir.exists() else 0
        report["slugs"][slug] = {
            "n_fake": n_f, "n_real": n_r,
            "sources": sorted({cid for _, _, _, cid in rows}),
            "real_source": provider_reals[0][3] if provider_reals else None,
            "real_tag": real_tag,
            "truncated": len(all_candidates[slug]) > args.per_gen,
            "candidates_available": len(all_candidates[slug])}

    # ---- P4-C22 dedup asserts (both directions + existing corpora)
    exam_hashes: set = set()
    for slug_dir in sorted(EXAM_ROOT.iterdir()):
        if not slug_dir.is_dir():
            continue
        for p in slug_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                exam_hashes.add(sha256_of(p))
    train_sets = existing_train_hashes()
    violations = {}
    for name, s in train_sets.items():
        inter = exam_hashes & s
        violations[name] = len(inter)
        if inter:
            print(f"FAIL: P4-C22 dedup violation exam ∩ {name} = {len(inter)}",
                  file=sys.stderr)
    report["dedup"] = {"exam_hashes": len(exam_hashes),
                       "train_sets": {k: len(v) for k, v in train_sets.items()},
                       "violations_both_directions": violations,
                       "pass": all(v == 0 for v in violations.values())}
    if not report["dedup"]["pass"]:
        return 1

    out_p = ART_ROOT / "exam_build_manifest.json"
    out_p.write_text(json.dumps(report, indent=2))
    print(f"[done] {len(report['slugs'])} slugs -> {EXAM_ROOT}", flush=True)
    for slug, s in sorted(report["slugs"].items()):
        print(f"  {slug}: n_fake={s['n_fake']} n_real={s['n_real']} "
              f"src={s['sources']}", flush=True)
    print(f"  wildaigc recall: {report['wildaigc_recall']}", flush=True)
    print(f"  dedup pass={report['dedup']['pass']} "
          f"violations={violations}", flush=True)
    print(f"manifest -> {out_p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
