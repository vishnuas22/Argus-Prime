"""Phase 4 Stage A — generator-intake acquisition (spec A.1, P4-C01/C02).

Fetches the pre-registered corpus plan into data/corpora/gen_intake/<corpus_id>/:
  openfaketiny        bitmind/OpenFakeTiny            EXAM-ONLY    12.8e9  CC-BY-NC-4.0
  omni-fake-ood       JamalLee/Omni-Fake-OOD          EXAM-ONLY    35.2e9  CC-BY-4.0
  aigenimages2026     sha6th/AIGenImages2026          TRAIN-ONLY   11.0e9  CC-BY-4.0
  wildaigc            yuzj1997yzj/WildAIGC-Bench      RECALL-ONLY   0.5e9  per card
  gpt4o-t2i-subset    yufan/GPT4O_Image_T2I           TRAIN-ONLY    5.0e9  license-gated
  ideogram-27k-subset bitmind/ideogram-27k            TRAIN-ONLY    5.0e9  license-gated

Fail-closed gates (no autonomous waivers, no mirror substitution):
  * repo-id verify via HF API — 404/gated -> REPO_ID_UNVERIFIED/GATED, skip, escalate.
  * license gate — gpt4o/ideogram subsets download ONLY if cardData.license is
    present and not '?'; else LICENSE_UNRESOLVED, skip (budget reverts to headroom).
  * P4-C01 cap — BEFORE any download: existing manifest bytes + remaining planned
    budgets must be <= 80_000_000_000 or exit 2. Cumulative bytes are written to
    artifacts/generator_intake/manifest.json after EVERY corpus.

Idempotent: a corpus with a completed MANIFEST.json is skipped (size-verified).
NO rm -rf. HF cache may temporarily double-store; the cap counts
data/corpora/gen_intake bytes only (noted in the stage report).

Usage:
  .venv/bin/python deepfake/scripts/fetch_generator_intake.py [--only id,...] [--dry-run]
"""
import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
DEST_ROOT = WS / "data" / "corpora" / "gen_intake"
ART_ROOT = WS / "artifacts" / "generator_intake"
CUM_MANIFEST = ART_ROOT / "manifest.json"
ACQUISITION = ART_ROOT / "acquisition.json"
HF_API = "https://huggingface.co/api/datasets"
ACQ_CAP = 80_000_000_000  # P4-C01

# P4-C02 corpus plan (role, repo id, method, budget bytes, license, why).
PLAN = [
    {"corpus_id": "openfaketiny", "repo": "bitmind/OpenFakeTiny",
     "role": "EXAM-ONLY", "method": "snapshot_full", "budget": 12.8e9,
     "license": "CC-BY-NC-4.0",
     "why": "75 model classes incl. imagen-4.0/flux.2-dev/sd-3.5; per-image model+type+release_date"},
    {"corpus_id": "omni-fake-ood", "repo": "JamalLee/Omni-Fake-OOD",
     "role": "EXAM-ONLY", "method": "snapshot_full", "budget": 35.2e9,
     "license": "CC-BY-4.0", "why": "purpose-built held-out generator field"},
    {"corpus_id": "aigenimages2026", "repo": "sha6th/AIGenImages2026",
     "alt_repo": "pthan12/AIGenImages2026",
     "role": "TRAIN-ONLY", "method": "snapshot_full", "budget": 11.0e9,
     "license": "CC-BY-4.0",
     "why": "19 contemporary 2025 T2I models, wild-collected"},
    {"corpus_id": "wildaigc", "repo": "yuzj1997yzj/WildAIGC-Bench",
     "role": "RECALL-ONLY", "method": "snapshot_full", "budget": 0.5e9,
     "license": "per card", "why": "407 news-event AIGC imgs (P4-C24 recall probe only)"},
    {"corpus_id": "gpt4o-t2i-subset", "repo": "yufan/GPT4O_Image_T2I",
     "role": "TRAIN-ONLY", "method": "shard_selective", "budget": 5.0e9,
     "license": "?", "license_gate": True,
     "why": "GPT-Image family train exposure"},
    {"corpus_id": "ideogram-27k-subset", "repo": "bitmind/ideogram-27k",
     "role": "TRAIN-ONLY", "method": "shard_selective", "budget": 5.0e9,
     "license": "?", "license_gate": True,
     "why": "Ideogram community held-out target"},
]

JUNK_DIRS = {"__macosx"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".parquet",
            ".json", ".jsonl", ".csv", ".txt", ".md", ".zip", ".tar", ".gz",
            ".tar.gz", ".sha256", ".webmanifest"}


def _junk(rel: str) -> bool:
    parts = Path(rel).parts
    base = Path(rel).name
    return (any(p.lower() in JUNK_DIRS for p in parts)
            or base.startswith("._") or base == ".DS_Store")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def api_dataset(repo: str):
    """GET HF dataset API record. Returns (status, json|None)."""
    import requests
    try:
        r = requests.get(f"{HF_API}/{repo}", timeout=60)
    except Exception as e:
        return f"ERROR:{type(e).__name__}", None
    if r.status_code == 200:
        return "OK", r.json()
    return f"HTTP_{r.status_code}", None


def verify_repo(entry: dict) -> dict:
    """Repo-id + license verify (A.1). Fail-closed; no mirror substitution."""
    out = {"corpus_id": entry["corpus_id"], "repo": entry["repo"],
           "status": None, "license_api": None, "gated": None,
           "used_storage": None, "note": None}
    status, j = api_dataset(entry["repo"])
    if status != "OK" and entry.get("alt_repo"):
        out["note"] = (f"primary {entry['repo']} -> {status}; "
                       f"trying spec-listed alt {entry['alt_repo']}")
        status, j = api_dataset(entry["alt_repo"])
        if status == "OK":
            out["repo"] = entry["alt_repo"]
    if status != "OK":
        code = "REPO_ID_GATED" if "401" in status or "403" in status else "REPO_ID_UNVERIFIED"
        out["status"] = code
        out["note"] = ((out["note"] + "; ") if out["note"] else "") + (
            f"HF API {status} for {out['repo']} — skipped, NO mirror substitution; "
            "escalate at user checkpoint")
        return out
    card = j.get("cardData") or {}
    out["license_api"] = card.get("license")
    out["gated"] = j.get("gated")
    out["used_storage"] = j.get("usedStorage")
    if j.get("gated"):
        out["status"] = "REPO_ID_GATED"
        out["note"] = "repo gated on HF — skipped, escalate"
        return out
    if entry.get("license_gate"):
        lic = out["license_api"]
        if lic in (None, "", "?"):
            out["status"] = "LICENSE_UNRESOLVED"
            out["note"] = ("cardData.license missing/'?' — fail-closed SKIP "
                           "(budget reverts to headroom)")
            return out
    out["status"] = "CLEARED"
    return out


def planned_download_bytes(checks: dict, on_disk: set) -> int:
    total = 0
    for e in PLAN:
        cid = e["corpus_id"]
        if cid in on_disk:
            continue
        if checks.get(cid, {}).get("status") == "CLEARED":
            total += int(e["budget"])
    return total


def corpus_complete(dest: Path) -> bool:
    man = dest / "MANIFEST.json"
    if not man.exists():
        return False
    try:
        m = json.loads(man.read_text())
    except Exception:
        return False
    files = m.get("files") or []
    if not files:
        return False
    return all((dest / f["file"]).exists() and (dest / f["file"]).stat().st_size == f["bytes"]
               for f in files)


def extract_tar_junk_safe(archive: Path, outdir: Path) -> int:
    """Stream-extract a tar(.gz); skip macOS junk; streamed .part writes; no rm."""
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(archive, "r:*") as t:
        for m in t.getmembers():
            if not m.isfile() or _junk(m.name):
                continue
            dst = outdir / m.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = t.extractfile(m)
            if src is None:
                continue
            if dst.exists() and dst.stat().st_size == m.size:
                n += 1
                continue
            part = dst.with_name(dst.name + ".part")
            with open(part, "wb") as f:
                shutil.copyfileobj(src, f)
            part.replace(dst)
            n += 1
    return n


def build_manifest(dest: Path, entry: dict, check: dict) -> dict:
    """Hash every non-junk file under dest -> per-corpus MANIFEST.json."""
    files = []
    total = 0
    for p in sorted(dest.rglob("*")):
        if not p.is_file():
            continue
        rel = str(p.relative_to(dest))
        if rel.startswith(".cache/") or _junk(rel):
            continue
        b = p.stat().st_size
        files.append({"file": rel, "bytes": b, "sha256": sha256_of(p),
                      "source_url": f"https://huggingface.co/datasets/{check['repo']}",
                      "license": check.get("license_api") or entry["license"],
                      "role": entry["role"], "generator_field": None})
        total += b
    man = {"corpus_id": entry["corpus_id"], "repo": check["repo"],
           "role": entry["role"], "license": check.get("license_api") or entry["license"],
           "license_api": check.get("license_api"), "fetched_at":
               datetime.now(timezone.utc).isoformat(),
           "n_files": len(files), "bytes": total, "files": files}
    (dest / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    return man


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="comma-separated corpus_ids (default: all cleared)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify + cap-check only; download nothing")
    args = ap.parse_args()

    only = set(args.only.split(",")) if args.only else None
    ART_ROOT.mkdir(parents=True, exist_ok=True)
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    # ---- verify all six (repo-id + license), record regardless of download
    checks = {}
    for e in PLAN:
        c = verify_repo(e)
        checks[e["corpus_id"]] = c
        print(f"[verify] {e['corpus_id']}: {c['status']} repo={c['repo']} "
              f"license={c['license_api']} gated={c['gated']}"
              + (f" | {c['note']}" if c.get("note") else ""), flush=True)

    # ---- which corpora are already complete on disk
    on_disk = {e["corpus_id"] for e in PLAN if corpus_complete(DEST_ROOT / e["corpus_id"])}
    for cid in sorted(on_disk):
        print(f"[skip] {cid}: complete on disk (size-verified)", flush=True)

    # ---- P4-C01 cap: existing cumulative + remaining planned budgets <= 80e9
    existing = 0
    if CUM_MANIFEST.exists():
        try:
            existing = int(json.loads(CUM_MANIFEST.read_text()).get("cumulative_bytes", 0))
        except Exception:
            existing = 0
    will_fetch = []
    for e in PLAN:
        cid = e["corpus_id"]
        if cid in on_disk or (only and cid not in only):
            continue
        if checks[cid]["status"] != "CLEARED":
            continue
        will_fetch.append(e)
    planned_remaining = sum(int(e["budget"]) for e in will_fetch)
    projected = existing + planned_remaining
    print(f"[cap] existing_cumulative={existing} planned_remaining={planned_remaining} "
          f"projected={projected} cap={ACQ_CAP}", flush=True)
    if projected > ACQ_CAP:
        print(f"FAIL: projected acquisition {projected} exceeds P4-C01 cap {ACQ_CAP} "
              f"— ABORT before download (exit 2)", file=sys.stderr)
        return 2

    if args.dry_run:
        print("[dry-run] verify+cap only; nothing downloaded")
        ACQUISITION.write_text(json.dumps(
            {"checks": checks, "on_disk": sorted(on_disk),
             "projected_bytes": projected, "cap": ACQ_CAP}, indent=2))
        return 0

    # ---- fetch loop; manifest (cumulative) updated after EVERY corpus
    acq = {"checks": checks, "fetched": [], "skipped": []}
    for e in PLAN:
        cid = e["corpus_id"]
        dest = DEST_ROOT / cid
        if cid in on_disk:
            acq["skipped"].append({"corpus_id": cid, "reason": "complete_on_disk"})
            continue
        if checks[cid]["status"] != "CLEARED":
            acq["skipped"].append({"corpus_id": cid,
                                   "reason": checks[cid]["status"],
                                   "note": checks[cid].get("note")})
            print(f"[skip] {cid}: {checks[cid]['status']}", flush=True)
            continue
        if only and cid not in only:
            continue

        repo = checks[cid]["repo"]
        print(f"[fetch] {cid} <- {repo} ({e['method']}, budget {e['budget']:.3g})",
              flush=True)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as err:
            raise SystemExit(f"FAIL: huggingface_hub missing: {err}")
        snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(dest))

        # aigenimages2026 ships one tar.gz — extract (junk-safe, idempotent) so
        # per-image sha256 dedup (P4-C22) can see the train-side images.
        if cid == "aigenimages2026":
            for arc in sorted(dest.rglob("*.tar.gz")):
                if ".cache" in arc.parts or _junk(str(arc.relative_to(dest))):
                    continue
                outdir = arc.parent / arc.name[: -len(".tar.gz")]
                marker = outdir / ".extracted"
                if marker.exists():
                    print(f"[extract] {outdir.name}: already extracted", flush=True)
                    continue
                print(f"[extract] {arc.name} -> {outdir.name}/", flush=True)
                n = extract_tar_junk_safe(arc, outdir)
                marker.write_text(json.dumps({"archive": arc.name, "n_files": n,
                                              "at": datetime.now(timezone.utc).isoformat()}))
                print(f"[extract] {outdir.name}: {n} files", flush=True)

        # size-verify against API usedStorage (warn below 90%; never delete)
        got = sum(p.stat().st_size for p in dest.rglob("*")
                  if p.is_file() and not str(p.relative_to(dest)).startswith(".cache/"))
        us = checks[cid].get("used_storage")
        if us and got < 0.9 * us:
            print(f"WARN: {cid} on-disk {got/1e9:.2f} GB < 90% of API usedStorage "
                  f"{us/1e9:.2f} GB", file=sys.stderr)

        man = build_manifest(dest, e, checks[cid])
        # generator_field discovered by build_gen_exam.py; record plan default
        field = {"omni-fake-ood": "generator", "aigenimages2026": "generator",
                 "wildaigc": None}.get(cid)
        for f in man["files"]:
            f["generator_field"] = field
        (dest / "MANIFEST.json").write_text(json.dumps(man, indent=2))

        existing += man["bytes"]
        if existing > ACQ_CAP:
            print(f"FAIL: cumulative {existing} exceeds P4-C01 cap {ACQ_CAP} "
                  f"after {cid} (exit 2)", file=sys.stderr)
            CUM_MANIFEST.write_text(json.dumps(
                {"cap": ACQ_CAP, "cumulative_bytes": existing,
                 "updated": datetime.now(timezone.utc).isoformat(),
                 "corpora": _cum_corpora(CUM_MANIFEST, cid, man)}, indent=2))
            return 2
        _write_cum(cid, man, existing)
        acq["fetched"].append({"corpus_id": cid, "bytes": man["bytes"],
                               "n_files": man["n_files"], "license": man["license"],
                               "role": man["role"]})
        print(f"[done] {cid}: {man['n_files']} files, {man['bytes']/1e9:.3f} GB; "
              f"cumulative {existing/1e9:.3f} GB / {ACQ_CAP/1e9:.0f} GB", flush=True)

    ACQUISITION.write_text(json.dumps(acq, indent=2))
    total = sum(f["bytes"] for f in acq["fetched"])
    print(f"[summary] fetched {len(acq['fetched'])} corpora "
          f"({total/1e9:.3f} GB), skipped {len(acq['skipped'])}; "
          f"cumulative {existing/1e9:.3f} GB <= {ACQ_CAP/1e9:.0f} GB", flush=True)
    return 0


def _cum_corpora(man_p: Path, cid: str, man: dict) -> dict:
    corpora = {}
    if man_p.exists():
        try:
            corpora = dict(json.loads(man_p.read_text()).get("corpora") or {})
        except Exception:
            corpora = {}
    corpora[cid] = {"bytes": man["bytes"], "n_files": man["n_files"],
                    "license": man["license"], "role": man["role"],
                    "repo": man["repo"]}
    return corpora


def _write_cum(cid: str, man: dict, cumulative: int) -> None:
    CUM_MANIFEST.write_text(json.dumps(
        {"cap": ACQ_CAP, "cumulative_bytes": cumulative,
         "updated": datetime.now(timezone.utc).isoformat(),
         "corpora": _cum_corpora(CUM_MANIFEST, cid, man)}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
