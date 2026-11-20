"""W-AUDIT: never-seen endpoint battery against the LIVE deployed court.

POSTs every sample to http://127.0.0.1:8420/api/detect (the exact surface a
user hits) and records verdicts, abstentions, latency. Seed-42 samples:

  image : gen_exam NEVER-SEEN generators x10 fakes each (gpt4o hidream
          ideogram2 ideogram3 imagen3 imagen4 nanao-banana recraftv3
          seedream3-0 — flux/flux2/mj/sana/sdxl are IN the p6v11 diet, excluded)
          + real_ext.parquet x12 external reals
  audio : kokoro x8 (all) | dfeval fake x15 real x15 | itw fake x12 real x12
  video : genvidbench sora x8 kling x8 opentora x8 (T2V, never examined)
          | avlips fake x8 real x8 (talking-head) | ffpp test reals x6 control

Output: artifacts/audit_20261106/endpoint_{manifest.csv,results.csv,summary.json}
Usage: ./.venv/bin/python deepfake/scripts/audit_neverseen_endpoint.py
"""
import csv
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests

# P0b: repo root via ARGUS_ROOT (default: this file's repo-relative root).
import os
WS = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[2])
URL = "http://127.0.0.1:8420/api/detect"
OUT = WS / "artifacts" / "audit_20261106"
SEED = 42
IMG_FAMILIES = ["gpt4o", "hidream", "ideogram2", "ideogram3", "imagen3",
                "imagen4", "nanao-banana", "recraftv3", "seedream3-0"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".webp")


def build_manifest():
    rng = random.Random(SEED)
    rows = []

    def add(kind, family, label, path):
        rows.append({"kind": kind, "family": family, "label": label,
                     "path": str(path)})

    # ---- image ----
    for fam in IMG_FAMILIES:
        d = WS / "data/corpora/gen_exam" / fam
        fakes = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT
                       and "real" not in str(p.parent).lower())
        for p in rng.sample(fakes, min(10, len(fakes))):
            add("image", fam, 1, p)
    rp = WS / "data/raw/modern_exam/real_ext.parquet"
    if rp.exists():
        import pandas as pd
        df = pd.read_parquet(rp)
        col = "image" if "image" in df.columns else df.columns[0]
        idx = rng.sample(range(len(df)), min(12, len(df)))
        for i in idx:
            rec = df.iloc[i][col]
            b = rec["bytes"] if isinstance(rec, dict) else rec
            p = OUT / "_tmp_real_ext" / f"{i}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
            add("image", "real_ext", 0, p)

    # ---- audio ----
    for p in sorted((WS / "data/release_probe_unseen/audio/fake").glob("*kokoro*")):
        add("audio", "kokoro", 1, p)
    dfe = list(csv.DictReader(open(WS / "artifacts/exam_audio_dfeval_p4w8/audio_exam.csv")))
    base = WS / "data/dfeval/audio/audio-data/audio-data"
    fakes = [r for r in dfe if r["label"] == "1" and (base / f"{r['stem']}.mp3").exists()]
    reals = [r for r in dfe if r["label"] == "0" and (base / f"{r['stem']}.mp3").exists()]
    for r in rng.sample(fakes, min(15, len(fakes))):
        add("audio", "dfeval", 1, base / f"{r['stem']}.mp3")
    for r in rng.sample(reals, min(15, len(reals))):
        add("audio", "dfeval", 0, base / f"{r['stem']}.mp3")
    itw_meta = list(csv.DictReader(open(WS / "data/raw/itw/release_in_the_wild/meta.csv")))
    itw_base = WS / "data/raw/itw/release_in_the_wild"
    for lab_name, lab, n in (("spoof", 1, 12), ("bona-fide", 0, 12)):
        pool = [r for r in itw_meta if r["label"] == lab_name
                and (itw_base / r["file"]).exists()]
        for r in rng.sample(pool, min(n, len(pool))):
            add("audio", "itw", lab, itw_base / r["file"])

    # ---- video ----
    for fam, d, n in (("genvid_sora", WS / "data/raw/genvidbench/sora", 8),
                      ("genvid_kling", WS / "data/raw/genvidbench/kling", 8),
                      ("genvid_opentora", WS / "data/raw/genvidbench/opentora", 8)):
        vids = sorted(d.rglob("*.mp4"))
        for p in rng.sample(vids, min(n, len(vids))):
            add("video", fam, 1, p)
    for fam, d, lab, n in (("avlips", WS / "data/raw/avlips/videos/1_fake", 1, 8),
                           ("avlips", WS / "data/raw/avlips/videos/0_real", 0, 8)):
        vids = sorted(d.glob("*.mp4"))
        for p in rng.sample(vids, min(n, len(vids))):
            add("video", fam, lab, p)
    ffpp_real_dir = WS / "data/raw/ffpp_mirror/FaceForensics++_C23/real"
    if ffpp_real_dir.exists():
        vids = sorted(ffpp_real_dir.glob("*.mp4"))
        for p in rng.sample(vids, min(6, len(vids))):
            add("video", "ffpp_real", 0, p)
    else:
        print("ffpp control skipped: mirror dir absent", flush=True)
    return rows


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_manifest()
    with open(OUT / "endpoint_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "family", "label", "path"])
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {len(rows)} samples "
          f"({sum(r['kind']=='image' for r in rows)} img / "
          f"{sum(r['kind']=='audio' for r in rows)} aud / "
          f"{sum(r['kind']=='video' for r in rows)} vid)", flush=True)

    results = []
    for i, r in enumerate(rows):
        p = Path(r["path"])
        t0 = time.time()
        err = None
        body = {}
        try:
            with open(p, "rb") as fh:
                resp = requests.post(URL, files={"file": (p.name, fh)},
                                     timeout=600)
            lat = round(time.time() - t0, 2)
            if resp.status_code == 200:
                body = resp.json()
            else:
                err = f"HTTP {resp.status_code}: {resp.text[:120]}"
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            lat = round(time.time() - t0, 2)
        results.append({
            **r, "latency_s": lat, "error": err,
            "p_fake_fused": body.get("p_fake_fused"),
            "abstained": body.get("abstained"),
            "trust": body.get("trust"), "risk": body.get("risk"),
            "p_fake_audio_mean": (body.get("audio") or {}).get("p_fake_audio_mean"),
            "out_of_coverage": (body.get("audio") or {}).get("out_of_coverage"),
            "verdict": body.get("verdict") or body.get("label"),
        })
        tag = f"[{i + 1}/{len(rows)}] {r['kind']}:{r['family']}"
        if err:
            print(f"{tag} ERROR {err}", flush=True)
        else:
            print(f"{tag} p={results[-1]['p_fake_fused']} "
                  f"abst={results[-1]['abstained']} lat={lat}s", flush=True)

    with open(OUT / "endpoint_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    fams = defaultdict(list)
    for r in results:
        if r["error"] is None and r["p_fake_fused"] is not None:
            fams[(r["kind"], r["family"])].append(r)
    summary = {}
    for (kind, fam), rs in sorted(fams.items()):
        ps = [r["p_fake_fused"] for r in rs]
        fakes = [r for r in rs if r["label"] == 1]
        reals = [r for r in rs if r["label"] == 0]
        lats = sorted(r["latency_s"] for r in rs)
        summary[f"{kind}:{fam}"] = {
            "n": len(rs),
            "mean_p": round(sum(ps) / len(ps), 4),
            "catch_rate_p05": round(sum(1 for r in fakes
                                        if (r["p_fake_fused"] or 0) > 0.5)
                                    / max(1, len(fakes)), 3),
            "fpr_p05": round(sum(1 for r in reals
                                 if (r["p_fake_fused"] or 0) > 0.5)
                             / max(1, len(reals)), 3),
            "abstain_rate": round(sum(1 for r in rs if r["abstained"]) / len(rs), 3),
            "latency_p50_s": lats[len(lats) // 2],
        }
    (OUT / "endpoint_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
