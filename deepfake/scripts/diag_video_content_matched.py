"""V1f — content-matched OOD-video diagnostic (DEPLOYED court, read-only).

Question: the court's OOD-video exam compares faceless T2V fakes against FF++
face reals. Is the 1/27 catch a content-domain confound or a real detection
failure?

Method: POST clips to the LIVE /api/detect surface (same as the audit), three
arms in one run:
  real_msrvtt  150 fresh MSR-VTT reals   (content-matched: general YouTube video)
  real_ffpp     50 FF++ C23 reals        (the current exam's real arm, control)
  fake_opentora 150 fresh OpenTora T2V   (faceless generated video)

Pre-declared interpretation (fixed before scoring):
  AUC(fake vs msrvtt) >= 0.70 AND AUC(fake vs ffpp) <= 0.60  -> CONTENT_CONFOUND
      the exam's real arm, not the detector, drives the 1/27; a re-spec is justified
  AUC(fake vs msrvtt) <= 0.60 AND AUC(fake vs ffpp) <= 0.60  -> DETECTOR_BLIND
      the deployed court cannot detect T2V at all; 70 is closed
  both >= 0.70                                                -> OPERATING_POINT
      ranking exists; the failure is threshold placement
  otherwise                                                   -> MIXED

No pointer, threshold, stacker, or court file is changed. Output is a new
artifact directory only.
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

import requests

WS = Path("/Users/Dataghost/Argus_Prime")
URL = "http://127.0.0.1:8420/api/detect"
OUT = WS / "artifacts" / "v1f_content_matched"
SEED = 20261119
N_REAL_MSRVTT = 150
N_REAL_FFPP = 50
N_FAKE = 150

D3 = WS / "artifacts/d3_t2v_matched/scores.jsonl"
PRE = WS / "artifacts/d3_transfer_preflight/manifest.csv"
AUDIT_MAN = WS / "artifacts/audit_20261106/endpoint_manifest.csv"


def _mp4s(d: Path) -> list:
    return sorted(p for p in d.iterdir()
                  if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv")
                  and not p.name.startswith("._"))


def _used_msrvtt() -> set:
    used = set()
    for line in D3.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("generator") == "msrvtt":
                used.add(r["filename"])
    for r in csv.DictReader(open(PRE)):
        if r["arm"] == "calib_msrvtt":
            used.add(r["filename"])
    return used


def _used_opentora() -> set:
    used = set()
    for line in D3.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("generator") == "opentora":
                used.add(r["filename"])
    return used


def _audit_video_paths() -> set:
    out = set()
    for r in csv.DictReader(open(AUDIT_MAN)):
        if r["kind"] == "video":
            out.add(Path(r["path"]).name)
    return out


def build_manifest() -> list:
    rng = random.Random(SEED)
    rows = []

    d = WS / "data/raw/genvideo_val/real_msrvtt"
    used = _used_msrvtt()
    pool = [p for p in _mp4s(d) if p.name not in used]
    for p in sorted(rng.sample(pool, N_REAL_MSRVTT)):
        rows.append({"arm": "real_msrvtt", "label": 0, "path": p})

    d = WS / "data/raw/ffpp_mirror/FaceForensics++_C23/real"
    audit = _audit_video_paths()
    pool = [p for p in _mp4s(d) if p.name not in audit]
    for p in sorted(rng.sample(pool, N_REAL_FFPP)):
        rows.append({"arm": "real_ffpp", "label": 0, "path": p})

    d = WS / "data/raw/genvidbench/opentora"
    op_used = _used_opentora()
    pool = [p for p in _mp4s(d)
            if p.name not in op_used and p.name not in audit]
    for p in sorted(rng.sample(pool, N_FAKE)):
        rows.append({"arm": "fake_opentora", "label": 1, "path": p})

    return rows


def score(rows: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    man = OUT / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["arm", "label", "path"])
        w.writeheader()
        w.writerows(rows)

    res_path = OUT / "results.csv"
    done = set()
    if res_path.exists():
        for r in csv.DictReader(open(res_path)):
            done.add(r["path"])
    fields = ["arm", "label", "path", "p_fake_fused", "abstained", "trust",
              "risk", "route", "n_identities", "confidence_degraded",
              "out_of_conflict", "stacker", "elapsed_s", "error"]
    new = not res_path.exists()
    t0 = time.time()
    with open(res_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for i, row in enumerate(rows):
            if str(row["path"]) in done:
                continue
            rec = {k: "" for k in fields}
            rec.update({"arm": row["arm"], "label": row["label"],
                        "path": str(row["path"])})
            try:
                with open(row["path"], "rb") as fh:
                    r = requests.post(URL, files={"file": (Path(row["path"]).name, fh,
                                                           "video/mp4")},
                                      timeout=120)
                if r.status_code == 200:
                    v = r.json()
                    rec.update({
                        "p_fake_fused": v.get("p_fake_fused"),
                        "abstained": v.get("abstained"),
                        "trust": v.get("trust"),
                        "risk": v.get("risk"),
                        "route": v.get("route"),
                        "n_identities": v.get("n_identities"),
                        "confidence_degraded": v.get("confidence_degraded"),
                        "out_of_conflict": v.get("out_of_conflict"),
                        "stacker": v.get("stacker"),
                        "elapsed_s": v.get("elapsed_s"),
                    })
                else:
                    rec["error"] = f"http_{r.status_code}"
            except Exception as e:
                rec["error"] = f"{type(e).__name__}:{e}"
            w.writerow(rec)
            f.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(rows):
                print(f"[{i+1}/{len(rows)}] {row['arm']} "
                      f"p={rec['p_fake_fused']} err={rec['error']} "
                      f"({(time.time()-t0)/max(1,i+1):.2f}s/clip)", flush=True)


def analyse() -> dict:
    import numpy as np
    from sklearn.metrics import roc_auc_score
    rows = [r for r in csv.DictReader(open(OUT / "results.csv"))
            if not r["error"] and r["p_fake_fused"] not in ("", None)]
    for r in rows:
        r["p"] = float(r["p_fake_fused"])
        r["y"] = int(r["label"])

    def auc(a: str, b: str):
        ra = [r for r in rows if r["arm"] == a]
        rb = [r for r in rows if r["arm"] == b]
        if len(ra) < 10 or len(rb) < 10:
            return None
        y = [0] * len(ra) + [1] * len(rb)
        s = [r["p"] for r in ra] + [r["p"] for r in rb]
        return round(float(roc_auc_score(y, s)), 4)

    arms = {}
    for a in ("real_msrvtt", "real_ffpp", "fake_opentora"):
        sub = [r for r in rows if r["arm"] == a]
        arms[a] = {
            "n": len(sub),
            "mean_p": round(float(np.mean([r["p"] for r in sub])), 4) if sub else None,
            "median_p": round(float(np.median([r["p"] for r in sub])), 4) if sub else None,
            "abstain": round(float(np.mean([str(r["abstained"]).lower() in ("true", "1") for r in sub])), 4) if sub else None,
            "issue_rate": round(float(np.mean([r["p"] >= 0.5 for r in sub])), 4) if sub else None,
            "no_face": round(float(np.mean([str(r["n_identities"]) == "0" for r in sub])), 4) if sub else None,
        }

    auc_tt = auc("real_msrvtt", "fake_opentora")
    auc_tf = auc("real_ffpp", "fake_opentora")
    if auc_tt is None or auc_tf is None:
        call = "INCOMPLETE"
    elif auc_tt >= 0.70 and auc_tf <= 0.60:
        call = "CONTENT_CONFOUND"
    elif auc_tt <= 0.60 and auc_tf <= 0.60:
        call = "DETECTOR_BLIND"
    elif auc_tt >= 0.70 and auc_tf >= 0.70:
        call = "OPERATING_POINT"
    else:
        call = "MIXED"

    m = {"date": time.strftime("%Y-%m-%d"), "url": URL, "seed": SEED,
         "arms": arms, "auc_fake_vs_msrvtt": auc_tt,
         "auc_fake_vs_ffpp": auc_tf, "call": call,
         "interpretation_rule": {
             "CONTENT_CONFOUND": "AUC(msrvtt)>=0.70 and AUC(ffpp)<=0.60",
             "DETECTOR_BLIND": "AUC(msrvtt)<=0.60 and AUC(ffpp)<=0.60",
             "OPERATING_POINT": "both >=0.70",
             "MIXED": "otherwise"}}
    (OUT / "metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))
    return m


def main() -> int:
    rows = build_manifest()
    print(f"manifest {len(rows)} rows")
    score(rows)
    analyse()
    return 0


if __name__ == "__main__":
    sys.exit(main())
