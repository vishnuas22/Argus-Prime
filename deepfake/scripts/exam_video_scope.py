"""P2 offline video OOD/scope-gate exam.

Read-only on the deployed court: this script never edits argus_detect.py, the
server, the UI, or any stacker. It evaluates a deterministic, train-real-only
scope gate on existing feature CSVs (FF++/CDF/AVLips) and on a held-out T2V
sample plus a fixed-re-encode control.

Sealed spec: artifacts/p2_video_scope/SPEC.md (hash checked before scoring).
"""
import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))
sys.path.insert(0, str(WS / "deepfake" / "scripts"))

SPEC = WS / "artifacts" / "p2_video_scope" / "SPEC.md"
SPEC_HASH = WS / "artifacts" / "p2_video_scope" / "SPEC.sha256"
OUT_DEFAULT = WS / "artifacts" / "p2_video_scope"

TRAIN = WS / "artifacts" / "restack_video_v4e" / "dump_train" / "scores.csv"
CDF_TRAIN = WS / "artifacts" / "restack_video_v4e" / "dump_cdf" / "scores.csv"
FFPP = WS / "artifacts" / "restack_video_v4e" / "gate_ffpp" / "features.csv"
CDF = WS / "artifacts" / "restack_video_v4e" / "gate_celebdf" / "features.csv"
AVLIPS = WS / "artifacts" / "restack_video_v5" / "gate_avlips" / "features.csv"
T2V_SCORES = (WS / "artifacts" / "exam_video_t2v_20261011_arcface" /
              "scores.jsonl")
GENVID = WS / "data" / "raw" / "genvidbench"

FEATURES = [
    "scene_face_area_mean", "scene_face_area_p90", "scene_seam_delta",
    "scene_color_delta", "scene_freq_ratio", "scene_flicker_delta",
    "scene_context_sharp", "drift_p95_max", "n_identities", "faces_detected",
]
FACE_FEATURES = {"scene_face_area_mean", "scene_face_area_p90"}
SCENE_FEATURES = {
    "scene_seam_delta", "scene_color_delta", "scene_freq_ratio",
    "scene_flicker_delta", "scene_context_sharp",
}
TRACK_FEATURES = {"n_identities", "faces_detected"}
ALLOWED_REASONS = {
    "video_scope_face_area_low", "video_scope_face_area_high",
    "video_scope_face_area_both", "video_scope_scene_shift",
    "video_scope_temporal_shift", "video_scope_track_shift",
    "video_scope_feature_vote", "video_scope_no_face",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def verify_spec() -> str:
    expected = SPEC_HASH.read_text().split()[0]
    actual = sha256_file(SPEC)
    if actual != expected:
        raise SystemExit(
            f"SEALED SPEC MISMATCH: expected {expected}, got {actual}")
    return actual


def wilson(k: int, n: int, z: float = 1.96) -> list:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [round(max(0.0, centre - half), 6),
            round(min(1.0, centre + half), 6)]


def load_calibration() -> tuple:
    tr = pd.read_csv(TRAIN)
    cdf = pd.read_csv(CDF_TRAIN)
    cal = pd.concat([tr[tr["label"] == 0], cdf[cdf["label"] == 0]],
                    ignore_index=True)
    if len(cal) != 1411:
        raise SystemExit(f"calibration count {len(cal)} != sealed 1411")
    if not (cal["label"] == 0).all():
        raise SystemExit("calibration contains non-real rows")
    thresholds = {}
    for f in FEATURES:
        vals = cal[f].to_numpy(dtype=float)
        lo = float(np.nanquantile(vals, 0.005))
        hi = float(np.nanquantile(vals, 0.995))
        thresholds[f] = {"lo": round(lo, 8), "hi": round(hi, 8),
                         "n": int(np.isfinite(vals).sum())}
    meta = {"n_rows": int(len(cal)), "features": thresholds,
            "source": [str(TRAIN), str(CDF_TRAIN)],
            "fit_on": "FF++ train reals + CDF train reals only",
            "quantiles": [0.005, 0.995]}
    return thresholds, canonical_hash(meta), meta


def _reason_for(votes: list) -> tuple:
    reasons = set()
    for f in votes:
        if f in FACE_FEATURES:
            reasons.add("video_scope_face_area")
        elif f in SCENE_FEATURES:
            reasons.add("video_scope_scene_shift")
        elif f == "drift_p95_max":
            reasons.add("video_scope_temporal_shift")
        elif f in TRACK_FEATURES:
            reasons.add("video_scope_track_shift")
        else:
            reasons.add("video_scope_feature_vote")
    face_dirs = []
    for f in votes:
        if f in FACE_FEATURES:
            # direction is filled by the caller; placeholder
            face_dirs.append(f)
    if any(f in FACE_FEATURES for f in votes):
        primary = "video_scope_face_area_both"
    elif "video_scope_scene_shift" in reasons:
        primary = "video_scope_scene_shift"
    elif "video_scope_temporal_shift" in reasons:
        primary = "video_scope_temporal_shift"
    elif "video_scope_track_shift" in reasons:
        primary = "video_scope_track_shift"
    else:
        primary = "video_scope_feature_vote"
    # Refine face primary by direction when all face votes agree.
    dirs = set()
    for f in votes:
        if f in FACE_FEATURES:
            dirs.add(f)
    return primary, sorted(reasons)


def gate_row(row: dict, route: str, thresholds: dict) -> dict:
    if route == "image_court" or row is None:
        return {"abstained": True, "reason": "video_scope_no_face",
                "reasons": ["video_scope_no_face"], "votes": [],
                "n_votes": 0}
    votes, directions = [], {}
    for f in FEATURES:
        v = row.get(f)
        if v is None:
            votes.append(f)
            directions[f] = "missing"
            continue
        try:
            if math.isnan(float(v)):
                votes.append(f)
                directions[f] = "missing"
                continue
        except (TypeError, ValueError):
            votes.append(f)
            directions[f] = "missing"
            continue
        lo, hi = thresholds[f]["lo"], thresholds[f]["hi"]
        if v < lo:
            votes.append(f)
            directions[f] = "low"
        elif v > hi:
            votes.append(f)
            directions[f] = "high"
    if len(votes) < 2:
        return {"abstained": False, "reason": None, "reasons": [],
                "votes": votes, "n_votes": len(votes)}
    face_votes = [f for f in votes if f in FACE_FEATURES]
    if face_votes:
        face_dirs = {directions[f] for f in face_votes}
        if face_dirs == {"low"}:
            primary = "video_scope_face_area_low"
        elif face_dirs == {"high"}:
            primary = "video_scope_face_area_high"
        elif face_dirs == {"missing"}:
            primary = "video_scope_feature_vote"
        else:
            primary = "video_scope_face_area_both"
        reasons = {primary}
        for f in votes:
            if f in SCENE_FEATURES:
                reasons.add("video_scope_scene_shift")
            elif f == "drift_p95_max":
                reasons.add("video_scope_temporal_shift")
            elif f in TRACK_FEATURES:
                reasons.add("video_scope_track_shift")
        reasons = sorted(reasons)
    else:
        primary, reasons = _reason_for(votes)
    return {"abstained": True, "reason": primary, "reasons": reasons,
            "votes": votes, "n_votes": len(votes),
            "directions": directions}


def apply_gate(record: dict, gate: dict) -> dict:
    out = dict(record)
    out["p_before"] = out.get("p_fake_fused")
    if gate["abstained"]:
        out["abstained"] = True
        out["trust"] = 0.3
        out["risk"] = 0.8
        out["scope"] = {"abstained": True, "reason": gate["reason"],
                        "reasons": gate["reasons"], "votes": gate["votes"],
                        "n_votes": gate["n_votes"]}
    else:
        out["abstained"] = False
        out["scope"] = {"abstained": False, "reason": None,
                        "reasons": [], "votes": gate["votes"],
                        "n_votes": gate["n_votes"]}
    out["p_after"] = out.get("p_fake_fused")
    return out


def auc(rows: list) -> float | None:
    from sklearn.metrics import roc_auc_score
    vals = [(int(r["label"]), float(r["p_fake_fused"])) for r in rows
            if r.get("p_fake_fused") is not None]
    if len({v[0] for v in vals}) < 2:
        return None
    return round(float(roc_auc_score([v[0] for v in vals],
                                     [v[1] for v in vals])), 6)


def eval_csv(path: Path, name: str, thresholds: dict) -> tuple:
    df = pd.read_csv(path)
    rows = []
    for rec in df.to_dict(orient="records"):
        route = str(rec.get("route", "video_face"))
        p = rec.get("stacked")
        record = {"corpus": name, "file": rec.get("file"),
                  "label": int(rec.get("label")),
                  "method": rec.get("method"),
                  "route": route, "p_fake_fused": p}
        gate = gate_row(rec, route, thresholds)
        rows.append(apply_gate(record, gate))
    all_auc = auc(rows)
    issued = [r for r in rows if not r["abstained"]]
    issued_auc = auc(issued)
    by_label = {}
    for lab in (0, 1):
        sub = [r for r in rows if r["label"] == lab]
        k = sum(1 for r in sub if r["abstained"])
        by_label[str(lab)] = {
            "n": len(sub), "abstained": k,
            "rate": round(k / len(sub), 6) if sub else None,
            "wilson": wilson(k, len(sub)),
        }
    overall_k = sum(1 for r in rows if r["abstained"])
    metrics = {
        "n": len(rows),
        "abstained": overall_k,
        "abstain_rate": round(overall_k / len(rows), 6),
        "abstain_wilson": wilson(overall_k, len(rows)),
        "by_label": by_label,
        "auc_all": all_auc,
        "auc_issued": issued_auc,
        "n_issued": len(issued),
        "delta_auc_issued": (round(issued_auc - all_auc, 6)
                             if issued_auc is not None and all_auc is not None
                             else None),
    }
    return rows, metrics


def select_t2v(limit_per_gen: int | None = None,
               limit_image: int | None = None) -> tuple:
    rows = [json.loads(line) for line in T2V_SCORES.read_text().splitlines()
            if line.strip()]
    face = [r for r in rows if r.get("route") == "video_face"]
    image = [r for r in rows if r.get("route") == "image_court"]
    exploratory = set()
    for gen in ("sora", "kling", "opentora"):
        g = [r for r in face if r.get("generator") == gen]
        for r in g[:20]:
            exploratory.add((gen, r["file"]))
    remaining = [r for r in face
                 if (r.get("generator"), r["file"]) not in exploratory]
    take = {"sora": 25, "kling": 40, "opentora": 40}
    confirm = []
    for gen, n in take.items():
        g = [r for r in remaining if r.get("generator") == gen]
        g.sort(key=lambda r: hashlib.sha256(
            f"{gen}/{r['file']}".encode()).hexdigest())
        for r in g[:n]:
            confirm.append({**r, "kind": "face"})
    image_take = image[:limit_image] if limit_image else image
    for r in image_take:
        confirm.append({**r, "kind": "image_court"})
    if limit_per_gen:
        out, counts = [], {}
        for r in confirm:
            g = r.get("generator", "?")
            if r["kind"] == "face":
                if counts.get(g, 0) >= limit_per_gen:
                    continue
                counts[g] = counts.get(g, 0) + 1
            out.append(r)
        confirm = out
    return confirm, {"exploratory_face": len(exploratory),
                     "confirm_face": sum(1 for r in confirm
                                         if r["kind"] == "face"),
                     "confirm_image_court": sum(1 for r in confirm
                                                if r["kind"] == "image_court"),
                     "take": take}


def ffmpeg_bin() -> Path:
    from video_score import _ffmpeg_bin
    ff = _ffmpeg_bin()
    if ff is None:
        raise SystemExit("no working ffmpeg")
    return ff


def reencode(video: Path, out: Path, ff: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ff), "-y", "-loglevel", "error", "-i", str(video),
        "-map", "0:v:0", "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-g", "24", "-keyint_min", "24", "-sc_threshold", "0",
        "-pix_fmt", "yuv420p", "-map_metadata", "-1",
        "-movflags", "+faststart", str(out),
    ]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    rec = {"input": str(video), "output": str(out), "cmd": cmd,
           "returncode": r.returncode, "stderr": r.stderr[-2000:],
           "wall_s": round(time.time() - t0, 2)}
    if r.returncode != 0 or not out.exists() or out.stat().st_size < 1000:
        rec["error"] = "reencode_failed"
        return rec
    rec["input_bytes"] = video.stat().st_size
    rec["output_bytes"] = out.stat().st_size
    rec["output_sha256"] = sha256_file(out)
    return rec


def extract_features(video: Path) -> tuple:
    from video_score import visual_channel_bundle
    from video_channels import drift_features
    try:
        vis, tracks, _, _ = visual_channel_bundle(video, 12, with_scene=True)
    except Exception as e:
        return None, "error", f"{type(e).__name__}: {e}"
    if vis is None:
        return None, "image_court", None
    scene = vis.get("scene") or {}
    df = drift_features(tracks, spike_thresh=None)
    row = {f: scene.get(f) for f in FEATURES if f in scene}
    row["drift_p95_max"] = df.get("drift_p95")
    row["n_identities"] = vis.get("n_identities")
    row["faces_detected"] = vis.get("faces_detected")
    return row, "video_face", None


def eval_t2v(confirm: list, thresholds: dict, out: Path,
             max_per_gen: int | None = None) -> tuple:
    from video_face_track import set_embedder
    set_embedder("arcface")
    ff = ffmpeg_bin()
    rows, receipts = [], []
    reenc_dir = out / "reencoded"
    for i, item in enumerate(confirm):
        gen, fname = item["generator"], item["file"]
        src = GENVID / gen / fname
        row = {"corpus": "t2v", "kind": item["kind"], "generator": gen,
               "file": fname, "label": 1,
               "route_ref": item.get("route"),
               "p_ref": item.get("p_fake_fused")}
        feat_o, route_o, err_o = extract_features(src)
        gate_o = gate_row(feat_o, route_o, thresholds)
        row["route_orig"] = route_o
        row["feature_error_orig"] = err_o
        row["orig_abstained"] = bool(gate_o["abstained"])
        row["orig_reason"] = gate_o["reason"]
        row["orig_votes"] = gate_o["votes"]
        row["orig_features"] = feat_o
        # Contract fields for G6 (the original decision is the scored one).
        row["abstained"] = bool(gate_o["abstained"])
        row["scope"] = {"abstained": bool(gate_o["abstained"]),
                        "reason": gate_o["reason"],
                        "reasons": gate_o["reasons"],
                        "votes": gate_o["votes"],
                        "n_votes": gate_o["n_votes"]}
        row["p_before"] = row.get("p_ref")
        row["p_after"] = row.get("p_ref")
        row["trust"] = 0.3 if gate_o["abstained"] else 0.5
        row["risk"] = 0.8 if gate_o["abstained"] else 0.4
        reenc = reenc_dir / gen / fname
        rr = reencode(src, reenc, ff)
        receipts.append(rr)
        if rr.get("error"):
            row["reenc_error"] = rr["error"]
            row["reenc_route"] = "error"
            row["reenc_abstained"] = True
            row["reenc_reason"] = "video_scope_feature_vote"
            row["reenc_votes"] = []
            row["reenc_features"] = None
        else:
            feat_r, route_r, err_r = extract_features(reenc)
            gate_r = gate_row(feat_r, route_r, thresholds)
            row["reenc_route"] = route_r
            row["reenc_error"] = err_r
            row["reenc_abstained"] = bool(gate_r["abstained"])
            row["reenc_reason"] = gate_r["reason"]
            row["reenc_votes"] = gate_r["votes"]
            row["reenc_features"] = feat_r
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[t2v] {i+1}/{len(confirm)} {gen}/{fname}", flush=True)
    return rows, receipts


def summarize_t2v(rows: list) -> dict:
    def rate(sub, key):
        if not sub:
            return {"n": 0, "k": 0, "rate": None, "wilson": [0, 0]}
        k = sum(1 for r in sub if r[key])
        return {"n": len(sub), "k": k, "rate": round(k / len(sub), 6),
                "wilson": wilson(k, len(sub))}
    face = [r for r in rows if r["kind"] == "face"]
    image = [r for r in rows if r["kind"] == "image_court"]
    # Denominator is the full reference face set: a current no-face/error
    # result is itself a scope abstention, not an exclusion.
    face_orig = face
    face_re = face
    per_gen = {}
    for gen in ("sora", "kling", "opentora"):
        sub = [r for r in face_orig if r["generator"] == gen]
        per_gen[gen] = rate(sub, "orig_abstained")
    agreement = None
    if face:
        agreement = round(sum(1 for r in face
                              if bool(r["orig_abstained"]) ==
                              bool(r["reenc_abstained"])) / len(face), 6)
    return {
        "face_n": len(face),
        "face_orig": rate(face_orig, "orig_abstained"),
        "face_reenc": rate(face_re, "reenc_abstained"),
        "face_per_generator": per_gen,
        "image_n": len(image),
        "image_orig": rate(image, "orig_abstained"),
        "image_reenc": rate(image, "reenc_abstained"),
        "decision_agreement": agreement,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-t2v", action="store_true")
    ap.add_argument("--max-per-gen", type=int, default=None,
                    help="smoke only: cap face clips per generator")
    ap.add_argument("--max-image", type=int, default=None,
                    help="smoke only: cap image-court clips")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not args.force:
        print(f"ERROR: {out}/metrics.json exists — pass --force", file=sys.stderr)
        return 1

    spec_sha = verify_spec()
    print(f"[seal] spec sha256={spec_sha}", flush=True)
    thresholds, calib_sha, calib_meta = load_calibration()
    (out / "thresholds.json").write_text(json.dumps(
        {**calib_meta, "sha256": calib_sha}, indent=2))
    print(f"[cal] n={calib_meta['n_rows']} sha={calib_sha}", flush=True)

    all_rows, metrics = [], {"spec_sha256": spec_sha,
                             "calibration_sha256": calib_sha,
                             "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    for path, name in ((FFPP, "ffpp_test"), (CDF, "celebdf_test"),
                       (AVLIPS, "avlips")):
        rows, m = eval_csv(path, name, thresholds)
        all_rows.extend(rows)
        metrics[name] = m
        print(f"[csv] {name}: {json.dumps(m, sort_keys=True)}", flush=True)

    t2v_rows = []
    if not args.skip_t2v:
        confirm, sel = select_t2v(limit_per_gen=args.max_per_gen,
                                  limit_image=args.max_image)
        (out / "t2v_confirm_manifest.json").write_text(
            json.dumps({"selection": sel, "items": [
                {"generator": r["generator"], "file": r["file"],
                 "kind": r["kind"]} for r in confirm]}, indent=2))
        print(f"[t2v] selected {len(confirm)} clips: {sel}", flush=True)
        t2v_rows, receipts = eval_t2v(confirm, thresholds, out,
                                      max_per_gen=args.max_per_gen)
        (out / "reencode_receipt.json").write_text(
            json.dumps(receipts, indent=2))
        metrics["t2v"] = summarize_t2v(t2v_rows)
        all_rows.extend([{**r, "p_fake_fused": r.get("p_ref")}
                         for r in t2v_rows])
        print(f"[t2v] {json.dumps(metrics['t2v'], sort_keys=True)}", flush=True)

    # G1: in-domain non-suppression
    g1 = {}
    for name in ("ffpp_test", "celebdf_test"):
        m = metrics[name]
        ok = (m["abstain_rate"] <= 0.05 and
              all(v["rate"] is not None and v["rate"] <= 0.05
                  for v in m["by_label"].values()) and
              (m["delta_auc_issued"] is None or m["delta_auc_issued"] >= -0.01))
        g1[name] = {"pass": bool(ok), **m}
    metrics["G1"] = {"pass": all(v["pass"] for v in g1.values()), "corpora": g1}

    # G2: AVLips confirm split (deterministic 50/50 by method/file hash)
    av = pd.read_csv(AVLIPS)
    av["_h"] = [hashlib.sha256(f"{r.method}/{r.file}".encode()).hexdigest()
                for r in av.itertuples()]
    design, confirm = [], []
    for lab in (0, 1):
        sub = av[av["label"] == lab].sort_values("_h")
        design.extend(sub.iloc[::2].index.tolist())
        confirm.extend(sub.iloc[1::2].index.tolist())
    # Use the hash split directly on the AVLips eval rows. Key on
    # (method, file): AVLips can reuse bare filenames across classes.
    av_rows = [r for r in all_rows if r["corpus"] == "avlips"]
    by_key = {(r["method"], r["file"]): r for r in av_rows}
    conf_rows = [by_key[(av.loc[i, "method"], av.loc[i, "file"])]
                 for i in confirm
                 if (av.loc[i, "method"], av.loc[i, "file"]) in by_key]
    fake = [r for r in conf_rows if r["label"] == 1]
    real = [r for r in conf_rows if r["label"] == 0]
    fk = sum(1 for r in fake if r["abstained"])
    rk = sum(1 for r in real if r["abstained"])
    fr = fk / len(fake) if fake else 0.0
    rr = rk / len(real) if real else 0.0
    g2_ok = (fr >= 0.95 and rr >= 0.95 and abs(fr - rr) <= 0.05)
    metrics["G2"] = {
        "pass": bool(g2_ok), "confirm_n": len(conf_rows),
        "fake": {"n": len(fake), "k": fk, "rate": round(fr, 6),
                 "wilson": wilson(fk, len(fake))},
        "real": {"n": len(real), "k": rk, "rate": round(rr, 6),
                 "wilson": wilson(rk, len(real))},
        "parity": round(abs(fr - rr), 6),
        "full": metrics["avlips"],
    }

    if not args.skip_t2v:
        t = metrics["t2v"]
        g3_ok = (t["face_orig"]["rate"] is not None and
                 t["face_orig"]["rate"] >= 0.55 and
                 all(v["rate"] is not None and v["rate"] >= 0.45
                     for v in t["face_per_generator"].values()))
        metrics["G3"] = {"pass": bool(g3_ok), **t}
        g4_ok = (t["image_orig"]["rate"] is not None and
                 t["image_orig"]["rate"] >= 0.95)
        metrics["G4"] = {"pass": bool(g4_ok), **t["image_orig"]}
        g5_ok = (t["face_reenc"]["rate"] is not None and
                 t["face_reenc"]["rate"] >= 0.50 and
                 abs(t["face_orig"]["rate"] - t["face_reenc"]["rate"]) <= 0.10 and
                 t["decision_agreement"] is not None and
                 t["decision_agreement"] >= 0.75 and
                 t["image_reenc"]["rate"] is not None and
                 t["image_reenc"]["rate"] >= 0.95)
        metrics["G5"] = {
            "pass": bool(g5_ok),
            "face_orig_rate": t["face_orig"]["rate"],
            "face_reenc_rate": t["face_reenc"]["rate"],
            "face_diff": (round(abs(t["face_orig"]["rate"] -
                                    t["face_reenc"]["rate"]), 6)
                          if t["face_orig"]["rate"] is not None and
                          t["face_reenc"]["rate"] is not None else None),
            "agreement": t["decision_agreement"],
            "image_reenc_rate": t["image_reenc"]["rate"],
        }

    # G6: contract
    contract_bad = []
    for r in all_rows:
        if r.get("abstained"):
            reason = (r.get("scope") or {}).get("reason")
            if reason not in ALLOWED_REASONS:
                contract_bad.append((r.get("corpus"), r.get("file"), reason))
            if r.get("p_before") != r.get("p_after"):
                contract_bad.append((r.get("corpus"), r.get("file"), "p_changed"))
            if r.get("trust") != 0.3 or r.get("risk") != 0.8:
                contract_bad.append((r.get("corpus"), r.get("file"), "trust_risk"))
    metrics["G6"] = {"pass": not contract_bad, "violations": contract_bad[:20],
                     "n_violations": len(contract_bad)}

    # G7: determinism on 20 real fixture rows (the FF++ feature CSV).
    fixture = pd.read_csv(FFPP).head(20).to_dict(orient="records")
    det_a = [gate_row(r, "video_face", thresholds) for r in fixture]
    det_b = [gate_row(r, "video_face", thresholds) for r in fixture]
    metrics["G7"] = {"pass": det_a == det_b, "n": len(fixture),
                     "spec_sha256": spec_sha}

    # G8: calibration integrity
    metrics["G8"] = {"pass": calib_meta["n_rows"] == 1411 and
                     all(v["n"] > 0 for v in thresholds.values()),
                     "n_rows": calib_meta["n_rows"],
                     "calibration_sha256": calib_sha}

    bars = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8"]
    integrity = all(metrics.get(b, {}).get("pass", False)
                    for b in ("G4", "G6", "G7", "G8"))
    bars_pass = all(metrics.get(b, {}).get("pass", False) for b in bars)
    metrics["verdict"] = {
        "status": ("PASS" if bars_pass else
                   ("INTEGRITY_FAIL" if not integrity else "FAIL")),
        "bars": {b: bool(metrics.get(b, {}).get("pass", False)) for b in bars},
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "scores.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r, default=str) + "\n")
    if t2v_rows:
        flat = []
        for r in t2v_rows:
            rec = {**(r.get("orig_features") or {}),
                   "kind": r["kind"], "generator": r["generator"],
                   "file": r["file"],
                   "orig_abstained": r["orig_abstained"],
                   "reenc_abstained": r["reenc_abstained"]}
            rec.update({f"re_{k}": v for k, v in
                        (r.get("reenc_features") or {}).items()})
            flat.append(rec)
        pd.DataFrame(flat).to_csv(out / "t2v_features.csv", index=False)
    print(json.dumps(metrics["verdict"], indent=2), flush=True)
    return 0 if bars_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
