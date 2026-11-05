#!/usr/bin/env python
"""W-AV2 SPIKE (pre-seal research): perioral kinematics feasibility probe.

BioLip (arXiv 2604.16808) reports AUC 0.9843 on AVLips with perioral-landmark
kinematics (velocity/acceleration/jerk variance). This spike measures, with
OUR frame budget (12 frames) and OUR corpora, whether the feature family is
discriminative on avlips val content AND sane on RAVDESS reals — the two
conditions the AV-only gate failed (AUC ok, calibration transfer broken).

Spike gate: kinematics-only AUC on the avlips sample >= 0.75 -> strong-go for
the W-AV2 wave; 0.65-0.75 -> marginal (combine with diet decision); < 0.65 ->
no-go (0.98 was in-distribution-only).

D-050 receipt: mediapipe 1.2.2 (Apache-2.0, pip);
face_landmarker.task sha256 64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
(3,758,596 B, storage.googleapis.com float16/latest).
"""
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake" / "bench"))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

FRAMES = 12
MODEL = WS / "integrations" / "mediapipe" / "face_landmarker.task"
# outer-lip landmark indices (MediaPipe canonical 478 mesh)
LIP_IDX = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
           146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
EYE_L, EYE_R = 33, 263  # outer eye corners for scale normalization


def kinematics(video: Path):
    """Return kinematics features or None if no face track.

    v2: rigid-align each frame's landmarks to the first frame (similarity
    transform fit on stable points: eye corners, nose tip, chin, forehead)
    BEFORE measuring perioral displacement — isolates lip articulation from
    head motion (the v1 centroid feature was dominated by corpus recording
    style: RAVDESS studio-static vs avlips YouTube-style head movement)."""
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL)),
        running_mode=mp_vision.RunningMode.VIDEO, num_faces=1)
    det = mp_vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None
    idxs = np.linspace(0, max(n - 1, 1), FRAMES).astype(int)
    seq, ars = [], []
    ref = None
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = det.detect_for_video(mp_img, int(i) + 1)
        if not res.face_landmarks:
            continue
        fl = res.face_landmarks[0]
        lm = np.array([[p.x, p.y] for p in (fl.landmark if hasattr(fl, "landmark") else fl)])
        if ref is None:
            ref = lm
        # similarity alignment (no reflection): fit lm -> ref on stable pts
        src = lm[[EYE_L, EYE_R, 1, 152, 10]]
        dst = ref[[EYE_L, EYE_R, 1, 152, 10]]
        mu_s, mu_d = src.mean(0), dst.mean(0)
        X, Y = src - mu_s, dst - mu_d
        U, S, Vt = np.linalg.svd(X.T @ Y)
        R = U @ Vt
        if np.linalg.det(R) < 0:  # block reflection
            U[:, -1] *= -1
            R = U @ Vt
        scale = (S.sum() / (X ** 2).sum()) if (X ** 2).sum() > 0 else 1.0
        aligned = (lm - mu_s) @ R * scale + mu_d
        lip = aligned[LIP_IDX] / (scale + 1e-9)
        seq.append(lip)
        # mouth aspect: width / height of outer lip oval
        w = np.linalg.norm(aligned[61] - aligned[291])
        h = np.linalg.norm(aligned[0] - aligned[17]) + 1e-9
        ars.append(w / h)
    cap.release()
    if len(seq) < FRAMES - 2:
        return None
    P = np.array(seq)          # T x K x 2, aligned, scale-normalized
    D = P - P[0]               # displacement from reference frame
    V = np.diff(P, axis=0)     # per-landmark velocity
    A = np.diff(P, axis=0, n=2)
    J = np.diff(P, axis=0, n=3)
    ar = np.array(ars)
    return {
        "kin_disp_var": float(D.var()),
        "kin_vel_var": float(V.var()),
        "kin_acc_var": float(A.var()),
        "kin_jerk_var": float(J.var()),
        "kin_ar_var": float(ar.var()),
        "kin_ar_vel_var": float(np.diff(ar).var()),
        "kin_n_obs": len(seq),
    }


def sample_videos():
    import pandas as pd
    rng = random.Random(11)
    avl_root = WS / "data" / "raw" / "avlips" / "videos"
    man = json.loads((WS / "data" / "raw" / "avlips" / "manifest.json").read_text())
    fakes = [avl_root.parent / Path(p).relative_to("data/raw/avlips")
             for p in man["fake"]]
    reals = [avl_root.parent / Path(p).relative_to("data/raw/avlips")
             for p in man["real"]]
    pick_f = rng.sample(fakes, 120)
    pick_r = rng.sample(reals, 90)
    df = pd.read_csv(WS / "artifacts" / "restack_video_v5" / "dump_train" / "scores.csv")
    rav = df[(df["method"] == "real") & (df["avs_present"] == 1)].head(30)
    rav_files = []
    for base in rav["file"]:
        hits = sorted((WS / "data" / "raw" / "av_diet" / "ravdess").glob(f"*/{base}"))
        if hits:
            rav_files.append(hits[0])
    wav = sorted((WS / "data" / "raw" / "av_diet" / "gen" / "fakes").glob("*.mp4"))[:20]
    return ([(p, 1, "avlips_fake") for p in pick_f]
            + [(p, 0, "avlips_real") for p in pick_r]
            + [(p, 0, "ravdess_real") for p in rav_files]
            + [(p, 1, "wav2lip_fake") for p in wav])


def auc(pairs):
    s = sorted(pairs, key=lambda t: t[0])
    n1 = sum(l for _, l in pairs); n0 = len(pairs) - n1
    r, i, ranks = 0.0, 0, []
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + j - 1) / 2 + 1
        ranks.extend([avg] * (j - i)); i = j
    return (sum(rk for rk, (_, l) in zip(ranks, s) if l == 1) - n1 * (n1 + 1) / 2) / (n0 * n1)


def main():
    jobs = sample_videos()
    rows, noface = [], 0
    for i, (p, lbl, strat) in enumerate(jobs):
        f = kinematics(p)
        if f is None:
            noface += 1
            continue
        f.update(label=lbl, stratum=strat, file=p.name)
        rows.append(f)
        if (i + 1) % 40 == 0:
            print(f"[{i+1}/{len(jobs)}] scored, noface={noface}", flush=True)
    out = WS / "artifacts" / "restack_video_v5" / "spike_kinematics.json"
    avl = [r for r in rows if r["stratum"].startswith("avlips")]
    feats = ["kin_disp_var", "kin_vel_var", "kin_acc_var", "kin_jerk_var",
             "kin_ar_var", "kin_ar_vel_var"]
    aucs = {k: auc([(r[k], r["label"]) for r in avl]) for k in feats}
    best = max(aucs, key=aucs.get)
    a = aucs[best]
    per = {k: round(aucs[k], 4) for k in feats}
    import statistics as st
    groups = {}
    for strat in ("avlips_real", "avlips_fake", "ravdess_real", "wav2lip_fake"):
        sub = [r for r in rows if r["stratum"] == strat]
        groups[strat] = ({k: round(st.mean([r[k] for r in sub]), 6) for k in feats}
                         if sub else {"empty": True}) | {"n": len(sub)}
    result = {"n_scored": len(rows), "n_noface": noface,
              "avlips_best_feature": best, "avlips_kin_auc_best": round(a, 4),
              "per_feature_auc": {k: round(v, 4) for k, v in aucs.items()},
              "group_means": groups}
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
