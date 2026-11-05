"""Video restack pipeline — dump -> train -> gate -> verdict (spec item 3.3).

  DUMP    video_dump --split train (limit 1500) and --split val (limit 300);
          plus the D-021 domain diet: dump_cdf = --corpus celebdf (limit 1000,
          official CDF test list excluded — 5b.5); audio on a deterministic
          200-video subsample only (decision #3)
          W4 (D-066): --bootstrap-from (default artifacts/restack_video_v2)
          byte-preserves the incumbent FF++/CDF dumps instead of re-dumping
          (sha + rowcount receipts; law §4.4), then stage_dump_dfd appends
          the DeepFakeDetection actor-arm diet (fakes + their actor-original
          reals together, manifest-determined, --drift-cal pass-through).
  TRAIN   train_video_stacker on dump_train + dump_val + dump_cdf ->
          work/candidate/{stacker.joblib, stacker_meta.json,
          drift_calibration.json}
  GATE    A: full official FF++ test split (700 videos, one pass) — stacked
            >= 0.90 AND stacked >= naive_combined - 0.005 AND per-method bars
            (Deepfakes/Face2Face/FaceSwap >= 0.88, NeuralTextures >= 0.60)
          B: leave-one-method-out (4 feature-level retrains) >= 0.70 each
          C: Celeb-DF cross-dataset stacked >= 0.75 (0.70-0.75 = escalate)
          D: exam_video_dfd on held-out DFD sources (W4, D-066) — stacked
            >= 0.85 AND exam-real mean pf <= 0.35 AND frac(pf>0.5) <= 0.25
            (the L-CHANNEL-CONFOUND guard)
          W-AV (--av, D-076): dump stage bootstraps restack_video_v4's dumps
            byte-verbatim, text-splices the 6 AV columns (zeros — those rows
            predate the AV channel; NO pandas round-trip), then appends the
            av diet (matched Wav2Lip pairs + narration/music strata, audio
            ON, av-calibrated). Gates gain:
          E: exam_video_av — self-gen held-out stacked >= 0.85 (D1) AND
            AVLips-val external stacked >= 0.70 (D2) AND integrity clean
            (D3 hard stop: zero source straddles, ffprobe pair parity,
            dedup, license receipts complete)
          F: exam_video_av_confound — narration/music issued-fake FPR
            <= 0.10 (D4)
          NR1 (with --av): ffpp stacked must ALSO clear the incumbent
            baseline - 0.005 (v4e receipt), fail-closed if the receipt is
            missing.
  VERDICT PROMOTE iff A and B and C and D (and E and F with --av); --promote
          --promote-to <tag> copies the candidate to artifacts/<tag>/
          (explicit tag required, never auto-wired; --av adds
          av_calibration.json to the promoted bundle). A gate miss ESCALATES
          to the user — no autonomous waiver file is ever written (decision
          #2 / D-046).

Usage:
  .venv/bin/python deepfake/scripts/restack_video.py [--stage all|dump|train|gate|verdict]
      [--smoke] [--work artifacts/restack_video_v3] [--embedder arcface]
      [--bootstrap-from artifacts/restack_video_v2 | --no-bootstrap]
      [--av] [--promote --promote-to p6vid_v5]
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[2]
PY = WS / ".venv" / "bin" / "python"
SCRIPTS = WS / "deepfake" / "scripts"

GATES = {
    "ffpp": {"min_auc": 0.90, "vs_naive_tol": -0.005,
             # spec per-method bars (not a flat floor): the three identity-
             # visible forgeries must clear 0.88; NeuralTextures (mouth-
             # reenactment, weakest identity signal) keeps the 0.60 floor.
             "min_per_method": {"Deepfakes": 0.88, "Face2Face": 0.88,
                                "FaceSwap": 0.88, "NeuralTextures": 0.60},
             # W4b D4 elevation guard (user-sealed X=0.10, 2026-10-31):
             # stacked real-mean may exceed the naive real-mean by at most X
             # over this gate's features.csv real rows.
             "max_real_elevation": 0.10},
    "lomo": {"min_auc": 0.70},          # per held-out method (4 folds)
    "celebdf": {"min_auc": 0.75, "warn_auc": 0.70,
                "max_real_elevation": 0.10},
    # <0.75 = FAIL, 0.70-0.75 = escalate to the user (decision #2 — no auto-waive)
    # W4 (D-066) gate D: held-out DFD sources; D2 is the L-CHANNEL-CONFOUND
    # guard — if the diet taught "actor domain = fake", held-out actor reals
    # read fake-ward and D2 fires.
    "dfd": {"min_auc": 0.85, "max_real_mean_pf": 0.35,
            "max_real_frac_gt_half": 0.25,
            "max_real_elevation": 0.10},
    # W-AV (D-076 §6, user-sealed 2026-11-02): D1 self-gen held-out
    # (Wav2Lip matched pairs) + D2 AVLips-val external; D3 integrity is a
    # hard stop folded into av.pass (no numeric bar — any violation fails).
    "av": {"min_auc": 0.85, "avlips_min_auc": 0.70},
    # D4 confound guard: narration/music exam reals at the deployed operating
    # point must not read fake-ward above this issued-fake FPR.
    "av_confound": {"max_fpr_issued": 0.10},
}
METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
DFD_SPLIT_MANIFEST = "data/raw/ffpp_dfd/split.json"
BOOTSTRAP_DEFAULT = WS / "artifacts" / "restack_video_v2"
BOOTSTRAP_DUMPS = ("dump_train", "dump_val", "dump_cdf")
BOOTSTRAP_ROWS = {"dump_train": 1500, "dump_val": 300, "dump_cdf": 1000}
ESCALATE_BAND = (0.85, 0.90)            # FF++ stacked in this band => escalate
# W-AV (D-076): the av bootstrap source is PINNED to the v4 work dir (the
# incumbent being extended); its expected counts are read from the source's
# OWN receipts (dfd_dump_receipt totals + bootstrap_receipt cdf rows) — never
# hardcoded here.
AV_BOOTSTRAP_SOURCE = WS / "artifacts" / "restack_video_v4"
# NR1 (D-076): the candidate's ffpp stacked AUC may not drop more than this
# below the incumbent's receipted ffpp auc_stacked (v4e = 0.9121 -> bar
# max(0.90, 0.9071)).
NR1_BASELINE_WORK = AV_BOOTSTRAP_SOURCE
NR1_DELTA = -0.005
# D-076: the W4e recipe carries forward — the v5 candidate trains with lever
# A' (dfd sample-weight multiplier 4.0). The chain passes it explicitly;
# train_video_stacker.py defaults to 1.0 (pre-W4d behavior), which would
# silently train off-recipe. The W4d/W4e manual runs masked this gap because
# they invoked train_video_stacker.py directly with the flag.
DFD_WEIGHT_MULT_SEALED = 4.0


def run(cmd, **kw):
    print(f"[run] {' '.join(str(c) for c in cmd)}", flush=True)
    r = subprocess.run([str(c) for c in cmd], cwd=WS, **kw)
    if r.returncode != 0:
        raise SystemExit(f"command failed ({r.returncode}): {' '.join(map(str, cmd))}")
    return r


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_append(work: Path, lines: list):
    mp = work / "MANIFEST.md"
    with open(mp, "a") as f:
        f.write("\n".join(lines) + "\n")


def bootstrap_dumps(work: Path, src: Path) -> None:
    """W4 law §4.4: byte-preserve the incumbent's FF++/CDF diet rows.

    Copies dump_train/dump_val/dump_cdf from the bootstrap source verbatim
    (sha256 receipts pre/post; the pre-append receipt MUST equal 1500/300/1000
    rows — clean attribution: the only diet delta is DFD+actors, and dump
    re-scoring is not bit-guaranteed across runs). On any receipt failure this
    STOPS — a fresh full re-dump (~1.5h) is the user's fallback decision."""
    import json as _json
    receipt = {"source": str(src), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "dumps": {}}
    lines = ["## Bootstrap — byte-preserved diet rows (W4 law §4.4)", ""]
    for name in BOOTSTRAP_DUMPS:
        dst = work / name
        if (dst / "scores.csv").exists():
            print(f"[bootstrap] {name}: exists, skipped", flush=True)
            continue
        s_csv = src / name / "scores.csv"
        if not s_csv.exists():
            raise SystemExit(f"bootstrap source missing {s_csv} — cannot "
                             f"byte-preserve; STOP (fallback = fresh full "
                             f"re-dump, a user decision)")
        pre_sha = _sha256(s_csv)
        n_rows = len(s_csv.read_text().splitlines()) - 1
        if n_rows != BOOTSTRAP_ROWS[name]:
            raise SystemExit(f"bootstrap receipt FAILED: {name} has {n_rows} "
                             f"rows, expected {BOOTSTRAP_ROWS[name]} — STOP")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s_csv, dst / "scores.csv")
        if _sha256(dst / "scores.csv") != pre_sha:
            raise SystemExit(f"bootstrap copy sha mismatch for {name} — STOP")
        sidecar = {}
        for extra in ("drift_spike.json", "noface.jsonl"):
            if (src / name / extra).exists():
                shutil.copy2(src / name / extra, dst / extra)
                if extra == "drift_spike.json":
                    sidecar = _json.loads((dst / extra).read_text())
        import pandas as pd
        pre_df = pd.read_csv(dst / "scores.csv")
        receipt["dumps"][name] = {
            "rows": n_rows, "sha256": pre_sha, "sidecar": sidecar,
            "n_reals": int((pre_df["label"] == 0).sum()),
            "n_fakes": int((pre_df["label"] == 1).sum()),
            "per_method": {m: int((pre_df["method"] == m).sum())
                           for m in sorted(pre_df["method"].unique())}}
        lines.append(f"- {name}: {n_rows} rows, sha256 {pre_sha}, "
                     f"sidecar spike={sidecar.get('spike_thresh')} "
                     f"fit_on='{sidecar.get('fit_on')}' "
                     f"embedder={sidecar.get('embedder')}")
        print(f"[bootstrap] {name}: {n_rows} rows byte-preserved "
              f"(sha {pre_sha[:16]}…)", flush=True)
    (work / "bootstrap_receipt.json").write_text(_json.dumps(receipt, indent=2))
    _manifest_append(work, lines)


def _splice_av_columns(csv_path: Path, av_names: list) -> dict:
    """Text-level splice (W-AV D-076): insert the 6 AV columns (zeros) into a
    bootstrapped 42-col CSV, BEFORE the trailing n_drift_tracks column.

    Byte law: NO pandas round-trip — the 38 non-AV values must pass through
    as written (a pandas read->write cycle re-formats floats and would break
    byte-attribution for rows that never re-featurize). Each data line is
    split on commas, the prefix (all fields but the last) is rejoined
    verbatim, the six zeros are inserted, and the original last field is
    appended; per-row prefix equality holds by construction and is asserted
    via field counts. A pandas read-back (read-only) verifies the shape."""
    import pandas as pd
    from video_features import AV_FEATURES
    assert av_names == list(AV_FEATURES), "splice order must match AV_FEATURES"
    zeros = ",".join(["0"] * len(av_names))
    lines = csv_path.read_text().splitlines()
    hdr = lines[0].split(",")
    assert hdr[-1] == "n_drift_tracks", f"unexpected trailing column: {hdr[-1]}"
    assert len(hdr) == 42, f"expected 42-col incumbent CSV, got {len(hdr)}"
    new_hdr = hdr[:-1] + av_names + [hdr[-1]]
    out = [",".join(new_hdr)]
    for ln in lines[1:]:
        if not ln.strip():
            continue
        f = ln.split(",")
        assert len(f) == 42, f"row field count {len(f)} != 42 in {csv_path}"
        # prefix (fields 0..40) verbatim + 6 zeros + original last field
        out.append(",".join(f[:-1]) + "," + zeros + "," + f[-1])
    csv_path.write_text("\n".join(out) + "\n")
    df = pd.read_csv(csv_path)          # read-only post-condition check
    assert list(df.columns) == new_hdr, "spliced header mismatch"
    assert len(df) == len(out) - 1, "spliced row count mismatch"
    return {"rows": len(out) - 1, "cols": len(new_hdr)}


def bootstrap_dumps_av(work: Path, src: Path) -> None:
    """W-AV (D-076): byte-preserve the v4 dumps, then splice the 6 AV columns.

    Expected counts are read from the SOURCE's own receipts (dfd receipt
    totals for dump_train/dump_val; the bootstrap receipt's cdf rows) — never
    hardcoded. The splice inserts the AV columns as ZEROS: those rows predate
    the AV channel, and the missingness contract (avs absent -> zeros) is
    exactly what the stacker trained on for them. Receipts record pre/post
    sha256 per dump; any mismatch STOPS."""
    import json as _json
    import pandas as pd
    import sys as _sys
    _sys.path.insert(0, str(WS / "deepfake" / "bench"))
    from video_features import AV_FEATURES
    dfd_rcp = src / "dfd_dump_receipt.json"
    boot_rcp = src / "bootstrap_receipt.json"
    if not dfd_rcp.exists() or not boot_rcp.exists():
        raise SystemExit(f"av bootstrap source receipts missing in {src} "
                         f"(dfd_dump_receipt.json / bootstrap_receipt.json) "
                         f"— expected counts must come from the source's own "
                         f"receipts; STOP")
    dfd_r = _json.loads(dfd_rcp.read_text())
    boot_r = _json.loads(boot_rcp.read_text())
    for split in ("train", "val"):
        if not dfd_r["splits"][split].get("reconciled"):
            raise SystemExit(f"av bootstrap source {src}: dfd receipt for "
                             f"{split} is not reconciled — STOP")
    expected = {
        "dump_train": int(dfd_r["splits"]["train"]["total_rows"]),
        "dump_val": int(dfd_r["splits"]["val"]["total_rows"]),
        "dump_cdf": int(boot_r["dumps"]["dump_cdf"]["rows"]),
    }
    receipt = {"source": str(src), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "av_columns": list(AV_FEATURES), "dumps": {}}
    lines = ["## AV bootstrap — byte-preserved v4 rows + AV-column splice "
             "(W-AV D-076)", ""]
    for name in BOOTSTRAP_DUMPS:
        dst = work / name
        if (dst / "scores.csv").exists():
            print(f"[bootstrap-av] {name}: exists, skipped", flush=True)
            continue
        s_csv = src / name / "scores.csv"
        if not s_csv.exists():
            raise SystemExit(f"av bootstrap source missing {s_csv} — STOP")
        pre_sha = _sha256(s_csv)
        n_rows = len(s_csv.read_text().splitlines()) - 1
        if n_rows != expected[name]:
            raise SystemExit(f"av bootstrap receipt FAILED: {name} has "
                             f"{n_rows} rows, source receipt says "
                             f"{expected[name]} — STOP")
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s_csv, dst / "scores.csv")
        if _sha256(dst / "scores.csv") != pre_sha:
            raise SystemExit(f"av bootstrap copy sha mismatch for {name} — STOP")
        for extra in ("drift_spike.json", "noface.jsonl"):
            if (src / name / extra).exists():
                shutil.copy2(src / name / extra, dst / extra)
        spliced = _splice_av_columns(dst / "scores.csv", list(AV_FEATURES))
        df = pd.read_csv(dst / "scores.csv")
        receipt["dumps"][name] = {
            "rows": n_rows, "sha256_pre_splice": pre_sha,
            "sha256_post_splice": _sha256(dst / "scores.csv"),
            "cols_after_splice": spliced["cols"],
            "n_reals": int((df["label"] == 0).sum()),
            "n_fakes": int((df["label"] == 1).sum()),
            "per_method": {m: int((df["method"] == m).sum())
                           for m in sorted(df["method"].unique())}}
        lines.append(f"- {name}: {n_rows} rows byte-preserved from v4 "
                     f"(pre-splice sha {pre_sha[:16]}…), spliced to "
                     f"{spliced['cols']} cols with 6 zeroed AV columns")
        print(f"[bootstrap-av] {name}: {n_rows} rows byte-preserved + "
              f"spliced to {spliced['cols']} cols", flush=True)
    (work / "av_bootstrap_receipt.json").write_text(_json.dumps(receipt, indent=2))
    _manifest_append(work, lines)


def stage_dump_dfd(work, smoke, embedder):
    """W4 (D-066): append the DFD actor-arm diet into dump_<split>/.

    Fakes and their actor-original reals enter TOGETHER per source (same
    split; L-CHANNEL-CONFOUND law). Every invocation passes the TRAIN dump's
    sidecar as --drift-cal (spike pass-through law §4.3 — no run refits or
    rewrites existing rows). Row-count deltas are reconciled against the
    split manifest + noface.jsonl and receipted into MANIFEST.md."""
    cal = work / "dump_train" / "drift_spike.json"
    if not cal.exists():
        raise SystemExit(f"missing {cal} — bootstrap the dumps first")
    for split in ("train", "val"):
        cmd = [PY, SCRIPTS / "video_dump.py", "--split", split,
               "--corpus", "dfd", "--no-audio", "--out", work,
               "--embedder", embedder, "--drift-cal", cal]
        if smoke:
            cmd.append("--smoke")
        run(cmd)
    _dfd_receipts(work, smoke)


def _dfd_receipts(work: Path, smoke: bool) -> None:
    """Reconcile appended rows against the split manifest + noface.jsonl."""
    import json as _json
    import pandas as pd
    sources = _json.loads(
        (WS / "data" / "raw" / "ffpp_dfd" / "sources.json").read_text())
    boot = _json.loads((work / "bootstrap_receipt.json").read_text())
    cap = {"fakes": 4, "reals": 2} if smoke else None
    expected = {}
    for split in ("train", "val"):
        n_fakes = sum(len(r["fakes"]) for r in sources.values()
                      if r["split"] == split)
        n_reals = sum(1 for r in sources.values() if r["split"] == split)
        expected[split] = {
            "fakes": min(n_fakes, cap["fakes"]) if cap else n_fakes,
            "reals": min(n_reals, cap["reals"]) if cap else n_reals}
    receipt = {"smoke": bool(smoke), "splits": {}}
    lines = ["## DFD diet append receipts (W4 stage_dump_dfd)", ""]
    for split in ("train", "val"):
        d = work / f"dump_{split}"
        df = pd.read_csv(d / "scores.csv")
        dups = int(df.duplicated(subset=["method", "file"]).sum())
        assert dups == 0, f"{d}: {dups} duplicated (method,file) rows"
        pre = boot["dumps"][f"dump_{split}"]
        noface = {"0": 0, "1": 0}
        nf_path = d / "noface.jsonl"
        if nf_path.exists():
            seen_nf = set()
            for l in nf_path.read_text().splitlines():
                if l.strip():
                    r = _json.loads(l)
                    k = (r.get("method", ""), r["file"])
                    if k in seen_nf:
                        continue          # re-attempted on resume; count once
                    seen_nf.add(k)
                    noface[str(int(r["label"]))] += 1
        got_fakes = int((df["label"] == 1).sum()) - pre["n_fakes"]
        got_reals = int((df["label"] == 0).sum()) - pre["n_reals"]
        exp = expected[split]
        ok = (got_fakes + noface["1"] == exp["fakes"]
              and got_reals + noface["0"] == exp["reals"])
        per_method = {m: int((df["method"] == m).sum())
                      for m in sorted(df["method"].unique())}
        receipt["splits"][split] = {
            "appended_fakes": got_fakes, "appended_reals": got_reals,
            "noface_fakes": noface["1"], "noface_reals": noface["0"],
            "expected_fakes": exp["fakes"], "expected_reals": exp["reals"],
            "reconciled": bool(ok), "total_rows": len(df),
            "per_method": per_method}
        lines.append(
            f"- dump_{split}: appended {got_fakes} fakes + {got_reals} reals "
            f"(noface {noface['1']}/{noface['0']}) vs manifest-expected "
            f"{exp['fakes']}/{exp['reals']} -> "
            f"{'RECONCILED' if ok else 'MISMATCH'}; total rows {len(df)}; "
            f"per-method {per_method}")
        print(f"[dfd-receipt] {split}: fakes+{got_fakes} reals+{got_reals} "
              f"noface {noface['1']}+{noface['0']} vs expected "
              f"{exp['fakes']}/{exp['reals']} -> "
              f"{'OK' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            raise SystemExit(f"dfd append receipt MISMATCH for {split} — STOP")
    (work / "dfd_dump_receipt.json").write_text(_json.dumps(receipt, indent=2))
    _manifest_append(work, lines)


def stage_dump_av(work, smoke, embedder):
    """W-AV (D-076): bootstrap v4 byte-verbatim + AV splice, then append the
    av diet into dump_<split>/.

    Matched Wav2Lip fakes + their source reals enter TOGETHER per source
    (same split; L-CHANNEL-CONFOUND law), narration/music confound strata on
    train (+exam, which the confound exam reads directly). Audio is ON for
    every av row; the av dump fits av_calibration.json on TRAIN reals only
    (D-035) before writing rows. --drift-cal pass-through: appended rows
    reuse the bootstrapped train sidecar's spike threshold."""
    cal = work / "dump_train" / "drift_spike.json"
    if not cal.exists():
        # D-076: the av bootstrap (byte-preserved v4 dumps + AV splice) is the
        # step that materializes the sidecars this stage consumes. Invoke it
        # here so --stage all and --stage dump are both self-contained — the
        # bootstrap is idempotent (existing dumps are skipped, sha-receipted).
        bootstrap_dumps_av(work, AV_BOOTSTRAP_SOURCE)
    if not cal.exists():
        raise SystemExit(f"missing {cal} — av bootstrap did not produce it; STOP")
    for split in ("train", "val"):
        cmd = [PY, SCRIPTS / "video_dump.py", "--split", split,
               "--corpus", "av", "--out", work,
               "--embedder", embedder, "--drift-cal", cal]
        if smoke:
            cmd.append("--smoke")
        run(cmd)
    _av_receipts(work, smoke)


def _av_receipts(work: Path, smoke: bool) -> None:
    """Reconcile the av append against the diet manifests + gen receipts."""
    import json as _json
    import pandas as pd
    diet = WS / "data" / "raw" / "av_diet"
    splitj = _json.loads((diet / "split.json").read_text())
    genr = _json.loads((diet / "gen_receipt.json").read_text())
    boot = _json.loads((work / "av_bootstrap_receipt.json").read_text())
    av_methods = ("Wav2Lip", "real", "narration", "music")
    exp_key = {"Wav2Lip": "fakes", "real": "reals", "narration": "narration",
               "music": "music"}
    failed = genr.get("failed", {})
    expected = {}
    for split in ("train", "val"):
        keys = [k for k in splitj["splits"][split] if k not in failed]
        expected[split] = {
            "fakes": len(keys), "reals": len(keys),
            "narration": len(genr.get("narration", {}).get(split, [])),
            "music": len(genr.get("music", {}).get(split, []))}
    receipt = {"smoke": bool(smoke), "splits": {}}
    lines = ["## AV diet append receipts (W-AV stage_dump_av)", ""]
    for split in ("train", "val"):
        d = work / f"dump_{split}"
        df = pd.read_csv(d / "scores.csv")
        dups = int(df.duplicated(subset=["method", "file"]).sum())
        assert dups == 0, f"{d}: {dups} duplicated (method,file) rows"
        pre = boot["dumps"][f"dump_{split}"]
        noface = {m: 0 for m in av_methods}
        nf_path = d / "noface.jsonl"
        if nf_path.exists():
            seen_nf = set()
            for l in nf_path.read_text().splitlines():
                if l.strip():
                    r = _json.loads(l)
                    k = (r.get("method", ""), r["file"])
                    if k in seen_nf or r.get("method") not in noface:
                        continue      # bootstrapped rows / re-attempted resume
                    seen_nf.add(k)
                    noface[r["method"]] += 1
        got = {m: int((df["method"] == m).sum()) - pre["per_method"].get(m, 0)
               for m in av_methods}
        exp = expected[split]
        ok = all(got[m] + noface[m] == exp[exp_key[m]] for m in av_methods)
        receipt["splits"][split] = {
            "appended": got, "noface": noface, "expected": exp,
            "reconciled": bool(ok), "total_rows": len(df),
            "per_method": {m: int((df["method"] == m).sum())
                           for m in sorted(df["method"].unique())}}
        lines.append(
            f"- dump_{split}: appended fakes {got['Wav2Lip']}/{exp['fakes']}, "
            f"reals {got['real']}/{exp['reals']}, narration "
            f"{got['narration']}/{exp['narration']}, music "
            f"{got['music']}/{exp['music']} (noface "
            f"{sum(noface.values())}) -> "
            f"{'RECONCILED' if ok else 'MISMATCH'}; total rows {len(df)}")
        print(f"[av-receipt] {split}: fakes {got['Wav2Lip']}/{exp['fakes']} "
              f"reals {got['real']}/{exp['reals']} narr "
              f"{got['narration']}/{exp['narration']} music "
              f"{got['music']}/{exp['music']} noface "
              f"{sum(noface.values())} -> {'OK' if ok else 'MISMATCH'}",
              flush=True)
        if not ok:
            raise SystemExit(f"av append receipt MISMATCH for {split} — STOP")
    (work / "av_dump_receipt.json").write_text(_json.dumps(receipt, indent=2))
    _manifest_append(work, lines)


def stage_dump(work, smoke, embedder, bootstrap_from=None, full_redump=False):
    src = Path(bootstrap_from) if bootstrap_from else None
    if src is not None and src.exists():
        # W4 (D-066): byte-preserve the incumbent diet, then append DFD.
        bootstrap_dumps(work, src)
        stage_dump_dfd(work, smoke, embedder)
        return
    if src is not None:
        print(f"[dump] bootstrap source {src} not found — legacy dump path",
              flush=True)
    limits = (40, 20) if smoke else (1500, 300)
    audio = 0 if smoke else 200
    for split, lim in zip(("train", "val"), limits):
        out = work / f"dump_{split}"
        if (out / "scores.csv").exists():
            print(f"[dump] {split}: exists, skipped", flush=True)
        else:
            cmd = [PY, SCRIPTS / "video_dump.py", "--split", split, "--limit", lim,
                   "--audio-limit", audio, "--out", work, "--embedder", embedder]
            if audio == 0:
                cmd.append("--no-audio")
            run(cmd)
        if split == "train":
            # D-021 diet stage BETWEEN the train-dump and the val-dump (5b.6)
            stage_dump_cdf(work, smoke, embedder)
    if full_redump:
        # W4c §3: the fresh ffpp/cdf dumps are receipted as the pre-append
        # state, then the DFD diet appends exactly as in the bootstrap path
        # (same stage_dump_dfd, same receipts; §3 order preserved).
        _fresh_dump_receipt(work)
        stage_dump_dfd(work, smoke, embedder)


def _fresh_dump_receipt(work: Path) -> None:
    """W4c §3 full-redump pre-receipt: snapshot the freshly dumped ffpp/cdf
    CSVs into the bootstrap_receipt.json schema so stage_dump_dfd's append
    reconciliation has its pre-state. No byte-preservation happened — the
    mode field says so, and the MANIFEST section is labeled accordingly."""
    import json as _json
    import pandas as pd
    receipt = {"mode": "full-redump fresh-dump pre-receipt (W4c §3 — no "
               "bootstrap source; counts snapshot the fresh ffpp/cdf dumps "
               "before the DFD append)",
               "created": time.strftime("%Y-%m-%d %H:%M:%S"), "dumps": {}}
    lines = ["## Full re-dump pre-receipt (W4c §3 — fresh dumps, no bootstrap)", ""]
    for name in BOOTSTRAP_DUMPS:
        p = work / name / "scores.csv"
        if not p.exists():
            raise SystemExit(f"full-redump pre-receipt: missing {p}")
        df = pd.read_csv(p)
        receipt["dumps"][name] = {
            "rows": int(len(df)),
            "n_reals": int((df["label"] == 0).sum()),
            "n_fakes": int((df["label"] == 1).sum()),
            "per_method": {m: int((df["method"] == m).sum())
                           for m in sorted(df["method"].unique())}}
        lines.append(f"- {name}: {len(df)} rows (fresh dump, pre-DFD-append)")
        print(f"[full-redump] pre-receipt {name}: {len(df)} rows", flush=True)
    (work / "bootstrap_receipt.json").write_text(_json.dumps(receipt, indent=2))
    _manifest_append(work, lines)


def stage_dump_cdf(work, smoke, embedder):
    """D-021 domain diet (5b.5): Celeb-DF TRAIN videos (test list excluded by
    video_dump.enumerate_celebdf) so the stacker sees both verdict spaces on
    BOTH domains — an FF++-only diet would teach 'spec always wins' and risk
    gate C. SIZE=1000 (500/500) per the user's binding decision."""
    out = work / "dump_cdf"
    if (out / "scores.csv").exists():
        print("[dump] cdf: exists, skipped", flush=True)
        return
    lim = 40 if smoke else 1000
    run([PY, SCRIPTS / "video_dump.py", "--split", "train", "--corpus", "celebdf",
         "--limit", lim, "--no-audio", "--out", work, "--embedder", embedder])


def stage_train(work, embedder, av=False):
    out = work / "candidate"
    if (out / "stacker.joblib").exists():
        print("[train] candidate exists, skipped", flush=True)
        return
    # D-021: ALL three dumps are the diet — a missing dump_cdf would silently
    # train FF++-only (the exact stale-diet failure class 5b.5 exists to fix).
    csvs = []
    for name in ("dump_train", "dump_val", "dump_cdf"):
        p = work / name / "scores.csv"
        if not p.exists():
            raise SystemExit(f"missing {p} — run the dump stage first")
        csvs.append(str(p))
    run([PY, SCRIPTS / "train_video_stacker.py",
         "--csv", *csvs, "--out", out, "--embedder", embedder,
         "--dfd-weight-mult", str(DFD_WEIGHT_MULT_SEALED)])
    if av:
        # W-AV (D-076): the AV coverage calibration rides the bundle — the
        # serve path derives it from VIDEO_STACKER.parent (same law as the
        # drift calibration). Fail-closed: no av_calibration, no candidate.
        av_cal = work / "av_calibration.json"
        if not av_cal.exists():
            raise SystemExit(f"missing {av_cal} — the av dump stage fits it "
                             f"on train reals (D-035); run the dump stage first")
        shutil.copy(av_cal, out / "av_calibration.json")
        n_cal = json.loads(av_cal.read_text()).get("n_videos")
        print(f"[train] av_calibration.json -> candidate "
              f"(fit on {n_cal} train reals)", flush=True)
    meta = json.loads((out / "stacker_meta.json").read_text())
    if av:
        # D-046 tripwire (2026-11-05): the artifact must match the seal
        # BEFORE any gate burns compute — an off-recipe candidate (e.g.
        # dfd_weight_mult drifting from the sealed value) silently
        # invalidates every downstream gate. Catch it seconds after train,
        # not hours later at verdict read.
        if meta.get("dfd_weight_mult") != DFD_WEIGHT_MULT_SEALED:
            raise SystemExit(
                f"off-recipe candidate: stacker_meta dfd_weight_mult="
                f"{meta.get('dfd_weight_mult')} != sealed "
                f"{DFD_WEIGHT_MULT_SEALED} (D-076) — STOP before gates")
        import sys as _sys
        _sys.path.insert(0, str(WS / "deepfake" / "bench"))
        from video_features import AV_FEATURES
        names = meta.get("feature_names", [])
        if len(names) != 32 + len(AV_FEATURES) or \
                names[-len(AV_FEATURES):] != list(AV_FEATURES):
            raise SystemExit(
                f"off-recipe candidate: feature_names tail is not the "
                f"sealed 32+{len(AV_FEATURES)} AV append (D-076) — "
                f"STOP before gates")
    print(f"[train] candidate: val_auc={meta['val_auc']} "
          f"features={len(meta['feature_names'])} embedder={embedder} "
          f"dfd_weight_mult={meta.get('dfd_weight_mult')}", flush=True)


def _gate_exam(out_dir, smoke, build_cmd):
    """Run an exam once; reuse its on-disk metrics afterwards."""
    if not (out_dir / "metrics.json").exists():
        run(build_cmd(out_dir))
    return json.loads((out_dir / "metrics.json").read_text())


def _real_elevation(features_csv: Path) -> dict:
    """W4b lever B — D4 elevation guard: how far the stacker shifts REAL rows
    fake-ward relative to the naive channel, per gate. Computed over the
    label==0 rows of the gate's features.csv; recorded per gate in gates.json
    and enforced against GATES[<gate>]["max_real_elevation"] (smoke runs
    record but do not enforce)."""
    import pandas as pd
    df = pd.read_csv(features_csv)
    reals = df[df["label"] == 0]
    naive = float(reals["naive_combined"].mean())
    stacked = float(reals["stacked"].mean())
    return {"naive_real_mean": round(naive, 4),
            "stacked_real_mean": round(stacked, 4),
            "real_elevation": round(stacked - naive, 4),
            "n_reals": int(len(reals))}


def stage_gate(work, smoke, embedder, av=False):
    cand = work / "candidate"
    cal = cand / "drift_calibration.json"
    # full = the official FF++ test split: 140 reals + 4x140 fakes = 700.
    # (A half-sample gate (per-method 70) flipped the delta-vs-naive sign and
    # hid the Deepfakes bar crossing — n is part of the gate contract.)
    nr, pm, cdf_lim = ((8, 4, 30) if smoke else (140, 140, 0))
    gate_dir = work / "gate_ffpp"

    ffpp = _gate_exam(gate_dir, smoke, lambda o: [
        PY, SCRIPTS / "exam_video_ffpp.py", "--n-real", nr, "--per-method", pm,
        "--frames", 12, "--video-stacker", cand, "--drift-cal", cal,
        "--embedder", embedder, "--out", o])
    auc = ffpp.get("auc", {})
    stacked, naive = auc.get("stacked"), auc.get("naive_combined")
    bars = GATES["ffpp"]["min_per_method"]
    per_meth_ok = all(
        (ffpp.get("per_method", {}).get(m, {}).get("stacked") or 0)
        >= bars[m] for m in METHODS) if not smoke else True
    delta = None if (stacked is None or naive is None) else stacked - naive
    elev_ffpp = _real_elevation(gate_dir / "features.csv")
    elev_ok = (elev_ffpp["real_elevation"] <= GATES["ffpp"]["max_real_elevation"]
               if not smoke else True)
    nr1_base = nr1_min = None
    if av:
        # NR1 (D-076): stacked must ALSO clear the incumbent baseline - 0.005
        # (v4e receipt 0.9121 -> bar max(0.90, 0.9071)). Fail-closed without
        # the incumbent's gates receipt.
        nr1_path = NR1_BASELINE_WORK / "gates.json"
        if not nr1_path.exists():
            raise SystemExit(f"NR1 baseline missing: {nr1_path} — the --av "
                             f"run requires the incumbent's gates receipt "
                             f"(fail-closed)")
        nr1_base = float(json.loads(nr1_path.read_text())["ffpp"]["auc_stacked"])
        nr1_min = max(GATES["ffpp"]["min_auc"], nr1_base + NR1_DELTA)
    nr1_ok = (stacked is not None and nr1_min is not None
              and stacked >= nr1_min) if av else True
    ffpp_pass = bool(stacked is not None and naive is not None
                     and stacked >= GATES["ffpp"]["min_auc"]
                     and delta >= GATES["ffpp"]["vs_naive_tol"] and per_meth_ok
                     and elev_ok and nr1_ok)
    g_ffpp = {"auc_stacked": stacked, "auc_naive": naive,
              "auc_identity": auc.get("identity_video_score"),
              "delta_vs_naive": None if delta is None else round(delta, 4),
              "per_method": ffpp.get("per_method", {}),
              "min_per_method": bars,
              "naive_real_mean": elev_ffpp["naive_real_mean"],
              "stacked_real_mean": elev_ffpp["stacked_real_mean"],
              "real_elevation": elev_ffpp["real_elevation"],
              "n_reals": elev_ffpp["n_reals"],
              "max_real_elevation": GATES["ffpp"]["max_real_elevation"],
              "nr1_baseline": nr1_base, "nr1_min": nr1_min,
              "embedder": ffpp.get("embedder"),
              "n": ffpp.get("n_scored"),
              "pass": ffpp_pass}
    print(f"[gate] ffpp: stacked={stacked} naive={naive} delta={delta} "
          f"elevation={elev_ffpp['real_elevation']} "
          f"(max {GATES['ffpp']['max_real_elevation']})"
          + (f" nr1_min={nr1_min} (baseline {nr1_base})" if av else "")
          + f" -> {'PASS' if ffpp_pass else 'FAIL'}", flush=True)

    # --- Gate B: leave-one-method-out (pure feature-level) -----------------
    import pandas as pd
    from train_video_stacker import fit_stacker
    from video_features import FEATURE_NAMES_FULL, row_to_X
    # W4d: the LOMO retrains mirror the CANDIDATE's recipe — its recorded
    # dfd_weight_mult (lever A'), defaulting to the sealed 1.0 for pre-W4d
    # candidates. One recipe, one feature space, gates included.
    _wmult = float(json.loads(
        (cand / "stacker_meta.json").read_text()).get("dfd_weight_mult", 1.0))
    train_df = pd.read_csv(work / "dump_train" / "scores.csv")
    test_df = pd.read_csv(gate_dir / "features.csv")
    reals = test_df[test_df["label"] == 0]
    lomo, lomo_ok = {}, True
    for m in METHODS:
        sub = train_df[~((train_df["label"] == 1) & (train_df["method"] == m))]
        held = test_df[test_df["method"] == m]
        if smoke:
            held = held.head(8)
        eval_df = pd.concat([held, reals], ignore_index=True)
        clf, _va, _X, _y = fit_stacker(sub, weight_mult=_wmult)
        X = np.vstack([row_to_X(r, FEATURE_NAMES_FULL)
                       for r in eval_df.to_dict("records")])
        p = clf.predict_proba(X)[:, 1]
        y = eval_df["label"].astype(int).to_numpy()
        from sklearn.metrics import roc_auc_score
        a = float(roc_auc_score(y, p)) if len(set(y)) > 1 else None
        lomo[m] = {"auc": None if a is None else round(a, 4),
                   "n_fake": int((y == 1).sum()), "n_real": int((y == 0).sum())}
        ok = a is not None and a >= GATES["lomo"]["min_auc"]
        lomo_ok = lomo_ok and ok
        print(f"[gate] lomo {m}: auc={a} -> {'PASS' if ok else 'FAIL'}", flush=True)
    g_lomo = {"per_method_auc": lomo, "min_auc": GATES["lomo"]["min_auc"],
              "pass": bool(lomo_ok)}

    # --- Gate C: Celeb-DF cross-dataset ------------------------------------
    cdf = _gate_exam(work / "gate_celebdf", smoke, lambda o: [
        PY, SCRIPTS / "exam_video_celebdf.py", "--frames", 12,
        "--stacker", cand, "--drift-cal", cal, "--embedder", embedder, *(
            ["--limit", cdf_lim] if cdf_lim else []), "--out", o])
    c_auc = cdf.get("auc", {}).get("stacked")
    elev_cdf = _real_elevation(work / "gate_celebdf" / "features.csv")
    c_elev_ok = (elev_cdf["real_elevation"]
                 <= GATES["celebdf"]["max_real_elevation"] if not smoke else True)
    if c_auc is None or c_auc < GATES["celebdf"]["warn_auc"]:
        c_verdict = "FAIL"
    elif c_auc < GATES["celebdf"]["min_auc"]:
        c_verdict = "WARN_ESCALATE"
    elif c_elev_ok:
        c_verdict = "PASS"
    else:
        c_verdict = "FAIL"              # D4 elevation miss at passing AUC
    g_cdf = {"auc_stacked": c_auc, "auc_naive": cdf.get("auc", {}).get("naive"),
             "auc_identity": cdf.get("auc", {}).get("identity"), "n": cdf.get("n"),
             "naive_real_mean": elev_cdf["naive_real_mean"],
             "stacked_real_mean": elev_cdf["stacked_real_mean"],
             "real_elevation": elev_cdf["real_elevation"],
             "n_reals": elev_cdf["n_reals"],
             "max_real_elevation": GATES["celebdf"]["max_real_elevation"],
             "embedder": cdf.get("embedder"),
             "verdict": c_verdict, "pass": c_verdict == "PASS",
             "warn": c_verdict == "WARN_ESCALATE"}
    print(f"[gate] celebdf: stacked={c_auc} "
          f"elevation={elev_cdf['real_elevation']} "
          f"(max {GATES['celebdf']['max_real_elevation']}) -> {c_verdict}",
          flush=True)

    # --- Gate D: held-out DFD sources (W4, D-066) ---------------------------
    dfd = _gate_exam(work / "gate_dfd", smoke, lambda o: [
        PY, SCRIPTS / "exam_video_dfd.py", "--frames", 12,
        "--video-stacker", cand, "--drift-cal", cal, "--embedder", embedder,
        *(["--limit", "4"] if smoke else []), "--out", o])
    d_auc = (dfd.get("auc") or {}).get("stacked")
    d_real_mean = dfd.get("real_mean_pf")
    d_real_frac = dfd.get("real_frac_gt_half")
    bars_d = GATES["dfd"]
    elev_dfd = _real_elevation(work / "gate_dfd" / "features.csv")
    d_elev_ok = (elev_dfd["real_elevation"] <= bars_d["max_real_elevation"]
                 if not smoke else True)
    dfd_pass = bool(d_auc is not None and d_real_mean is not None
                    and d_real_frac is not None
                    and d_auc >= bars_d["min_auc"]
                    and d_real_mean <= bars_d["max_real_mean_pf"]
                    and d_real_frac <= bars_d["max_real_frac_gt_half"]
                    and d_elev_ok)
    g_dfd = {"auc_stacked": d_auc,
             "auc_naive": (dfd.get("auc") or {}).get("naive_combined"),
             "auc_identity": (dfd.get("auc") or {}).get("identity_video_score"),
             "real_mean_pf": d_real_mean,
             "real_frac_gt_half": d_real_frac,
             "naive_real_mean": elev_dfd["naive_real_mean"],
             "stacked_real_mean": elev_dfd["stacked_real_mean"],
             "real_elevation": elev_dfd["real_elevation"],
             "max_real_elevation": bars_d["max_real_elevation"],
             "n_fakes": dfd.get("n_fakes"), "n_reals": dfd.get("n_reals"),
             "n_sources": dfd.get("n_sources"),
             "split_manifest": DFD_SPLIT_MANIFEST,
             "min_auc": bars_d["min_auc"],
             "max_real_mean_pf": bars_d["max_real_mean_pf"],
             "max_real_frac_gt_half": bars_d["max_real_frac_gt_half"],
             "embedder": dfd.get("embedder"),
             "pass": dfd_pass}
    print(f"[gate] dfd: stacked={d_auc} real_mean_pf={d_real_mean} "
          f"real_frac_gt_half={d_real_frac} "
          f"elevation={elev_dfd['real_elevation']} "
          f"(max {bars_d['max_real_elevation']}) "
          f"(n_fakes={dfd.get('n_fakes')} n_reals={dfd.get('n_reals')} "
          f"n_sources={dfd.get('n_sources')}) -> "
          f"{'PASS' if dfd_pass else 'FAIL'}", flush=True)

    # --- Gate E: W-AV self-gen held-out + AVLips external (D-076 §6) --------
    g_av = g_avc = None
    if av:
        av_ex = _gate_exam(work / "gate_av", smoke, lambda o: [
            PY, SCRIPTS / "exam_video_av.py", "--corpus", "av", "--frames", 12,
            "--video-stacker", cand, "--drift-cal", cal, "--embedder", embedder,
            "--av-cache", str(work / "av_cache"),
            "--out", o])
        av_auc = (av_ex.get("auc") or {}).get("stacked")
        avlips_ex, avlips_auc = None, None
        avlips_dir = WS / "data" / "raw" / "avlips"
        if avlips_dir.exists() and any(avlips_dir.iterdir()):
            avlips_ex = _gate_exam(work / "gate_avlips", smoke, lambda o: [
                PY, SCRIPTS / "exam_video_av.py", "--corpus", "avlips",
                "--frames", 12, "--video-stacker", cand, "--drift-cal", cal,
                "--embedder", embedder,
                "--av-cache", str(work / "av_cache"),
                "--out", o])
            avlips_auc = (avlips_ex.get("auc") or {}).get("stacked")
        elif not smoke:
            raise SystemExit(f"AVLips val data missing at {avlips_dir} — D2 is "
                             f"a sealed bar (>= 0.70); download + receipt it "
                             f"first (fail-closed)")
        else:
            print("[gate] avlips: SKIPPED (smoke run, no AVLips val data)",
                  flush=True)
        integ = av_ex.get("integrity") or {}
        integ_ok = all([
            integ.get("straddle_violations") == 0,
            integ.get("ffprobe_parity_violations") == 0,
            integ.get("dedup_clean") is True,
            integ.get("license_complete") is True])
        bars_av = GATES["av"]
        avlips_ok = (avlips_auc is not None
                     and avlips_auc >= bars_av["avlips_min_auc"])
        if smoke and avlips_auc is None:
            avlips_ok = True       # smoke: D2 skipped mechanics-only; the full
                                   # run fail-closes above when data is missing
        av_pass = bool(av_auc is not None and av_auc >= bars_av["min_auc"]
                       and avlips_ok and integ_ok)
        g_av = {"auc_stacked": av_auc,
                "auc_naive": (av_ex.get("auc") or {}).get("naive_combined"),
                "auc_identity": (av_ex.get("auc") or {}).get("identity_video_score"),
                "n_fakes": av_ex.get("n_fakes"), "n_reals": av_ex.get("n_reals"),
                "n_sources": av_ex.get("n_sources"),
                "manifest": av_ex.get("manifest"),
                "integrity": integ,
                "in_coverage_rate": av_ex.get("in_coverage_rate"),
                "avlips_auc_stacked": avlips_auc,
                "avlips_n": (avlips_ex or {}).get("n_scored"),
                "min_auc": bars_av["min_auc"],
                "avlips_min_auc": bars_av["avlips_min_auc"],
                "embedder": av_ex.get("embedder"),
                "pass": av_pass}
        print(f"[gate] av: stacked={av_auc} (bar {bars_av['min_auc']}) "
              f"avlips={avlips_auc} (bar {bars_av['avlips_min_auc']}) "
              f"integrity_clean={integ_ok} -> {'PASS' if av_pass else 'FAIL'}",
              flush=True)

        # --- Gate F: narration/music confound (D4) --------------------------
        avc = _gate_exam(work / "gate_av_confound", smoke, lambda o: [
            PY, SCRIPTS / "exam_video_av_confound.py", "--frames", 12,
            "--video-stacker", cand, "--drift-cal", cal, "--embedder", embedder,
            "--av-cache", str(work / "av_cache"),
            "--out", o])
        fpr = avc.get("fpr_issued")
        bars_avc = GATES["av_confound"]
        avc_pass = bool(fpr is not None and fpr <= bars_avc["max_fpr_issued"])
        g_avc = {"fpr_issued": fpr,
                 "in_coverage_rate": avc.get("in_coverage_rate"),
                 "mean_pf": avc.get("mean_pf"), "n_rows": avc.get("n_rows"),
                 "max_fpr_issued": bars_avc["max_fpr_issued"],
                 "embedder": avc.get("embedder"), "pass": avc_pass}
        print(f"[gate] av_confound: fpr_issued={fpr} "
              f"(max {bars_avc['max_fpr_issued']}) -> "
              f"{'PASS' if avc_pass else 'FAIL'}", flush=True)

    gates = {"ffpp": g_ffpp, "lomo": g_lomo, "celebdf": g_cdf, "dfd": g_dfd,
             "av": g_av, "av_confound": g_avc,
             "gates_spec": GATES, "smoke": bool(smoke),
             "embedder": embedder,
             "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (work / "gates.json").write_text(json.dumps(gates, indent=2))
    return gates


def stage_verdict(work, promote, promote_to, av=False):
    gp = work / "gates.json"
    if not gp.exists():
        raise SystemExit("no gate results — run the gate stage first")
    gates = json.loads(gp.read_text())
    if gates.get("smoke"):
        print("[verdict] gates.json is from a --smoke run — verdict skipped "
              "(run the full gate stage before any verdict)", flush=True)
        return 0
    a, b, c = (gates["ffpp"]["pass"], gates["lomo"]["pass"],
               gates["celebdf"]["pass"])
    d = gates.get("dfd", {}).get("pass")
    e = gates.get("av", {}).get("pass") if av else None
    f = gates.get("av_confound", {}).get("pass") if av else None
    if av and (e is None or f is None):
        raise SystemExit("gates.json lacks av/av_confound results — run the "
                         "gate stage of this --av run first (fail-closed)")
    stacked = gates["ffpp"]["auc_stacked"]
    all_pass = bool(a and b and c and d and (not av or (e and f)))
    missed = []
    if not a:
        f = gates["ffpp"]
        missed.append({"gate": "ffpp", "stacked": f.get("auc_stacked"),
                       "bar": 0.90, "delta_vs_naive": f.get("delta_vs_naive"),
                       "delta_bar": -0.005, "per_method": f.get("per_method"),
                       "naive_real_mean": f.get("naive_real_mean"),
                       "stacked_real_mean": f.get("stacked_real_mean"),
                       "real_elevation": f.get("real_elevation"),
                       "max_real_elevation": f.get("max_real_elevation")})
    if not b:
        missed.append({"gate": "lomo", "per_method_auc":
                       gates["lomo"].get("per_method_auc"), "bar": 0.70})
    if not c:
        g = gates["celebdf"]
        missed.append({"gate": "celebdf", "stacked": g.get("auc_stacked"),
                       "bar": 0.75,
                       "naive_real_mean": g.get("naive_real_mean"),
                       "stacked_real_mean": g.get("stacked_real_mean"),
                       "real_elevation": g.get("real_elevation"),
                       "max_real_elevation": g.get("max_real_elevation")})
    if not d:
        g = gates.get("dfd", {})
        missed.append({"gate": "dfd", "auc_stacked": g.get("auc_stacked"),
                       "bar": 0.85, "real_mean_pf": g.get("real_mean_pf"),
                       "real_mean_bar": 0.35,
                       "real_frac_gt_half": g.get("real_frac_gt_half"),
                       "real_frac_bar": 0.25,
                       "naive_real_mean": g.get("naive_real_mean"),
                       "stacked_real_mean": g.get("stacked_real_mean"),
                       "real_elevation": g.get("real_elevation"),
                       "max_real_elevation": g.get("max_real_elevation")})
    if av and not e:
        g = gates.get("av", {})
        missed.append({"gate": "av", "auc_stacked": g.get("auc_stacked"),
                       "bar": GATES["av"]["min_auc"],
                       "avlips_auc_stacked": g.get("avlips_auc_stacked"),
                       "avlips_bar": GATES["av"]["avlips_min_auc"],
                       "integrity": g.get("integrity"),
                       "in_coverage_rate": g.get("in_coverage_rate")})
    if av and not f:
        g = gates.get("av_confound", {})
        missed.append({"gate": "av_confound", "fpr_issued": g.get("fpr_issued"),
                       "max_fpr_issued": GATES["av_confound"]["max_fpr_issued"],
                       "mean_pf": g.get("mean_pf"), "n_rows": g.get("n_rows")})
    print(f"\nVERDICT: {'PROMOTE' if all_pass else 'HOLD'}  "
          f"(ffpp={a} [stacked={stacked}, naive={gates['ffpp']['auc_naive']}], "
          f"lomo={b}, celebdf={c} [{gates['celebdf']['verdict']}], "
          f"dfd={d} [stacked={gates.get('dfd', {}).get('auc_stacked')}, "
          f"real_mean_pf={gates.get('dfd', {}).get('real_mean_pf')}]"
          + (f", av={e} [stacked={gates.get('av', {}).get('auc_stacked')}, "
             f"avlips={gates.get('av', {}).get('avlips_auc_stacked')}, "
             f"av_confound={f} "
             f"[fpr_issued={gates.get('av_confound', {}).get('fpr_issued')}]]"
             if av else "") + ")", flush=True)
    if stacked is not None and ESCALATE_BAND[0] <= stacked < ESCALATE_BAND[1]:
        print(f"  ESCALATE: FF++ stacked AUC {stacked} is in the "
              f"{ESCALATE_BAND[0]}-{ESCALATE_BAND[1]} waiver band — per user "
              f"decision #2 this is NOT waived autonomously; user decides.",
              flush=True)
    if gates["celebdf"]["warn"]:
        print(f"  ESCALATE: Celeb-DF stacked AUC {gates['celebdf']['auc_stacked']} "
              f"is in the 0.70-0.75 warn band — user decides.", flush=True)
    if not all_pass:
        print("  HOLD: gates incomplete/failed — no promotion, no waiver "
              "written. Missed bar(s) per D-046:", flush=True)
        for m in missed:
            print(f"    - {json.dumps(m)}", flush=True)
    receipt = {"decision": "PROMOTE" if all_pass else "HOLD",
               "exit_code": 0 if all_pass else 2,
               "gates": {k: gates[k].get("pass") for k in
                         ("ffpp", "lomo", "celebdf", "dfd")
                         + (("av", "av_confound") if av else ())},
               "missed": missed,
               "smoke": bool(gates.get("smoke")),
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    (work / "verdict.json").write_text(json.dumps(receipt, indent=2))
    print(f"verdict receipt: {work / 'verdict.json'}", flush=True)
    if not all_pass:
        return 2
    if promote:
        if not promote_to:
            raise SystemExit("--promote requires --promote-to <tag>")
        dst = WS / "artifacts" / promote_to
        dst.mkdir(parents=True, exist_ok=True)
        names = ["stacker.joblib", "stacker_meta.json",
                 "drift_calibration.json"]
        if av:
            names.append("av_calibration.json")   # rides VIDEO_STACKER.parent
        for name in names:
            shutil.copy(work / "candidate" / name, dst / name)
        print(f"promoted -> {dst} (wire argus_detect.VIDEO_STACKER = "
              f"artifacts/{promote_to}/stacker.joblib to activate)", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "dump", "train", "gate", "verdict"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--work", default=None,
                    help="work dir (default artifacts/restack_video_v2[_smoke]; "
                         "MUST be fresh — _gate_exam reuses existing metrics.json "
                         "and stage_dump skips existing scores)")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--promote-to", default=None)
    ap.add_argument("--bootstrap-from", default=str(BOOTSTRAP_DEFAULT),
                    help="W4: byte-preserve this run's dump_train/dump_val/"
                         "dump_cdf instead of re-dumping (law §4.4)")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="disable the byte-preserve bootstrap (legacy "
                         "from-scratch dump path)")
    ap.add_argument("--full-redump", action="store_true",
                    help="W4c §3: ignore --bootstrap-from and re-dump ALL "
                         "corpora from scratch (ffpp train + cdf + ffpp val, "
                         "then the DFD appends via stage_dump_dfd — the exact "
                         "v3b provenance sequence, extended featurizer)")
    ap.add_argument("--embedder", choices=["sface", "arcface"], default="arcface",
                    help="identity embedder for ALL stages — dump/train/gate "
                         "must match, and the serve path reads it from "
                         "stacker_meta.embedder (phase default: arcface, D-042)")
    ap.add_argument("--av", action="store_true",
                    help="W-AV (D-076): bootstrap restack_video_v4 byte-verbatim, "
                         "splice the 6 AV columns (zeros), append the av diet, "
                         "and run the 6-gate verdict (adds av + av_confound; "
                         "NR1 incumbent baseline enforced). Default work "
                         "artifacts/restack_video_v5[_smoke]")
    args = ap.parse_args()
    if args.promote and not args.promote_to:
        raise SystemExit("--promote requires --promote-to <tag>")

    if args.work:
        work = Path(args.work)
    else:
        base = "restack_video_v5" if args.av else "restack_video_v2"
        work = WS / "artifacts" / (base + "_smoke" if args.smoke else base)
    work.mkdir(parents=True, exist_ok=True)
    bootstrap_from = None if (args.no_bootstrap or args.full_redump) \
        else args.bootstrap_from
    stages = (["dump", "train", "gate", "verdict"] if args.stage == "all"
              else [args.stage])
    for st in stages:
        if st == "dump":
            if args.av:
                stage_dump_av(work, args.smoke, args.embedder)
            else:
                stage_dump(work, args.smoke, args.embedder, bootstrap_from,
                           full_redump=args.full_redump)
        elif st == "train":
            stage_train(work, args.embedder, av=args.av)
        elif st == "gate":
            stage_gate(work, args.smoke, args.embedder, av=args.av)
        elif st == "verdict":
            return stage_verdict(work, args.promote, args.promote_to, av=args.av)
    return 0


if __name__ == "__main__":
    sys.exit(main())
