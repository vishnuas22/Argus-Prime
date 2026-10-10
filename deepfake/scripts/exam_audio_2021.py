"""Modern voice-clone exam: ASVspoof 2021 LA eval vs the 2019-trained XLS-R probe.

The wired audio probe (argus_detect.AUDIO_PROBE, D-019 lineage) was trained on
ASVspoof 2019 LA; the 2021 eval uses different/reworked attack algorithms +
channel compression — a genuine held-out exam. Samples a balanced subset from
one parquet shard (labels: 0=fake, 1=real) and scores every utterance through
argus_detect.detect_audio.

Usage: .venv/bin/python deepfake/scripts/exam_audio_2021.py [--n-per-class 200]
"""
import argparse
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
# 2021 LA eval shards are label-sorted: shard 0 = fakes, shard 13 = reals
# (verified via datasets-server /rows probes 2026-09-03)
SHARDS = {
    0: ("https://huggingface.co/datasets/MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized/"
        "resolve/main/data/test-00000-of-00014.parquet", 0),
    13: ("https://huggingface.co/datasets/MoaazTalab/ASVspoof_2021_LA_Balanced_Normalized/"
         "resolve/main/data/test-00013-of-00014.parquet", 1),
}
SHARD_DIR = WS / "data" / "raw"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--out", default=None,
                    help="output dir (default artifacts/exam_audio_2021_<stacker>_<YYYYMMDD>)")
    ap.add_argument("--force", action="store_true",
                    help="write into an output dir that already contains results")
    args = ap.parse_args()

    import numpy as np
    import pyarrow.parquet as pq
    import requests
    from sklearn.metrics import roc_auc_score

    sys.path.insert(0, str(WS / "deepfake"))
    from train_face_probe import eer_from
    from argus_detect import AUDIO_PROBE, STACKER

    out = Path(args.out) if args.out else WS / "artifacts" / (
        f"exam_audio_2021_{STACKER.parent.name}_{time.strftime('%Y%m%d')}")
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"ERROR: output dir {out} already contains results — pass --force to overwrite",
              file=sys.stderr)
        return 1
    out.mkdir(parents=True, exist_ok=True)

    # gather balanced utterances from the label-sorted shards
    picks = {0: [], 1: []}  # dataset label -> list[(shard_idx, row_bytes)]
    for shard_no, (url, want_label) in SHARDS.items():
        shard = SHARD_DIR / f"asvspoof21_la_test-{shard_no:05d}.parquet"
        if not shard.exists() or shard.stat().st_size < 10_000_000:
            shard.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading shard {shard_no} (~445MB)...", flush=True)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(shard.with_suffix(".part"), "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            shutil.move(shard.with_suffix(".part"), shard)
        tab = pq.read_table(shard)
        acol = tab["audio"]
        lcol = tab["label"]
        got = 0
        for i in range(len(tab)):
            if got >= args.n_per_class:
                break
            if lcol[i].as_py() != want_label:
                continue
            rec = acol[i].as_py()
            data = rec["bytes"] if isinstance(rec, dict) else rec
            if data:
                picks[want_label].append(data)
                got += 1
        print(f"shard {shard_no}: {got} x label={want_label}", flush=True)

    from argus_detect import detect_audio
    rows = []
    y, p = [], []
    tmp = Path(tempfile.mkdtemp(prefix="audio_exam_")) / "u.wav"
    from PIL import Image  # noqa: F401  (parquet decode parity)
    for lab in (0, 1):
        for data in picks[lab]:
            tmp.write_bytes(data)
            try:
                out = detect_audio(tmp)
                score = out.get("p_fake_audio_mean")
            except Exception:
                score = None
            if score is None:
                continue
            # dataset label: 0=fake, 1=real -> detector label 1=fake
            y.append(1 - lab)
            p.append(score)
            rows.append({"dataset_label": lab, "p_fake": score})
            if len(rows) % 50 == 0:
                print(f"[{len(rows)}] scored", flush=True)
    shutil.rmtree(tmp.parent, ignore_errors=True)

    (out / "scores.json").write_text(json.dumps(rows))
    auc = float(roc_auc_score(y, p))
    eer = eer_from(np.array(y), np.array(p))
    res = {"auc": round(auc, 4), "eer": round(eer, 4), "n": len(y),
           "n_real": int(sum(1 for v in y if v == 0)),
           "n_fake": int(sum(1 for v in y if v == 1)),
           "source": "ASVspoof 2021 LA eval (MoaazTalab balanced shard 0)",
           "probe": f"{AUDIO_PROBE.relative_to(WS)} (trained ASVspoof 2019 LA)"}
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    print("AUDIO EXAM:", json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
