"""W4 DFD diet prep (D-066, spec artifacts/generator_intake/w4_video_dfd_spec.md).

Parses the mirror's 1000 DeepFakeDetection fake names into (target, sequence)
SOURCE keys, validates every name against the official FaceForensics filename
list, splits the SOURCES 60/15/25 (seed 11), and writes the split manifest
ONCE into data/raw/ffpp_dfd/ — dump (video_dump --corpus dfd) and the exam
(exam_video_dfd) both READ it; it is never regenerated after training starts.

Source-key law (spec §2): a DFD fake is named
    <target>_<source>__<sequence>__<experiment-id>.mp4
and its source original (an ACTOR-dataset video) is <target>__<sequence>.mp4.
The trailing experiment-id token is NOT the source video: official counts are
3068 fakes -> 358 unique (target,sequence) sources but 530 unique experiment
ids. Splitting by experiment id would place fakes of the SAME source video in
different splits — a leak. The split groups by (target, sequence) source key.

Usage: .venv/bin/python deepfake/scripts/ffpp/dfd_prepare.py
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

WS = Path(__file__).resolve().parents[3]
MIRROR_DFD = WS / "data" / "raw" / "ffpp_mirror" / "FaceForensics++_C23" / \
    "fake" / "DeepFakeDetection"
OUT = WS / "data" / "raw" / "ffpp_dfd"
OFFICIAL_URL = ("https://kaldir.vc.in.tum.de/faceforensics/v3/misc/"
                "deepfake_detection_filenames.json")
SEED = 11
PROPORTIONS = (0.60, 0.15, 0.25)          # train / val / exam of SOURCES
SPLITS = ("train", "val", "exam")
NAME_RE = re.compile(r"^\d{2}$")   # DFD actor ids are 2-digit (01..28)


def parse_fake_name(stem: str):
    """'<t>_<s>__<sequence>__<expid>' -> (target, source, sequence, expid).

    Returns None when the name does not match the official DFD shape (the
    caller asserts zero unparseable names)."""
    parts = stem.split("__")
    if len(parts) != 3:
        return None
    ids = parts[0].split("_")
    if len(ids) != 2 or not all(NAME_RE.match(i) for i in ids):
        return None
    target, source, sequence, expid = ids[0], ids[1], parts[1], parts[2]
    if not sequence or not expid:
        return None
    return target, source, sequence, expid


def fetch_official(path: Path) -> dict:
    """Fetch the official filename list once and save it as the receipt."""
    if path.exists():
        return json.loads(path.read_text())
    req = urllib.request.Request(OFFICIAL_URL, headers={"User-Agent": "Mozilla/5.0"})
    data = json.loads(urllib.request.urlopen(req, timeout=120).read())
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    return data


def split_sources(sources: list, seed: int = SEED) -> dict:
    """The pre-registered split rule (spec §2): sort source keys
    alphabetically, rng.permutation(N, seed), assign 60/15/25 in permuted
    order (round() for the first two cuts, exam takes the remainder).
    Pure + deterministic — replayed twice in main() as the receipt."""
    keys = sorted(sources)
    n = len(keys)
    rng = np.random.default_rng(seed)
    order = [keys[i] for i in rng.permutation(n)]
    n_train = round(PROPORTIONS[0] * n)
    n_val = round(PROPORTIONS[1] * n)
    out = {}
    for i, k in enumerate(order):
        out[k] = "train" if i < n_train else (
            "val" if i < n_train + n_val else "exam")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mirror_names = sorted(v.stem for v in MIRROR_DFD.glob("*.mp4"))
    print(f"mirror DeepFakeDetection fakes: {len(mirror_names)}", flush=True)

    official = fetch_official(OUT / "official_filenames.json")
    official_fakes = set(official["DeepFakesDetection"])
    official_actors = set(official["actors"])
    print(f"official list: {len(official_fakes)} DFD fakes / "
          f"{len(official_actors)} actor originals", flush=True)

    parsed, unparseable = {}, []
    for stem in mirror_names:
        p = parse_fake_name(stem)
        if p is None:
            unparseable.append(stem)
            continue
        parsed[stem] = p
    assert not unparseable, f"unparseable mirror names: {unparseable}"
    assert len(parsed) == len(mirror_names) == 1000, \
        f"expected exactly 1000 mirror DFD fakes, got {len(parsed)}"

    missing_official = [s for s in parsed if s not in official_fakes]
    assert not missing_official, \
        f"mirror names absent from the official list ({len(missing_official)}): " \
        f"{missing_official[:5]}"
    print("official-name membership: 1000/1000 mirror DFD fakes present in "
          "the official DeepFakesDetection list", flush=True)

    sources = {}
    for stem, (target, _src, sequence, expid) in parsed.items():
        key = f"{target}__{sequence}"
        rec = sources.setdefault(key, {"original": f"{key}.mp4",
                                       "fakes": [], "expids": []})
        rec["fakes"].append(stem)
        rec["expids"].append(expid)
    missing_actors = [k for k, r in sources.items()
                      if r["original"][:-len(".mp4")] not in official_actors]
    assert not missing_actors, \
        f"source originals absent from the official actors list " \
        f"({len(missing_actors)}): {missing_actors[:5]}"
    n_sources = len(sources)
    n_expids = len({e for r in sources.values() for e in r["expids"]})
    multi = sum(1 for r in sources.values() if len(r["expids"]) > 1)
    print(f"RECONCILIATION: {len(mirror_names)} mirror fakes -> "
          f"{n_expids} unique experiment ids -> {n_sources} unique "
          f"(target,sequence) sources ({multi} sources carry >1 expid, "
          f"max {max(len(r['expids']) for r in sources.values())})", flush=True)

    assignment = split_sources(list(sources))
    replay = split_sources(list(sources))
    assert assignment == replay, "split replay diverged — not deterministic"
    for k, v in assignment.items():
        sources[k]["split"] = v
    straddle = {}
    for k, r in sources.items():
        straddle.setdefault(r["split"], []).append(k)
    assert sum(len(v) for v in straddle.values()) == n_sources, \
        "source partition does not cover every source exactly once"

    counts = {s: {"n_sources": 0, "n_fakes": 0, "n_originals": 0}
              for s in SPLITS}
    fps = {s: [] for s in SPLITS}
    for k, r in sources.items():
        c = counts[r["split"]]
        c["n_sources"] += 1
        c["n_fakes"] += len(r["fakes"])
        c["n_originals"] += 1
        fps[r["split"]].append(len(r["fakes"]))
    for s in SPLITS:
        dist = fps[s]
        counts[s]["fakes_per_source"] = {
            "min": min(dist), "max": max(dist),
            "mean": round(sum(dist) / len(dist), 2)} if dist else None

    (OUT / "split.json").write_text(json.dumps(assignment, indent=2, sort_keys=True))
    (OUT / "sources.json").write_text(json.dumps(sources, indent=2, sort_keys=True))

    lines = [
        "# DFD diet split manifest (W4, D-066) — PRODUCED ONCE, never regenerated",
        f"",
        f"- generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- rule: sort (target,sequence) source keys alphabetically; "
        f"numpy default_rng permutation(seed={SEED}); first "
        f"round(0.60*N) -> train, next round(0.15*N) -> val, remainder -> exam",
        f"- proportions: {PROPORTIONS} of SOURCES (fakes never straddle; "
        f"an original always lands in its fakes' split)",
        f"- input: {len(mirror_names)} mirror fakes "
        f"(data/raw/ffpp_mirror/FaceForensics++_C23/fake/DeepFakeDetection)",
        f"- official receipt: official_filenames.json "
        f"({len(official_fakes)} fakes / {len(official_actors)} actors, "
        f"fetched {OFFICIAL_URL})",
        f"- integrity: 1000/1000 mirror names in the official list; "
        f"{n_sources}/{n_sources} source originals in the official actors "
        f"list; zero unparseable names; zero source straddle (by "
        f"construction, asserted)",
        f"- RECONCILIATION: unique experiment ids = {n_expids} (the peak "
        f"plan's '418' class of count), unique (target,sequence) sources = "
        f"{n_sources} — the split law uses SOURCES; {multi} sources carry "
        f">1 experiment id (max {max(len(r['expids']) for r in sources.values())})",
        "",
        "| split | sources | fakes | originals | fakes/source min/mean/max |",
        "|---|---|---|---|---|",
    ]
    for s in SPLITS:
        c, d = counts[s], counts[s]["fakes_per_source"]
        lines.append(f"| {s} | {c['n_sources']} | {c['n_fakes']} | "
                     f"{c['n_originals']} | "
                     f"{d['min'] if d else '-'}/{d['mean'] if d else '-'}/"
                     f"{d['max'] if d else '-'} |")
    lines += [
        "",
        "Exam sources are HELD OUT ENTIRELY: their fakes and originals are "
        "never dumped; the exam (exam_video_dfd) scores fakes + originals of "
        "the exam sources only.",
        "Deterministic replay: split_sources run twice in-process — "
        "IDENTICAL (asserted).",
        "Files: split.json (source->split), sources.json (per-source fakes/"
        "expids/original), official_filenames.json (server receipt).",
    ]
    (OUT / "MANIFEST.md").write_text("\n".join(lines) + "\n")

    print(f"split: " + ", ".join(
        f"{s}={counts[s]['n_sources']}src/{counts[s]['n_fakes']}fakes/"
        f"{counts[s]['n_originals']}originals" for s in SPLITS), flush=True)
    print(f"wrote split.json / sources.json / MANIFEST.md / "
          f"official_filenames.json under {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
