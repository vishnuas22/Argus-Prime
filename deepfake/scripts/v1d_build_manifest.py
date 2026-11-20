"""V1d stage 1 — build the sample manifest (decode-filtered, resumable).

Sealed spec: memory/spec_v1d_feasibility_v1.md
Determinism: random.Random(20261120), per-domain shuffle, then keep the first
200 clips per class that DECODE. Every decode failure is recorded with a reason.

Writes artifacts/v1d_restrav/manifest.json. Re-running is idempotent: if the
manifest exists it is reused unless --force is given.

Usage:
    .venv/bin/python deepfake/scripts/v1d_build_manifest.py [--force] [--n 300] [--keep 200]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "deepfake"))
sys.path.insert(0, str(WS / "deepfake" / "bench"))

SEED = 20261120
OUT = WS / "artifacts" / "v1d_restrav"

FFPP = WS / "data/raw/ffpp_mirror/FaceForensics++_C23"
CDF = WS / "data/raw/celebdf/Celeb-DF-v2"
GB = WS / "data/raw/genbuster_plusplus/extracted/video"
GV = WS / "data/raw/genvidbench"

# domain -> (class -> directory or glob, kind)
DOMAINS = {
    "genbuster_t2v": {
        "real": sorted((GB / "real").glob("*.mp4")),
        "fake": sorted((GB / "fake").glob("*.mp4")),
        "kind": "t2v_stock_vs_generated",
    },
    "genvid_opentora": {
        "real": None,                      # reals come from the shared GenBuster real pool
        "fake": sorted((GV / "opentora").glob("*.mp4")),
        "kind": "t2v_opentora",
    },
    "genvid_kling": {
        "real": None,
        "fake": sorted((GV / "kling").glob("*.mp4")),
        "kind": "t2v_kling",
    },
    "genvid_sora": {
        "real": None,
        "fake": sorted((GV / "sora").glob("*.mp4")),
        "kind": "t2v_sora",
    },
    "ffpp_curated": {
        "real": sorted((FFPP / "real").glob("*.mp4")),
        "fake": sorted((FFPP / "fake" / "Deepfakes").glob("*.mp4")),
        "kind": "curated_face_swap",
    },
    "cdf_curated": {
        "real": sorted((CDF / "YouTube-real").glob("*.mp4")),
        "fake": sorted((CDF / "Celeb-synthesis").glob("*.mp4")),
        "kind": "curated_face_swap",
    },
}

# Domains that share the GenBuster real pool must use the SAME real clips so the
# comparison is not confounded by different real sets.
SHARED_REAL_DOMAIN = "genbuster_t2v"


def build(n=300, keep=200, force=False, device="mps"):
    OUT.mkdir(parents=True, exist_ok=True)
    mpath = OUT / "manifest.json"
    if mpath.exists() and not force:
        print(f"manifest exists, reusing: {mpath}")
        return json.load(open(mpath))

    from restrav_feats import decode_clip  # local sealed extractor

    rng = random.Random(SEED)
    manifest = {"seed": SEED, "n_sampled": n, "n_keep": keep, "domains": {}}

    # 1. real pool: shuffle once, decode-filter, keep `keep`
    real_files = DOMAINS[SHARED_REAL_DOMAIN]["real"]
    rng.shuffle(real_files)
    reals, real_fail = [], []
    for f in real_files:
        if len(reals) >= keep:
            break
        try:
            decode_clip(str(f), T=24, window_sec=2.0)
            reals.append(str(f))
        except Exception as e:
            real_fail.append({"file": str(f), "error": f"{type(e).__name__}: {e}"[:160]})
    print(f"real pool: kept {len(reals)}/{keep}  failed {len(real_fail)}")
    manifest["real_pool"] = reals
    manifest["real_pool_decode_failures"] = real_fail

    # 2. per-domain fakes (and curated reals)
    for name, spec in DOMAINS.items():
        if name == SHARED_REAL_DOMAIN:
            fakes, fake_fail = [], []
            for f in spec["fake"]:
                if len(fakes) >= keep:
                    break
                try:
                    decode_clip(str(f), T=24, window_sec=2.0)
                    fakes.append(str(f))
                except Exception as e:
                    fake_fail.append({"file": str(f),
                                      "error": f"{type(e).__name__}: {e}"[:160]})
            manifest["domains"][name] = {
                "kind": spec["kind"],
                "real": reals,
                "fake": fakes,
                "decode_failures": {"real": real_fail, "fake": fake_fail},
                "n_real": len(reals), "n_fake": len(fakes),
            }
            print(f"{name}: real={len(reals)} fake={len(fakes)} "
                  f"(fail real={len(real_fail)} fake={len(fake_fail)})")
            continue

        fakes, fake_fail = [], []
        for f in spec["fake"]:
            if len(fakes) >= keep:
                break
            try:
                decode_clip(str(f), T=24, window_sec=2.0)
                fakes.append(str(f))
            except Exception as e:
                fake_fail.append({"file": str(f),
                                  "error": f"{type(e).__name__}: {e}"[:160]})
        dom_reals, dom_real_fail = reals, real_fail
        if spec["real"] is not None:
            dom_reals, dom_real_fail = [], []
            for f in spec["real"]:
                if len(dom_reals) >= keep:
                    break
                try:
                    decode_clip(str(f), T=24, window_sec=2.0)
                    dom_reals.append(str(f))
                except Exception as e:
                    dom_real_fail.append({"file": str(f),
                                          "error": f"{type(e).__name__}: {e}"[:160]})
        manifest["domains"][name] = {
            "kind": spec["kind"],
            "real": dom_reals,
            "fake": fakes,
            "decode_failures": {"real": dom_real_fail, "fake": fake_fail},
            "n_real": len(dom_reals), "n_fake": len(fakes),
        }
        print(f"{name}: real={len(dom_reals)} fake={len(fakes)} "
              f"(fail real={len(dom_real_fail)} fake={len(fake_fail)})")

    json.dump(manifest, open(mpath, "w"), indent=2)
    print(f"\nwrote {mpath}")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--keep", type=int, default=200)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    build(n=args.n, keep=args.keep, force=args.force)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()