"""Deterministic P0 corpus preparation.

Builds from /tmp/val2017 (5000 imgs):
  /tmp/argus_corpus/reference   2500   (consumed by Argus_Master build_reference.py)
  /tmp/argus_corpus/calib        400   (consumed by build_envelope.py + calibration parts)
  <workspace>/data/corpora/eval_real  500 (disjoint scoring slice)

Slices are sorted-filename disjoint, mirroring Argus_Master's own protocol.
"""
import shutil
import sys
from pathlib import Path

SRC = Path("/tmp/val2017")
CORPUS = Path("/tmp/argus_corpus")
EVAL_REAL = Path(__file__).resolve().parents[2] / "data" / "corpora" / "eval_real"

N_REF, N_CAL, N_EVAL = 2500, 400, 500


def main() -> int:
    files = sorted(SRC.glob("*.jpg"))
    assert len(files) >= N_REF + N_CAL + N_EVAL, f"need >= {N_REF+N_CAL+N_EVAL} images, got {len(files)}"
    splits = {
        CORPUS / "reference": files[0:N_REF],
        CORPUS / "calib": files[N_REF:N_REF + N_CAL],
        EVAL_REAL: files[N_REF + N_CAL:N_REF + N_CAL + N_EVAL],
    }
    for dest, chunk in splits.items():
        dest.mkdir(parents=True, exist_ok=True)
        for f in chunk:
            target = dest / f.name
            if not target.exists():
                shutil.copy(f, target)
        print(f"{dest}: {len(list(dest.glob('*.jpg')))} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
