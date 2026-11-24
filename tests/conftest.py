"""Pytest path bootstrap.

Mirrors the runtime ``sys.path`` layout the CLI and API set up, so fast-tier
tests can import harness modules (``calibration``, ``channels``, ...) exactly as
production code does.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

for rel in (
    "deepfake",
    "deepfake/bench",
    "deepfake/scripts",
    "deepfake/api",
    "integrations/Argus_Master/backend",
):
    p = str(REPO / rel)
    if p not in sys.path:
        sys.path.insert(0, p)