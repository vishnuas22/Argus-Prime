"""Fast tier: the P0b config seam (no models, no corpora, no artifacts).

These tests run on a fresh clone with an empty artifacts/ dir. They lock the
property that every artifact pointer resolves through ARGUS_ARTIFACTS_DIR and
that the default is repo-relative, so the serve path does not depend on any one
machine's absolute paths.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[1]

_PROBE = (
    "import sys; sys.path.insert(0, 'deepfake');"
    "import argus_detect as a;"
    "print(a.STACKER); print(a.VIDEO_STACKER); print(a.AUDIO_PROBE)"
)


def _run(env_overrides=None):
    env = dict(os.environ)
    env.pop("ARGUS_ROOT", None)
    env.pop("ARGUS_ARTIFACTS_DIR", None)
    if env_overrides:
        env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env, cwd=str(REPO),
    )


def test_default_paths_are_repo_relative():
    r = _run()
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) == 3, lines
    stacker, video_stacker, audio_probe = lines
    root = str(REPO)
    for p in (stacker, video_stacker, audio_probe):
        assert p.startswith(root), f"{p} is not under {root}"
    assert stacker.endswith("artifacts/p6v11/stacker.joblib"), stacker
    assert video_stacker.endswith("artifacts/p6vid_v4/stacker.joblib"), video_stacker
    assert audio_probe.endswith("artifacts/p4/audio_probe.joblib"), audio_probe


def test_artifacts_dir_override():
    r = _run({"ARGUS_ARTIFACTS_DIR": "/tmp/argus-bundle"})
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    assert lines[0] == "/tmp/argus-bundle/p6v11/stacker.joblib", lines[0]
    assert lines[1] == "/tmp/argus-bundle/p6vid_v4/stacker.joblib", lines[1]
    assert lines[2] == "/tmp/argus-bundle/p4/audio_probe.joblib", lines[2]


def test_root_override_relocates_repo_paths():
    r = _run({"ARGUS_ROOT": "/tmp/argus-root"})
    assert r.returncode == 0, r.stderr
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    # ARGUS_ROOT moves the repo-relative anchors; artifacts default under it.
    assert lines[0] == "/tmp/argus-root/artifacts/p6v11/stacker.joblib", lines[0]


def test_no_hardcoded_machine_paths_in_serve_path():
    """Regression guard for the 21 de-hardcoded files (P0b).

    Needles are assembled from fragments so this file does not itself contain
    the literals (the CI `git grep` guard would otherwise match this test).
    """
    needles = ("/Users/" + "Dataghost", "/private/var/" + "folders")
    offenders = []
    for sub in ("deepfake", "tools", "tests"):
        for p in (REPO / sub).rglob("*"):
            if not p.is_file() or p.suffix not in (".py", ".sh", ""):
                continue
            if p.suffix == "" and p.parent.name != "bin":
                continue
            try:
                text = p.read_text(errors="ignore")
            except OSError:
                continue
            if any(n in text for n in needles):
                offenders.append(str(p.relative_to(REPO)))
    assert not offenders, f"hardcoded machine-local paths: {offenders}"