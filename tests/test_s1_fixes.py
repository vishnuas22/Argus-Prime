"""Fast tier: S1 correctness fixes (D-115).

Locks three fixes:
  1. R2 recovery uses the BOUND alpha (operator form alpha, else stacker meta),
     not the vendored default 0.10.
  2. The video stacker lazy load is fail-closed: a failed load never leaves
     loaded=True with clf=None (no silent None on later calls).
  3. The audio model lazy load is lock-guarded and publishes only on success.
"""
import sys
from pathlib import Path

import pytest

WS = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(WS / "deepfake"), str(WS / "deepfake" / "bench"),
                str(WS / "integrations" / "Argus_Master" / "backend")]

pytestmark = pytest.mark.fast


# --- fix 1: R2 uses the bound alpha -----------------------------------------

def test_r2_reads_alpha_from_extras():
    """R2 must resolve alpha from extras['stacker_clfs'][*]['alpha']."""
    import inspect
    import recovery
    src = inspect.getsource(recovery._r2_refuse)
    assert 'stacker_clfs' in src, "R2 no longer reads the extras alpha contract"
    assert 'conformal_set(probs, alpha=alpha)' in src, \
        "R2 must pass the bound alpha to conformal_set"


def test_r2_falls_back_when_alpha_absent():
    """With no alpha in extras, R2 must still work (vendored default)."""
    import recovery
    import fusion
    from schemas import DegradationState, ModuleOutput

    calls = []
    orig = fusion.conformal_set
    fusion.conformal_set = lambda probs, alpha=None: (
        calls.append(alpha), ["camera_original"])[1]
    try:
        d = DegradationState()
        m = ModuleOutput(module_id="metadata", version="v1",
                         verdict_direction="authentic",
                         evidence_score=0.1, reliability_score=0.9,
                         confidence_score=0.9, artifacts=[], compute_ms=1)
        extras = {"outputs": [m], "d": d, "survivors": [m],
                  "stacker_clfs": {"bundle": {"alpha": None}}}
        recovery._r2_refuse(None, extras, [])
    finally:
        fusion.conformal_set = orig
    assert calls and calls[0] is None, f"expected None alpha, got {calls}"


# --- fix 2: video stacker lazy load is fail-closed --------------------------

def test_video_stacker_load_is_double_checked_and_locked():
    """loaded is set only after success; a lock guards the load."""
    import inspect
    import argus_detect
    src = inspect.getsource(argus_detect._load_video_stacker)
    assert "_VID_STACKER_LOCK" in src, "load is not lock-guarded"
    # the public entry must NOT set loaded before calling the worker
    assert 'loaded"] = True' not in src, \
        "public entry sets loaded before the load completes (fail-open)"
    worker = inspect.getsource(argus_detect._load_video_stacker_locked)
    assert worker.rstrip().endswith('_VID_STACKER_CACHE["loaded"] = True') or \
        '_VID_STACKER_CACHE["loaded"] = True' in worker, \
        "worker must publish loaded=True"


def test_video_stacker_failure_does_not_poison_cache():
    """A failed load must leave loaded=False so the error surfaces again."""
    import importlib
    import argus_detect
    importlib.reload(argus_detect)

    argus_detect._VID_STACKER_CACHE.update(
        {"loaded": False, "clf": None, "embedder": None, "names": None})

    # force the worker to fail
    def boom():
        raise RuntimeError("simulated load failure")
    argus_detect._load_video_stacker_locked = boom
    try:
        with pytest.raises(RuntimeError):
            argus_detect._load_video_stacker()
        # the failure must not have been marked as loaded
        assert argus_detect._VID_STACKER_CACHE["loaded"] is False
        assert argus_detect._VID_STACKER_CACHE["clf"] is None
    finally:
        importlib.reload(argus_detect)


# --- fix 3: audio model load is lock-guarded --------------------------------

def test_audio_model_load_is_lock_guarded():
    import inspect
    import argus_detect
    src = inspect.getsource(argus_detect._get_audio_model)
    assert "_AUDIO_MODEL_LOCK" in src, "audio model load is not lock-guarded"
    # double-checked
    assert src.count("_AUDIO_MODEL is None") >= 2, \
        "audio model load is not double-checked"


def test_audio_model_lock_exists():
    import argus_detect
    assert hasattr(argus_detect, "_AUDIO_MODEL_LOCK")
    assert hasattr(argus_detect._AUDIO_MODEL_LOCK, "acquire")