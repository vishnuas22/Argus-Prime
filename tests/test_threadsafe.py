"""Fast tier: TASK-S1 thread-safety primitives (no models, no corpora).

Locks three properties:
  1. ``alpha_scope`` isolates α per thread/context (the fix).
  2. The historical global monkey-patch DOES cross-bind under threads
     (the defect) — this test is the evidence that raising CapacityLimiter(1)
     without the fix is unsafe, not merely slower.
  3. ``locked_lazy`` builds exactly once under contention and does not cache a
     failed load (the fail-open fix).
"""
import threading

import pytest

from threadsafe import (
    SHARED_STATE,
    alpha_scope,
    current_alpha,
    install_conformal_shim,
    locked_lazy,
    shared_state_table,
    thread_local_cache,
    uninstall_conformal_shim,
)

pytestmark = pytest.mark.fast


class _FakeFusion:
    """Stand-in for the vendored fusion module's conformal_set."""

    def __init__(self):
        self.calls = []

    def conformal_set(self, probs, alpha=0.10):
        self.calls.append(alpha)
        return ["camera_original"]


# --- alpha_scope -------------------------------------------------------------

def test_alpha_scope_binds_and_restores():
    assert current_alpha() is None
    with alpha_scope(0.25):
        assert current_alpha() == 0.25
    assert current_alpha() is None


def test_alpha_scope_rejects_out_of_range():
    with pytest.raises(ValueError, match="alpha must be in"):
        with alpha_scope(1.5):
            pass


def test_alpha_scope_nests():
    with alpha_scope(0.1):
        with alpha_scope(0.9):
            assert current_alpha() == 0.9
        assert current_alpha() == 0.1
    assert current_alpha() is None


def test_alpha_scope_isolates_threads():
    """8 threads, 8 distinct α, no cross-binding (the S1 entry-ticket property)."""
    n = 8
    barrier = threading.Barrier(n)
    seen = {}
    errors = []

    def worker(i):
        a = round(0.05 + i * 0.1, 2)
        try:
            barrier.wait(timeout=5)
            with alpha_scope(a):
                # yield to maximise interleaving
                for _ in range(200):
                    pass
                seen[i] = current_alpha()
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, errors
    assert len(seen) == n
    for i in range(n):
        expected = round(0.05 + i * 0.1, 2)
        assert seen[i] == expected, f"thread {i} saw {seen[i]} expected {expected}"


# --- the defect this replaces ------------------------------------------------

def test_global_monkeypatch_cross_binds_under_threads():
    """Documents the C-1 defect: swapping a module global is NOT thread-safe.

    Two threads each install their own partial on the shared global, then both
    call it. The last writer wins, so both observe the same α — the exact
    contamination that makes raising CapacityLimiter(1) unsafe today.
    """
    import functools

    fusion = _FakeFusion()
    original = fusion.conformal_set
    barrier = threading.Barrier(2)
    observed = {}

    def worker(alpha):
        fusion.conformal_set = functools.partial(original, alpha=alpha)
        barrier.wait(timeout=5)
        for _ in range(200):
            pass
        observed[alpha] = fusion.conformal_set([], )

    t1 = threading.Thread(target=worker, args=(0.05,))
    t2 = threading.Thread(target=worker, args=(0.95,))
    t1.start(); t2.start()
    t1.join(timeout=10); t2.join(timeout=10)

    # Both threads installed a DIFFERENT alpha on the shared global, then both
    # called it. The last writer won, so both calls observed ONE alpha: the
    # cross-bind. (Two requests, one operating point.)
    assert sorted(observed.keys()) == [0.05, 0.95]
    assert len(fusion.calls) == 2
    assert len(set(fusion.calls)) == 1, (
        f"expected cross-binding (one alpha for two requests), got {fusion.calls}")
    # restore
    fusion.conformal_set = original


def test_alpha_scope_prevents_the_cross_bind():
    """Same shape as above, but with alpha_scope: each thread keeps its own α."""
    n = 16
    barrier = threading.Barrier(n)
    seen = {}
    errors = []

    def worker(i):
        a = round(0.02 + i * 0.05, 3)
        try:
            barrier.wait(timeout=5)
            with alpha_scope(a):
                for _ in range(300):
                    pass
                seen[i] = current_alpha()
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    assert not errors, errors
    assert seen == {i: round(0.02 + i * 0.05, 3) for i in range(n)}


# --- conformal shim ----------------------------------------------------------

def test_shim_reads_contextvar_and_passes_explicit_alpha():
    fusion = _FakeFusion()
    original = fusion.conformal_set

    assert install_conformal_shim(fusion) is True
    assert fusion.conformal_set([], ) == ["camera_original"]
    assert fusion.calls == [0.10]  # default preserved

    with alpha_scope(0.3):
        fusion.conformal_set([])
    assert fusion.calls[-1] == 0.3

    # explicit alpha still wins
    fusion.conformal_set([], alpha=0.42)
    assert fusion.calls[-1] == 0.42

    assert uninstall_conformal_shim(fusion) is True
    # bound methods compare equal by (function, instance), not by identity
    assert fusion.conformal_set == original


def test_shim_is_idempotent():
    fusion = _FakeFusion()
    assert install_conformal_shim(fusion) is True
    first = fusion.conformal_set
    assert install_conformal_shim(fusion) is True
    assert fusion.conformal_set is first


def test_shim_returns_false_without_conformal_set():
    class _Empty:
        pass
    assert install_conformal_shim(_Empty()) is False
    assert uninstall_conformal_shim(_Empty()) is False


# --- locked_lazy -------------------------------------------------------------

def test_locked_lazy_builds_once_under_contention():
    builds = []
    lock = threading.Lock()

    def factory():
        with lock:
            builds.append(1)
        return object()

    lazy = locked_lazy("thing", factory)
    results = []
    barrier = threading.Barrier(12)

    def worker():
        barrier.wait(timeout=5)
        results.append(lazy.get())

    ts = [threading.Thread(target=worker) for _ in range(12)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)

    assert lazy.build_count == 1
    assert len(set(id(r) for r in results)) == 1


def test_locked_lazy_does_not_cache_failure():
    """The fail-open fix: a raising factory must not leave built=True."""
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")
        return "ok"

    lazy = locked_lazy("flaky", factory)
    with pytest.raises(RuntimeError, match="boom"):
        lazy.get()
    assert lazy.built is False
    assert lazy.get() == "ok"
    assert lazy.build_count == 1


def test_locked_lazy_reset():
    lazy = locked_lazy("r", lambda: object())
    first = lazy.get()
    lazy.reset()
    assert lazy.built is False
    assert lazy.get() is not first


# --- thread_local_cache ------------------------------------------------------

def test_thread_local_cache_isolates_threads():
    cache = thread_local_cache("net", lambda: object())
    main_obj = cache.get()
    seen = []

    t = threading.Thread(target=lambda: seen.append(cache.get()))
    t.start(); t.join(timeout=5)

    assert seen and seen[0] is not main_obj
    assert cache.get() is main_obj


# --- audit table -------------------------------------------------------------

def test_shared_state_table_shape():
    rows = shared_state_table()
    assert len(rows) == len(SHARED_STATE) >= 15
    for r in rows:
        assert set(r) == {"component", "location", "state", "thread_safe", "note"}
        assert isinstance(r["thread_safe"], bool)

    unsafe = [r["component"] for r in rows if not r["thread_safe"]]
    # the load-bearing ones must be present so nobody "fixes" the limiter blindly
    assert "fusion.conformal_set" in unsafe
    assert "video_face_track._DET" in unsafe
    assert "video_face_track._ARC_NET" in unsafe