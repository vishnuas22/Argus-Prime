"""Slow tier: 8-way concurrency race test for the serve path (TASK-S1).

This is the S1 entry-ticket gate. It must pass BEFORE ``CapacityLimiter(1)`` is
raised. It needs the real artifact bundle, so it is ``slow`` (nightly only).

Run with:
    pytest -m slow tests/test_concurrency_races.py -q

IMPORTANT — how to read a PASS here (measured, 2026-11-20)
----------------------------------------------------------
This test PASSES today both with and without ``ARGUS_THREADSAFE_ALPHA=1``,
because ``CapacityLimiter(1)`` serialises every court request: the historical
global monkey-patch never interleaves, so it cannot cross-bind. The limiter is
load-bearing.

The test's value is as a **regression guard for the promotion step**: the moment
someone raises the limiter, this test is what fails if the α plumbing is still
the global monkey-patch. It is the gate, not the demonstration.

The defect itself is demonstrated directly (no limiter in the way) by
``tests/test_threadsafe.py::test_global_monkeypatch_cross_binds_under_threads``.
"""
import os
import threading

import pytest

pytestmark = pytest.mark.slow

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

N = 8
ALPHAS = [round(0.05 + i * 0.1, 2) for i in range(N)]


def _require_artifacts():
    from pathlib import Path
    root = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[1])
    stacker = root / "artifacts" / "p6v11" / "stacker.joblib"
    if not stacker.exists():
        pytest.skip(f"artifact bundle missing: {stacker} (P0a not bootstrapped)")
    return root


@pytest.fixture(scope="module")
def client():
    _require_artifacts()
    os.environ.setdefault("ARGUS_THREADSAFE_ALPHA", "1")
    from fastapi.testclient import TestClient
    from api.server import app, init_court
    init_court()
    with TestClient(app) as c:
        yield c


def _fixture_image():
    from pathlib import Path
    root = Path(os.environ.get("ARGUS_ROOT") or Path(__file__).resolve().parents[1])
    for cand in (
        root / "data" / "corpora" / "inv_aug" / "ffpp_real" / "real_147_frame_000.jpg",
    ):
        if cand.exists():
            return cand
    pytest.skip("no fixture image available")


def test_alpha_isolation_8_way(client):
    """8 concurrent requests with distinct alpha: each response must report ITS OWN alpha.

    Under the historical global monkey-patch this fails (all responses share one
    alpha). With ARGUS_THREADSAFE_ALPHA=1 the contextvar shim keeps them apart.
    """
    img = _fixture_image()
    results = {}
    errors = []
    barrier = threading.Barrier(N)

    def worker(i):
        a = ALPHAS[i]
        try:
            barrier.wait(timeout=10)
            with img.open("rb") as fh:
                r = client.post(
                    "/api/detect",
                    files={"file": (img.name, fh, "image/jpeg")},
                    data={"alpha": str(a)},
                )
            results[i] = (r.status_code, r.json().get("alpha"))
        except Exception as e:  # pragma: no cover
            errors.append(f"worker {i}: {e!r}")

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)

    assert not errors, errors
    assert len(results) == N
    for i in range(N):
        status, reported = results[i]
        assert status == 200, (i, status)
        # the response must echo the alpha THIS request asked for
        assert reported == ALPHAS[i], (
            f"request {i} asked alpha={ALPHAS[i]} but response reported "
            f"{reported} — cross-binding detected")


def test_concurrent_verdicts_are_not_contaminated(client):
    """Repeated identical requests must be deterministic under concurrency."""
    img = _fixture_image()
    outs = []
    lock = threading.Lock()
    barrier = threading.Barrier(4)

    def worker():
        try:
            barrier.wait(timeout=10)
            with img.open("rb") as fh:
                r = client.post("/api/detect",
                                files={"file": (img.name, fh, "image/jpeg")})
            with lock:
                outs.append(r.json().get("p_fake_fused"))
        except Exception as e:  # pragma: no cover
            with lock:
                outs.append(f"error:{e!r}")

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=120)

    assert len(outs) == 4
    assert not any(isinstance(o, str) for o in outs), outs
    assert len(set(outs)) == 1, f"non-deterministic verdicts: {outs}"