"""Fast tier: TASK-A6 graceful drain.

Locks:
  1. new requests are admitted while not draining
  2. begin() returns False once draining (503 path)
  3. wait_for_drain returns promptly when nothing is in flight
  4. wait_for_drain times out and reports abandoned work
  5. inflight accounting is balanced across begin/end
  6. max_inflight is tracked
"""
import time

import pytest

from drain import DrainState

pytestmark = pytest.mark.fast


def test_admits_while_not_draining():
    d = DrainState(timeout=1.0)
    assert d.begin() is True
    d.end()
    assert d.inflight == 0


def test_rejects_once_draining():
    d = DrainState(timeout=1.0)
    d.start_drain()
    assert d.draining is True
    assert d.begin() is False


def test_wait_returns_promptly_when_idle():
    d = DrainState(timeout=5.0)
    t0 = time.monotonic()
    summary = d.wait_for_drain()
    elapsed = time.monotonic() - t0
    assert summary["drained"] is True
    assert summary["abandoned"] == 0
    assert elapsed < 0.5


def test_wait_times_out_and_reports_abandoned():
    d = DrainState(timeout=0.3)
    d.begin()                      # simulate one in-flight request
    d.start_drain()
    t0 = time.monotonic()
    summary = d.wait_for_drain()
    elapsed = time.monotonic() - t0
    assert summary["drained"] is False
    assert summary["abandoned"] == 1
    assert 0.25 <= elapsed <= 1.5


def test_inflight_accounting_balanced():
    d = DrainState(timeout=1.0)
    for _ in range(5):
        assert d.begin() is True
    assert d.inflight == 5
    for _ in range(5):
        d.end()
    assert d.inflight == 0


def test_end_never_goes_negative():
    d = DrainState(timeout=1.0)
    d.end()
    d.end()
    assert d.inflight == 0


def test_max_inflight_tracked():
    d = DrainState(timeout=1.0)
    for _ in range(3):
        d.begin()
    assert d._max_inflight == 3
    d.end(); d.end(); d.end()
    for _ in range(2):
        d.begin()
    assert d._max_inflight == 3     # high-water mark, not current


def test_wait_completes_when_work_finishes():
    d = DrainState(timeout=5.0)
    d.begin()
    d.start_drain()

    import threading
    t = threading.Timer(0.2, d.end)
    t.start()
    summary = d.wait_for_drain()
    t.join()
    assert summary["drained"] is True
    assert summary["abandoned"] == 0