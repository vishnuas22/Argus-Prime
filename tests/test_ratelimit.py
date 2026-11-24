"""Fast tier: TASK-A2 sliding-window rate limiter.

Locks:
  1. allows up to capacity, then rejects
  2. window slides — old hits age out and capacity returns
  3. keys are isolated (one noisy client cannot starve another)
  4. retry_after is positive and shrinks as the window advances
  5. burst multiplies capacity
  6. env parsing rejects garbage (fail fast at boot)
"""
import time

import pytest

from ratelimit import (
    RateLimitInfo,
    SlidingWindowLimiter,
    _parse_limit,
    client_key,
    limiter_from_env,
)

pytestmark = pytest.mark.fast


def test_allows_capacity_then_rejects():
    lim = SlidingWindowLimiter(limit=3, period=60.0)
    now = 1000.0
    assert lim.check("k", now=now).allowed is True
    assert lim.check("k", now=now).allowed is True
    assert lim.check("k", now=now).allowed is True
    r = lim.check("k", now=now)
    assert r.allowed is False
    assert r.remaining == 0
    assert r.retry_after > 0


def test_window_slides():
    lim = SlidingWindowLimiter(limit=2, period=10.0)
    t = 100.0
    assert lim.check("k", now=t).allowed is True
    assert lim.check("k", now=t).allowed is True
    assert lim.check("k", now=t).allowed is False
    # after the window passes, capacity returns
    assert lim.check("k", now=t + 11.0).allowed is True


def test_keys_are_isolated():
    lim = SlidingWindowLimiter(limit=1, period=60.0)
    t = 500.0
    assert lim.check("a", now=t).allowed is True
    assert lim.check("a", now=t).allowed is False
    # a different key is unaffected
    assert lim.check("b", now=t).allowed is True


def test_burst_multiplies_capacity():
    lim = SlidingWindowLimiter(limit=2, period=60.0, burst=3)
    t = 0.0
    allowed = [lim.check("k", now=t).allowed for _ in range(7)]
    assert sum(allowed) == 6
    assert allowed[6] is False


def test_reset_clears_one_or_all():
    lim = SlidingWindowLimiter(limit=1, period=60.0)
    t = 0.0
    lim.check("a", now=t)
    lim.reset("a")
    assert lim.check("a", now=t).allowed is True

    lim.check("b", now=t)
    lim.reset()
    assert lim.check("b", now=t + 1).allowed is True


def test_parse_limit_variants():
    assert _parse_limit("10/minute") == (10, 60.0)
    assert _parse_limit("600/hour") == (600, 3600.0)
    # bare "<n>s" = n requests per 1 second
    assert _parse_limit("5s") == (5, 1.0)
    assert _parse_limit("") == (120, 60.0)


def test_parse_limit_rejects_garbage():
    for bad in ("abc/minute", "10/fortnight", "0/minute", "-1/minute"):
        with pytest.raises(SystemExit):
            _parse_limit(bad)


def test_client_key_prefers_api_key():
    class _C:
        host = "1.2.3.4"

    class _Req:
        client = _C()

    assert client_key(_Req(), "secretkey123").startswith("key:")
    assert client_key(_Req(), None).startswith("ip:")


def test_limiter_from_env(monkeypatch):
    monkeypatch.setenv("ARGUS_RATE_LIMIT", "7/minute")
    monkeypatch.setenv("ARGUS_RATE_BURST", "2")
    lim = limiter_from_env()
    assert lim.capacity == 14


def test_retry_after_shrinks_as_window_advances():
    lim = SlidingWindowLimiter(limit=1, period=60.0)
    t = 0.0
    lim.check("k", now=t)
    r1 = lim.check("k", now=t + 10.0)
    r2 = lim.check("k", now=t + 50.0)
    assert r1.retry_after > r2.retry_after
    assert r1.retry_after <= 60.0