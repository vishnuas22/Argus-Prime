"""Sliding-window rate limiting for the Argus-Prime serve path (TASK-A2).

Why this exists
---------------
The court is single-threaded (``CapacityLimiter(1)``); one client can starve
every other client. There is no per-client fairness today.

Design
------
* **Moving window, not fixed.** A fixed window lets a client send ``limit``
  requests at the end of one window and ``limit`` more at the start of the next,
  so the observed burst is ``2 x limit``. The literature on rate-limit replay
  reports exactly this leak. A per-key deque of timestamps has no such seam.
* **Keyed on the API key when present, else the client IP.** Keying on IP alone
    is wrong behind NAT/CGNAT — many legitimate users share one address.
* **In-process store.** Correct for a single worker. With multiple workers each
    has its own view; that needs Redis. Documented in ``deploy/``.
* **Zero new dependencies** — a deque and a lock.

Configuration
-------------
``ARGUS_RATE_LIMIT``     e.g. ``"10/minute"``, ``"600/hour"`` (default ``120/minute``)
``ARGUS_RATE_BURST``     integer burst multiplier (default ``1``)
"""
from __future__ import annotations

import os
import threading
import time
from collections import deque
from dataclasses import dataclass

_RATE_LIMIT_ENV = "ARGUS_RATE_LIMIT"
_RATE_BURST_ENV = "ARGUS_RATE_BURST"

_DEFAULT_LIMIT = 120
_DEFAULT_PERIOD = 60.0
_PERIODS = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}


@dataclass
class RateLimitInfo:
    """Result of a limiter check."""

    allowed: bool
    limit: int
    remaining: int
    retry_after: float
    scope: str


def _parse_limit(spec: str) -> tuple[int, float]:
    """Parse ``"<n>/<period>"`` into ``(limit, period_seconds)``."""
    spec = (spec or "").strip().lower()
    if not spec:
        return _DEFAULT_LIMIT, _DEFAULT_PERIOD
    try:
        left, period = spec.split("/", 1)
        limit = int(left.strip())
        period = period.strip()
        if period.endswith("s") and period[:-1].isdigit():
            return limit, float(period[:-1])
        if period not in _PERIODS:
            raise ValueError(period)
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        return limit, _PERIODS[period]
    except (ValueError, KeyError) as e:
        raise SystemExit(
            f"invalid {_RATE_LIMIT_ENV}={spec!r}: expected '<n>/<period>' with "
            f"period in {sorted(_PERIODS)} (or '<n>s'); got {e}"
        )


def _burst() -> int:
    raw = os.environ.get(_RATE_BURST_ENV, "1").strip()
    try:
        b = int(raw)
    except ValueError:
        raise SystemExit(f"invalid {_RATE_BURST_ENV}={raw!r}: must be an integer")
    if b < 1:
        raise SystemExit(f"{_RATE_BURST_ENV} must be >= 1, got {b}")
    return b


class SlidingWindowLimiter:
    """Per-key moving-window limiter.

    Thread-safe: the check-and-record is one critical section, so two concurrent
    requests cannot both observe the same remaining allowance.
    """

    def __init__(self, limit: int, period: float, burst: int = 1):
        self.limit = limit
        self.period = period
        self.capacity = limit * max(1, burst)
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> RateLimitInfo:
        """Record a hit for ``key`` and report whether it is allowed."""
        now = time.monotonic() if now is None else now
        cutoff = now - self.period
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.capacity:
                oldest = hits[0]
                retry = max(0.0, (oldest + self.period) - now)
                return RateLimitInfo(
                    allowed=False,
                    limit=self.capacity,
                    remaining=0,
                    retry_after=retry,
                    scope=key,
                )
            hits.append(now)
            return RateLimitInfo(
                allowed=True,
                limit=self.capacity,
                remaining=self.capacity - len(hits),
                retry_after=0.0,
                scope=key,
            )

    def reset(self, key: str | None = None) -> None:
        """Clear one key's history, or all of them."""
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


def limiter_from_env() -> SlidingWindowLimiter:
    """Build the process limiter from ``ARGUS_RATE_LIMIT`` / ``_BURST``."""
    limit, period = _parse_limit(os.environ.get(_RATE_LIMIT_ENV, ""))
    return SlidingWindowLimiter(limit=limit, period=period, burst=_burst())


def client_key(request, api_key: str | None) -> str:
    """Identify the caller: API key when present, else the peer IP."""
    if api_key:
        return f"key:{api_key[:8]}"
    client = getattr(request, "client", None)
    host = getattr(client, "host", None) if client else None
    return f"ip:{host or 'unknown'}"


def config_status() -> dict:
    """Describe the limiter configuration (used by ``/api/health``)."""
    limit, period = _parse_limit(os.environ.get(_RATE_LIMIT_ENV, ""))
    return {
        "limit": limit,
        "period_s": period,
        "burst": _burst(),
        "capacity": limit * max(1, _burst()),
        "store": "in-process",
        "scope": "api-key else client-ip",
    }