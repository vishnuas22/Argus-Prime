"""Graceful drain for the Argus-Prime serve path (TASK-A6).

Why this exists
---------------
The lifespan had no shutdown branch: on SIGTERM uvicorn stopped accepting and
immediately tore down the event loop, so any in-flight adjudication was dropped.
For a court that takes 1.5-10 s per clip, that means a client can receive a
truncated response or a dropped connection and cannot tell "no verdict" from
"processing".

Design
------
* Track in-flight requests with a counter + condition (no dependency).
* On shutdown: stop accepting (uvicorn does this first), then wait up to
  ``ARGUS_DRAIN_TIMEOUT`` seconds for in-flight work to finish.
* Anything still running after the timeout is abandoned and counted; new
  requests arriving during drain get **503** with ``Retry-After``.
* ``/api/health`` stays answerable during drain so an orchestrator can see
  "draining" rather than a hard failure.

Configuration
-------------
``ARGUS_DRAIN_TIMEOUT``  seconds to wait for in-flight work (default ``10.0``)
``ARGUS_DRAIN_GRACE``    seconds to keep serving 503s before exit (default ``0``)
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

_DRAIN_TIMEOUT_ENV = "ARGUS_DRAIN_TIMEOUT"
_DRAIN_DEFAULT = 10.0


@dataclass
class DrainState:
    """Tracks in-flight requests and the drain transition."""

    timeout: float = _DRAIN_DEFAULT
    _lock: threading.Lock = None  # type: ignore[assignment]
    _inflight: int = 0
    _draining: bool = False
    _max_inflight: int = 0
    _abandoned: int = 0

    def __post_init__(self):
        if self._lock is None:
            self._lock = threading.Lock()

    @property
    def draining(self) -> bool:
        return self._draining

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def begin(self) -> bool:
        """Register a new request. False when the server is draining."""
        with self._lock:
            if self._draining:
                return False
            self._inflight += 1
            if self._inflight > self._max_inflight:
                self._max_inflight = self._inflight
            return True

    def end(self) -> None:
        """Mark a request finished."""
        with self._lock:
            if self._inflight > 0:
                self._inflight -= 1

    def reset(self) -> None:
        """Return to the accepting state.

        Used by tests (which share the process-wide singleton) and by any
        in-process restart path. Not used in normal operation.
        """
        with self._lock:
            self._draining = False
            self._inflight = 0
            self._abandoned = 0
            self._max_inflight = 0

    def start_drain(self) -> None:
        """Stop accepting new work and record the transition time."""
        with self._lock:
            self._draining = True

    def wait_for_drain(self) -> dict:
        """Block until in-flight hits zero or the timeout expires.

        Returns a summary for the shutdown log.
        """
        deadline = time.monotonic() + max(0.0, self.timeout)
        while True:
            with self._lock:
                n = self._inflight
                if n == 0:
                    return {
                        "drained": True,
                        "abandoned": self._abandoned,
                        "max_inflight": self._max_inflight,
                        "waited_s": round(max(0.0, self.timeout -
                                             (deadline - time.monotonic())), 3),
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._abandoned = n
                    return {
                        "drained": False,
                        "abandoned": n,
                        "max_inflight": self._max_inflight,
                        "waited_s": round(self.timeout, 3),
                    }
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


_DRAIN = DrainState(timeout=float(
    os.environ.get(_DRAIN_TIMEOUT_ENV, "") or _DRAIN_DEFAULT))


def drain_state() -> DrainState:
    """The process drain tracker."""
    return _DRAIN


def configure(timeout: float | None = None) -> DrainState:
    """Set the drain timeout (called once at startup)."""
    if timeout is not None:
        _DRAIN.timeout = float(timeout)
    return _DRAIN