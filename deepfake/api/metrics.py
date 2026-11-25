"""Zero-dependency Prometheus-style metrics + structured JSON logging (TASK-A4).

Why this exists
---------------
The serve path had almost no observability: two ``print()`` calls in error
handlers and nothing else. In production you could not answer "what is p95?", "what
is the abstention rate?", or "how many 5xx?".

Design
------
* **No new dependency.** Prometheus' text exposition format is trivial to emit
  by hand, and ``prometheus_client`` is not in ``requirements.txt``. Adding it
  would be a serve-path dependency for a formatting concern.
* **In-process counters/histograms**, guarded by a lock. Correct for the single
  worker the court runs (CapacityLimiter(1)); multi-worker needs a real registry.
* **Structured logs**: one JSON object per line to stdout, so any log shipper
  (Loki, CloudWatch, Datadog) can index fields directly instead of regexing text.
* **Bounded histograms**: fixed buckets, no unbounded label cardinality. Labels
  are limited to ``modality``, ``route``, ``status`` and ``outcome`` — never a
  filename, path, or user id (that would be both a memory leak and a privacy leak).

Endpoint: ``GET /metrics`` (Prometheus text). Always available; no auth required
so scrapers do not need a key.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

# Latency buckets in seconds — chosen around the court's measured ~1.5-10 s range.
_LATENCY_BUCKETS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)

_LOG_RAW_ENV = "ARGUS_LOG_RAW"


@dataclass
class _Histogram:
    """Fixed-bucket histogram with a sum and count."""

    buckets: list[float]
    counts: list[int] = field(default_factory=list)
    total: float = 0.0
    count: int = 0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * (len(self.buckets) + 1)

    def observe(self, v: float) -> None:
        self.total += v
        self.count += 1
        for i, b in enumerate(self.buckets):
            if v <= b:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def cumulative(self) -> list[tuple[float, int]]:
        """Prometheus-style cumulative bucket counts.

        Prometheus requires ``le="X"`` to mean *at most* X, so each bucket must
        include everything in the lower buckets. The internal store is
        per-interval (cheaper to update); the cumulative form is derived here.
        """
        out = []
        running = 0
        for bound, c in zip(self.buckets, self.counts[:-1]):
            running += c
            out.append((bound, running))
        return out

    def snapshot(self) -> dict:
        return {
            # cumulative (Prometheus-correct) buckets
            "buckets": self.cumulative(),
            # per-interval counts, kept for tests/debugging
            "interval_counts": list(zip(self.buckets, self.counts[:-1])),
            "overflow": self.counts[-1],
            "sum": self.total,
            "count": self.count,
        }


class Metrics:
    """Process-wide counters, histograms and gauges."""

    def __init__(self):
        self._lock = threading.Lock()
        self._requests: dict[tuple, int] = defaultdict(int)
        self._latency: dict[tuple, _Histogram] = {}
        self._abstain: dict[tuple, int] = defaultdict(int)
        self._started = time.time()

    def _hist(self, key) -> _Histogram:
        h = self._latency.get(key)
        if h is None:
            h = _Histogram(list(_LATENCY_BUCKETS))
            self._latency[key] = h
        return h

    def record_request(
        self,
        modality: str,
        route: str,
        status: int,
        elapsed: float,
        abstained: bool,
        outcome: str = "none",
    ) -> None:
        """Record one completed adjudication."""
        key = (modality, route)
        with self._lock:
            self._requests[(modality, route, status)] += 1
            self._hist(key).observe(elapsed)
            if abstained:
                self._abstain[key] += 1

    def snapshot(self) -> dict:
        """Plain-dict snapshot (used by /metrics and tests)."""
        with self._lock:
            req = {f"{m}/{r}/{s}": c for (m, r, s), c in self._requests.items()}
            lat = {f"{m}/{r}": h.snapshot() for (m, r), h in self._latency.items()}
            ab = {f"{m}/{r}": c for (m, r), c in self._abstain.items()}
        return {
            "uptime_s": round(time.time() - self._started, 1),
            "requests_total": req,
            "latency": lat,
            "abstained_total": ab,
        }

    def render_prometheus(self) -> str:
        """Render the snapshot in Prometheus text exposition format."""
        snap = self.snapshot()
        lines = [
            "# HELP argus_uptime_seconds Process uptime in seconds.",
            "# TYPE argus_uptime_seconds gauge",
            f"argus_uptime_seconds {snap['uptime_s']}",
            "",
            "# HELP argus_requests_total Adjudications by modality, route and HTTP status.",
            "# TYPE argus_requests_total counter",
        ]
        for k, v in sorted(snap["requests_total"].items()):
            m, r, s = k.split("/")
            lines.append(
                f'argus_requests_total{{modality="{m}",route="{r}",status="{s}"}} {v}')
        lines += [
            "",
            "# HELP argus_request_latency_seconds Adjudication latency histogram.",
            "# TYPE argus_request_latency_seconds histogram",
        ]
        for k, h in sorted(snap["latency"].items()):
            m, r = k.split("/")
            for bound, cumulative in h["buckets"]:
                lines.append(
                    f'argus_request_latency_seconds_bucket{{modality="{m}",'
                    f'route="{r}",le="{bound}"}} {cumulative}')
            lines.append(
                f'argus_request_latency_seconds_bucket{{modality="{m}",'
                f'route="{r}",le="+Inf"}} {h["count"]}')
            lines.append(
                f'argus_request_latency_seconds_sum{{modality="{m}",'
                f'route="{r}"}} {h["sum"]}')
            lines.append(
                f'argus_request_latency_seconds_count{{modality="{m}",'
                f'route="{r}"}} {h["count"]}')
        lines += [
            "",
            "# HELP argus_abstained_total Abstentions by modality and route.",
            "# TYPE argus_abstained_total counter",
        ]
        for k, v in sorted(snap["abstained_total"].items()):
            m, r = k.split("/")
            lines.append(
                f'argus_abstained_total{{modality="{m}",route="{r}"}} {v}')
        return "\n".join(lines) + "\n"


_METRICS = Metrics()


def metrics() -> Metrics:
    """The process metrics registry."""
    return _METRICS


def _json_default(o):
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def log_event(event: str, **fields) -> None:
    """Emit one structured JSON log line to stdout.

    ``ARGUS_LOG_RAW=1`` falls back to plain text for local debugging.
    """
    rec = {"ts": round(time.time(), 3), "event": event}
    rec.update(fields)
    if os.environ.get(_LOG_RAW_ENV, "").strip().lower() in ("1", "true", "yes"):
        print(f"[{event}] " + " ".join(f"{k}={v}" for k, v in fields.items()),
              flush=True)
        return
    print(json.dumps(rec, default=_json_default), flush=True)