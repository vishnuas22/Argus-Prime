"""Fast tier: TASK-A4 metrics + structured logging.

Locks:
  1. counters increment per modality/route/status
  2. histogram buckets are cumulative and bounded (+Inf == count)
  3. abstention counted separately
  4. Prometheus text format parses (metric name, labels, value)
  5. labels never carry filenames or paths (cardinality + privacy)
  6. structured logs are one JSON object per line with ts + event
"""
import json
import re

import pytest

from api_metrics import (
    Metrics,
    _LATENCY_BUCKETS,
    log_event,
    metrics as get_metrics,
)

pytestmark = pytest.mark.fast


def test_counters_increment():
    m = Metrics()
    m.record_request("image", "image_court", 200, 0.5, False)
    m.record_request("image", "image_court", 200, 0.7, True)
    m.record_request("video", "video_face", 500, 1.0, False)
    snap = m.snapshot()
    assert snap["requests_total"]["image|image_court|200"] == 2
    assert snap["requests_total"]["video|video_face|500"] == 1
    assert snap["abstained_total"]["image|image_court"] == 1


def test_histogram_buckets_are_cumulative():
    """Prometheus requires cumulative buckets: le=X means AT MOST X.

    Internal storage is per-interval; ``snapshot()`` must expose the cumulative
    form so ``histogram_quantile`` is correct.
    """
    m = Metrics()
    for v in (0.1, 0.3, 0.6, 1.5, 100.0):
        m.record_request("image", "r", 200, v, False)
    h = m.snapshot()["latency"]["image|r"]
    assert h["count"] == 5

    buckets = h["buckets"]
    counts = [c for _, c in buckets]
    # cumulative => non-decreasing
    assert counts == sorted(counts)
    # le=0.25 -> 1 (only 0.1) ; le=0.5 -> 2 ; le=1.0 -> 3 ; le=2.0 -> 4
    assert counts[:4] == [1, 2, 3, 4]
    # the last finite bucket must equal count - overflow
    assert counts[-1] == h["count"] - h["overflow"]
    # 100.0 exceeds the largest bound (64) -> overflow
    assert h["overflow"] == 1


def test_prometheus_inf_bucket_equals_count():
    m = Metrics()
    for v in (0.1, 5.0, 500.0):
        m.record_request("image", "r", 200, v, False)
    text = m.render_prometheus()
    # the +Inf bucket must equal the total count (3)
    assert 'le="+Inf"} 3' in text
    assert "argus_request_latency_seconds_count" in text


def test_overflow_bucket_captures_large_values():
    m = Metrics()
    m.record_request("image", "r", 200, 999.0, False)
    h = m.snapshot()["latency"]["image|r"]
    assert h["overflow"] == 1
    assert h["count"] == 1


def test_prometheus_renders_parseable_text():
    m = Metrics()
    m.record_request("image", "image_court", 200, 0.4, False)
    m.record_request("image", "image_court", 200, 0.6, True)
    text = m.render_prometheus()

    # every non-comment, non-empty line is `name{labels} value` or `name value`
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        assert re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? \S+$', line), line

    assert 'argus_requests_total{modality="image",route="image_court",status="200"} 2' in text
    assert "argus_abstained_total" in text
    assert 'le="+Inf"' in text
    assert text.endswith("\n")


def test_labels_never_contain_paths_or_filenames():
    m = Metrics()
    m.record_request("image", "image_court", 200, 0.5, False)
    text = m.render_prometheus()
    for bad in ("/", ".jpg", ".mp4", "real_", "fake_", "Dataghost"):
        # "/" appears in label *values* only if a path leaked
        assert f'route="{bad}' not in text
        assert f'modality="{bad}' not in text


def test_log_event_emits_json(capsys):
    log_event("detect", kind="image", p_fake=0.25, abstained=False)
    out = capsys.readouterr().out.strip()
    rec = json.loads(out)
    assert rec["event"] == "detect"
    assert rec["kind"] == "image"
    assert rec["p_fake"] == 0.25
    assert "ts" in rec


def test_log_event_raw_mode(capsys, monkeypatch):
    monkeypatch.setenv("ARGUS_LOG_RAW", "1")
    log_event("detect", kind="image")
    out = capsys.readouterr().out
    assert "[detect]" in out
    assert "kind=image" in out


def test_route_paths_with_slashes_do_not_break_rendering():
    """Regression: route '/api/history' contains slashes.

    The first implementation joined composite keys with \"/\", so splitting on
    \"/\" produced 4 parts and /metrics returned 500. Keys must use a separator
    that cannot appear in a route.
    """
    m = Metrics()
    m.record_request("http", "/api/history", 200, 0.01, False)
    m.record_request("http", "/api/detect", 200, 1.5, False)
    text = m.render_prometheus()          # must not raise
    assert 'route="/api/history"' in text
    assert 'route="/api/detect"' in text
    assert 'status="200"' in text


def test_registry_is_process_singleton():
    assert get_metrics() is get_metrics()


def test_histogram_bucket_count_matches_spec():
    m = Metrics()
    m.record_request("a", "b", 200, 1.0, False)
    h = m.snapshot()["latency"]["a|b"]
    # one count per bucket + one overflow
    assert len(h["buckets"]) == len(_LATENCY_BUCKETS)
    assert h["overflow"] == 0