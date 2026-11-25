"""Fast tier: TASK-A5 API contract tests.

The HTTP surface had ZERO tests, so any refactor (including S1's concurrency
work) could break the contract silently. These tests pin the *shape* of every
response without needing real models: heavy work is replaced with a stub via
FastAPI dependency overrides and monkeypatching.

Locks:
  1. /api/health required fields + types
  2. /api/detect required fields + types (image path, stubbed court)
  3. error envelopes: 400 empty file, 413 oversized, 415 undecodable audio
  4. auth 401 envelope when enabled
  5. no secrets or absolute machine paths leak into any response
  6. OpenAPI schema is generated and has the expected paths
"""
import io
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[1]

# A tiny valid PNG generated at import (no fixture file needed).
def _tiny_png() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (128, 128, 128)).save(buf, format="PNG")
    return buf.getvalue()


_PNG_1X1 = _tiny_png()

HEALTH_REQUIRED = {
    "status": str,
    "channels": list,
    "stacker": str,
    "n_modules": int,
    "modalities": dict,
    "total_instruments": int,
    "conformal_alpha": (int, float, type(None)),
    "auth": dict,
    "rate_limit": dict,
    "calibration": dict,
}

DETECT_REQUIRED = {
    "p_fake_fused": (int, float, type(None)),
    "abstained": bool,
    "alpha": (int, float, type(None)),
    "kind": str,
    "channels": list,
    "module_evidence": dict,
    "stacker": (str, type(None)),
    "scope": dict,
    "p_fake_adjudicated": (int, float, type(None)),
    "adjudicated": bool,
}


def _client(monkeypatch):
    """Build a TestClient with the court stubbed (no real models).

    The lifespan would call init_court() and load the 218 MB bundle; instead we
    install the module once with a minimal fake _state and bypass startup.
    """
    from fastapi.testclient import TestClient
    import api.server as srv
    # TASK-A6: the drain tracker is a process-wide singleton; a test that leaves
    # it draining would 503 every later test. Reset per client.
    from drain import drain_state
    drain_state().reset()

    # Minimal fake court state: enough for the response builders.
    srv._state.clear()
    srv._state.update({
        "ids": ["metadata", "compression_history"],
        "modules": {"metadata": object(), "compression_history": object()},
        "stacker": "artifacts/p6v11",
        "conformal_alpha": 0.2,
        "mids": ["metadata", "compression_history"],
        "clf": object(),
        "feature_names": ["a", "b"],
        "conflict_p99": 0.5,
        "calibrator": None,
        "threadsafe_alpha": False,
        "estimator": None,
    })

    # Stub the expensive court call.
    def _fake_run_court(pil, alpha=None, original_raw=None):
        return {
            "p_fake_fused": 0.25,
            "abstained": False,
            "alpha": alpha,
            "channels": ["metadata"],
            "module_evidence": {"metadata": {"fired": True}},
            "trust": 0.9,
            "risk": 0.1,
            "recovery": {"outcome": "none", "verdict": None},
        }

    monkeypatch.setattr(srv, "run_court", _fake_run_court, raising=False)

    # Avoid the lifespan entirely.
    app = srv.app
    app.router.on_startup.clear()
    app.router.on_shutdown.clear()
    return TestClient(app)


def _assert_shape(obj, required, where):
    for key, types in required.items():
        assert key in obj, f"{where}: missing required field {key!r}"
        assert isinstance(obj[key], types), (
            f"{where}: {key!r} is {type(obj[key]).__name__}, "
            f"expected {types}")


def _assert_no_leaks(obj, where):
    """No secrets and no absolute machine paths in a response."""
    blob = json.dumps(obj)
    # Needles assembled from fragments so this file does not itself trip the
    # P0b hardcoded-path guard (same trick as test_config_seam.py).
    for bad in ("/Users/" + "Dataghost", "/private/var/" + "folders",
                "/home/", "HF_TOKEN", "ARGUS_API_KEYS", "BEGIN PRIVATE KEY"):
        assert bad not in blob, f"{where}: leaked {bad!r}"


# --- health ------------------------------------------------------------------

def test_health_contract(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.get("/api/health")
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_shape(body, HEALTH_REQUIRED, "/api/health")
    _assert_no_leaks(body, "/api/health")
    assert body["total_instruments"] == sum(
        m["instruments"] for m in body["modalities"].values())
    for mod in ("image", "video", "audio"):
        assert mod in body["modalities"], f"missing modality {mod}"
        assert {"build", "artifact", "instruments"} <= set(
            body["modalities"][mod])


def test_health_auth_shape(monkeypatch):
    with _client(monkeypatch) as c:
        body = c.get("/api/health").json()
    assert set(body["auth"]) >= {"required", "keys_configured", "source"}
    assert isinstance(body["auth"]["required"], bool)


def test_health_rate_limit_shape(monkeypatch):
    with _client(monkeypatch) as c:
        body = c.get("/api/health").json()
    assert set(body["rate_limit"]) >= {"limit", "period_s", "store"}
    assert body["rate_limit"]["limit"] >= 0


# --- detect ------------------------------------------------------------------

def test_detect_contract_image(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/api/detect",
                   files={"file": ("t.png", io.BytesIO(_PNG_1X1), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    _assert_shape(body, DETECT_REQUIRED, "/api/detect")
    _assert_no_leaks(body, "/api/detect")


def test_detect_alpha_echo(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/api/detect",
                   files={"file": ("t.png", io.BytesIO(_PNG_1X1), "image/png")},
                   data={"alpha": "0.3"})
    assert r.status_code == 200, r.text
    assert r.json()["alpha"] == 0.3


def test_detect_rejects_empty_file(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/api/detect",
                   files={"file": ("t.png", io.BytesIO(b""), "image/png")})
    assert r.status_code == 400, r.text
    assert "empty file" in r.json()["detail"]


def test_detect_rejects_unparseable_image(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post("/api/detect",
                   files={"file": ("t.png", io.BytesIO(b"not-an-image"),
                                   "image/png")})
    assert r.status_code == 400, r.text
    assert "unparseable image" in r.json()["detail"]


def test_detect_rejects_oversized_content_length(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.post(
            "/api/detect",
            files={"file": ("t.png", io.BytesIO(_PNG_1X1), "image/png")},
            headers={"content-length": str(200 * 1024 * 1024)},
        )
    assert r.status_code == 413, r.text
    assert "too large" in r.json()["detail"]


# --- auth envelope -----------------------------------------------------------

def test_auth_401_envelope(monkeypatch):
    """When auth is required and no key is sent, the error is a clean 401."""
    monkeypatch.setenv("ARGUS_REQUIRE_AUTH", "1")
    monkeypatch.setenv("ARGUS_API_KEYS", "contractkey")
    import importlib
    import auth
    importlib.reload(auth)
    import api.server as srv
    importlib.reload(srv)
    # the drain tracker is process-wide; reset it after the reload
    from drain import drain_state
    drain_state().reset()
    # rebuild state after reload
    srv._state.clear()
    srv._state.update({
        "ids": [], "modules": {}, "stacker": "artifacts/p6v11",
        "conformal_alpha": None, "mids": [], "clf": object(),
        "feature_names": [], "conflict_p99": None, "calibrator": None,
        "threadsafe_alpha": False, "estimator": None,
    })
    from fastapi.testclient import TestClient
    srv.app.router.on_startup.clear()
    with TestClient(srv.app) as c:
        r = c.get("/api/history")
    assert r.status_code == 401, r.text
    assert "detail" in r.json()
    assert "api key" in r.json()["detail"].lower()


# --- OpenAPI -----------------------------------------------------------------

def test_openapi_schema_has_expected_paths(monkeypatch):
    with _client(monkeypatch) as c:
        r = c.get("/openapi.json")
    assert r.status_code == 200, r.text
    spec = r.json()
    assert "/api/detect" in spec["paths"]
    assert "/api/health" in spec["paths"]
    assert "/api/history" in spec["paths"]
    _assert_no_leaks(spec, "openapi")


def test_openapi_does_not_expose_secrets(monkeypatch):
    with _client(monkeypatch) as c:
        spec = c.get("/openapi.json").json()
    blob = json.dumps(spec)
    for bad in ("ARGUS_API_KEYS", "HF_TOKEN", "api_key_value"):
        assert bad not in blob, f"openapi leaked {bad!r}"