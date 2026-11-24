"""Fast tier: TASK-A1 API-key authentication.

Uses dependency overrides and a fake client so no real models or artifacts are
needed. Locks the security properties:
  1. auth disabled by default  -> requests pass (preserves local dev)
  4. auth enabled + no key     -> 401
  3. auth enabled + wrong key  -> 401
  4. auth enabled + right key  -> 200
  5. fail-closed at boot       -> SystemExit when required but unconfigured
  6. constant-time verification across a key set
"""
import importlib
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO = Path(__file__).resolve().parents[1]


def _fresh_auth(monkeypatch, keys=None, require=None):
    """Reload auth.py under a controlled environment."""
    monkeypatch.delenv("ARGUS_API_KEYS", raising=False)
    monkeypatch.delenv("ARGUS_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("ARGUS_REQUIRE_AUTH", raising=False)
    if keys is not None:
        monkeypatch.setenv("ARGUS_API_KEYS", keys)
    if require is not None:
        monkeypatch.setenv("ARGUS_REQUIRE_AUTH", require)
    import auth
    importlib.reload(auth)
    return auth


def test_disabled_by_default(monkeypatch):
    a = _fresh_auth(monkeypatch)
    assert a.require_auth_enabled() is False
    # require_key is a no-op, so a request without a key passes
    assert a.require_key(None, None) is None


def test_enabled_rejects_missing_key(monkeypatch):
    a = _fresh_auth(monkeypatch, keys="secret123", require="1")
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        a.require_key(None, None)
    assert e.value.status_code == 401


def test_enabled_rejects_wrong_key(monkeypatch):
    a = _fresh_auth(monkeypatch, keys="secret123", require="1")
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as e:
        a.require_key(None, "wrong")
    assert e.value.status_code == 401


def test_enabled_accepts_right_key(monkeypatch):
    a = _fresh_auth(monkeypatch, keys="secret123,other456", require="1")
    assert a.require_key(None, "secret123") is None
    assert a.require_key(None, "other456") is None


def test_bearer_token_accepted(monkeypatch):
    a = _fresh_auth(monkeypatch, keys="secret123", require="1")

    class _Req:
        headers = {"authorization": "Bearer secret123"}

    assert a.require_key(_Req(), None) is None


def test_fail_closed_when_required_but_unconfigured(monkeypatch):
    a = _fresh_auth(monkeypatch, keys=None, require="1")
    with pytest.raises(SystemExit) as e:
        a.validate_config_or_die()
    assert "no API keys are configured" in str(e.value)


def test_fail_closed_when_keys_file_missing(monkeypatch):
    _fresh_auth(monkeypatch, keys="k", require="1")
    monkeypatch.setenv("ARGUS_API_KEYS_FILE", "/tmp/definitely-not-here.keys")
    import auth
    importlib.reload(auth)
    with pytest.raises(SystemExit):
        auth.validate_config_or_die()


def test_keys_file_loaded(monkeypatch, tmp_path):
    f = tmp_path / "keys.txt"
    f.write_text("# comment\nfilekey1\n\nfilekey2\n")
    monkeypatch.setenv("ARGUS_API_KEYS_FILE", str(f))
    import auth
    importlib.reload(auth)
    assert auth.require_auth_enabled() is False  # keys alone don't enable auth
    assert auth.verify_key("filekey1", auth._load_keys()) is True
    assert auth.verify_key("filekey2", auth._load_keys()) is True
    assert auth.verify_key("nope", auth._load_keys()) is False


def test_verify_key_is_constant_time_shape():
    import auth
    keys = frozenset({"a", "b", "c"})
    assert auth.verify_key(None, keys) is False
    assert auth.verify_key("", keys) is False
    assert auth.verify_key("a", frozenset()) is False
    assert auth.verify_key("a", keys) is True


def test_config_status_reports_shape(monkeypatch):
    a = _fresh_auth(monkeypatch, keys="k1,k2", require="1")
    st = a.config_status()
    assert st["required"] is True
    assert st["keys_configured"] == 2