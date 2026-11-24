"""API key authentication for the Argus-Prime serve path (TASK-A1).

Why this exists
---------------
The serve path already has a cross-origin POST gate (``_origin_gate`` in
``server.py``), which stops *browsers* from forging requests. It does **not**
stop a script: a client that sends no ``Origin`` header is allowed through. So
anything that can reach the port can spend GPU time on ``/api/detect`` and read
``/api/history`` (up to 200 prior verdicts).

Design
------
* FastAPI's own ``APIKeyHeader`` + ``Security`` (OWASP FastAPI Cheat Sheet).
* Keys come from ``ARGUS_API_KEYS`` (comma-separated) or ``ARGUS_API_KEYS_FILE``.
* Comparison uses ``hmac.compare_digest`` (constant time) — no timing oracle.
* **Default OFF.** With ``ARGUS_REQUIRE_AUTH`` unset, ``require_key`` is a no-op
  and behaviour is identical to before. This keeps local dev unchanged.
* **Fail closed.** With ``ARGUS_REQUIRE_AUTH=1`` and no keys configured, boot
  must refuse to start — never run open while claiming auth.

Scope
-----
``/api/health`` stays open so orchestrators can probe liveness/readiness.
Everything else (``/api/detect``, ``/api/explain``, ``/api/history``) is gated.
"""
from __future__ import annotations

import hmac
import os
from pathlib import Path

from fastapi import Header, HTTPException, Request, status

API_KEY_HEADER = "X-API-Key"

_REQUIRE_AUTH_ENV = "ARGUS_REQUIRE_AUTH"
_KEYS_ENV = "ARGUS_API_KEYS"
_KEYS_FILE_ENV = "ARGUS_API_KEYS_FILE"

_TRUTHY = {"1", "true", "yes", "on"}


def require_auth_enabled() -> bool:
    """True when the deployment demands an API key."""
    return os.environ.get(_REQUIRE_AUTH_ENV, "").strip().lower() in _TRUTHY


def _load_keys() -> frozenset[str]:
    """Resolve the configured key set from env and/or a file.

    Both sources are additive so a file can hold the long-lived client keys while
    an env var carries a break-glass key.
    """
    keys: set[str] = set()

    inline = os.environ.get(_KEYS_ENV, "")
    for k in inline.split(","):
        k = k.strip()
        if k:
            keys.add(k)

    path = os.environ.get(_KEYS_FILE_ENV, "").strip()
    if path:
        p = Path(path)
        if p.exists():
            for line in p.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    keys.add(line)
        elif require_auth_enabled():
            raise SystemExit(
                f"{_REQUIRE_AUTH_ENV} is set but {_KEYS_FILE_ENV} does not "
                f"exist: {path}"
            )
    return frozenset(keys)


def verify_key(presented: str | None, expected: frozenset[str]) -> bool:
    """Constant-time check of ``presented`` against the configured key set.

    Iterates over every configured key so the running time does not depend on
    which key matched (or whether any did).
    """
    if not presented or not expected:
        return False
    ok = False
    for k in expected:
        if hmac.compare_digest(presented, k):
            ok = True
    return ok


def config_status() -> dict:
    """Describe the auth configuration (used by ``/api/health``)."""
    keys = _load_keys()
    return {
        "required": require_auth_enabled(),
        "keys_configured": len(keys),
        "source": "file" if os.environ.get(_KEYS_FILE_ENV, "").strip() else "env",
    }


def validate_config_or_die() -> None:
    """Fail closed at boot when auth is required but no keys exist."""
    if require_auth_enabled() and not _load_keys():
        raise SystemExit(
            f"{_REQUIRE_AUTH_ENV} is set but no API keys are configured.\n"
            f"  set {_KEYS_ENV}=key1,key2  or  {_KEYS_FILE_ENV}=/path/to/keys\n"
            "  refusing to start open while claiming authentication is required"
        )


def _extract(request: Request | None, x_api_key: str | None) -> str | None:
    """Pull the presented key from the explicit arg, the header, or a bearer token.

    ``x_api_key`` arrives via FastAPI's ``Header(...)`` on the dependency, but the
    rate limiter calls this without it — so the header is read here too. Without
    that, every rate-limited request would fall back to the shared ``ip:`` bucket.
    """
    if x_api_key:
        return x_api_key.strip()
    if request is None:
        return None
    header = request.headers.get(API_KEY_HEADER)
    if header:
        return header.strip()
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def require_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias=API_KEY_HEADER),
) -> None:
    """FastAPI dependency: 401 unless a valid key is presented.

    A no-op when ``ARGUS_REQUIRE_AUTH`` is unset, so the default local
    deployment behaves exactly as before.
    """
    if not require_auth_enabled():
        return
    presented = _extract(request, x_api_key)
    if not verify_key(presented, _load_keys()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid API key",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )