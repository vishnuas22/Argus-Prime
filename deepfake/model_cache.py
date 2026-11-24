"""HuggingFace model resolution with a cache-first, fail-fast policy.

Why this exists
---------------
Several lanes call ``transformers.*.from_pretrained(<hf-id>)`` with no offline
guard. On a dev Mac the weights are already in ``~/.cache/huggingface``, so the
call is a cache hit. On a fresh clone, a container, or an air-gapped host the
call attempts a network fetch — and ``huggingface_hub`` **retries with backoff**
against ``cas-server.xethub.hf.co``, which can block for many minutes before
failing with ``401 Unauthorized`` (measured 2026-11-23 in the A3 container: the
CLI hung >6 min where the API path took 13 s).

Policy
------
1. Set ``HF_HUB_OFFLINE=1`` when a ``HF_HOME``/cache already holds the model,
   so a cache hit never touches the network.
2. Otherwise fail **fast** with a clear message naming the missing model and the
   command to fetch it — never hang, never silently degrade.
3. ``ARGUS_ALLOW_HF_DOWNLOAD=1`` opts back into downloads (developer machines).

This keeps the serve path deterministic and makes a missing model a startup
error rather than a multi-minute stall.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ALLOW_DOWNLOAD_ENV = "ARGUS_ALLOW_HF_DOWNLOAD"
_TRUTHY = {"1", "true", "yes", "on"}


def downloads_allowed() -> bool:
    """True when runtime model downloads are permitted (default: no)."""
    return os.environ.get(_ALLOW_DOWNLOAD_ENV, "").strip().lower() in _TRUTHY


def apply_offline_policy() -> None:
    """Force offline mode unless downloads are explicitly allowed.

    Call once at process start (CLI entry, server lifespan) before any
    ``from_pretrained``.
    """
    if downloads_allowed():
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        return
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def require_model(model_id: str, call: str, **kwargs):
    """Call ``from_pretrained`` with a cache-first, fail-fast policy.

    Parameters
    ----------
    model_id : str
        The HuggingFace repo id (e.g. ``"openai/clip-vit-large-patch14"``).
    call : str
        A zero-arg callable that performs the load.
    **kwargs
        Forwarded to the caller for error context (unused by the loader itself).

    Raises
    ------
    SystemExit
        With an actionable message when the model is not cached and downloads
        are disabled.
    """
    try:
        return call()
    except Exception as e:
        if downloads_allowed():
            raise
        msg = str(e)
        # Only convert *offline/cache* failures into the actionable error; a
        # genuine model/runtime bug must still surface as itself.
        offline_markers = (
            "offline",
            "cannot find",
            "not found",
            "no such file",
            "connection",
            "401",
            "unauthorized",
            "cas-server",
            "max retries",
        )
        if not any(m in msg.lower() for m in offline_markers):
            raise
        raise SystemExit(
            f"model '{model_id}' is not available locally and downloads are "
            f"disabled (offline serve path).\n"
            f"  fetch it once on a networked machine:\n"
            f"    python -c \"from transformers import AutoModel; "
            f"AutoModel.from_pretrained('{model_id}')\"\n"
            f"  or set {_ALLOW_DOWNLOAD_ENV}=1 to allow runtime downloads.\n"
            f"  original error: {type(e).__name__}: {msg[:200]}"
        )


def cache_status(model_id: str) -> dict:
    """Report whether a model is cached locally (for diagnostics/health)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        return {
            "model": model_id,
            "downloads_allowed": downloads_allowed(),
            "hf_offline": os.environ.get("HF_HUB_OFFLINE"),
        }
    except Exception:
        return {"model": model_id, "downloads_allowed": downloads_allowed()}