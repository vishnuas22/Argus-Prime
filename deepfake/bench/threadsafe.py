"""Thread-safety primitives for the serve path (TASK-S1 scaffold).

Why this module exists (measured, not assumed)
----------------------------------------------
The API serialises every adjudication through ``anyio.CapacityLimiter(1)``
(``deepfake/api/server.py``). That limiter is **load-bearing**: it is what makes
the following shared-state defects safe today. Raising it without the fixes below
produces cross-request contamination, not just contention.

SHARED_STATE is the audit table. Read it before touching concurrency.

C-1  server.py            global monkey-patch of ``fusion.conformal_set``
C-3  argus_detect.py      global ``USER_ALPHA``
C-4  video_face_track.py  lazy ``_DET``  (cv2.FaceDetectorYN, stateful)
C-5  video_face_track.py  lazy ``_REC``  (cv2.FaceRecognizerSF)
C-6  video_face_track.py  lazy ``_ARC_NET`` (cv2.dnn, stateful)
C-7  video_face_track.py  global ``EMBEDDER_NAME`` + ``set_embedder``
C-8  video_face_track.py  global ``_SCORE_THRESHOLD`` + ``set_score_threshold``
C-9  argus_detect.py      ``_VID_STACKER_CACHE`` loaded-flag set BEFORE load
C-10 argus_detect.py      ``_AUDIO_MODEL`` lazy singleton
C-11 registry.py          ``_cache``
C-12 channels.py          realness / spectral / perturbation caches
C-13 dinov2 service       ``_cache``
C-14 server.py            ``_state`` dict mutated during init
C-15 server.py            ``sys.path`` growth on import
C-16 audio_fusion.py      ``torch.set_grad_enabled`` global toggle
C-17 audio path           ``torch.set_num_threads`` / manual_seed

What this module provides
-------------------------
``alpha_scope``            ContextVar-based, thread-safe replacement for the
                           ``fusion.conformal_set`` monkey-patch. ``anyio.to_thread``
                           copies the context into the worker, so each request
                           keeps its own α without touching a module global.
``install_conformal_shim`` Installs a ContextVar-aware ``conformal_set`` on the
                           vendored fusion module (opt-in; D-009 — the vendored
                           FILE is never edited, only its module attribute).
``locked_lazy``            Double-checked locking that spans the *whole* load.
                           The current lazy singletons set ``loaded=True`` before
                           the load completes, so a concurrent caller can observe
                           a half-built object (or a failed load as success).
``SHARED_STATE``           The audit table as data, for docs and tests.

What this module deliberately does NOT do
-----------------------------------------
* It does not raise ``CapacityLimiter(1)``. That is S1's promotion step and
  requires the 8-way race test to pass first.
* It does not convert the stateful YuNet/ArcFace objects to thread-local caches.
  Doing so changes object identity and would create one model instance per pool
  thread, raising memory and losing the process-cached property. That trade-off
  belongs to S1, measured.
* It does not fix the R2 α inconsistency or the ``conformal.alpha`` reporting.
  Both change behaviour/output and need a sealed spec + owner approval.
"""
from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

_CONFORMAL_ALPHA: contextvars.ContextVar[float | None] = (
    contextvars.ContextVar("argus_conformal_alpha", default=None)
)

SHIM_FLAG = "_argus_contextvar_shim"


@dataclass(frozen=True)
class SharedState:
    """One row of the serve-path shared-state audit."""

    component: str
    location: str
    state: str
    thread_safe: bool
    note: str = ""


SHARED_STATE: tuple[SharedState, ...] = (
    SharedState("fusion.conformal_set", "server.py run_court",
                "module-global function replaced per request",
                False, "C-1; replaced by alpha_scope"),
    SharedState("argus_detect.USER_ALPHA", "argus_detect.py",
                "module global int", False, "C-3"),
    SharedState("video_face_track._DET", "video_face_track.py get_det",
                "cv2.FaceDetectorYN (setInputSize/setInput per frame)",
                False, "C-4; stateful, must not be shared across threads"),
    SharedState("video_face_track._REC", "video_face_track.py get_rec",
                "cv2.FaceRecognizerSF", False, "C-5"),
    SharedState("video_face_track._ARC_NET", "video_face_track.py get_arcface",
                "cv2.dnn.ReadNet (setInput per call)", False, "C-6; stateful"),
    SharedState("video_face_track.EMBEDDER_NAME", "video_face_track.py set_embedder",
                "module global str", False, "C-7"),
    SharedState("video_face_track._SCORE_THRESHOLD", "video_face_track.py set_score_threshold",
                "module global float + detector rebuild", False, "C-8"),
    SharedState("argus_detect._VID_STACKER_CACHE", "argus_detect.py",
                "dict with loaded-flag set before load", False, "C-9; fail-open on exception"),
    SharedState("argus_detect._AUDIO_MODEL", "argus_detect.py",
                "lazy singleton", False, "C-10"),
    SharedState("registry._cache", "registry.py", "module dict", False, "C-11; vendored"),
    SharedState("channels caches", "channels.py", "module dicts", False, "C-12"),
    SharedState("dinov2 _cache", "dinov2 service", "module singleton", False, "C-13"),
    SharedState("server._state", "server.py", "dict mutated during init", True,
                "C-14; read-only after init_court"),
    SharedState("sys.path", "server.py", "list mutated on import", True,
                "C-15; import-time only"),
    SharedState("torch grad mode", "audio_fusion.py", "global thread-local toggle",
                True, "C-16; torch guards per-thread"),
    SharedState("torch threads/seed", "audio path", "global process state", False, "C-17"),
)


def alpha_scope(alpha: float | None):
    """Context manager binding the conformal α for the current context only.

    Thread-safe replacement for swapping ``fusion.conformal_set``. Because
    ``contextvars`` are copied into ``anyio.to_thread`` workers, concurrent
    requests each see their own α with no shared mutable state.

    Usage::

        with alpha_scope(0.209291):
            core, *_ = fuse(outputs, d, judge=judge)
    """
    return _AlphaScope(alpha)


class _AlphaScope:
    def __init__(self, alpha: float | None):
        if alpha is not None:
            alpha = float(alpha)
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha_scope: alpha must be in (0, 1), got {alpha}")
        self._alpha = alpha
        self._token = None

    def __enter__(self):
        self._token = _CONFORMAL_ALPHA.set(self._alpha)
        return self._alpha

    def __exit__(self, exc_type, exc, tb):
        if self._token is not None:
            _CONFORMAL_ALPHA.reset(self._token)
            self._token = None
        return False


def current_alpha() -> float | None:
    """The α bound by the nearest enclosing :func:`alpha_scope`, or None."""
    return _CONFORMAL_ALPHA.get()


def install_conformal_shim(fusion_module) -> bool:
    """Make the vendored ``conformal_set`` honour :func:`alpha_scope`.

    Installs a wrapper on the *module attribute* only. The vendored file on disk
    is never modified (D-009). Idempotent.

    Returns True if the shim is installed (or was already present), False if the
    module does not expose ``conformal_set``.
    """
    original = getattr(fusion_module, "conformal_set", None)
    if original is None:
        return False
    if getattr(original, SHIM_FLAG, False):
        return True

    def conformal_set(probs, alpha=None):
        if alpha is None:
            alpha = _CONFORMAL_ALPHA.get()
        if alpha is None:
            return original(probs)
        return original(probs, alpha=alpha)

    setattr(conformal_set, SHIM_FLAG, True)
    setattr(conformal_set, "_argus_original", original)
    fusion_module.conformal_set = conformal_set
    return True


def uninstall_conformal_shim(fusion_module) -> bool:
    """Restore the original ``conformal_set``. Idempotent."""
    current = getattr(fusion_module, "conformal_set", None)
    original = getattr(current, "_argus_original", None) if current else None
    if original is None:
        return False
    fusion_module.conformal_set = original
    return True


class locked_lazy:
    """Lazily build a value exactly once, even under concurrent first access.

    Fixes the fail-open pattern where ``loaded = True`` is set before the load
    completes: if the load raises, later callers see a "loaded" singleton that
    was never built.

    Usage::

        _DET = locked_lazy("yunet", _build_detector)
        det = _DET.get()
    """

    def __init__(self, name: str, factory: Callable[[], Any]):
        self.name = name
        self._factory = factory
        self._lock = threading.Lock()
        self._value: Any = None
        self._built = False
        self.build_count = 0

    @property
    def built(self) -> bool:
        return self._built

    def get(self) -> Any:
        if self._built:
            return self._value
        with self._lock:
            if not self._built:
                value = self._factory()
                self.build_count += 1
                self._value = value
                self._built = True
            return self._value

    def reset(self) -> None:
        """Drop the cached value (used when a config key changes)."""
        with self._lock:
            self._value = None
            self._built = False


class thread_local_cache:
    """Per-thread cache for stateful objects (e.g. cv2.dnn nets).

    Provided for S1's measured decision. Not wired by default: converting the
    process-cached YuNet/ArcFace objects to thread-local would create one model
    instance per pool thread, trading memory for safety.
    """

    def __init__(self, name: str, factory: Callable[[], Any]):
        self.name = name
        self._factory = factory
        self._local = threading.local()

    def get(self) -> Any:
        if not hasattr(self._local, "value"):
            self._local.value = self._factory()
        return self._local.value


def shared_state_table() -> list[dict]:
    """The audit as plain dicts (docs, tests, reports)."""
    return [
        {
            "component": r.component,
            "location": r.location,
            "state": r.state,
            "thread_safe": r.thread_safe,
            "note": r.note,
        }
        for r in SHARED_STATE
    ]