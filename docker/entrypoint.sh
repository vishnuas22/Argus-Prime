#!/bin/sh
# Argus-Prime container entrypoint (TASK-A3).
#
# Verifies (and, if an HF token is present, fetches) the artifact bundle BEFORE
# starting the API. Fails closed: a missing bundle is a startup error, never a
# silently-degraded serve path.
set -eu

cd "$ARGUS_ROOT"

echo "[entrypoint] verifying artifact bundle at $ARGUS_ARTIFACTS_DIR"

if ! python deepfake/bootstrap.py --verify; then
    echo "[entrypoint] artifact bundle missing or unverified." >&2
    if [ -n "${HF_TOKEN:-}" ] && [ -n "${ARGUS_ARTIFACTS_REPO:-}" ]; then
        echo "[entrypoint] HF_TOKEN and ARGUS_ARTIFACTS_REPO set — fetching." >&2
        if ! python deepfake/bootstrap.py; then
            echo "[entrypoint] fetch failed; refusing to start." >&2
            exit 1
        fi
    else
        echo "[entrypoint] set ARGUS_ARTIFACTS_REPO + HF_TOKEN, or mount a" >&2
        echo "[entrypoint] pre-populated bundle at ARGUS_ARTIFACTS_DIR." >&2
        exit 1
    fi
fi

echo "[entrypoint] bundle OK; starting API on :${ARGUS_PORT}"

exec python -m uvicorn deepfake.api.server:app \
    --host 0.0.0.0 \
    --port "${ARGUS_PORT}" \
    --workers 1