#!/bin/bash
# V1d resumable watchdog: restart the stage if it dies, until its output is complete.
# Usage: nohup bash deepfake/scripts/v1d_watchdog.sh > /tmp/v1d_watch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

STAGE="${1:-score}"
case "$STAGE" in
  score)  SCRIPT=deepfake/scripts/v1d_score.py;    OUT=artifacts/v1d_restrav/scores.jsonl;   TARGET=1651 ;;
  feats)  SCRIPT=deepfake/scripts/v1d_features.py; OUT=artifacts/v1d_restrav/features.jsonl; TARGET=1651 ;;
  *) echo "unknown stage $STAGE"; exit 2 ;;
esac

ATTEMPT=0
while true; do
  HAVE=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  if [ "$HAVE" -ge "$TARGET" ]; then
    echo "[watchdog] $STAGE complete ($HAVE/$TARGET)"
    exit 0
  fi
  ATTEMPT=$((ATTEMPT+1))
  echo "[watchdog] attempt $ATTEMPT: $STAGE at $HAVE/$TARGET — launching"
  ./.venv/bin/python "$SCRIPT" >> "/tmp/v1d_${STAGE}.log" 2>&1
  RC=$?
  echo "[watchdog] $STAGE exited rc=$RC; will resume from $(wc -l < "$OUT" 2>/dev/null || echo 0)"
  sleep 10
done