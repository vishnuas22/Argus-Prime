#!/bin/bash
# Phase 4 D-046 AB3 control — pure-retrain ablation (7 original domains, zero new).
#
#   1. launch AB3 (restack_r9_ab3, --exclude-domains = all 8 NEW domains)
#      with caffeinate attached; heartbeat every 300s
#   2. fail-closed: AB3 gates.json must exist on exit
#   3. run ab3_readout.py -> artifacts/restack_r9_ab3_readout.json (c1/c2)
#      -> append readout path to the DONE marker
#
# NEVER: touches artifacts/restack_r9/{candidate,gates.json}, --promote, memory.
set -euo pipefail

WS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WS"
PY="$WS/.venv/bin/python"
AB3_WORK="$WS/artifacts/restack_r9_ab3"
ART="$WS/artifacts"
CHAIN_LOG=/tmp/ablation_chain3.log
AB3_LOG=/tmp/restack_ab3.log
FAILED="$ART/restack_r9_ab3.FAILED"
DONE="$ART/restack_r9_ab3.DONE"
READOUT="$ART/restack_r9_ab3_readout.json"
# AB3 train diet: 7 original domains, ZERO new (pure retrain control).
AB3_EXCLUDE="flux-1-pro,gpt4o,hidream,ideogram3,imagen4,nanao-banana,seedream3-0,flux2-klein-4b"

rm -f "$FAILED" "$DONE"

hb() { echo "[ablation_chain3] heartbeat $(date -u +%FT%TZ) $1" >> "$CHAIN_LOG"; }

fail() {
  {
    echo "reason: $1"
    echo "at: $(date -u +%FT%TZ)"
    echo "ab3_log: $AB3_LOG"
    echo "chain_log: $CHAIN_LOG"
  } > "$FAILED"
  hb "FAILED reason='$1'"
  echo "[ablation_chain3] FAILED: $1 (marker: $FAILED)" >&2
  exit 1
}

# --- launch AB3 (direct nohup on python so $! is the restack PID) ------------
echo "[ablation_chain3] $(date -u +%FT%TZ) launching AB3 -> $AB3_LOG"
hb "startup launching ab3 diet=7orig exclude=$AB3_EXCLUDE chain_pid=$$"
nohup "$PY" "$WS/deepfake/scripts/restack.py" \
  --work "$AB3_WORK" \
  --gate-limit 100 \
  --exclude-domains "$AB3_EXCLUDE" \
  >> "$AB3_LOG" 2>&1 &
AB3_PID=$!
disown "$AB3_PID" 2>/dev/null || true
sleep 4
kill -0 "$AB3_PID" 2>/dev/null || fail "AB3 (PID $AB3_PID) died within 4s of launch"
caffeinate -is -w "$AB3_PID" >/dev/null 2>&1 &
echo "[ablation_chain3] AB3_PID=$AB3_PID caffeinate attached"
hb "ab3_pid=$AB3_PID caffeinate=yes"

while kill -0 "$AB3_PID" 2>/dev/null; do
  ps -p "$AB3_PID" -o command= 2>/dev/null | grep -q "[r]estack_r9_ab3" || break
  hb "ab3_pid=$AB3_PID alive=yes"
  sleep 300
done
hb "ab3_pid=$AB3_PID exited"

# --- fail-closed: AB3 must have produced gates.json --------------------------
[ -s "$AB3_WORK/gates.json" ] || fail "AB3 exited without artifacts/restack_r9_ab3/gates.json"

# --- readout (c1/c2) + DONE ---------------------------------------------------
"$PY" "$WS/deepfake/scripts/ab3_readout.py" || fail "ab3_readout.py exited non-zero"
[ -s "$READOUT" ] || fail "readout $READOUT not produced"

{
  echo "completed_at: $(date -u +%FT%TZ)"
  echo "ab3_gates: $AB3_WORK/gates.json"
  echo "readout: $READOUT"
  echo "ab3_pid: $AB3_PID"
  echo "note: diagnostic control — r9 workdir/candidate untouched, NO promotion, memory not written"
} > "$DONE"
hb "DONE readout=$READOUT"
echo "[ablation_chain3] DONE (marker: $DONE)"
