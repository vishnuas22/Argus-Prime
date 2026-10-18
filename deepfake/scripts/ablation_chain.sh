#!/bin/bash
# Phase 4 D-046 HOLD-cycle ablation chain — AB1 -> AB2, sequential, NO promote.
#
#   1. wait for AB1 (restack_r9_ab1, no flux-1-pro diet) to exit
#   2. fail-closed: AB1 gates.json must exist
#   3. launch AB2 (restack_r9_ab2, train diet = 7 original + flux2-klein-4b)
#      with caffeinate attached; heartbeat every 300s
#   4. fail-closed: AB2 gates.json must exist -> run ablation_compare.py ->
#      append summary path to the DONE marker
#
# NEVER: touches artifacts/restack_r9/{candidate,gates.json}, --promote, memory.
set -euo pipefail

WS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WS"
PY="$WS/.venv/bin/python"
AB1_PID="${1:-69170}"
AB1_WORK="$WS/artifacts/restack_r9_ab1"
AB2_WORK="$WS/artifacts/restack_r9_ab2"
ART="$WS/artifacts"
CHAIN_LOG=/tmp/ablation_chain.log
AB1_LOG=/tmp/restack_ab1.log
AB2_LOG=/tmp/restack_ab2.log
FAILED="$ART/restack_r9_ablations.FAILED"
DONE="$ART/restack_r9_ablations.DONE"
SUMMARY="$ART/restack_r9_ablations_summary.json"
# AB2 train diet: 7 original domains + flux2 only (every other NEW domain out).
# Matches restack.py --exclude-domains semantics (filters CORPORA train side;
# GATES/TOL untouched -> all 13 gates still evaluated).
AB2_EXCLUDE="flux-1-pro,gpt4o,hidream,ideogram3,imagen4,nanao-banana,seedream3-0"

rm -f "$FAILED" "$DONE"

hb() { echo "[ablation_chain] heartbeat $(date -u +%FT%TZ) $1" >> "$CHAIN_LOG"; }

fail() {
  {
    echo "reason: $1"
    echo "at: $(date -u +%FT%TZ)"
    echo "ab1_log: $AB1_LOG"
    echo "ab2_log: $AB2_LOG"
    echo "chain_log: $CHAIN_LOG"
  } > "$FAILED"
  hb "FAILED reason='$1'"
  echo "[ablation_chain] FAILED: $1 (marker: $FAILED)" >&2
  exit 1
}

echo "[ablation_chain] $(date -u +%FT%TZ) start; waiting for AB1 PID $AB1_PID"
hb "startup ab1_pid=$AB1_PID waiting=yes chain_pid=$$"
while kill -0 "$AB1_PID" 2>/dev/null; do
  # PID+cmdline guard: only treat as live if it is still our AB1 restack
  ps -p "$AB1_PID" -o command= 2>/dev/null | grep -q "[r]estack_r9_ab1" || break
  hb "ab1_pid=$AB1_PID alive=yes"
  sleep 300
done
hb "ab1_pid=$AB1_PID exited"

# --- fail-closed: AB1 must have produced gates.json --------------------------
[ -s "$AB1_WORK/gates.json" ] || fail "AB1 exited without artifacts/restack_r9_ab1/gates.json"
echo "[ablation_chain] AB1 gates.json present ($(wc -c < "$AB1_WORK/gates.json" | tr -d ' ') bytes)"

# --- launch AB2 (direct nohup on python so $! is the restack PID) ------------
echo "[ablation_chain] launching AB2 -> $AB2_LOG"
nohup "$PY" "$WS/deepfake/scripts/restack.py" \
  --work "$AB2_WORK" \
  --gate-limit 100 \
  --exclude-domains "$AB2_EXCLUDE" \
  >> "$AB2_LOG" 2>&1 &
AB2_PID=$!
disown "$AB2_PID" 2>/dev/null || true
sleep 4
kill -0 "$AB2_PID" 2>/dev/null || fail "AB2 (PID $AB2_PID) died within 4s of launch"
caffeinate -is -w "$AB2_PID" >/dev/null 2>&1 &
echo "[ablation_chain] AB2_PID=$AB2_PID caffeinate attached"
hb "ab2_pid=$AB2_PID diet=7orig+flux2 exclude=$AB2_EXCLUDE caffeinate=yes"

while kill -0 "$AB2_PID" 2>/dev/null; do
  ps -p "$AB2_PID" -o command= 2>/dev/null | grep -q "[r]estack_r9_ab2" || break
  hb "ab2_pid=$AB2_PID alive=yes"
  sleep 300
done
hb "ab2_pid=$AB2_PID exited"

[ -s "$AB2_WORK/gates.json" ] || fail "AB2 exited without artifacts/restack_r9_ab2/gates.json"

# --- comparison + DONE --------------------------------------------------------
"$PY" "$WS/deepfake/scripts/ablation_compare.py" \
  || fail "ablation_compare.py exited non-zero"
[ -s "$SUMMARY" ] || fail "summary $SUMMARY not produced"

{
  echo "completed_at: $(date -u +%FT%TZ)"
  echo "ab1_gates: $AB1_WORK/gates.json"
  echo "ab2_gates: $AB2_WORK/gates.json"
  echo "summary: $SUMMARY"
  echo "ab1_pid: $AB1_PID"
  echo "ab2_pid: $AB2_PID"
  echo "note: diagnostic ablation only — r9 workdir/candidate untouched, NO promotion, memory not written"
} > "$DONE"
hb "DONE summary=$SUMMARY"
echo "[ablation_chain] DONE (marker: $DONE)"
