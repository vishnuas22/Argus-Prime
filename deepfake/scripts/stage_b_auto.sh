#!/bin/bash
# Phase 4 Stage B — unattended remainder launcher (spec B.4; D-049).
#
# Fires automatically AFTER the flux2-klein-4b batch (B.1) exits:
#   1. verify batch outputs (fail-closed -> stage_b_auto.FAILED marker)
#   2. holdout split / legacy real-side repair (idempotent; --split-only)
#   3. B.4: FULL restack r9 dump->train->gate->verdict at --gate-limit 100,
#      work dir artifacts/restack_r9, log /tmp/restack_r9_full.log
#
# INTENTIONALLY OUT OF SCOPE — NEVER AUTOMATED:
#   - B.5 promotion: restack is invoked WITHOUT any promote flag. Promotion
#     (--promote --promote-to p6v11 + argus_detect STACKER rewire) requires
#     EXPLICIT user instruction on a later turn.
#   - B.6 non-regression: scoreboard purge + 5-exam golden/p6v10 comparison,
#     recovery clean-path spotcheck, git-diff asserts — all deferred to an
#     orchestrator-confirmed turn (recorded as TODO in the DONE marker).
set -euo pipefail

WS="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$WS"
PY="$WS/.venv/bin/python"
BATCH_PID=81244
BATCH_LOG=/tmp/gen_flux2.log
RESTACK_LOG=/tmp/restack_r9_full.log
TRAIN_DIR="$WS/data/corpora/gen_intake_train/flux2-klein-4b"
EXAM_FAKE="$WS/data/corpora/gen_exam/flux2-klein-4b/fake"
ART="$WS/artifacts/generator_intake"
FAILED="$ART/stage_b_auto.FAILED"
DONE="$ART/stage_b_auto.DONE"

rm -f "$FAILED" "$DONE"
mkdir -p "$ART"

fail() {
  {
    echo "reason: $1"
    echo "at: $(date -u +%FT%TZ)"
    echo "batch_log: $BATCH_LOG"
    echo "restack_log: $RESTACK_LOG"
  } > "$FAILED"
  echo "[stage_b_auto] FAILED: $1 (marker: $FAILED)" >&2
  exit 1
}

# --- 1. wait for the batch (spec: until ! kill -0 81244, poll 300s) ---------
echo "[stage_b_auto] $(date -u +%FT%TZ) waiting for batch PID $BATCH_PID..."
while kill -0 "$BATCH_PID" 2>/dev/null; do
  # PID-reuse guard: if 81244 is no longer our generator, treat as exited
  ps -p "$BATCH_PID" -o command= 2>/dev/null | grep -q "[g]en_modern_corpus" || break
  sleep 300
done
echo "[stage_b_auto] batch PID $BATCH_PID exited at $(date -u +%FT%TZ)"

# --- 2. output assertions (fail-closed) -------------------------------------
[ -s "$TRAIN_DIR/manifest.jsonl" ] || fail "missing $TRAIN_DIR/manifest.jsonl"
rows=$(wc -l < "$TRAIN_DIR/manifest.jsonl" | tr -d ' ')
[ "$rows" -eq 1100 ] || fail "manifest.jsonl has $rows rows, expected 1100"
n_jpg=$(find "$TRAIN_DIR" -maxdepth 1 -name '*.jpg' | wc -l | tr -d ' ')
if [ "$n_jpg" -eq 1100 ]; then
  # generation complete, split not yet applied
  echo "[stage_b_auto] train dir at 1100 jpgs (split pending)"
elif [ "$n_jpg" -eq 1000 ]; then
  # the batch's built-in legacy split already moved 100 fakes to the exam
  n_exam=$(find "$EXAM_FAKE" -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  [ "$n_exam" -eq 100 ] || fail "train at 1000 jpgs but exam fake has $n_exam (expected 100)"
  echo "[stage_b_auto] legacy split already applied (exam fake $n_exam); real side will be repaired"
else
  if grep -q "Traceback" "$BATCH_LOG" 2>/dev/null; then
    fail "batch log contains Traceback; train dir has $n_jpg jpgs (expected 1000 or 1100)"
  fi
  fail "train dir has $n_jpg jpgs, expected 1000 (post-split) or 1100 (pre-split)"
fi

# --- 3. holdout split / legacy real-side repair (idempotent) ----------------
"$PY" "$WS/deepfake/scripts/gen_modern_corpus.py" --model flux2-klein-4b --split-only \
  || fail "holdout-split (--split-only) failed"

# --- 3b. D-049 carve-out assert: exam reals DISJOINT from gen_train/real ----
EXAM_REAL="$WS/data/corpora/gen_exam/flux2-klein-4b/real"
GEN_REAL="$WS/data/corpora/gen_train/real"
"$PY" - "$EXAM_REAL" "$GEN_REAL" <<'PYEOF' || fail "D-049 carve-out assert failed (exam real overlap with gen_train/real, or bad counts)"
import hashlib, sys
from pathlib import Path

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

exam, train = Path(sys.argv[1]), Path(sys.argv[2])
exam_h = {sha256(p) for p in exam.iterdir() if p.is_file()}
train_h = {sha256(p) for p in train.iterdir()
           if p.is_file() and p.suffix.lower() != ".jsonl"}
assert len(exam_h) == 100, f"exam real count {len(exam_h)} != 100"
assert len(train_h) == 1000, f"gen_train/real count {len(train_h)} != 1000"
overlap = exam_h & train_h
assert not overlap, f"D-049 carve-out violated: {len(overlap)} exam/train real overlap"
print(f"[stage_b_auto] carve-out OK: exam 100, train 1000, overlap 0", flush=True)
PYEOF

# --- 4. B.4 full restack r9 (NO promote flag anywhere — B.5 is user-gated) --
echo "[stage_b_auto] starting B.4 restack r9 -> $RESTACK_LOG"
"$PY" "$WS/deepfake/scripts/restack.py" \
  --work "$WS/artifacts/restack_r9" \
  --gate-limit 100 \
  > "$RESTACK_LOG" 2>&1 \
  || fail "restack r9 exited non-zero (see $RESTACK_LOG)"

[ -s "$WS/artifacts/restack_r9/gates.json" ] \
  || fail "restack r9 exited zero but artifacts/restack_r9/gates.json missing"

# --- 5. DONE marker ---------------------------------------------------------
{
  echo "completed_at: $(date -u +%FT%TZ)"
  echo "gates_summary: $WS/artifacts/restack_r9/gates.json"
  echo "restack_log: $RESTACK_LOG"
  echo "verdict: restack ran WITHOUT any promote flag — candidate stays in artifacts/restack_r9/candidate/; HOLD applies unless user overrides"
  echo "TODO_B5: promotion (copy candidate -> artifacts/p6v11 + rewire argus_detect.STACKER + grep consumers) requires EXPLICIT user instruction — NOT automated"
  echo "TODO_B6: non-regression — purge artifacts/scoreboard/p6v11 cache; 5 exams vs golden (TOL -0.01) AND vs pre-recorded p6v10 (aigc 0.9999 / cdf 1.0000 / ffpp 0.6614 / mj 0.9906 / flux 0.9502); recovery clean-path on 5 eval_real images bit-identical + not_attempted; git diff asserts; final Stage B report (incl. HiDream NO-GO evidence, acquisition/gen/shard tables, deviations, file audit, git status) — deferred to orchestrator-confirmed turn, NOT automated"
} > "$DONE"
echo "[stage_b_auto] DONE (marker: $DONE)"
