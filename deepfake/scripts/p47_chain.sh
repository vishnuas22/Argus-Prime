#!/bin/bash
# P4.7 final assembly chain (user-approved "run as registered", D-046/HOLD cycle).
# B2 candidate (7 shards excluded from train diet) + B3 per-candidate conformal
# recalibration (train-only, D-035) + B4 abstention-parity gate. NO promotion
# parameter passsed — B.5 stays user-worded.
# Dump reuse: AB2 dump CSVs are byte-identical train rows for this diet; fresh
# gate dirs get scored by the post-B3 benchmark code (old at vendored alpha,
# candidate at its meta alpha).
set -euo pipefail
# P0b: ARGUS_ROOT overridable (default: this script's repo-relative root).
ARGUS_ROOT="${ARGUS_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$ARGUS_ROOT"
exec >> /tmp/restack_p47.log 2>&1
echo "==== [p47] chain start $(date -u +%FT%TZ) ===="

SRC=artifacts/restack_r9_ab2
DST=artifacts/restack_p47
mkdir -p "$DST"

n=0
for d in "$SRC"/dump_*/; do
  name=$(basename "$d")
  [ -d "$DST/$name" ] || cp -R "$d" "$DST/$name"
  n=$((n+1))
done
echo "[p47] reused $n AB2 dump dirs -> $DST"

./.venv/bin/python deepfake/scripts/restack.py \
  --work "$DST" \
  --exclude-domains "flux-1-pro,gpt4o,hidream,ideogram3,imagen4,nanao-banana,seedream3-0" \
  --gate-limit 100 &
RPID=$!
caffeinate -is -w "$RPID" &
echo "[p47] restack pid $RPID; caffeinate watching"

if wait "$RPID"; then
  cat > artifacts/restack_p47.DONE <<EOF
completed_at: $(date -u +%FT%TZ)
  work: $ARGUS_ROOT/artifacts/restack_p47
  gates: $ARGUS_ROOT/artifacts/restack_p47/gates.json
diet: 8 base corpora + flux2-klein-4b (7 shards excluded, B2)
verdict: printed by restack stage; orchestrator re-derives independently (spec step 5)
note: NO promotion (B.5 user-worded)
EOF
  echo "[p47] DONE marker written $(date -u +%FT%TZ)"
else
  rc=$?
  printf 'failed_at: %s\nrestack_rc: %s\n' "$(date -u +%FT%TZ)" "$rc" > artifacts/restack_p47.FAILED
  echo "[p47] FAILED marker written (rc=$rc)"
fi
echo "==== [p47] chain end $(date -u +%FT%TZ) ===="
