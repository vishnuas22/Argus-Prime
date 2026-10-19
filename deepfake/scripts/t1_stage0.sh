#!/bin/bash
# T1 stage-0 chain — wild-real FPR measurement (pre-registered spec):
# artifacts/generator_intake/t1_wild_fpr_spec.md · deployed stacker p6v11
# Executes ONLY via idle-court; sha-disjoint assert gates the scoring run.
set -uo pipefail
cd /Users/Dataghost/Argus_Prime
echo "[t1] assert start $(date -u +%FT%TZ)"
./.venv/bin/python deepfake/scripts/t1_assert.py || {
  echo "[t1] ASSERT FAILED — scoring refused (fail-closed)"
  echo "FAILED $(date -u +%FT%TZ) (assert)" > artifacts/t1_stage0.FAILED
  exit 1
}
echo "[t1] scoring start $(date -u +%FT%TZ) real_test+fake_test limit=2000 no-ladder stacker=p6v11 --channels all (spec-registered; omission would silently degrade to 5 core modules)"
PATH="$PWD/tools/bin:$PATH" caffeinate -i ./.venv/bin/python deepfake/bench/run_benchmark.py \
  --real data/corpora/wild_sample/real_test \
  --fake data/corpora/wild_sample/fake_test \
  --no-ladder --channels all --limit 2000 \
  --with-stacker artifacts/p6v11/stacker.joblib \
  --out artifacts/exam_wild_real
rc=$?
echo "[t1] bench rc=$rc"
if [ $rc -eq 0 ]; then
  echo "DONE $(date -u +%FT%TZ)" > artifacts/t1_stage0.DONE
else
  echo "FAILED $(date -u +%FT%TZ) (bench rc=$rc)" > artifacts/t1_stage0.FAILED
fi
