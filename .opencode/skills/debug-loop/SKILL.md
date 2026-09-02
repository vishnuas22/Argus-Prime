---
name: debug-loop
description: Systematic debugging methodology - reproduce minimally, isolate via bisection, test ranked hypotheses one variable at a time, minimal root-cause fix, mandatory regression test, lesson logged to memory.
---

# Debug Loop

1. REPRODUCE: exact command plus observed vs expected behavior. Reduce to the smallest reliable reproduction. If you cannot reproduce, gather environment details first — no guessing from vibes.
2. EVIDENCE: read errors, logs, and stack traces fully before theorizing. Bisect where useful: git history, binary-search over config or code space. Instrument if needed.
3. HYPOTHESES: rank by likelihood times cheapness-to-test. Test ONE variable at a time. Discard hypotheses on evidence, not attachment.
4. FIX: smallest change that addresses the ROOT CAUSE. Symptom patches are forbidden unless explicitly requested as an interim measure.
5. REGRESSION: add a test that fails without the fix and passes with it. No regression test means not fixed.
6. CLOSE: run the full suite. Then append to `memory/lessons.md`: root cause → misleading symptom → the check or heuristic that would catch this class of bug earlier.

## Banned anti-patterns

- Shotgun edits: multiple simultaneous changes hoping one lands.
- "Let me just try random things" sequences without evidence between attempts.
- Declaring victory when the bug is hidden rather than fixed.
