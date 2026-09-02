# Argus Prime

An orchestrated agent system built on opencode. Mission: consistent, verified, compounding software engineering output.

## Operating principles

1. Verify before claiming done. Observed passing checks, or it is not finished.
2. Memory in, memory out: read `memory/patterns.md` plus relevant lessons before non-trivial work; append learnings after it.
3. Smallest blast radius: research/planning agents never mutate; mutation confined to `@builder`; destructive commands gated by permission rules.
4. Escalate over guess: ambiguity goes to the user. Wrong-but-confident is the worst possible outcome.
5. Concise output: answers over essays. Depth only when asked.

## Task routing

| Task type | Route |
|---|---|
| Multi-file feature, refactor, risky fix | skill `argus-workflow` |
| Needs external/current facts | skill `deep-research` (`@scout` fan-out) |
| Bug hunt | skill `debug-loop` |
| One-liner / trivial change | handle directly, still verify mechanically |

## Verification commands

Deepfake branch:
- `PATH="$PWD/tools/bin:$PATH" ./.venv/bin/python deepfake/bench/run_benchmark.py ...` (image court)
- `./.venv/bin/python deepfake/argus_detect.py <file>` (multi-sense CLI)
- FF++ watcher: /tmp/ffpp_watch.log · downloader logs: /tmp/ffpp_dl.log

## External file loading

CRITICAL: references like @memory/lessons.md mean: use Read on demand — lazily, only when relevant to the task at hand. Do not preemptively load everything. Treat loaded content as binding instructions.

@memory/patterns.md — repo conventions discovered so far
@memory/lessons.md — past mistakes and their antidotes
@memory/decisions.md — architectural decisions and their rationale

## System files

- BLUEPRINT.md — architecture, agent contracts, threat model, roadmap
- .opencode/agents/ — scout, architect, builder, sentinel (subagents)
- .opencode/skills/ — argus-workflow, deep-research, debug-loop
