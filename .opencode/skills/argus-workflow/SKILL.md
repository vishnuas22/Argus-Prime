---
name: argus-workflow
description: Master build loop for non-trivial tasks - plan with architect, get user confirmation, build with builder, verify with fresh sentinel, log lessons to memory. Use for features, refactors, multi-file fixes, or any work lacking obvious verification.
---

# Argus Workflow

The default loop for work that deserves rigor. Trivial-task escape hatch defined below.

## Loop

0. LOAD: read `memory/patterns.md` fully; skim `memory/lessons.md` headings for relevance; read `AGENTS.md`.
1. FRAME: restate the goal plus success criteria in three lines or fewer. Ask now if anything material is unclear.
2. PLAN: dispatch `@architect`. Present its spec to the user compactly. WAIT for confirmation unless the user pre-authorized autonomy.
3. BUILD: dispatch `@builder` with the confirmed spec verbatim — subagents inherit nothing from this conversation.
4. VERIFY: dispatch a FRESH `@sentinel` with spec + `git diff` only — never builder's report.
5. LOOP CONTROL: on REJECT, fix via `@builder`, then review with a new fresh `@sentinel`. Maximum two cycles; after that, ESCALATE to the human with both verdicts attached.
6. CLOSE: all checks green → summarize honestly: steps completed, deviations from spec, residual risks.
7. REMEMBER: append dated entries — `memory/lessons.md` if something went wrong or surprised us; `memory/patterns.md` if we learned a repo convention; `memory/decisions.md` for architectural choices with rationale.

## Trivial-task heuristic (steps 2 and 4 may be skipped)

ALL must be true: single file · fewer than ~30 lines changed · no new dependencies · obvious mechanical verification exists.
Even then: run the mechanical check yourself, and still do step 7.

## Non-negotiables

- Nothing is done because it looks done. Done = checks observed passing.
- Never mark a todo complete while a check fails.
- The user confirms the plan before mutation, unless they explicitly delegated that gate.
