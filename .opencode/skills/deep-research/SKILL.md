---
name: deep-research
description: Full research methodology for questions requiring current external knowledge - technology comparisons, best practices, ecosystem state. Fans out parallel scout queries, cross-references sources, outputs confidence-rated synthesis.
---

# Deep Research

1. SCOPE: convert the question into decision-relevant sub-questions. Confirm scope with the user if it implies a large time investment or many parallel scouts.
2. FAN OUT: dispatch parallel `@scout` tasks, one per independent sub-question. Each prompt carries explicit context: constraints, stack, current date, what decision the answer feeds.
3. CROSS-REFERENCE: merge scout reports. Dedupe overlaps. Resolve conflicts into an explicit contradiction matrix rather than silently picking winners.
4. SYNTHESIZE: recommendation first, then evidence. Per-claim confidence ratings. A GAPS section stating what remains unknown.
5. DELIVER: Summary → Findings (attributed inline: URL + date) → Contradictions → Gaps → Recommendation.
6. REMEMBER: durable facts discovered (e.g., "library X v3 broke API Y") go to `memory/patterns.md` so future sessions inherit them.

## Rules

- Primary sources over commentary; docs and specs over blog posts about them.
- Recency matters: weight sources from the last twelve months highest; flag older ones.
- Two independent confirmations minimum for load-bearing claims.
- "Unknown" beats invented — always.
