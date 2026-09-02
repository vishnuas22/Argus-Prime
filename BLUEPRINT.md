# Argus Prime Blueprint

Orchestrated agent system: Planner–Executor–Critic core, research fan-out, self-updating memory. Built on opencode custom agents + skills + rules. Design grounded in current production practice for multi-agent orchestration: verification-driven loops, blast-radius isolation, anti-drift critics, explicit contracts.

## Architecture

```
            USER ──► Primary agent (opencode Build/Plan)
                        │  routes via skills
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
   argus-workflow  deep-research     debug-loop
        │               │                │
   @architect      @scout ×N         evidence/bisect
   (spec)          (parallel)        
        │               │                
   @builder ◄──── spec              
     (diff + checks)                 
        │                             
   @sentinel (fresh instance)          
        │                             
   memory/* ◄── append lessons/patterns/decisions after every task
```

## Agent contracts

| Agent | Input | Output | Tool surface | Temp |
|---|---|---|---|---|
| scout | research question | attributed report w/ confidence | web search/fetch, read-only | 0.3 |
| architect | goal + constraints | structured spec (steps/files/verify/risks/rollback) | read-only | 0.1 |
| builder | spec verbatim | diff + honest check results | edit + bash | 0.2 |
| sentinel | spec + diff ONLY | APPROVE / REJECT / ESCALATE verdict | git read cmds; other bash ask-gated | 0.1 |

## Threat model → countermeasures

| Threat | Countermeasure |
|---|---|
| Critic drift (rubber-stamping) | Sentinel sees spec+diff only; fresh instance per loop; graded against spec invariants |
| Optimistic success reporting | Mechanical checks mandatory; only observed passes count as done |
| Context evaporation across handoffs | Subagents inherit nothing; specs passed verbatim; durable state in files |
| Blast-radius leaks | Mutation confined to @builder; destructive bash globally ask-gated |
| Runaway loops | Max two reject-cycles → human escalation with both verdicts |
| Hallucinated facts | Scout attribution rules; ≥2 independent sources for load-bearing claims; "unknown" permitted |

## Roadmap

- Phase 1 ✅ Foundation — four subagents, three skills, safety gates, AGENTS.md brainstem, memory templates.
- Phase 2 Hardening — eval harness (golden tasks), `/argus` custom command, budget caps per agent, MCP integrations where they earn their keep.
- Phase 3 Expansion — general-assistant workflows beyond SWE: research pipelines, file management, daily operations.
- Phase 4 Proof — real task through the full loop; measure consistency and quality against unstructured baseline.

## Operating model

Memory compounds: every completed task appends lessons, patterns, decisions. Each session starts smarter than the last. The multiplier comes from engineered context, parallel fan-out, verification catching errors before the user sees them, and memory that never forgets — not from prompt incantations.
