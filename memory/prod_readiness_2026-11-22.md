# Argus-Prime — Production-Readiness To-Do (authoritative)

**Generated:** 2026-11-22. Supersedes prior production framing.
**Purpose:** convert a research prototype into something a client can run.
**Court:** 64.8/100 (detection score — NOT the subject of this list).
**Rule:** only items with a **clear contribution to production readiness**. Items that only chase the detection score without moving it are excluded.
**Stop ruling (D-105):** training/promotion/pointer change/commit still need approval.

---

## Current state (verified today, 2026-11-22)

| Capability | Status | Evidence |
|---|---|---|
| Fresh-clone boot | ✅ **DONE** (P0a, D-120) | private HF bundle, pinned rev, `bootstrap.py --verify` → ALL-SHAS-OK |
| No machine paths | ✅ **DONE** (P0b, D-108) | `git grep` over tracked code = 0 hits |
| Licence | ✅ **MIT** (U6, D-114) | `LICENSE` + `WEIGHTS-NOTICES.md` |
| Fast test tier + CI | ✅ **DONE** (P0e, D-108) | `pytest -m fast` 48 passed; two-tier CI |
| Calibration plumbing | ✅ **DONE** (D-109), dormant | flag OFF by default; fitting showed it hurts (D-111) |
| Thread-safety primitives | ✅ **DONE** (D-110) | `threadsafe.py` + 14 tests; limiter NOT raised |
| Upload cap | ✅ 100 MB, 413 | `MAX_UPLOAD`, server.py:37,520 |
| Input validation | ✅ 400 on unparseable | server.py:537-540 |
| CORS | ✅ configured | server.py:54 |
| Health endpoint | ✅ 11 fields | status/channels/stacker/alpha/instruments/ledger |
| **Auth** | ❌ **absent** | no API key, no bearer, no mTLS |
| **Rate limiting** | ❌ **absent** | only `CapacityLimiter(1)` (concurrency, not rate) |
| **Metrics/tracing** | ❌ **absent** | no Prometheus, no OpenTelemetry; `print()` only |
| **API tests** | ❌ **absent** | no `tests/test_api*.py`; only module-level tests |
| **Container** | ❌ **absent** | no Dockerfile, no compose, no .dockerignore |
| **Structured logging** | ❌ **absent** | `print()` to stdout/stderr |
| **Graceful shutdown** | ⚠️ partial | uvicorn handles SIGTERM; no drain of in-flight |
| **Concurrency** | ❌ serial | `CapacityLimiter(1)`; raising it CRASHES (D-118) |
| **ArcFace licence** | ⚠️ NC | blocks commercial video; SFace swap = ~0.013 AUC (D-122/123) |
| **Audio wild** | ❌ broken | AUC 0.5593, abstains 83.9% (not a prod blocker, a product-scope blocker) |

---

## A. Production blockers (must fix before any client deployment)

| # | Item | Why it blocks | Risk if skipped |
|---|---|---|---|
| **A1** | **Authentication** | anyone who can reach the port can spend GPU and read `/api/history` | open compute, data exposure |
| **A2** | **Rate limiting** | one client can starve all others (single-threaded court) | DoS, no fair use |
| **A3** | **Container image** | no reproducible deploy unit; every host drifts | "works on my machine" |
| **A4** | **Metrics + structured logs** | cannot see p50/p95, error rate, abstention rate in prod | blind ops |
| **A5** | **API test suite** | zero tests on the HTTP surface; any refactor can break it silently | regression |
| **A6** | **Graceful shutdown / drain** | in-flight requests dropped on SIGTERM | partial verdicts |
| **A7** | **ArcFace → SFace** (commercial only) | NC weight blocks commercial video | legal exposure |

## B. Production hardening (should fix)

| # | Item | Why |
|---|---|---|
| **B1** | Concurrency: per-thread YuNet/ArcFace + α ContextVar + lazy-singleton locks | raising the limiter currently crashes (D-118); needed for any throughput |
| **B2** | ONNX / OpenCV-DNN export per frozen channel | 9.7 s/clip is too slow for interactive use |
| **B3** | `/api/batch` endpoint with length-aware grouping | bulk scoring is the realistic workload |
| **B4** | Health readiness vs liveness split | orchestrators need `/readyz` vs `/livez` |
| **B5** | Request IDs + audit trail per verdict | forensics needs traceability |
| **B6** | Config validation at boot (fail fast on bad env) | misconfig currently surfaces as a runtime 500 |
| **B7** | Docs: API reference (OpenAPI), deployment guide, runbook | client handover |
| **B8** | `WEIGHTS-NOTICES` → SBOM + licence scan in CI | supply-chain hygiene |

## C. Product-scope decisions (owner, not engineering)

| # | Item | Decision needed |
|---|---|---|
| **C1** | Audio modality | **ship as-is with a "research" badge, or remove from the product story?** At AUC 0.56 in wild it is a liability. |
| **C2** | Commercial posture | research-only vs commercial. If commercial → A7 (SFace) is mandatory. |
| **C3** | Freshness policy | what does the client do about 2026 generators? Currently near chance. |
| **C4** | Support model | who answers when it abstains on a client's content? |

## D. Detection-score items (NOT production — excluded here)

Everything in `memory/backlog_2026-11-21.md` §C (V1d-2, S1-promote, R1/R2, AB1/AB2, MM1, A4, D1-D3) is **detection accuracy**, not production readiness. It is tracked there, not here. **None of it blocks a deployment** — they block a *better* score.

## E. Closed — do not reopen

C1 calibration (hurts) · V1d ReStraV (B2 fail) · V1h router (no label-free signal) · V1d-2 V-JEPA2 (FF++ chance) · V1c ASMR (non-face) · GenD (CDF −0.116, ensemble −0.0004) · D3/SPLIT/WaveRep · A1 A0z · P8a DF-Eval (rejected) · S1-promote pilot (crashes).

## F. Recommended order

```
A1 Auth ──┐
A2 Rate   ├──> A3 Container ──> A4 Metrics/Logs ──> A5 API tests ──> A6 Drain ──> [A7 if commercial]
          │
          └──> B4 readz/livez, B5 request IDs, B6 config validation, B7 docs
                     |
                     v
              B1 concurrency -> B2 ONNX -> B3 batch (throughput, not correctness)
```

**A1→A6 are the minimum for a client handover.** B1–B3 are throughput/scale, not correctness. C1–C4 are your calls.

## G. What "production ready" means here (acceptance)

1. `docker build` → `docker run` → `/livez` 200, `/readyz` 200 after bundle fetch.
2. Auth: request without a key → 401; with a valid key → 200.
8. Rate limit: N+1 requests in a window → 429 on the N+1th.
4. Metrics: `GET /metrics` returns p50/p95, error rate, abstention rate per modality.
5. `pytest -m fast` green in CI; `pytest tests/test_api*.py` green against a live app.
6. SIGTERM mid-request → request completes or returns 503, never a partial verdict.
7. OpenAPI at `/docs` matches the implementation.
8. A deployment runbook a third party can follow without reading the source.

## H. Score reconciliation

- Detection score stays **64.8/100**. This list does not change it.
- Production readiness is orthogonal: a client cares about A1–A6, not the AUC.
- Both rubrics agree ~70–78 is the public-data detection ceiling.