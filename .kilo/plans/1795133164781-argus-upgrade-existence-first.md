# Production-Readiness Plan — Argus-Prime

**Status:** Implementation-ready plan. No source edits made.
**Goal:** make Argus-Prime deployable by a third party.
**Detection score:** unchanged (64.8/100) — this plan is orthogonal to accuracy.
**Full to-do list:** `memory/prod_readiness_2026-11-22.md`.

---

## 1. Verified starting position

| Capability | State |
|---|---|
| Fresh-clone boot | ✅ private HF bundle + `bootstrap.py --verify` |
| Machine paths | ✅ 0 hits in tracked code |
| Licence | ✅ MIT |
| Fast tests + CI | ✅ 48 passed, two-tier |
| Upload cap / 400 / CORS | ✅ 100 MB, 413, 400, CORS allow-list |
| **Cross-origin POST gate** | ⚠️ partial — `_origin_gate` (server.py:63-74) blocks *browser* cross-origin POST, but **not** a non-browser client (no `Origin` header ⇒ allowed) |
| **Auth** | ❌ none |
| **Rate limiting** | ❌ none |
| **Metrics / structured logs** | ❌ `print()` only |
| **API tests** | ❌ none |
| **Container** | ❌ none |
| **Graceful drain** | ❌ none |
| **readz/livez** | ❌ single `/api/health` |
| **Concurrency** | ❌ `CapacityLimiter(1)`; raising it crashes (D-118) |

---

## 2. Task list (ordered; dependencies only, no time estimates)

### A1 — Authentication (blocks everything)
**Problem:** `_origin_gate` stops browsers, not scripts. Anyone who can reach the port can spend GPU and read `/api/history` (returns up to 200 prior verdicts).

**Design (OWASP FastAPI Cheat Sheet + FastAPI docs):**
- `deepfake/api/auth.py`: `APIKeyHeader(name="X-API-Key", auto_error=False)` + `Security(...)`.
- Keys from `ARGUS_API_KEYS` (comma-separated) or `ARGUS_API_KEYS_FILE`. Compare with `hmac.compare_digest`.
- **Fail-closed:** `ARGUS_REQUIRE_AUTH=1` with no keys ⇒ refuse to start (never run open while claiming auth).
- **Default:** unset ⇒ auth disabled, preserving current local behaviour (zero blast radius).
- Apply via `dependencies=[Security(require_key)]` on `/api/detect`, `/api/explain`, `/api/history`.
- `/api/health` stays open for orchestrator probes.
- **Do not touch** vendored `integrations/` (D-009).

**Acceptance:** no key → 401 · valid key → 200 · `ARGUS_REQUIRE_AUTH=1` with no keys → boot fails with a clear message.

### A2 — Rate limiting (depends on A1)
**Problem:** one client starves all others; the court is single-threaded.

**Design (OWASP "Step 3" — no new dependency):**
- `deepfake/api/ratelimit.py` — sliding-window limiter, ~40 lines.
- Key on **API key** when present, else client IP (IP alone is wrong behind NAT).
- Limits from `ARGUS_RATE_LIMIT` (e.g. `10/minute`) + `ARGUS_RATE_BURST`.
- **Moving window, not fixed** — fixed windows leak badly (published replay: 19 requests through a 10/min limit).
- In-process store is correct only for one worker; document that multi-worker needs Redis.

**Acceptance:** Nth request → 200; N+1th → **429** with `Retry-After`.

### A3 — Container image (depends on A1, A2)
- Multi-stage `Dockerfile`, non-root user, `ARGUS_ARTIFACTS_DIR` as a volume.
- Entrypoint runs `bootstrap.py --verify` and fails fast if the bundle is missing.
- `.dockerignore` excludes `data/`, `logs/`, `.venv/`, `artifacts/` (bundle is fetched, not baked).
- `docker-compose.yml` for local dev with a `/api/health` healthcheck.

**Acceptance:** `docker build` → `docker run` → `/api/health` 200 after bundle fetch.

### A4 — Metrics + structured logs (depends on A3)
- `GET /metrics` in Prometheus text format (no new heavy dependency): per-modality `p50`, `p95`, `abstention_rate`, `error_rate`, `request_total{status}`.
- Structured JSON logs (one line per request: `request_id`, `path`, `status`, `elapsed_ms`, `modality`, `abstained`, `alpha`).
- Keep `print()` out of the request path.

**Acceptance:** `curl /metrics` returns parseable metrics; logs are one JSON object per line.

### A5 — API test suite (depends on A1, A2)
- `tests/test_api_auth.py` (fast): 401 without key, 200 with key, 429 on burst — using `dependency_overrides`, no real models.
- `tests/test_api_detect.py` (slow): one real detect per modality against a live app.
- `tests/test_api_contract.py` (fast): OpenAPI shape — required fields present, no secrets in schema.

**Acceptance:** `pytest -m fast` includes the auth + contract tests; `pytest -m slow tests/test_api_detect.py` green with the bundle.

### A6 — Graceful shutdown / drain (depends on A3)
- Track in-flight requests; on SIGTERM stop accepting, drain with a timeout, then exit.
- Return `503` for requests that arrive after drain starts.

**Acceptance:** SIGTERM mid-request ⇒ the request completes or returns 503; never a partial verdict.

### A7 — ArcFace → SFace (commercial only, depends on U8)
- Already priced: **~0.013 AUC** (D-122/D-123).
- Requires a sealed restack spec (p6vid_v4 → v5) with its own bars.

**Acceptance:** no NC weight in the commercial image; `WEIGHTS-NOTICES.md` updated.

### B1 — Concurrency (depends on S1-scaffold, already done)
- Per-thread YuNet/ArcFace nets (C-4/C-6) · per-request embedder (C-3) · lazy-singleton locks spanning the full load (C-4..C-8) · vendored caches harness-side (C-11..C-13).
- **Do not raise `CapacityLimiter(1)` until the 8-way race test passes.**

**Acceptance:** `pytest -m slow tests/test_concurrency_races.py` green *with the limiter raised*.

### B2 — ONNX / OpenCV-DNN export (depends on B1)
- Export frozen channels; parity gate `max|p_onnx − p_torch| < 1e-3`.

### B3 — `/api/batch` (depends on B2)
- Length-aware grouping for bulk scoring; never used for single-clip latency.

### B4 — readz / livez (depends on A3)
- `/livez` = process alive; `/readyz` = court built + bundle verified + auth configured.

### B5 — Request IDs + audit trail (depends on A4)
- Every verdict gets an id; `/api/history` filterable by id.

### B6 — Config validation at boot (depends on A1)
- Fail fast on bad env (invalid rate-limit string, missing bundle, auth misconfig).

### B7 — Docs (depends on A5)
- OpenAPI at `/docs`, deployment guide, runbook, `LIMITATIONS.md` refresh.

### B8 — SBOM + licence scan in CI (depends on A3)

---

## 3. Risks

| Risk | Mitigation |
|---|---|
| Auth breaks the local dev flow | Default OFF (`ARGUS_REQUIRE_AUTH` unset); CI runs both ways |
| Rate limiter in-process lies under multiple workers | Documented; Redis path named for multi-worker |
| Container bloat / macOS-only deps | Multi-stage; mflux/diffusers excluded (training extra) |
| Raising the limiter crashes | Gated behind the 8-way race test (D-118) |
| ArcFace swap regresses video | Sealed restack spec with FF++/CDF/DFD bars |
| Scope creep into detection accuracy | This plan is production-only; detection items stay in `backlog_2026-11-21.md` §C |

## 4. Out of scope

- Any detection-accuracy improvement (V1d-2, R1/R2, AB1/AB2, MM1, D1–D3).
- Audio modality replacement (product-scope decision C1).
- Multi-region / HA deployment (single-box first).
- GPU scheduling.

## 5. Open questions (owner)

1. **U8** commercial vs research posture → decides whether A7 (SFace) is mandatory.
2. **C1** audio modality: ship with a research badge, or remove from the product story?
3. **Deployment target**: single box, or orchestrated (k8s)? Decides whether the in-process limiter is enough.
7. **Auth model**: static API keys, or OIDC/JWT for client SSO?