# Argus-Prime — MASTER To-Do (authoritative)

**Generated:** 2026-11-24. Supersedes `backlog_2026-11-21.md`, `backlog_2026-11-22.md`, `prod_readiness_2026-11-22.md`, `MASTER_TODO_2026-11-22.md`.
**Court:** **64.8/100** measured and booked. Target 70 = **+5.2**.
**Score baseline:** repo ledger 64.8 (NOT the external 42/100 audit).
**STOP ruling (D-105):** training / wiring / promotion / pointer change / commit need explicit approval.
**Latest decisions:** D-101 → D-112b. **Git HEAD:** `2da9036` (A1/A2; message is mislabelled `WaveSP-Net.pt.training` but the content is auth+ratelimit+server+decisions).

---

## 0. Done — do not redo

| Area | Result |
|---|---|
| **P0a** artifact bundle | private HF `thedataghost22/Argus_prime` @ `1699da0f`; fresh clone boots identically |
| **P0b–P0e** | config seam, 21 files de-hardcoded, `init_court` guard, requirements split, pytest markers, two-tier CI |
| **P0f** | LICENSE (MIT) + WEIGHTS-NOTICES + README |
| **U6/U7/U8** | MIT · K1-binds/ECE-report-only · commercial achievable at ~0.013 AUC |
| **C1** | scaffold done; **fit CLOSED 0** (calibration hurts, D-111) |
| **S1-scaffold** | threadsafe primitives + α ContextVar; bug fixes (R2 α, conformal.alpha, lazy-singleton fail-open); **limiter NOT raised** |
| **Detection** | **EXHAUSTED** — temporal stats, ReStraV, V-JEPA2, GenD, ASMR image-court, MotionPhys (no code), WaveSP-Net (CUDA+gated) all measured/closed |
| **IDC** | **CLOSED** — scale artifact (D-130) |
| **Reliability table** | **MEASURED** (D-131): `learned_aigc_probe` 0.955 vs hardcoded 0.5; 8/11 channels mis-weighted. **Needs your approval to promote.** |
| **A1 auth** | **DONE** — `deepfake/api/auth.py`, verified live (401/200) |
| **A2 rate limit** | **DONE** — `deepfake/api/ratelimit.py`, moving window, per-key isolation proven |
| **A3 container** | **BUILT + ACCEPTED** — 11.7 GB, auth+rate-limit+bootstrap verified in-container |
| **Portability fixes** | requirements pinned to models.lock (D-112b); `model_cache.py` offline/fail-fast; `ARGUS_DB_PATH` separates mutable state |

---

## A. Production — remaining (client handover blockers)

| # | Item | Why it matters | Cost |
|---|---|---|---|
| **A4** | **Metrics + structured logs** (`/metrics` Prometheus text; one JSON line per request) | cannot see p50/p95, error rate, abstention rate in prod | ~2 h |
| **A5** | **API contract tests** (`tests/test_api_contract.py`, fast) — required fields, no secrets in schema | zero tests on the HTTP surface; any refactor can break it silently | ~1 h |
| **A6** | **Graceful shutdown / drain** — stop accepting, drain in-flight, 503 after | in-flight requests dropped on SIGTERM | ~1 h |
| **B4** | **readz / livez split** (`/livez` alive, `/readyz` court built + bundle verified + auth configured) | orchestrators need to distinguish "starting" from "dead" | ~30 min |
| **B5** | **Request IDs + audit trail** per verdict | forensics needs traceability | ~1 h |
| **B6** | **Config validation at boot** beyond auth (bad rate-limit string, missing bundle) | misconfig currently surfaces as a runtime 500 | ~30 min |
| **B7** | **Docs** — OpenAPI at `/docs`, deployment guide, runbook | client handover | ~2 h |
| **B8** | **SBOM + licence scan in CI** | supply-chain hygiene | ~1 h |

**Minimum for client handover: A4, A5, A6, B4, B7.**

## B. Production — scale (not correctness)

| # | Item | Blocker |
|---|---|---|
| **B1** | Concurrency: per-thread YuNet/ArcFace (C-4/C-6), per-request embedder (C-3), lazy-singleton locks (C-4..C-8), vendored caches harness-side (C-11..C-13) | raising `CapacityLimiter(1)` **crashes today** (D-118) |
| **B2** | ONNX / OpenCV-DNN export + parity <1e-3 | 9.7 s/clip is too slow for interactive use |
| **B3** | `/api/batch` with length-aware grouping | bulk scoring is the realistic workload |
| **A7** | ArcFace → SFace (commercial only) | NC weight blocks commercial video; priced ~0.013 AUC |

## C. Detection — gated, low probability

| # | Item | Realistic Y | P(success) |
|---|---|---|---|
| **C1** | **Reliability-table promotion** (measured D-131) | abstention only, not AUC | med |
| **C2** | **V1d-2** new video-OOD signal class | +4.8–5.8 if bars clear | low |
| **C3** | **A4** 2025–26 open audio model with published ITW FPR+ECE | unknown | unknown |
| **C4** | **R1** codec-diverse retraining | +1.5 if retention bars pass | low |
| **C5** | **R2** compression detector + 2nd threshold | +0.5–1.0 | low |
| **C6** | **AB1/AB2** abstention retries | +1.0–1.5 / +0.7–1.0 | low |
| **C7** | **MM1** new AV signal | +0.7 | low |
| **C8** | **D1** disjoint licence-clean provider-real corpus | +0.5–1.0 data | med |
| **C9** | **D2/D3/D4** AV-Deepfake1M++ / LAV-DF / AIGIBench | data only | low (disk/gated) |
| **C10** | **V1c** Video Reality Test (MIT, 4.3 GB) | guard only, 0 score | — |
| **C11** | **V4** AIGVDBench (378 GB) | data + OOD-video | **disk-blocked** (~60 GiB free) |

## D. Blocked — needs you

| Item | Blocker |
|---|---|
| **P0a extensions** (HF hosting of ArcFace) | owner decision — currently excluded as NC |
| **U9** audio modality: ship with a "research" badge, or remove from the product story? | product scope |
| **U10** auth model: static API keys vs OIDC/JWT? | A1 design (static keys shipped; OIDC would replace) |
| **U11** deployment target: single box vs orchestrated? | A2 (in-process vs Redis limiter) |
| **U12** reliability-table promotion | your approval + sealed spec |

## E. Closed at 0 — do not re-litigate without a new pre-registered hypothesis

C1 calibration (hurts) · V1d ReStraV (B2 fail) · V1h router (no label-free signal) · V1d-2 V-JEPA2 (FF++ chance) · V1c/V1c-2 ASMR (non-face, 20% FPR) · GenD (CDF −0.116, ensemble −0.0004) · IDC (scale artifact) · D3/SPLIT/WaveRep (temporal stats) · A1 A0z (G3 23/300) · P8a DF-Eval (rejected) · S1-promote (crashes) · MLLM-as-detector (chance) · HOLA · per-generator heads · ASVspoof5 standalone · WaveSP-Net (CUDA + gated) · MotionPhys (no code) · chasing 100/100.

## F. Score reality

- Detection: **64.8/100**. Every model-replacement path measured and closed.
- **No live detection lead remains** except the reliability-table promotion (abstention, not AUC) and a genuinely new hypothesis.
- Production readiness is **orthogonal** to score: A4/A5/A6/B4/B7 block a client handover, not accuracy.
- Public-data ceiling ~73–76; 77–78 needs a new video-OOD channel; 100 unreachable.

## G. Recommended order

```
A4 metrics ──> A5 API tests ──> A6 drain ──> B4 readz/livez ──> B7 docs
   (all fast, no models needed, all improve real operability)

then, if pursuing scale:  B1 concurrency ──> B2 ONNX ──> B3 batch
then, if pursuing score:  C1 reliability promotion (your approval)
```

**A4→B7 are ~7 h total and are the only remaining work with unambiguous value and no gate.**