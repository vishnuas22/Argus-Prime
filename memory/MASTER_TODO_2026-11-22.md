# Argus-Prime — MASTER To-Do (authoritative, supersedes all prior lists)

**Generated:** 2026-11-22. Supersedes `backlog_2026-11-21.md`, `backlog_2026-11-20.md`, `prod_readiness_2026-11-22.md`.
**Court:** **64.8/100** measured and booked. Target 70 = **+5.2**.
**Score baseline:** repo ledger 64.8 (NOT the external 42/100 audit).
**STOP ruling (D-105):** training / wiring / promotion / pointer change / commit need explicit approval.
**Latest decisions:** D-101 → D-129.

---

## 0. What has been done (do not redo)

| Area | Result |
|---|---|
| **P0a** artifact bundle | **CLOSED** — private HF `thedataghost22/Argus_prime` @ `1699da0f`; fresh clone boots identically |
| **P0b–P0e** | config seam, 21 files de-hardcoded, `init_court` guard, requirements split, pytest markers, two-tier CI (D-108) |
| **P0f** | LICENSE (MIT) + WEIGHTS-NOTICES + README (D-114) |
| **U6** | code licence **MIT** (D-114) |
| **U7** | audio target: **K1 binds, ECE report-only** (D-095/D-116) |
| **U8** | ArcFace→SFace priced at **~0.013 AUC** (D-122/D-123) |
| **C1** | scaffold DONE (D-109); **fit CLOSED 0** — calibration hurts (D-111) |
| **S1** | scaffold DONE (D-110); bug fixes DONE (D-115); **promotion STOPPED — crashes** (D-118) |
| **V1d/V1h** | ReStraV geometry **CLOSED** (B2 fail; no label-free router) (D-112/D-113) |
| **V1d-2** | V-JEPA2 frozen SSL **CLOSED** (FF++ 0.4935 at chance) (D-125) |
| **V1c/V1c-2** | ASMR non-face; image-court on T2V **20% FPR → CLOSED** (D-124) |
| **GenD** | fetch-gate PASS (FF++ 0.9956) but CDF **0.8832 vs 0.9992** → **CLOSED** (D-127/D-128) |
| **A1** | A0z 1B **CLOSED 0** (G3 23/300) (D-117) |
| **OOD-video rescore** | **CLOSED** — 27.7% @ 4.8% FPR real, but AUC 0.653 caps it (D-121) |
| **P8a** | DF-Eval access **REJECTED** (D-102) |

**Detection search is exhausted:** temporal stats (D-099/100/104), ReStraV (D-112/113), V-JEPA2 (D-125), GenD (D-127/128), image-court (D-124). Incumbent p6vid_v4 hits **0.9992 on CDF** in the paired sample.

---

## 1. THE LIVE INVENTION — IDC (Inverted-Disagreement Calibration)

**Derivation:** `memory/invention_2026-11-22_idc.md`. Measured on **this repo's own 475 real image verdicts** in `artifacts/verdicts.db` — an asset no published dataset has.

**Finding that contradicts every published fusion rule:**

| Test | Result |
|---|---:|
| r(channel disagreement, abstained) | **−0.2543** |
| r(channel disagreement, \|p−0.5\|) | **+0.2685** |

> Everyone assumes *disagree → abstain*. Our data says **disagree → issue, agree → abstain**. The sign is inverted (p<0.001, n=475).

**Mechanism (first principles):** agreement among weak signals is not corroboration — it is **shared blindness** (correlated errors from one training distribution). High disagreement = one channel confidently catching a domain-specific artifact.

**Corroborating dead weight:** the `reliability` field has **28 unique values across ~3,800 channel outputs** (mean 0.473, std 0.124) — effectively a constant.

**Invented mechanism:**
```
current (wrong):  abstain if disagreement high
IDC:              abstain if (agreement high) AND (|p−0.5| small)
                  issue   if (disagreement high) OR (|p−0.5| large)
```

**Pre-registered bar:** coverage must improve (abstention 65–73% → ≤55%) **without** issued-FPR rising above 2%. Fail → closed at 0.

**First check (mandatory, cheap):** re-test with **per-channel z-scored** disagreement — the inverted sign may be a scale artifact (`learned_mj_probe` mean 0.933 vs `real_distribution_probe` −0.225).

---

## 2. Production readiness (client deployability) — from `prod_readiness_2026-11-22.md`

| # | Item | Blocks | Status |
|---|---|---|---|
| **A1** | Authentication (`X-API-Key`, `Security`, fail-closed) | any deployment | ❌ absent — `_origin_gate` blocks browsers, **not scripts** |
| **A2** | Rate limiting (sliding window, per-key) | A1 | ❌ absent |
| **A3** | Container (multi-stage, non-root, bootstrap at start) | A1,A2 | ❌ absent |
| **A4** | Metrics (`/metrics`) + structured JSON logs | A3 | ❌ `print()` only |
| **A5** | API test suite (`test_api_auth`, `test_api_contract`) | A1,A2 | ❌ none |
| **A6** | Graceful drain on SIGTERM | A3 | ❌ absent |
| **A7** | ArcFace→SFace (commercial only) | U8 | priced ~0.013 |
| **B1** | Concurrency (per-thread YuNet/ArcFace; then raise limiter) | S1-scaffold ✅ | ❌ limiter=1 |
| **B2** | ONNX/OpenCV-DNN export + parity <1e-3 | B1 | ❌ no onnxruntime |
| **B3** | `/api/batch` length-aware grouping | B2 | ❌ |
| **B4** | readz/livez split | A3 | ❌ single `/api/health` |
| **B5** | Request IDs + audit trail | A4 | ❌ |
| **B6** | Config validation at boot | A1 | ❌ |
| **B7** | Docs: OpenAPI, deployment, runbook | A5 | ❌ no README beyond P0f |
| **B8** | SBOM + licence scan in CI | A3 | ❌ |

**Minimum for client handover: A1→A6.**

---

## 3. Detection improvements — new leads from the 10-scientist research

Source: `memory/research_2026-11-22_10scientists.md`. **Research only, nothing executed.**

| # | Candidate | Evidence | Cost | Bar | Novelty |
|---|---|---|---|---|---|
| **D-1** | **WaveSP-Net** (ICASSP 2026, MIT, HF ckpt) | DF-Eval **0.9426** vs our 0.5593; 1.3% trainable | ~1 h (py3.8 sidecar) | audio AUC ≥0.85 | the plan named it, **never fetched** |
| **D-2** | **MotionPhys** (arXiv 2608.20770) | **GenBuster 0.872** vs our **0.590**; optical-flow curvature/angular velocity/jerk | ~2 h | GenBuster ≥0.75 + motion-bias control | not temporal-stats, not face-based |
| **D-3** | **Rank-space / evidential re-fusion** (DiCoME ICML 2026) | repo already has `conflict_metric` | ~1 h, existing dumps | beats global stacker | exploits 11 channels |
| **D-4** | **Input-adaptive gating** (TriMoE CVPRW 2026, MoSTE 2611.12443) | "static fusion is the flaw" — our exact architecture | ~3 h | beats global stacker | #1 literature rec |
| **D-5** | **COF uncertainty fusion** (arXiv 2606.06666) | ρ(UQ, error) 0.41–0.47 → ~0 externally | ~2 h | ρ>0.3 held-out | abstention→coverage |

**Threat to D-2:** arXiv 2607.00948 — motion detectors degrade on rebalanced datasets (AI videos move less). Must clear a motion-bias control.

---

## 4. Closed at 0 — do not re-litigate without a new pre-registered hypothesis

C1 calibration (hurts) · V1d ReStraV (B2 fail) · V1h router (no label-free signal) · V1d-2 V-JEPA2 (FF++ chance) · V1c/V1c-2 ASMR (non-face, 20% FPR) · GenD (CDF −0.116, ensemble −0.0004) · D3/SPLIT/WaveRep (temporal stats) · A1 A0z (G3 fail) · P8a DF-Eval (rejected) · S1-promote (crashes) · MLLM-as-detector (chance) · HOLA · per-generator heads · ASVspoof5 standalone · chasing 100/100.

---

## 5. Ranked next actions (cheapest falsification first)

| Rank | Action | Cost | Gate | Why |
|---|---|---|---|---|
| **1** | **IDC falsification** — z-scored disagreement vs correctness on the 475 verdicts | **~1 h, no inference** | coverage ↑ without FPR >2% | Uses an asset nobody else has; inverts the field's fusion rule |
| **2** | **WaveSP-Net probe** | ~1 h | audio AUC ≥0.85 (from 0.5593) | MIT, checkpoint exists, cheapest path to fixing the worst modality |
| **3** | **MotionPhys features** | ~2 h | GenBuster ≥0.75 + motion-bias control | 0.872 vs our 0.590 on the same corpus |
| **4** | **A1 auth + A2 rate limit** | ~2 h | 401/429 behaviour | production blocker; default-OFF so zero risk |
| **5** | **Rank/evidential re-fusion** | ~1 h | beats global stacker | exploits the 11-channel investment |

---

## 6. Open owner decisions

| ID | Question | Unblocks |
|---|---|---|
| **U6** | ✅ **RULED: MIT** | P0f done |
| **U7** | ✅ **RULED: K1 binds, ECE report-only** | A1 closed |
| **U8** | ✅ **RULED: commercial achievable at ~0.013 AUC** | A7 |
| **U9** | **Audio modality**: ship with a "research" badge, or remove from the product story? (AUC 0.56 wild) | product scope |
| **U10** | **Auth model**: static API keys vs OIDC/JWT? | A1 design |
| **U11** | **Deployment target**: single box vs orchestrated? | A2 (in-process vs Redis limiter) |
| **U12** | **IDC**: proceed to falsification? | #1 action |

---

## 7. Score reality

- Detection: **64.8/100**. Every model-replacement path measured and closed.
- **IDC is the only live detection lead** — and it does not need a new model.
- Production readiness is **orthogonal** to score: A1–A6 block a client handover, not accuracy.
- Public-data ceiling ~73–78; 77–78 needs a new video-OOD channel; 100 unreachable.