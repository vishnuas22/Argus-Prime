# Argus Deepfake Detection — Limitations & Honest Capabilities

**Status:** Beta (supervised screening). Snapshot date: 2026-10-15.
**Stack:** image + video + audio multi-sense fusion with calibrated confidence and abstention.

This document states what the system demonstrably does, where it fails, and
how to use it safely. It is the authoritative scope statement for client use.

---

## 1. What it is

A multi-sense deepfake detection pipeline. For a given file it produces:

- a fake/real **probability** (not a binary verdict),
- a **confidence** score with per-sense evidence (image forensics, video
  temporal consistency, audio spectral/prosodic cues),
- an **abstention** state ("needs human review") when evidence is insufficient
  or senses conflict.

The recovery layer is designed to degrade gracefully: when a primary signal is
unavailable or unreliable, fallback checks run and the output is explicitly
flagged as lower-confidence rather than guessed.

## 2. Measured performance (curated exam sets)

All numbers below are from held-out exam corpora, run against the current
production model. "AUC" 1.0 = perfect separation; 0.5 = chance.

| Domain | Exam | AUC | Notes |
|---|---|---|---|
| Image (modern AI-generated) | AIGC-2026 | 0.9999 | strongest domain |
| Image (face-swap/reenactment) | CDF | 1.0000 | |
| Image (Midjourney v6-era) | MJ | 0.9906 | |
| Image (FLUX.1 era) | FLUX | 0.9502 | |
| Video (manipulation) | FF++ | 0.6614 | weakest gated domain; see §3.2 |
| Audio (conversion/voice-swap) | DF-Eval | ~0.97 | strong on conversion; see §3.3 |

**Important:** these are curated corpora. Real-world traffic always shifts
distribution; treat these as upper bounds, not guarantees.

## 3. Known limitations (current, measured)

### 3.1 False-positive rate on real images — the primary risk
At the default operating threshold the system flags **~12% of verified-real
photographs** as suspicious, and **up to ~37%** of real images drawn from
wild/social-media-like distributions (provider-matched reals). This is the
single most important limitation for deployment.

**Consequence:** on real-heavy workloads, a non-trivial fraction of innocent
content will be marked suspicious. Outputs must never be treated as
authoritative accusations.

### 3.2 Video domain
FF++ AUC 0.66 — the video layer is meaningfully weaker than image. For video
inputs, weight the image-frame evidence and expect more abstentions.

### 3.3 Audio domain
Strong on studio-style conversion, but an unresolved **~7% in-domain
false-alarm residual** exists (0.0697 out-of-calibration rate on in-domain
audio). Audio-only verdicts should be treated as advisory.

### 3.4 Coverage gap: today's most-used generators
The four most widely deployed commercial generators of 2026 (GPT-Image-2,
Nano Banana 2, Seedream 4.5, Grok) are **not yet covered by trained models** —
images from them are currently misclassified at elevated rates. Open-weight
and corpus-recoverable generators are being absorbed on a rolling basis; the
ten generators measured this cycle all fell below our internal quality bar,
and remediation is in progress (current model-generation cycle).

### 3.5 Distribution shift
No calibration has been performed on any specific client's traffic. Thresholds
are set on public research corpora. Client-specific calibration is a required
step before production use (see §5).

## 4. Intended use vs out-of-scope use

**Intended (supported):**
- Triage/screening of user-submitted media with **human review of flagged items**,
- prioritization queues where false positives are acceptable cost,
- research and benchmarking,
- trend monitoring (aggregate rates over time, not per-item verdicts).

**Out of scope (do not deploy as):**
- authoritative real/fake classification for moderation, legal, or disciplinary
  outcomes,
- fully automated takedown/banning decisions,
- evidence in legal proceedings,
- any use where a false accusation against a real person is unacceptable.

## 5. Required before production use

1. **Client calibration pass** — run a labeled sample of client-representative
   traffic; set the operating threshold to the client's false-positive budget.
2. **Human-in-the-loop** — all "suspicious" outputs route to review; the system
   abstains rather than guessing when uncertain.
3. **Scope agreement** — §4 intended-use signed off; out-of-scope uses
   contractually excluded.
4. **Re-test on current generators** — refresh the generator exam suite at least
   quarterly; generative models rotate faster than detection models train.

## 6. Ongoing work (affects future revisions of this document)

- Absorbing 2026 open-weight/corpus generators into the training diet
  (current cycle), with per-generator regression gates.
- Root-causing the false-positive gap on wild/provider-matched real images.
- Residual audio false-alarm investigation.
- Rolling generator-exam refresh pipeline (new generators = routine gated
  operation, not a rebuild).

Performance numbers in this document are superseded by any later revision
issued after the above land. Clients receive the revision at each delivery.

---

*Snapshot generated 2026-10-15. Questions on any number in this document can be
traced to the corresponding exam artifacts in `artifacts/`.*
