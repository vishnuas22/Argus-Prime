# Argus Deepfake Detection — Limitations & Honest Capabilities

**Status:** Beta (supervised screening). Snapshot date: 2026-10-19 (p6v11 promoted under user waiver — `artifacts/p6v11/WAIVER.md`; rows repaired to artifact truth 2026-10-19).
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
| Image (modern AIGC corpora) | AIGC | 0.9993 | strongest domain |
| Image (celeb face-swap) | CDF | 1.0000 | |
| Image (Midjourney v6-era) | MJ | 0.9806 | within tolerance of golden |
| Image (FaceSwap / FF++ faces) | FF++ faces | 0.6479 | weakest gated image domain; lineage + waiver item |
| Image (FLUX.1 era) | FLUX | 0.9297 | waived family cost (§3.6); incumbent scored 0.9611 |
| Image (FLUX.2-klein, self-generated) | FLUX2 | 0.9436 | NEW this cycle: +0.1725 coverage gain |
| Image (2026 commercial gens, measured) | 7 domains | 0.67–0.89 | ideogram3 0.8643 · imagen4 0.6731 · nano-banana 0.8913 · seedream3-0 0.7709 · flux-1-pro 0.7988 · gpt4o 0.7823 · hidream 0.8538 — §3.4/§3.6 |
| Video (FF++ official identity-disjoint split) | Gate A | 0.9016 | p6vid_v2 stacker (F2F 0.876 vs 0.88 waived); Celeb-DF 0.9411; §3.2 |
| Audio (wild audio, DF-Eval 2024 protocol) | D-035 coverage-gated | raw TPR 0.8766 · in-coverage TPR 0.6909 · abstain 83.9% (n=703) — FPR 0.79 raw / 0.60 in-coverage: **UNTRUSTED, fails closed** | artifacts/exam_dfeval/audio_exam_summary.json; §3.3 |

**Important:** these are curated corpora. Real-world traffic always shifts
distribution; treat these as upper bounds, not guarantees.

## 3. Known limitations (current, measured)

### 3.1 False-positive rate on real images — the primary risk
**Measured on wild reals (T1 stage-0, 2026-10-19, n=2000, sha-asserted):**
score-level FPR@0.5 = **0.25%** (bootstrap CI 0.05–0.50%); among ISSUED
verdicts **0.00%** (0/701) — every score-level FP abstains instead. Operating
point TPR@FAR=5% = 77.1% raw / 90.7% among issued calls. FP attribution
(face-forensics channels): `artifacts/exam_wild_real/EXAM_REPORT.md`.

Historical framing: pre-recalibration regimes were estimated at **~12% FPR on
verified-real photographs** and **up to ~37%** on wild/social-media-like
(provider-matched) reals; provider-matched reals still carry a recorded
0.37 score-level FPR residue from the intake-design history (T1 readouts
extend to them before client claims widen). The single most important
deployment rule is unchanged: never treat outputs as authoritative
accusations.

**Why the issued-FPR collapses to zero:** the court **abstains first** under
the deployed p6v11 operating point — "not confident" replaces a suspicious
call whenever top probability mass falls inside the calibrated band
(per-candidate alpha 0.209291, D-051): measured abstention aigc 26.5% /
cdf 54% / ffpp-face images 63.5% (vs 7.5%/4.5%/26% under p6v10).

### 3.2 Video domain
Deployed p6vid_v2 (identity-aware stacker, D-044): FF++ official split Gate A
**0.9016** stacked (naive 0.7487), Celeb-DF cross-dataset **0.9411**, one
per-method bar waived on record (Face2Face 0.876 vs 0.88). In-the-wild video
remains **UNTRUSTED** (GenVidBench exam-only; no wild video diet). Expect more
abstentions than image; route to image-frame evidence on faceless clips.

### 3.3 Audio domain
Strong on studio-style conversion, but an unresolved **~7% in-domain
false-alarm residual** exists (0.0697 out-of-calibration rate on in-domain
audio). Audio-only verdicts should be treated as advisory.

### 3.4 Coverage gap: today's most-used generators
The four most widely deployed commercial generators of 2026 (GPT-Image-2,
Nano Banana 2, Seedream 4.5, Grok) are **not yet covered by trained models**
(no open weights, no funded API access — unreachable on record). The 2026
generators measured this cycle: 10 exams, AUC **0.67–0.94** (flux2-klein 0.9436
added this cycle) — **above published
ecosystem parity** (external benchmark: state-of-the-art detectors score
18–31% accuracy zero-shot on unseen frontier generators), so the residual gap
is a *coverage* problem (training-data diversity), not an architecture one.
Remediation is rolling intake (D-049 family-substitute rule), not a rebuild.

### 3.5 Distribution shift
No calibration has been performed on any specific client's traffic. Thresholds
are set on public research corpora. Client-specific calibration is a required
step before production use (see §5).

### 3.6 Waived residuals (p6v11 promotion, 2026-10-18)
The promotion shipped the FLUX.2-klein family coverage (+0.1725 AUC,
0.7712 → 0.9436, parity PASS) under an explicit user waiver of measured
residuals: AUC family-cost mj −0.0169 / flux −0.0391 / commercial-gen shards
−0.0101…−0.0371 (current standing AUCs across the waived domains span
**0.67–0.97**: mj 0.9737, flux 0.9111, imagen4 0.6731 — the only one below
0.77; source `artifacts/restack_p47/gates.json`), and the calibrated
abstention shift on core domains (§3.1). The stranded structural finding:
a single global train-only recalibration cannot cross-domain restore old
abstention exactly (D-051); the product absorbs this under the
abstention-first UI default (D-053). Full per-gate table:
`artifacts/p6v11/WAIVER.md`. Where numbers here and there disagree, the
waiver timestamp is authoritative.

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
- Root-causing the false-positive gap on wild/provider-matched real images
  (**T1 stage-0 DONE 2026-10-19**: wild FPR@0.5 0.25% score-level, 0.00%
  issued-verdict; face-forensics channels carry the residual FP surface —
  `artifacts/exam_wild_real/EXAM_REPORT.md`. r10 fix-candidate design stays
  pre-registered; upside-parity of TPR@FAR5 remains open, user-gated).
- **Golden re-baselined (Phase R, user-ruled 2026-10-19)** to the measured p6v11
  cells; future candidates fail-closed against them. Historical trace of the
  waived decomposition (ffpp/flux) stays in the B.6 record and WAIVER.md.
- Per-domain abstention-calibration followup (design-B / domain-aware
  recalibration) — open, user-gated (D-051).
- Residual audio false-alarm investigation.
- Rolling generator-exam refresh pipeline (new generators = routine gated
  operation, not a rebuild).

Performance numbers in this document are superseded by any later revision
issued after the above land. Clients receive the revision at each delivery.

---

*Snapshot generated 2026-10-19 (p6v11 waiver-wave + T1 wild measurement). Questions on any number in this document can be
traced to the corresponding exam artifacts in `artifacts/`.*
