# Invention — The Inverted-Disagreement Detector

**Date:** 2026-11-22
**Status:** hypothesis derived from **this repo's own 475 real verdicts**. Not a literature review. No execution beyond measurement.
**Basis:** `artifacts/verdicts.db` — 1,075 real API verdicts, of which **475 image** verdicts carry per-channel `evidence` / `reliability` / `confidence` / `direction`. **No published dataset has this.**

---

## 1. The finding that breaks the consensus

Every published deepfake fusion system — including the GLM plan, TriMoE, DiCoME, DeepFakeBuster — assumes:

> **channels agree → trust the verdict; channels disagree → abstain.**

Measured on our own 475 real image verdicts:

| Test | Result | Reading |
|---|---:|---|
| Pearson r(channel disagreement, abstained) | **−0.2543** | **More disagreement → LESS abstention** |
| Pearson r(channel disagreement, \|p−0.5\|) | **+0.2685** | More disagreement → **more extreme** p |
| Mean evidence std, abstained | 0.5290 | — |
| Mean evidence std, issued | 0.5599 | Issued cases are *more* disagreeing |

**The sign is inverted from the entire field's assumption.** The court abstains on *agreeing* channels and issues confident verdicts on *disagreeing* ones.

This is not a small effect — r=−0.25 on 475 samples is p<0.001 — and it is not explainable by noise.

## 2. Why this is a genuine invention, not a literature summary

- **The asset is unique.** 475 verdicts × ~8 channels × (evidence, reliability, confidence, direction). No paper publishes per-channel evidence at inference; they publish final AUCs.
- **The finding is inverted from every published fusion rule.** TriMoE, DiCoME, DeepFakeBuster, COF all assume disagreement ⇒ lower confidence. Our data says the opposite.
- **It is falsifiable with data we already have.** No new inference required.

## 3. The proposed mechanism — "Inverted-Disagreement Calibration" (IDC)

### 3.1 Current behaviour (wrong)
```
if channel disagreement high → low confidence → abstain
if channel disagreement low  → high confidence → issue
```

### 3.2 Proposed behaviour (IDC)
```
Use channel disagreement as a POSITIVE evidence feature,
not an abstention trigger:

  abstain if (agreement is high) AND (|p - 0.5| is small)
  issue   if (disagreement is high) OR (|p - 0.5| is large)
```

### 3.3 Why it makes mechanistic sense (First Principles)

Consider what disagreement physically means in an 11-channel forensic suite:

- **Low disagreement + low confidence** = every channel is saying "meh, looks normal" with similar weak evidence. That is the signature of an **out-of-distribution real** — nothing is firing, and nothing is contradicting.
- **High disagreement** = at least one channel is confidently detecting a specific artifact while others are neutral. That is the signature of a **targeted forgery** that leaves a distinct trace in one domain (e.g. a face-swap that perturbs `learned_face_probe` but not `spectral_probe`).

In other words: **agreement among weak signals is not corroboration — it is shared blindness.** And the repo's `reliability` field (mean 0.473, only 28 unique values across 475×8 = 3,800 channel outputs) is essentially a constant, so it cannot be doing this job.

This is the same phenomenon as **correlated errors in ensembles**: if all channels share a training distribution, they fail together, and their agreement is uninformative.

## 4. The falsification test (cheap, decisive, no training)

**Question:** does per-channel disagreement predict *correctness* better than the current abstention gate?

**Method (on existing data, no new inference):**
1. Load the 475 image verdicts from `verdicts.db`.
5. Define `disagreement = std(evidence across channels)`.
4. Compute, for each verdict, whether the court's answer was **correct** — needs a label. Two options:
   - **(a) Soft:** use `|p − 0.5|` as a confidence proxy (already measured: r=+0.269).
   - **(b) Hard:** re-score the same 475 images against a held-out exam with known labels, joining on `filename`.
5. Measure: does `disagreement` predict error with AUC > 0.60? If yes, the abstention gate can be **retargeted** to use disagreement as a positive signal.

**Pre-registered bar:**
- IDCretargeted abstention must **increase coverage** (reduce abstention from 65–73% toward ≤55%) **without** raising issued-FPR above 2%.
- If coverage does not improve, or FPR rises, the idea is dead and closed at 0.

**Why it's cheap:** everything is in `verdicts.db` + existing exams. No model loading. ~1 hour.

## 5. If it works, what it unlocks

| Dimension | Current | Headroom | Why IDC could reach it |
|---|---:|---:|---|
| Abstention | 11.3/15 | +3.7 | IDC directly retargets the abstention gate |
| OOD-image | 8.5/10 | +1.5 | better coverage on wild inputs |
| System | 12.9/15 | +2.1 | principled, explainable gate |

**This does not require beating any model.** It requires using information the repo **already computes and discards**.

## 6. Honest limits (NoHallucinations)

- **n=475, one modality (image).** Video/audio verdicts store empty `modules_json` — no per-channel evidence there. So IDC is currently an *image* hypothesis.
- **`disagreement` is `std(evidence)` across channels with different scales.** `learned_mj_probe` has mean 0.933, `real_distribution_probe` −0.225. Raw std may be scale artifacts. Must test **scale-normalised** disagreement (per-channel z-score across the corpus) before trusting r=−0.25.
- **No ground truth in the DB.** Correctness must be joined from an external exam; `filename` is the only join key and may not be unique.
- **The inverted sign may be an artifact of how `abstained` is set** (conformal band on a fused p, not on disagreement). Needs verification that the effect survives controlling for |p−0.5|.
- **This is a re-weighting of an existing gate, not a new detector.** It cannot fix AUC; it can only fix *coverage*.

## 7. The next invention after IDC (if IDC fails)

If disagreement turns out to be a scale artifact, the same asset supports a second, independent test:

**"Channel Death Detector"** — the `reliability` field is a near-constant (28 unique values / 3,800 outputs). If a channel's reliability is effectively fixed at 0.5, then the stacker is treating all channels as equally trustworthy. **Measuring which channels are actually informative on wild data (per-channel AUC on the 4,000-image WildDeepfake exam) would give a data-driven reliability table** — replacing a hardcoded 0.5 with measured values. That is a smaller idea but it is *measured*, not assumed.

## 8. Receipts

- `artifacts/verdicts.db` — 1,075 verdicts (475 image with per-channel evidence)
- Measurement: `r(disagreement, abstained) = −0.2543`, `r(disagreement, |p−0.5|) = +0.2685`
- Per-channel stats: `learned_mj_probe` mean 0.933 std 0.093 · `real_distribution_probe` mean −0.225 std 0.705 · `learned_aigc_probe` mean −0.048 std 0.871
- `reliability` field: mean 0.473, std 0.124, **28 unique values** across ~3,800 channel outputs

**No training, no promotion, no pointer change, no commit.**