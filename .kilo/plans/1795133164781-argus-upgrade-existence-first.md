# Argus-Prime Video Lane — Maximum-Performance Plan

**Scope:** video modality only. Planning document — **no execution performed**.
**Court:** 64.8/100 (repo ledger baseline). OOD-video 1.2/10 is the only dimension that can reach 70.
**Rule applied:** only items with **confirmed, cited evidence of improvement** are in scope. Anything without confirmation is listed as excluded (§5) — not planned.
**Stop ruling (D-105):** training, wiring, promotion, pointer change, and commit all require explicit approval.

---

## 1. Verified incumbent baseline

From `artifacts/benchmarks.json` (as_of 2026-11-04), video build **p6vid_v4e**, 32-feature stacker:

| Benchmark | p6vid_v4 | Scope |
|---|---:|---|
| FF++ c23 test (n=700) | **0.9121** | DF 0.9884 · F2F 0.8938 · FS 0.9745 · NT 0.7917 |
| Celeb-DF-v2 | **0.9482** | cross-dataset |
| DFD (DeepfakeDetection) | **0.8857** | cross-dataset |
| DFDC | **never measured** | — |
| GenBuster++ T2V | **0.590** | measured this session (D-125) |
| GenVidBench t2v | kling 0.6035 · sora 0.6812 · opentora 0.7497 | informational |

Feature contract (`deepfake/bench/video_features.py:31-46`): 12 GEN + 8 SPEC + 1 drift + 3 audio + 8 SCENE = 32.
Serve path: YuNet → ArcFace `alignCrop` → dual CLIP ViT-L/14 heads → 32-feature HistGBM stacker.

---

## 2. Task list — confirmed improvements only

### V2a — Replace the face-probe head with GenD (CLIP ViT-L/14 + LN-tuning)

**Status:** **CONFIRMED improvement, commercial-safe, near-zero marginal cost.**

**Evidence (WACV 2026, arXiv 2508.06248, MIT licence, `github.com/yermandy/GenD`):**

| Benchmark | p6vid_v4 (ours) | GenD (CLIP) published | Δ |
|---|---:|---:|---:|
| FF++ in-domain (mean of DF/F2F/FS/NT) | 0.9121 | **0.980** (99.5/98.1/98.7/95.5) | **+0.068** |
| CDFv2 | 0.9482 | **0.960** | +0.012 |
| DFD | 0.8857 | **0.970** | **+0.084** |
| DFDC | not measured | **0.871** | new coverage |
| FFIW | not measured | 0.928 | new coverage |

Weakest sub-method is the tell: GenD's worst (NT 95.5) is **+16 pts** above our worst (NT 79.17).

**Checkpoint (verified live):**
- `huggingface.co/yermandy/GenD_CLIP_L_14` — 0.3B params, updated 2025-11-13
- `huggingface.co/yermandy/deepfake-detection` — TorchScript export (`model.torchscript`)
- Licence: **MIT** (both code and weights)

**Why this is the cheapest possible win:** GenD is **the same backbone we already load** (CLIP ViT-L/14) with only the LayerNorm parameters tuned (0.03% of params). The existing `p3_full/face_probe.joblib` and `p3_ffpp/ffpp_only_probe.joblib` occupy exactly this slot (`video_score.py:38-42`). Swapping the head changes **no backbone cost** — the marginal inference cost is approximately zero.

**Pre-registered bars:**
- Reproduce CDFv2 ≥ 0.950 and DFD ≥ 0.930 on our exam sets.
- FF++ c23 ≥ 0.940 (no regression vs 0.9121).
- Abstention parity B4: `ab_new ≤ ab_old + 0.10`.
- If any bar fails → HOLD + WAIVER.md, never silently absorb.

**Known risk — preprocessing parity (fetch-gate, must clear first):**
The README states: *"To get the same results as in the paper, you need to preprocess images using DeepfakeBench preprocessing pipeline."* Our face crops come from **YuNet → ArcFace `alignCrop`**, which is **not** DeepfakeBench's pipeline. The published AUCs may not reproduce on our crops. **Gate:** run GenD on our existing FF++ crops first; if CDF < 0.93, re-run with DeepfakeBench-compatible crops before judging.

### V2b — Add DFDC to the exam suite (measurement gap, prerequisite for any claim)

**Status:** **CONFIRMED gap.** DFDC is the one benchmark where every competitor reports a number and we report nothing. Without it, "improvement" cannot be evidenced externally.

**Evidence:** DFDC is the standard cross-dataset test (ForAda 0.872, GenD 0.871, UDD 0.812, TALL++ 0.785). Our `benchmarks.json` has no DFDC row.

**Task:** acquire a DFDC sample (or a documented public subset), score p6vid_v4 and the GenD head on it, publish with scope label. **No training.** This is measurement infrastructure that V2a depends on for its acceptance claim.

**Risk:** DFDC full is ~470 GB; disk is ~60 GiB free. Requires a documented subset (e.g. the 3,000-clip DFDC-Preview) — license is research-use.

---

## 3. Execution order

```
V2b (DFDC exam plumbing, no model work)
   └──> V2a (GenD head swap, fetch-gate: preprocessing parity)
          └──> sealed restack spec (p6vid_v4 -> v5) with B1-B3 above
                 └──> D-number + benchmarks.json regeneration
```

V2b first because V2a's acceptance is defined on DFDC. V2a's fetch-gate (preprocessing parity) must clear before any restack.

---

## 4. Risks

| Risk | Mitigation |
|---|---|
| Preprocessing mismatch invalidates published AUCs | Fetch-gate: score on our crops first; if CDF < 0.93, switch to DeepfakeBench-compatible crops |
| GenD head trained on FF++ c23 only may not transfer | That is exactly what the CDF/DFD bars measure |
| Restack parity laws block promotion | Budget one HOLD; never waive silently (repo law) |
| GenD is frame-based; our lane is 12-frame face-crop | Compatible — same as existing p3_full/p3_ffpp probes |
| NC-licensed competitors (ForAda, TALL++, SBI) tempt a shortcut | Excluded — MIT-only roster for commercial posture (D-122/D-123) |
- **V1d-2 T2V specialist (V-JEPA2):** signal **confirmed** (GenBuster T2V 0.9458 confound-clean, vs p6vid_v4 0.590) but **deployment unproven** — no label-free router exists (D-113), and it scores 0.4935 on FF++ (at chance). Excluded until a router clears a pre-registered bar.
- **ForAda** (CVPR 2025, CDF 0.957 / DFD 0.972 / DFDC 0.872) — **CC BY-NC 4.0**, excluded for commercial posture. Re-open only if U8 rules research-only.
- **UDD** (AAAI 2025, 0.931/0.955/0.812) — licence unconfirmed, weaker than GenD. Excluded.
- **TALL++** (IJCV 2024, 0.920/–/0.785) — CC BY-NC weights. Excluded.
- **SBI / Face-X-Ray / SCENE family** — NC, already in incumbent lineage (D-070/D-075). Excluded.
- **Temporal statistics** (D3/SPLIT/WaveRep) — CLOSED (D-099/D-100/D-104).
- **ReStraV geometry** (V1d/V1h) — CLOSED (B2a/B2b fail; no label-free router, D-113).
- **V-JEPA2 as wholesale replacement** — REFUTED (D-125: FF++ 0.4935, CDF 0.8288).
- **p6vid_v5-on-SFace** — commercial enabler, **0 detection Y** (D-123). Tracked separately as U8 follow-up.

---

## 6. Validation

```bash
# V2b: DFDC exam plumbing
#   - a DFDC subset scorer exists and runs end-to-end
#   - benchmarks.json gains a video.scopes.dfdc entry with n + scope label

# V2a: GenD head
python deepfake/argus_detect.py tests/fixtures/wildface.mp4 --json | jq '.video_score'
#   - CDF >= 0.950, DFD >= 0.930, FF++ >= 0.940
#   - abstention parity: ab_new <= ab_old + 0.10
#   - AUC invariance where applicable (monotone law)

# restack
python deepfake/scripts/verify_dump_parity.py && echo PARITY-OK
jq '.gates | map(select(.pass==false))' artifacts/<new>/gates.json   # empty or waived
```

---

## 7. Open questions (owner)

1. **U6** code licence — ruled **MIT** (D-114). GenD is MIT-compatible. ✅
2. **U8** commercial posture — **ruled commercial achievable at ~0.013 AUC** (D-122/D-123). MIT-only roster is therefore correct; NC methods stay excluded.
3. **DFDC acquisition** — is a research-use DFDC subset acceptable for the exam, given disk limits (~60 GiB free vs 470 GB full)?
4. **V1d-2 T2V specialist** — pursue a router feasibility pilot, or leave closed? (Router is the single bar; without it the +0.36 T2V AUC is unrealisable.)