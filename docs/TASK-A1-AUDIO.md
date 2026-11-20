# TASK-A1 — Audio Modality Replacement (WaveSP-Net-class + Second Opinion)

**Phase:** A1 · **Window:** weeks 5-10 (after P0, parallel with I1/V1) · **Effort:** 4-6 engineer-weeks

## Evidence (why replace)

- **Wild audio is broken**: raw AUC **0.5593**, FPR@0.5 **79.5%**, coverage-gate
  abstention **83.9%**, in-coverage residue AUC 0.6204
  (`artifacts/exam_dfeval/audio_exam_summary.json`). r8 retrain already failed+reverted (D-036).
- **The number was measured on ~3% of the protocol**: 703 matched-stem clips vs the
  official 21,347-segment test. All audio decisions to date rest on a biased subset.
- Current architecture = frozen XLS-R-300m + LogisticRegression (13 KB joblib),
  4 s windows capped at first 8, trained on ASVspoof 2019 LA only.
- Published context: off-the-shelf open models on DF-Eval audio: AASIST 0.43,
  RawNet2 0.53, P3 0.58 AUC. Fine-tuned: P3 **0.920**, AASIST 0.906,
  **WaveSP-Net 0.9426** (best permissively-licensed), XLS-R-1B AntiDeepfake 0.9435.
  Best commercial: 0.93.

## Target architecture

| Slot | Component | Why | License | Trap avoided |
|---|---|---|---|---|
| Primary | **WaveSP-Net-class head on frozen XLS-R** (ICASSP 2026, arXiv 2510.05305) | best verified DF-Eval audio among permissive options: AUC 0.9426 / EER 10.58; only 4.1M trainable params; transformers-native | MIT (code; verify ckpt at fetch) | — |
| Second opinion (research) | **AntiDeepfake W2V-Large** (`nii-yamagishilab/wav2vec-large-anti-deepfake`) | best published zero-shot wild: In-the-Wild EER 1.91; DF-Eval FT EER 10.52 @30 s | **CC BY-NC-SA 4.0 — research only** | license |
| Fallback baseline | P3 (Zenodo 8337778, BSD-3 per record — re-verify license text at download) | AUC 0.920 FT on DF-Eval | BSD-3 | **fairseq cannot import on Python ≥3.12** (verified ImportError) — if used, run in a py3.10 sidecar container |
| Keep | spectral DSP channel (from dormant W9 fused court, `audio_fusion.py:45-75`) | cheap, decorrelated | — | — |
| Architecture | activate the dormant 3-channel fused audio court (p13_fused path, W9/D-079) | the design already exists and is fail-closed | — | — |

## Steps

1. **Stage 0 — full-protocol baseline (week 1 of track, before any training):** rerun
   the CURRENT p4 probe on the full official 21,347-segment DF-Eval audio protocol
   (public ungated mirror `xxuan-speech/Deepfake-Eval-2024-audio` exists for research
   dev — but its "MIT" tag conflicts with the source's CC-BY-SA-4.0 + gating ToS:
   research use only, compliance review before any product use). No more decisions on
   3% subsets. Record the true baseline with scope label.
2. **Fetch-and-verify WaveSP-Net**: checkpoint URL from the paper repo; verify
   license file, load, one-inference smoke gate under pinned transformers. If
   unfetchable/non-permissive → kill-gate K3 (P3 sidecar or self-train the 4.1M head
   on public corpora — recipe published).
3. **Fine-tune (Stage 1, public corpora):** ITW (In-the-Wild, research EULA) + MLAAD
   (CC-BY-NC) + partialspoof + ASVspoof 19LA/21LA union with RawBoost-style codec
   augmentation (MUSAN-style background augmentation recommended by the AntiDeepfake
   authors — DF-Eval error analysis: music −17.9% acc, silence −35.4%).
4. **Windows**: support 30 s windows (longer windows cut EER 2-4 pts published;
   current code caps at 8×4 s — extend `argus_detect.py:437-438`), with 4 s fallback
   for short clips. Audio advisory in video lane inherits the better head.
5. **Activate the fused court** (3 channels: primary + second opinion + spectral)
   via the existing `audio_fusion.py` bundle contract — it is already fail-closed and
   has an abstention rule; re-derive its coverage gates on the new channels' train
   distributions (D-035 train-only law).
6. **Calibration pack** (with TASK-C1): logits output, LR court calibration,
   class-pure cohort packs per codec/source, LTT thresholds.
7. **Gate battery**: re-use `run_gate_battery_w9.py` structure; exams on ITW, 21LA,
   21DF, DF-Eval-full (if access). D-number entries + benchmarks.json regeneration.
8. **UI/health lockstep** (audio instrument count 3 → keep 3 if fused court counts
   as one bundle, or update counts everywhere).

## Acceptance criteria + verification commands

```bash
# A1: full-protocol baseline recorded first (Stage 0 artifact)
jq '.audio.scope_wild_full_protocol' artifacts/benchmarks.json   # exists, n=21347

# A2: new head live with long-window support
python deepfake/argus_detect.py tests/fixtures/40s.wav --json \
  | jq '.segments_analyzed, .duration_s'    # 10 segments (not capped at 8), ~40 s

# A3: fused court fail-closed contract
python - <<'EOF'
import audio_fusion  # deliberate missing-asset fixture -> must raise FusedBundleError
EOF

# A4: Stage-1 cross-corpus performance
#   ITW+MLAAD held-out: AUC ≥ 0.85, fixed-threshold FPR ≤ 10% after calibration
#   (from current: AUC 0.5593 / FPR 79.5%)

# A5: abstention
#   wild abstention ≤ 30%   (from 83.9%)

# A6: DF-Eval full protocol (if access granted): AUC ≥ 0.90, stretch 0.94
```

## Staged targets (honest)
- **Stage 1 (public):** cross-corpus AUC ≥0.85, FPR ≤10% post-calibration, abstention ≤30%.
- **Stage 2 (DF-Eval):** ≥0.90 full protocol; stretch 0.94 (= published WaveSP-Net).
  Commercial ceiling context: 0.93 — matching it open-source is realistic HERE first.
- Codec/partial-spoof robustness: measure (21LA codec set, partialspoof segment-level);
  accept documented degradation, do not hide it.

## Risks
- WaveSP-Net checkpoint availability (paper is ICASSP 2026; repo may be thin) — K3
  fallbacks are concrete: P3-sidecar (py3.10) or self-train (4.1M params on frozen
  XLSR is a days-not-weeks train on one GPU).
- AntiDeepfake NC license — fine for research; excluded from any commercial build.
- Fused-court re-activation may surface the W9.1 HOLD causes — read D-079 first and
  fix the root cause, do not bypass the fail-closed contracts.
