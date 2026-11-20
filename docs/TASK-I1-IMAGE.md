# TASK-I1 — Image Modality Upgrade (Tuned-Probe Ensemble)

**Phase:** I1 · **Window:** weeks 5-10 (after P0) · **Effort:** 4-6 engineer-weeks
**Prereq:** TASK-P0 complete (bootstrap, config seam, CI fast tier)

## Evidence (why replace)

- Wild image performance was NEVER measured on Deepfake-Eval-2024 (exam blocked on
  gated HF registration — `benchmarks.json:63`).
- On WildDeepfake (n=4000): issued FPR 0.0% (0/701) but **abstention 65-73%** — the
  court answers only ~1/3 of wild inputs. TPR@FAR5 rate-level 0.771.
- External generators: 0.67-1.00 across 13 gated domains with **5 AUC FAILs + 5 core
  parity FAILs waived** (`artifacts/p6v11/WAIVER.md`); probes are generator-keyed
  (`real_distribution_probe` flips direction across generators, D-026).
- The serve path is macOS-locked where it is heaviest: mflux/MLX verified broken on
  Linux (`libmlx.so` ImportError); SD-VAE inversion round-trips are DIRE-class
  (published: 6.98 s/image on an A10 GPU — minutes on CPU).
- Published context: off-the-shelf open SOTA on DF-Eval image = **0.56 AUC (UnivFD)**.
  A tuned linear probe reaches **81% accuracy** on the same benchmark (arXiv
  2509.04150) — level with commercial (82% acc). NPR fine-tuned "0.73 AUC" is
  internally conflicted (supplement S9 says 0.55) — do not anchor on it.

## Target architecture (3 decorrelated families ≈ 1.2-1.5 s/image CPU)

| Slot | Model | Role | License | Verified numbers |
|---|---|---|---|---|
| A | **SAFE** (KDD 2025) | 60 ms noise/wavelet pre-filter + expert | Apache-2.0 | GenImage 95.6 acc; FLUX 99.3 / SD3 99.4 acc |
| B | **Tuned linear probe on CLIP-L/SigLIP** (arXiv 2509.04150 recipe) | semantic expert — PRIMARY mechanism | per backbone (MIT for CLIP) | 81% acc on DF-Eval (published recipe) |
| C | **Detectra-v1 / CommFor-384** ONNX ViT-S | web-pipeline-hardened expert | MIT | 253 ms @384; CommFor 0.968 AUC on 4,803-model corpus |
| GPU-opt | Smaug-AI-Image-Detector | 2026-generator coverage (GPT-Image 2, FLUX.2, MJ7) | Apache-2.0 (Gemma terms) | 0.973 AUC OpenFake (self-reported) |
| keep | ELA, spectral, metadata, C2PA channels | cheap, decorrelated, already built | — | — |
| drop (serve) | mflux/FLUX probe, SD-VAE inversion probe | macOS-only / CPU-infeasible | — | — |

Checkpoints (all verified HTTP 200 in audit): SAFE `github.com/Ouxiang-Li/SAFE`
(in-repo `checkpoint/checkpoint-best.pth`, 5.8 MB) · UnivFD `huggingface.co/slxhere/UnivFD`
(ViT-L-14.pt 933 MB, MIT) · Detectra `huggingface.co/TensorFold/detectra-v1`
(model.onnx 44 MB) · Smaug `huggingface.co/abacusai/Smaug-AI-Image-Detector` (878 MB).

## Steps

1. **Fetch-and-load smoke gates** (before scheduling anything): each checkpoint
   downloads, loads, and runs one inference on a fixture under the repo's pinned
   library versions. SAFE needs `kornia` + `pytorch_wavelets` — pin working versions.
   Also evaluate **FerretNet** (arXiv 2509.20890, 1.06M params, 97.1 acc — beats SAFE
   on the same eval) as slot-A alternative if its license checks out.
2. **New bench channels** `safe_probe`, `univfd_tuned`, `detectra_probe` in
   `deepfake/bench/channels.py` (follow `learned_aigc_probe` pattern: frozen backbone
   service + sklearn/torch head + joblib artifact). Respect the registry contract
   (`CORE_IDS` untouched; add to `OPTIONAL_CHANNELS`).
3. **Tuned-probe training** (slot B): follow arXiv 2509.04150 — backbone choice,
   LP-FT style tuning, their augmentation set. Training data (Stage 1, public):
   CommunityForensics (CC-BY-4.0) + OpenFake (CC-BY-NC — research only; do not ship
   weights trained on it in a commercial product without review).
4. **Retrain the image stacker** (p6 → p7) with the new channel features via the
   existing `train_stacker.py` + sealed-spec amendment (new D-number; spec before
   scoring — repo law).
5. **Reliability LUTs** for new channels in `integrations/Argus_Master/backend/reliability.py`
   (currently default 0.5) + evidence thresholds.
6. **Restack + gate battery**: 13-domain external-generator exam re-run (the 5 waived
   AUC FAILs must be re-attempted — target: no new waivers), screenshot-launder exam,
   abstention-parity B4 (`ab_new ≤ ab_old + 0.10`).
7. **Wild evaluation**: if DF-Eval access granted (kill-gate K1): run the image exam
   on the full protocol. Else: OpenFake + CommunityForensics-Eval + a WildDeepfake
   image-side run, scope-labeled.
8. **UI/health lockstep**: `web/index.html` CHANNELS array + panel captions,
   `server.py:279-284` health instrument counts, `e2e_ui_test.py` totals (currently
   asserts img 11 / vid 5 / aud 3).
9. Regenerate `benchmarks.json` with new scopes; update `LIMITATIONS.md`.

## Acceptance criteria + verification commands

```bash
# A1: channels registered and live
python deepfake/argus_detect.py tests/fixtures/wild1.jpg --json \
  | jq '.module_evidence | keys'          # ⊇ {safe_probe, univfd_tuned, detectra_probe}

# A2: gates — no silent waivers
jq '.[] | select(.pass==false)' artifacts/<new-tag>/gates.json   # empty OR each has WAIVER.md + user sign-off

# A3: abstention target (wild mix, held out)
# issued-FPR ≤ 2% AND abstention ≤ 55%   (from 65-73%)

# A4: latency budget
python deepfake/scripts/perf_bench.py --modality image    # ensemble ≤ 1.5 s/image warm (8-core)

# A5: CI fast tier still green after channel swap
pytest -m fast -q
```

## Staged targets (honest)
- **Stage 1 (public corpora):** ensemble beats every single member on OpenFake AND
  CommunityForensics-Eval; abstention ≤55% @ FPR<2%.
- **Stage 2 (DF-Eval, if access):** 0.60-0.65 AUC interim; 0.73 stretch ONLY
  (conflicted source number). Never promise externally.
- **Ceiling statement for all comms:** best evidenced open result on this benchmark
  is 81% accuracy (tuned probe); commercial 0.90 AUC is not reachable open-source.

## Risks
- kornia/pytorch_wavelets version churn on Python 3.13 (fetch-gate first).
- Probe training data licensing (OpenFake NC) — keep commercial-safe roster:
  CommunityForensics + self-generated corpora only if commercial product intended.
- Stacker retrain can shift conformal alpha — re-derive per `train_stacker.py:80-86`
  and re-bind at serve time; smoke-test the API alpha override afterwards.
