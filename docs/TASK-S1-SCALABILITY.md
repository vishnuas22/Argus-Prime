# TASK-S1 — Scalability and Serving 2.0 (Always LAST)

**Phase:** S1 · **Window:** weeks 12-16 (after I1/V1/A1 feature spaces freeze) · **Effort:** 3-5 engineer-weeks

## Evidence (current state)

- Single-process synchronous court; API serializes ALL adjudications through
  `anyio.CapacityLimiter(1)` (`server.py:96,392-441`) — one request at a time.
- The serializer is also a SAFETY mechanism: the conformal-α rebind
  (`server.py:199-210` monkey-patch save/restore), the global embedder mutation
  (`set_embedder`, `argus_detect.py:241-243`), and lazy singletons
  (`_VID_STACKER_CACHE`, `_AUDIO_MODEL`, `_FaceScorer._inst`) are NOT thread-safe.
- ~3.7 s/clip baseline; the UPGRADED pipeline's corrected budget is 2-4 s/clip CPU
  wall-clock (master plan §9) — sub-second needs GPU, state it honestly.
- Verified serving facts (audit): a shared ONNX Runtime session serializes concurrent
  requests (parallelism only from N sessions); ORT gives 1.4-1.9× on transformer-class
  CPU inference but ViT results are mixed (benchmark per channel; test OpenVINO EP on
  Intel); INT8 dynamic quantization of wav2vec2-class models costs ~0.1% accuracy for
  3.6× size reduction; naive padding batching LOSES 5-8× throughput on CPU
  (length-aware grouping only, throughput mode only).

## Steps (order is mandatory)

1. **Thread-safety first** (this is the S1 entry ticket):
   a. Parameterize conformal α per request (pass through the call chain; delete the
      monkey-patch).
   b. Per-request embedder handle instead of the global `set_embedder` mutation.
   c. Locks around lazy singletons; then RAISE the CapacityLimiter.
   d. Race test: 8 concurrent `/api/detect` with distinct α values — every response's
      conformal decision must match ITS OWN α.
2. **Async fan-out**: `asyncio` + `run_in_executor` (or ORT `run_async`) so the three
   modality chains of a video clip run concurrently (audio ∥ visual). Keep the worker
   pool sized to cores.
3. **N sessions per channel**: each ONNX channel gets its own
   `onnxruntime.InferenceSession` count sized to cores; pin
   `intra_op_num_threads` per session (ORT threading docs: global pool or affinity
   partitioning when running multiple sessions).
4. **ONNX exports + parity gates per channel** (frozen models only):
   export CLIP heads, SBI/EffNet, wav2vec2-class encoders, SAFE; gate =
   `max|p_onnx − p_torch| < 1e-3` on a fixed fixture set PLUS the repo's own
   dump-parity law (`verify_dump_parity.py` re-based per channel).
5. **INT8 dynamic quantization** for wav2vec2-class encoders (verify EER delta ≤0.3%
   on the 21LA/ITW exams before promoting); ViT quantization only with PTQ4ViT-class
   twin-uniform method (naive INT8 ViT loses >1%).
6. **perf receipts**: extend `perf_bench.py` to record p50/p95 per modality before/after
   into `benchmarks.json` (scope-labeled); CI nightly records the trend.
7. **Throughput mode (optional)**: length-aware batching for bulk scoring jobs only —
   never for single-clip latency; document that a batch server (Triton/Ray Serve/
   BentoML) only pays off when a single box saturates — no evidence of benefit below
   saturation (audit finding).
8. **Horizontal scale path (design doc only, do not build)**: when p95 SLOs breach on
   a saturated box — per-channel containers behind a queue; keep the court stateless.

## Acceptance criteria + verification commands

```bash
# A1: concurrency race test (the entry-ticket gate)
pytest -m slow tests/test_concurrency_races.py -q      # 8-way α isolation, green

# A2: latency budget (8-core box)
python deepfake/scripts/perf_bench.py --all
# image p50 ≤ 1.5 s | video p50 ≤ 4.0 s | audio p50 ≤ 2.0 s (30 s clip)
# before/after table present in artifacts/benchmarks.json

# A3: ONNX parity per exported channel
pytest -m slow tests/test_onnx_parity.py -q            # max|Δp| < 1e-3

# A4: INT8 accuracy gate
# wav2vec2-class: EER delta ≤ 0.3% on 21LA + ITW exams vs fp32

# A5: throughput (optional mode)
python deepfake/scripts/bench_throughput.py            # length-aware batching ≥ naive
```

## Risks
- Doing S1 before the feature spaces freeze = redoing every parity gate per channel
  (executor's warning; this is why S1 is scheduled last).
- ONNX export quirks per architecture (timm EfficientNet key formats, wav2vec2
  conv-feature extractors) — budget one export-spike day per channel family.
- Quantization accuracy cliffs on some channels — keep fp32 fallback behind a flag
  and let the parity gate decide promotions.

## Honest bounds (repeat in every status report)
- Single-clip sub-second latency = GPU territory; do not chase it on CPU.
- The corrected CPU target for the full upgraded pipeline is 2-4 s/clip wall-clock;
  throughput scales with sessions/cores, latency does not.
