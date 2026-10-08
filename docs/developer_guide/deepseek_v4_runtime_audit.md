# DeepSeek V4 runtime interface audit

Audit date: 2026-10-08. Runtime baseline: `12a2d000da700df91bb6ccbc8ebcdc066ac41473`.
Behavioral reference: `primatrix/sglang-jax` epic/dsv4 at
`ce1ebb6375c5ed0fc5eec6aa360adb22c49ed32d`. This is a pinned comparison,
not a claim about the latest remote branch.

The purpose is to catch integration errors locally before another full static
checkpoint load. New Falcon submissions are paused pending review and user
confirmation. This audit does not establish TPU compilation or model accuracy.

The previous three-layer CPU runtime fixture uses hidden size 128, two query
heads and head dimension 128. That geometry selects generic HCA; CPU dispatch
also selects the reference indexer and mHC implementation. It validates R's
serving lifecycle but misses Flash's production adapters, v7x schedule lookup
and static-FP8 sequence-parallel projections. The new gate supplements those
tests with production dimensions and platform dispatch.

## Findings and corrections

| Boundary | Finding | Correction and regression |
| --- | --- | --- |
| Unified B backend → H mixin → `run_hca` | Current H passes `self.compressor_hidden_size`; the unified backend did not initialize it. The reference execution hardcoded 4096, so copying the reference constructor alone did not satisfy the newer H contract. | Store the config's hidden size and use the same value in the specialized-path geometry check. Flash-geometry tests reach the actual shared H execution adapter. |
| M → B → H normalization | B accepted `norm_eps` but omitted it from the HCA call. The shipped `1e-6` default concealed the mismatch. | Forward the model's epsilon. An execution-level regression checks a nondefault `3e-5` value. |
| M padded RoPE → B fallback | M stores 64 logical rotary columns in a 128-column cache. Without optional pre-split tables, B split at the padded midpoint, producing two 64-column tables instead of `[positions,32]`. | Split by `rope_head_dim`; exercise both pre-split and fallback paths. A regression uses different cos, sin and padding values, checking that padding is excluded. Normal model execution already supplies pre-split tables; this defect affected the optional fallback. |
| CSA sequence parallelism → shared `QuantizedLinear` | `DSV4_LOWRANK_AG=1` passes `P("tensor",None)` inputs to replicated q-lora/KV projections. Main's layer fixed input rows to `P("data",None)`. The reference's row-axis adaptation was missing. | Restore committed row-axis selection from the reference, retaining the contraction-axis restriction. Real FP8 per-channel projections, with BF16 or FP8 activations, match the replicated-input result exactly on logical CPU devices. Flash static-FP8 8K tracing exercises the actual row-local M/CSA path. |

Two previously fixed failures are also included in the audit: mHC's `TPU7x`
schedule and StreamIndex's static `return_scores` argument. The mHC tuning
module matches the pinned reference's executable AST. StreamIndex's score-return
function matches the reference after ignoring its docstring; the target-aware
SparseCore capability helper is retained for offline compilation.

The latest approved experiment, `exp-c9r8emv2bv`, used the baseline commit.
It failed at the missing HCA hidden-size attribute. Prewarm took 174.9 seconds,
M took 27.9 seconds, E took 353.4 seconds, and combined weight loading took
381.3 seconds. Readiness took 645.7 seconds; zero of 13 generation checks
completed. A fresh 2026-10-08 10:40:05 UTC Falcon resource snapshot showed
no lease, no Running/Pending allocation, and a Failed Pod for the experiment.
The first resource response after failure was stale and was not used as
evidence of release.

## Comparison scope

### Completed-batch snapshot correction

Experiment `exp-kc4pihy5rh` used `3deae8eaf` and reached service readiness
in 655.9 seconds after M/E loading in 28.1/350.2 seconds. Its first prefill
result failed in `reclaim_batch_swa`: the overlap result queue uses
`ScheduleBatch.copy()`, which preserved `seq_lens` only when hidden states
were requested. Ordinary generation retained the request list but lost its
completed-forward lengths. No generation check completed.

RFC #1727 items 2 and 4, and C's #1688 completed-forward boundary, require
the previous-result processor to reclaim only consumed SWA pages. The epic
reference calls C with the live request's committed length; this runtime uses
a temporary request view with the completed batch's length to tolerate the
next overlap submission advancing the shared request. That adaptation must
preserve `seq_lens` in every result-queue snapshot, independently of hidden
state collection. C's helper and full-request release path remain unchanged.

The old overlap test constructed a `SimpleNamespace` with lengths already
present, so it did not test their production. It now uses the actual batch
copy and mutates both the source array and shared request before reclaiming.
Additional tests call real prefill/decode output processors and C's helpers
for regular, chunked, mixed, completion and chunk-abort results, in both
overlap and ordinary modes, with page sizes 128/256 and JIT/AOT runner pools.
The worker result and stream output are substituted; these are host lifecycle
tests, not a complete scheduler event loop or TPU numerical verification.

After this correction, the complete runtime test module passed 119 cases:
eight real-copy snapshot cases, 56 result-processing lifecycle cases, and
55 existing runner/config/transport/precompile checks. Both C lifecycle
functions match the pinned epic reference's AST exactly; the chunk-cache
implementation differs only in documentation and a return annotation.

An AST comparison covered 57 tracked modules: the unified backend, DSV4
attention/execution/metadata/compressor/indexer, DSA and HCA kernels, mHC,
M's model graph, config, E's MoE/loader, C's physical pools/allocator, and
the shared quantized linear/matmul boundaries. It checked 532 resolvable
direct function calls and found no remaining explicit argument-binding or
local import errors after the corrections. Class methods, dynamically built
arguments and kernel bodies require additional review or execution; the
static count is not proof that every possible call is valid.

Fourteen dynamic call sites were reviewed separately:

| Sites | Expanded contract |
| --- | --- |
| Five HCA kernel dispatch calls | `shared` supplies compression ratio, head dimension, epsilon, dtype, fused projection and schedule; attention options supply scale, window and both page geometries. Keys match the current compression and attention entry points. |
| One shared H execution call | Mode, schedule, scale, epsilon, dimensions and page geometry reach `hca_step`; fused weight is supplied once. |
| One unified H mixin call | Compressor tensors, cos/sin, input, sink, epsilon and optional fused weight reach `run_hca`; mesh, hidden size, page/context capacity and live cache arguments come from the mixin. |
| Two row-sharded compressor calls | The plan supplies `axis_size`, `rows`, `slots`, `extra`, `local`; both KV and indexer branches use the same plan. |
| Two compressor weight expansions | Only `wkv`, `wgate`, `ape`, `norm_weight`, `cos_sin_cache` are forwarded. Pre-split H tables and fused weights are excluded from generic CSA compressor kwargs. The preceding `functools.partial` compressor construction uses the same filtering. |
| Two ratio-metadata calls | Both ratios receive the same positions, masks, request ownership, lengths, page geometry and padded capacities. |
| One state-reset kernel wrapper | Pallas supplies the five buffer/semaphore refs and the wrapper supplies `capacity`. Kernel-body tracing remains outside the shape-only check. |

## Config and physical layouts

| Surface | Flash contract checked |
| --- | --- |
| Attention projections | hidden 4096, 64 query heads, head dimension 512, rotary dimension 64, 8 output groups, TP8 |
| CSA | ratio 4, index heads 64, index dimension 128, top-K 512, original-token pages 128/256 give compressed pages 32/64 |
| HCA specialization | hidden 4096, heads 64, head dimension 512, rotary dimension 64, window 128; other geometries use the generic implementation |
| HCA C owners | KV `[pages,1,P/128,512]`; FP32 state `[slots,128,2,512]`; updates preserve these physical shapes |
| CSA C owners | BF16 window/compressed/indexer buffers and FP32 continuation state; execution returns the original owner shapes |
| mHC | four 4096-wide residual streams, FP32 gate parameters, config-derived epsilon and Sinkhorn iteration count, v7x schedule |
| Weight preparation | Static dense FP8 projections and scale layouts, grouped/fused output projection, fused HCA compressor weights, padded and pre-split RoPE tables |

All ten attributes read by the HCA mixin are available on the constructed
unified backend after the hidden-size correction. Binding sets actual request
capacity before model freezing. Pool layout, metadata and update schemas are
checked with real C resources, rather than replacement dictionaries.

Some source differences are intentional:

- Current H retains native paged caches and its newer explicit VMEM/schedule
  calculations. Replacing the entire HCA implementation with the older reference
  would discard these dependency changes.
- Main's DSA reference and sparse-attention modules contain shared upstream
  improvements. CSA caller signatures and the optional LSE/score outputs are
  checked against those current APIs.
- M delegates MoE to E and mHC arithmetic to H. The reference's fused seams and
  its monolithic MoE class are not the current dependency interface.
- R owns host metadata transport, immutable overlap snapshots, both-owner
  update commits, lifecycle and precompile capacities. Target selection uses
  the compilation target for CPU-hosted TPU export.

## Local gate and limits

`test/srt/test_deepseek_v4_tpu_contract.py` is registered in the CPU test suite
with eight logical devices. It traces production platform dispatch and actual
M static-FP8 projections, B adapters, C metadata/pools and H execution:

- Both page sizes, all three layer families, and backend-only/model-projection
  entry points.
- First 128-token prefill, an unaligned continuation, ordinary decode, a
  compression-boundary decode, long-context decode and 8K prefill.
- CSA's row-sharded compressor/indexer, including row-local low-rank inputs.
- mHC pre/post/head with 1, 128, 768 and 8192 rows.

Pallas calls are replaced at their output-shape/alias boundary. Their actual
host-side launch construction runs, but mathematical kernel bodies and TPU
lowering do not. Alias indices, shapes and dtypes are checked. These tests
therefore detect attribute, argument, dimension,
dtype, sharding and update-contract failures without allocating full model
weights. Separate numerical regressions check actual H adapter parameter
propagation and the shared row-sharded linear output. Existing small-model
runtime tests continue to cover scheduler, warmup, donation and lifecycle.

Reproduce the new gate:

```sh
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=8 PYTHONPATH=python \
  python -m pytest -q test/srt/test_deepseek_v4_tpu_contract.py
JAX_PLATFORMS=cpu JAX_NUM_CPU_DEVICES=8 PYTHONPATH=python \
  python -m pytest -q python/sgl_jax/test/kernels/quantized_linear_test.py \
  -k preserves_sequence_parallel_rows
```

Validated with JAX/jaxlib 0.11.1 and Flax 0.12.9 on eight logical CPU devices:

| Check | Result |
| --- | --- |
| New Flash production-dispatch, model-projection and mHC contracts | 76 passed |
| CPU-supported shared quantized-linear regressions, including two new row-parallelism checks | 20 passed |
| Runtime, unified backend, packed metadata, StreamIndex API, mHC, config and M/E loading regressions | 185 passed, 2 skipped |
| Pallas alias index/shape/dtype validation in the new contract gate | 76 cases passed; same contract cases, not additional tests |
| Repository pre-commit, including mypy | Passed |

The two local groups contain 281 distinct passing cases in total. Three
TPU-only blockwise matmul cases were deselected from the CPU linear group;
their platform limitation is described below. Passing shape checks are not
numerical verification of the substituted Pallas computations.

CPU results cannot establish SparseCore lowering, compiler VMEM allocation,
kernel math, full static-weight accuracy, TPU performance, or the RFC's six
acceptance rows. The existing blockwise quantized-matmul execution tests require
TPU: three cases fail on CPU because their Pallas calls do not enable interpret
mode. They are not counted as passing CPU checks. Before another full model
experiment, complete the local gates, review the candidate, and obtain user
confirmation for the exact Falcon submission.

## Attribution

The row-axis adaptation comes from the reference's
`python/sgl_jax/srt/layers/linear.py`, lines 492–503, authored by Greg Huang
(`55ae1f7424`). The backend binding and padded-cache fixes and the new
production-contract regressions are integration corrections on this branch.
