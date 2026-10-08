# DeepSeek V4 Flash runtime

This is Task R's serving binding for [RFC #1727](https://github.com/sgl-project/sglang-jax/issues/1727).
The numerical model, attention, MoE and mHC implementations are supplied by
M, A/B, E and H. Resource layout, planning, allocation and lifecycle helpers
are supplied by C. R connects these contracts to the existing serving loops.

## Supported configuration

The initial path serves Flash text inference with one DP rank, BF16 KV and
indexer buffers, FP32 compressor state, and original-token pages of 128 or
256. Prefill, chunk continuation, decode and scheduler mixed batches are
supported. `ScheduleBatch.mix_with_running` represents mixed batches as
EXTEND with a one-token query for each running decode request.

Use `--disable-radix-cache`. Prefix reuse, MTP/speculative decoding, PD
disaggregation, FP8 KV, LoRA and expert placement remapping are rejected
before loading the model. V4 Pro is outside the validated scope.

For example, on the RFC's target topology:

```sh
python -m sgl_jax.launch_server \
  --model-path /path/to/deepseek-v4-flash-static-expert-fp8 \
  --tp-size 8 --ep-size 8 --dp-size 1 \
  --dtype bfloat16 --kv-cache-dtype auto \
  --disable-radix-cache --page-size 128 \
  --max-running-requests 112 \
  --max-prefill-tokens 8192 --chunked-prefill-size 8192
```

The checkpoint format and validation remain M/E's responsibility; consult
`deepseek_v4_m_contract.md` and `deepseek_v4_attention_contract.md` for the
module contracts. The example is a launch configuration, not a measured
performance result.

## Pool construction and update ownership

`ModelRunner` selects `DeepseekV4AttentionBackend` independently of the
ordinary MLA/FA setting. V4 does not enable V3 MLA absorption or DSA flags.
Its model graph is frozen after the runtime binds the actual request capacity.

`_init_deepseek_v4_memory_pool` derives `DeepseekV4CacheSpec` from the model
configuration and passes post-weight available bytes, minus execution and
embedding reservations, to C's `plan_deepseek_v4_pools`. The planner accounts
for every KV family, page-zero padding and the extra state request slot. It
honors `max_total_tokens`, the SWA/history ratio and the SWA admission floor.
`build_deepseek_v4_pools` constructs the request pool, two device owners and
C's request-owned allocator; there is no additional runtime allocator.

Both serving JIT entry points validate the complete updates before donation:
owner keys, family keys, layer counts, shapes and dtypes. The existing
`MemoryPools.replace_all` call immediately after model dispatch commits
`token_to_kv_pool` and `compressor_state_pool` once per step. Following steps
consume the replacement arrays, including in overlap mode.

## Metadata and request lifecycle

`ModelRunner.get_attention_metadata` passes the live host request pool and
allocator to B's single `get_forward_metadata` producer. B owns compression
events, masks, read tables and write-address derivation.

`ForwardBatch.init_new` transports B's packed vector together with ordinary
batch metadata through the shared `packed_device_array` helper. The large
cache-address vector keeps its ordinary separate upload. Compatible scalar
fields are grouped by canonical dtype and leading-axis sharding, packed per
DP rank, and unpacked by a cached JIT. Every host group owns an immutable
snapshot so overlapping batches cannot overwrite staging data.

V4 metadata is a dynamic child of `ForwardBatch`, not a second copy in the
runner's backend. Inside the model JIT it is attached to the trace-local
backend. The overlap worker supplies its already-produced host metadata to
`ForwardBatch.init_new`, avoiding a second derivation or V4-only transfer.

The cache factory selects `DeepseekV4ChunkCache` for C's allocator. Chunk
continuation retains the request's committed mapping. Normal completion,
abort and retraction use the existing `release_kv_cache` entry point, which
calls C's `release_req` to release the committed prefix and allocated tail
as one extent. Decode admission and retraction use C's exact history/SWA
allocation demand, including a missing SWA mapping on an existing history page.

`ScheduleBatch.maybe_evict_swa` returns early for V4. Both prefill and decode
result processing call `reclaim_completed_v4_swa` only after resolving the
submitted result. Reclamation uses the completed batch's sequence-length
snapshot: a shared request may already describe the next decode in overlap
mode. C advances the reclamation cursor; R preserves the live request length.
A reused state slot is initialized by B's zero-prefix initialization mask on
its first forward.

## Precompile and executable store

`CompilationManager.iter_precompile_batches` supplies both parallel
compile preparation and serial warmup. V4 dummy requests are all inactive,
use C's padding request slot and have no write addresses. Executing warmup
does not alter any KV or compressor-state buffer or host allocation ledger.

The plan enumerates reachable power-of-two CSA and HCA capacity combinations,
not just a single-request context ladder. EXTEND CSA uses total batch history;
tuned HCA uses maximum per-request history by default. Page-128 DECODE uses
request-local CSA tables. These dimensions must cover uneven multi-request
lengths as well as chunk continuations. Host-only overrides prepare each
metadata shape and are restored before the batch is yielded.

The offline exporter uses the same planner, C pool layouts, B metadata
producer and runtime capacity variants. Device buffers and dummy weights
remain abstract on the compilation host. Kernel selection honors the target
mesh/compilation target rather than the CPU host. `serving.json` records V4
capacities per model bucket; sampler variants are deduplicated by output
structure, shape, dtype and sharding.

For `--save-aot`, supply explicit `--max-running-requests` and
`--max-total-tokens`, along with the existing TPU topology options. Restore
with `--aot-model-dir` using the same model configuration, resource capacities,
page size, precompile paddings, mesh, kernel environment and JAX stack. The
store validates the retained-input signature and program hash and rejects a
missing or incompatible executable without compiling it.

## Local validation and pending TPU gate

`test/srt/test_deepseek_v4_runtime.py` exercises a real three-layer
SWA/CSA/HCA ModelRunner, with dummy weights and CPU reference kernels. It
covers JIT/parallel warmup, AOT dispatch/serial warmup, pages 128/256, chunked
prefill, decode, mixed batches, both-owner donation and commit, state-slot
reuse, overlap reclamation, retraction/abort, shared transfer snapshots and
saved executable loading/execution. Post-warmup serving checks prohibit
backend compilation and count model lowerings. The full serving-bundle test
substitutes a CPU mesh for topology creation, while exercising the exporter
and loading each serving bucket.

The CPU suite passes 59 runtime cases (58 serving cases plus target selection).
Related resource/scheduler/compiler regressions pass 180 cases with one skip;
kernel/mHC regressions pass 43 cases with one skip. The fixture mocks available device
memory and disables the TPU-only fused output projection and routing kernel.
It uses JAX/jaxlib 0.11.1 and Flax 0.12.9. It does
not establish full-model accuracy, TPU HBM usage or TPU performance.

The first approved Falcon smoke run, `exp-jgt28sijsg` on 2026-10-08,
tested commit `6dc9c772b` on v7x 2x2x1, tp8/ep8, with the
[published static expert-FP8 checkpoint](https://github.com/primatrix/sglang-jax/pull/345).
Remote source-tree and runner hashes matched the reviewed inputs. The
eight-device probe, checkpoint metadata/shard-size validation and M-owned
parameter loading completed. The service then failed in E's MoE loader:
`jax.device_put` received a prototype `NamedSharding` backed by `AbstractMesh`.
The loader now binds that partition spec to the layer's concrete device mesh
when materializing gate, bias and shared-expert parameters. Eight additional
CPU cases exercise this actual `nnx.eval_shape` serving boundary for both
checkpoint formats, both routing modes and EP1/EP2. TPU revalidation of the
fix is pending; the failed run reached no generation or acceptance workload.

No TPU acceptance row has been measured or posted. The six RFC rows (single-request decode, 8K and
32K TTFT, cc64 and cc256 throughput/latency, GSM8K) remain pending on v7x
2x2x1, tp8/ep8, with the static expert-FP8 checkpoint and the reference's 5%
reproducibility band. Keep other hardware results separate.

## Source and attribution

Behavioral reference: `primatrix/sglang-jax` `epic/dsv4`,
`ce1ebb6375c5ed0fc5eec6aa360adb22c49ed32d`. The adaptation targets the
runtime branch's C/M/A/B/E/H interfaces, rather than replacing complete main
files with epic files. The shared transfer helper was selectively restored
from #1717, merge `5f085ce60cadf35eb0325905da2672aaac9e7763` (the broader
main change was reverted in #1739).

The following is a source-fragment blame table. Counts refer to the source
ranges before adaptation, not an assertion that every line was imported.
Paths in the first six rows are relative to `python/sgl_jax/srt/`.

| Source file / fragment | Source lines | Source blame |
| --- | --- | --- |
| `model_executor/model_runner.py`, update validation | 106–114 | Brian, 9 |
| `model_executor/model_runner.py`, resource binding | 923–931 | Brian, 9 |
| `model_executor/model_runner.py`, metadata producer entry | 933–944 | Brian, 12 |
| `model_executor/model_runner.py`, inactive dummy entry | 946–953 | Brian, 8 |
| `model_executor/model_runner_kv_cache_mixin.py`, V4 detection | 903–907 | Brian, 5 |
| `model_executor/model_runner_kv_cache_mixin.py`, pool initialization | 909–959 | Brian, 41; Greg Huang, 6; Yun Ting, 4 |
| #1717 `utils/jax_utils.py`, `_metadata_unpacker` | 300–328 | Brian, 29 |
| #1717 `utils/jax_utils.py`, `packed_device_array` | 331–377 | Brian, 47 |

The file-level adaptations also touch `forward_batch_info.py`, both workers,
`model_forward.py`, compilation/export inputs and resources, scheduler batch,
policy, cache factory and result processing. Capacity-combination enumeration,
completed-length snapshots, abstract C pool support, target-aware dispatch,
masked short-context page anchors and the runtime tests are integration work
on this branch. Numerical operators and C ownership formulas remain with
those dependencies. The commit carries source-author co-author trailers.
