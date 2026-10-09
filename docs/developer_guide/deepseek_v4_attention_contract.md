# DeepSeek V4 A/B attention dependency

This change supplies the missing attention dependency for R (#1727). It follows
the functional ownership in RFC #1678 and the A/B contract in #1693: A computes
compression, selection and attention; B produces metadata, dispatches layers and
packages replacement arrays. M owns model parameters, projections and RoPE.
R owns serving transport, update commits, precompilation and scheduling.

## Dependency stack

Base: official `sgl-project/sglang-jax` main at
`61b304ce8a09af214626cf7a77f59f92e3185d4b`.

| PR | Source commit | Runtime branch treatment |
| --- | --- | --- |
| C #1695 | `e130489dee433335672adf14615f3bd898cc5eec` | Merge already in base |
| CSA compressor #1663 | `220d0442c92c254d66a4e875b76895010cd7e357` | Merge already in base |
| HCA adapter #1726 | `f27d3a92300c0ac35047b99826b8d02518968547` | Merge already in base |
| CSA attention #1664 | `5b01168e501546cdaca9244218e549a135515f7c` | Cherry-picked with `-x` |
| H #1737 | `f252b6240ae430557a793fe1bf17312bd2087fba` | Cherry-picked with `-x` |
| E #1740 | `1047e37bbc02afde21b58bcacdc22e56b7cf33d0` | Cherry-picked with `-x` |
| M #1742 | `3d8ab279de0abcde00cedda908dad833be064f97` | Cherry-picked with `-x` |

## Backend contract

`DeepseekV4AttentionBackend` provides a single model-facing entry point for
physical ratios 0, 4 and 128. Bind the C request pool and allocator with
`bind_resources`; resource owners are passed to metadata preparation and remain
outside the Flax model graph.

`get_forward_metadata(batch, request_pool=..., allocator=...)` validates positions,
request slots and C's SWA ownership mapping. It alone derives read tables,
completed compression boundaries, write locations and first-execution reset masks.
It returns `DeepseekV4RuntimeMetadata` containing one **host NumPy int32 vector**
and a static pytree layout. `resolve()` and `hca_metadata()` decode that vector
inside execution. The caller must transport the vector using the shared serving
path; the backend does not call `device_put` or introduce a submission path.

The model passes projected queries and shared KV, `CompressorWeights`,
`IndexerInputs`, layer identity and attention parameters. The backend returns
`(attention_output, per_layer_updates)`. `pack_pool_updates` uses C's
`build_buffer_updates` to assemble complete `token_to_kv_pool` and
`compressor_state_pool` payloads, retaining untouched layers and families. It
never commits those payloads or changes host allocation state.

CSA emits completed groups only, retains the eight-row continuation ring across
chunks, and selects visible index entries before joint window/compressed attention.
HCA keeps C's native `[pages, 1, P/128, D]` KV and `[slots, 128, 2, D]` state layouts.
The generic HCA executor uses a packed state view internally and returns the
original physical shapes. Fresh/recycled requests reset state only when their
prefix length is zero; padding slots and unrelated requests remain unchanged.

The default production execution is the pinned epic path: native CSA compressor
and indexer, fused CSA prefill, request-local decode and the existing HCA mixin.
CPU uses dense attention and the reference indexer; aligned compressor kernels
can run in Pallas interpret mode. The additional operators from #1663/#1664 are
present in the dependency stack. Selection wiring and a same-hardware comparison
of those alternative implementations remain separate from this baseline port,
as allowed by the A/B discussion's R-side dispatch agreement.

## Adaptations to main

- Metadata preparation returns a host vector; the epic-specific direct upload
  and lazy-host transport dependency are removed.
- The standalone dummy-batch preparation helper is excluded. R registers and
  prepares serving/precompile batches separately.
- Generic HCA resets/executes a packed state view and uses buffer-rank-aware
  partition specs so C's native four-dimensional buffers remain valid.
- `state_init.py` gains epic's optional reset-kernel switch; the existing DMA
  implementation is reused.
- The shared DSA query-block kernel gains optional `return_lse` and aligned
  single-row reads for sparse CSA's sink normalization. Main's native 4-D cache
  handling and writeback remain intact. Existing callers default to array output.
- Focused backend and numerical tests are registered in `test/srt/run_suite.py`.

No model-runner registration, pool initialization/budget integration, batch-device
transfer, scheduler lifecycle call sites, runtime update commit, or serving/AOT
bucket registration is implemented by this dependency commit.

## Source and attribution

Port source: [`primatrix/sglang-jax` epic/dsv4](https://github.com/primatrix/sglang-jax/tree/ce1ebb6375c5ed0fc5eec6aa360adb22c49ed32d).
The following file-level table compares this port with that pinned source.
Author counts come from `git blame --line-porcelain` on the **source** file;
changes made for main are reported separately rather than assigned epic blame.
Brian: `donghouze666@outlook.com`; Greg Huang: `debin.huang@gmail.com`;
Lifei Chen: `hustclf@gmail.com`.

| Imported/adapted file | Delta vs source (+/- lines) | Source blame (lines) |
| --- | --- | --- |
| `python/sgl_jax/srt/kernels/csa_decode.py` | +0 / -0 | Brian: 20, Greg: 344 |
| `python/sgl_jax/srt/kernels/dsa/sparse_mla_prefill_qblock.py` | +28 / -13 | Greg: 704 |
| `python/sgl_jax/srt/kernels/dsv4/compressor_tail.py` | +0 / -0 | Brian: 3, Greg: 178 |
| `python/sgl_jax/srt/kernels/dsv4/csa_decode_attention.py` | +0 / -0 | Greg: 134 |
| `python/sgl_jax/srt/kernels/dsv4/csa_flash_attention.py` | +0 / -0 | Brian: 1, Greg: 141 |
| `python/sgl_jax/srt/kernels/dsv4/state_init.py` | +2 / -1 | Greg: 97 |
| `python/sgl_jax/srt/kernels/dsv4/topk_threshold.py` | +0 / -0 | Greg: 111 |
| `python/sgl_jax/srt/layers/attention/deepseek_v4_backend.py` | +12 / -31 | Brian: 306, Greg: 257, Lifei: 8 |
| `python/sgl_jax/srt/layers/attention/dsv4/attention.py` | +0 / -0 | Brian: 204, Greg: 206 |
| `python/sgl_jax/srt/layers/attention/dsv4/compressor.py` | +0 / -0 | Brian: 262, Greg: 74 |
| `python/sgl_jax/srt/layers/attention/dsv4/decode.py` | +0 / -0 | Brian: 54, Greg: 286 |
| `python/sgl_jax/srt/layers/attention/dsv4/dispatch.py` | +10 / -31 | Brian: 316, Greg: 490 |
| `python/sgl_jax/srt/layers/attention/dsv4/execution.py` | +11 / -5 | Brian: 242, Greg: 99 |
| `python/sgl_jax/srt/layers/attention/dsv4/indexer.py` | +0 / -0 | Brian: 46, Greg: 191 |
| `python/sgl_jax/srt/layers/attention/dsv4/metadata.py` | +0 / -0 | Brian: 460 |
| `python/sgl_jax/srt/layers/attention/dsv4/ref/__init__.py` | +0 / -0 | Brian: 1 |
| `python/sgl_jax/srt/layers/attention/dsv4/ref/compressor.py` | +0 / -0 | Brian: 72 |
| `python/sgl_jax/srt/layers/attention/dsv4/ref/decode_attention.py` | +0 / -0 | Brian: 24 |
| `python/sgl_jax/srt/layers/attention/dsv4/ref/indexer.py` | +0 / -0 | Brian: 155 |
| `python/sgl_jax/srt/layers/attention/dsv4/ref/topk_threshold.py` | +0 / -0 | Brian: 14 |
| `test/srt/test_deepseek_v4_compressor.py` | +0 / -0 | Brian: 57, Greg: 62 |
| `test/srt/test_deepseek_v4_compressor_row_shard.py` | +0 / -0 | Brian: 4, Greg: 128 |
| `test/srt/test_deepseek_v4_csa_attention.py` | +0 / -0 | Brian: 19, Greg: 62 |
| `test/srt/test_deepseek_v4_csa_decode.py` | +0 / -0 | Brian: 85, Greg: 98 |
| `test/srt/test_deepseek_v4_csa_decode_segments.py` | +0 / -0 | Brian: 2, Greg: 161 |
| `test/srt/test_deepseek_v4_indexer_kernel.py` | +0 / -0 | Brian: 3, Greg: 171 |
| `test/srt/test_deepseek_v4_indexer_row_shard.py` | +0 / -0 | Brian: 7, Greg: 129 |
| `test/srt/test_deepseek_v4_packed_metadata.py` | +0 / -0 | Greg: 103 |
| `test/srt/test_deepseek_v4_topk_threshold.py` | +0 / -1 | Brian: 2, Greg: 173 |

The shared DSA file's delta includes retained upstream improvements because this
is a selective adaptation, not a replacement with the old epic file. The new
`test_deepseek_v4_backend.py` is original integration coverage at the B/C boundary.
The port commit credits Brian and Greg Huang in `Co-authored-by` trailers.

## Validation

CPU tests use JAX/jaxlib 0.11.1 and Flax 0.12.9. Logical multi-device tests validate
array sharding and operator interfaces; they do not establish multi-host serving
or DP>1 model support. HCA host-metadata coverage uses the published v7x schedule
with eight local heads; it does not execute a TPU kernel.

Coverage includes host-only metadata generation, inconsistent ownership rejection,
reordered/mixed and padded requests, page-size-128 decode tables, functional pool
updates, unaligned chunk continuation across CSA/HCA boundaries, recycled state
slots, numerical compressor/attention/selector checks, row sharding and existing
DSA query-block parity. TPU serving/performance acceptance remains with R after
this dependency is reviewed.

Completed checks:

| Check | Result |
| --- | --- |
| Unified backend and packed metadata | 20 passed |
| Compressor/indexer row sharding, eight logical CPU devices | 11 passed |
| Existing C/config/M/E/H regressions | 123 passed, 22 platform/kernel skips, 30 subtests passed |
| Imported compressor/CSA attention/decode/selector and existing DSA parity | Passed; two platform skips in the indexer/DSA group |
| Three-layer M construction with real A/B imports (ratios 0/4/128) | Passed on a two-axis explicit mesh |
| Repository pre-commit hooks, all files | Passed, including mypy |

Focused checks can be reproduced in a Python environment with the repository's
dependencies:

```sh
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 PYTHONPATH=python \
  python -m pytest -q test/srt/test_deepseek_v4_backend.py \
  test/srt/test_deepseek_v4_packed_metadata.py
JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=python \
  python -m pytest -q test/srt/test_deepseek_v4_compressor_row_shard.py \
  test/srt/test_deepseek_v4_indexer_row_shard.py
pre-commit run --all-files --hook-stage manual
```

The local run used `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` to isolate it from unrelated
plugins installed in the reused dependency environment. No Falcon TPU job or
serving benchmark has been run for this dependency commit.
