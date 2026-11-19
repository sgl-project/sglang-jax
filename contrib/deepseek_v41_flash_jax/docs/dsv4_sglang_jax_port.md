# Porting DeepSeek V4 / V4.1 Flash to SGL-JAX (TPU v6e)

Draft, 2026-11-16. Based on a read of the main branches of sgl-project/sglang (47361c5), sgl-project/sglang-jax (b3e3e16, Oct 20 shallow clone) and vllm-project/tpu-inference (da76a6a). Nothing here has been run.

## Goal
Serve `deepseek-ai/DeepSeek-V4.1-Flash-1017` (552B backbone + 196B Engram, 510 GB checkpoint, FP4 experts / FP8 rest) on a v6e-32 with SGLang's JAX backend. V4 Flash (284B) comes first because it is a subset of the work.

## Status of others (searched 2026-11-16)
- No issue or PR for V4.1 or Engram in sglang-jax, tpu-inference or primatrix/sglang-jax. Only tpu-inference #3538 / #3656 (KV-spec ownership) mention `deepseek_v41`.
- SGLang (GPU/AMD) has the full V4.1 stack and an active roadmap (#42170). This is the reference.
- tpu-inference has V4 (not V4.1) as experimental, including an MXFP4 W4A16 expert path.
- Before starting: post on sglang-jax asking whether V4/V4.1 is planned, to avoid duplicate work.

## What SGL-JAX already has (V4 building blocks)
`configs/deepseek_v4.py` (expert_dtype fp4), `layers/deepseek_v4_mhc.py`, `layers/attention/dsv4/hca.py`, `kernels/dsv4/{paged_row_write,state_init}.py`, `mem_cache/deepseek_v4/{allocator,capacity,pool,state}.py`. No V4 model file in `srt/models/` (only `deepseek_v3.py`).

## Gaps, in dependency order
| # | Piece | GPU reference (SGLang) | Notes for JAX/TPU |
|---|---|---|---|
| 1 | V4 model file (embeddings, 40 layers, MoE routing: sqrtsoftplus, noaux_tc, hash routing, shared expert) | `srt/models/deepseek_v4.py` (6k lines) | Start from `deepseek_v3.py`; tpu-inference `models/vllm/experimental/deepseek_v4.py` is a second reference. |
| 2 | Weight loading + FP4/FP8 handling | `deepseek_v4.py`, `inference/convert.py` in the HF repo | v6e has no FP4/FP8 MXU: dequantize experts to INT8 (32-element scales) or keep packed with in-kernel dequant. tpu-inference `layers/vllm/quantization/mxfp4.py` shows the dequant-then-requantize path. |
| 3 | Attention: CSA / HCA / SWA with compress ratios (0, 2 and 1 here; 4/128 in V4), indexer top-k | `layers/attention/deepseek_v4_backend.py`, `dsv4/indexer.py`, `dsv4/v41_indexer/` | Pallas kernels needed: tpu-inference already has `kernels/experimental/deepseek_v4/` (mla, mla_swa, sparse_mla, csa_gather, streamindex_topk, compress_and_store) that can be reused or adapted. |
| 4 | mHC residual mixing (hc_mult 4, Sinkhorn) | `deepseek_v4_mhc.py` | SGL-JAX has a layer; tpu-inference has Pallas kernels. V4.1 uses a revised single-pass variant. |
| 5 | Memory pools for compressed KV, SWA, compressor state | `mem_cache/deepseek_v4_memory_pool.py` | SGL-JAX `mem_cache/deepseek_v4/` covers the V4 Flash backbone. |
| 6 | **V4.1 only: Engram** (2 layers, ~196B params as 2 hashed n-gram tables, 8 heads x 256 dim, ngram size up to 4) | `srt/layers/engram.py`, `engram_staging.py`, `kernels/ops/embeddings/engram_*` | Lookup depends only on token ids, so it can be hashed and prefetched ahead of the layer. Tables are FP8 in the checkpoint; on v6e dequantize rows after gather. Keep tables in host RAM (or HBM on v6e-32: ~196 GB). Needs a host-gather + device_put pipeline in the scheduler. |
| 7 | **V4.1 only: Causal Encoder-Decoder** (20+20 layers; decoder KV projected from final encoder states; 8B active at prefill / 16B at decode) | `deepseek_v4.py`, `deepseek_v4_boundary.py` | Changes the KV cache and prefill/decode execution paths, not just layers. Largest design risk. |
| 8 | **V4.1 only: SWA bounded replay** | `deepseek_v4_replay_tail.py` | Can be skipped at first (persist SWA KV; costs memory). |
| 9 | Vision tower / DSpark speculative decoding / MTP | `deepseek_v4_dspark.py` | Out of scope for the first version (text-only, no speculation). |

## Phases
1. **V4 Flash text-only on SGL-JAX** (items 1, 2, 3, 4, 5). Correctness first: compare logits per layer against SGLang GPU or the HF reference on short prompts. Serves as the vLLM fallback.
2. **V4.1 backbone without Engram** (7) to validate the encoder-decoder path (output quality will be wrong without Engram; check shapes and logits against the reference only).
3. **Engram** (6): hash + gather + gate; start with tables in HBM on v6e-32, then host offload.
4. Optimizations: bounded replay (8), INT8 experts, prefix caching.

## Validation
- Per-layer activation diffs vs. a GPU reference run (needs a few GPUs or the hosted model with logprobs).
- End-to-end: scripts/eval_quality.sh (GSM8K, MMLU-Pro, IFEval) vs DeepSeek's model-card scores; scripts/bench_sweep.sh for throughput.

## Risks / unknowns
- Effort: this is weeks of work, not days (the GPU model file alone is ~6k lines plus kernels). I have not estimated it more precisely.
- No TPU kernels for V4.1 indexer (`v41_indexer`), candidate-block selection, or Engram gather/gate; they are Triton/CUDA/HIP on GPU and need Pallas or plain JAX versions.
- Multi-host (v6e-32 = 8 hosts x 4 chips) with expert parallelism: unverified for these layers in SGL-JAX.
- SGL-JAX clone is from Oct 20 on a shallow checkout; re-check main before starting.
- Hosted comparison API for V4.1 quality is not set up; we compare to published scores instead.

## Next actions
1. Ask in sglang-jax whether anyone is porting V4/V4.1.
2. Decide the TPU slice for development (a small v6e-8 cannot hold V4.1; needs v6e-32, or dev with random-init tiny config on CPU/v6e-4).
3. Build a tiny-config V4.1 (few layers, small dims) to develop and test on CPU/v6e-4 before using the big slice.
