# Multi-head Latent Attention (MLA) Design

## Summary

This document describes the design and testing plan for the `MLAAttention` layer in sglang-jax, targeting models that use the MLA architecture (DeepSeek-V2/V3, etc.).

The current implementation uses **non-absorbed mode**: `MLAAttention` fully decompresses the latent state into Q/K/V tensors during forward (V is zero-padded to match K's head_dim for KV cache compatibility), then reuses the existing `RadixAttention` + `MHATokenToKVPool` infrastructure without relying on any MLA-specific attention kernel or compressed KV cache. The goal is to correctly integrate the MLA data flow with minimal system changes.

`MLAAttention` is a reusable layer module located at `srt/layers/mla.py`, alongside other layers (`linear.py`, `layernorm.py`, etc.). It does not introduce a new attention backend — attention computation is dispatched through `RadixAttention` to the existing backend. A future absorbed mode implementation would require adding a dedicated backend under `srt/layers/attention/` along with `MLATokenToKVPool`.

## Background and Goals

### Goals

- Provide a reusable MLA layer that correctly implements the MLA Q/K/V projection pipeline.
- Integrate with existing attention infrastructure (`RadixAttention`, `MHATokenToKVPool`) without modifying these components.
- Handle K_dim != V_dim by padding V to match K's head_dim, enabling use of the fused `MHATokenToKVPool`.

### Non-Goals

- Absorbed mode (caching compressed state instead of decompressed K/V). Requires a custom attention kernel and is a separate effort.
- Model-level integration (weight loading, model config parsing). This layer provides the building block; model files compose it.

## Design Overview

`MLAAttention` is implemented as a Flax NNX module. It performs the full MLA data flow internally and outputs standard attention results, making it a drop-in replacement for standard multi-head attention in model definitions.

**Data flow:**

```mermaid
graph LR
    hidden[hidden] --> q_a_proj --> q_norm[RMSNorm] --> q_b_proj --> q_split["split(q_nope, q_rope)"]
    hidden --> kv_a_proj --> kv_split["split(compressed, k_rope_raw)"]
    kv_split -- compressed --> kv_norm[RMSNorm] --> kv_b_proj --> kv_split2["split(k_nope, v)"]
    q_split -- q_rope --> RoPE
    kv_split -- k_rope_raw --> RoPE
    RoPE -- q_rope' --> assemble["concat -> Q, K"]
    q_split -- q_nope --> assemble
    kv_split2 -- k_nope --> assemble
    RoPE -- k_rope' --> assemble
    assemble --> RadixAttention
    kv_split2 -- v --> pad["pad V to qk_head_dim"]
    pad --> RadixAttention
    RadixAttention --> strip["strip V padding"]
    strip --> o_proj --> output[hidden]
```

**Key design decisions:**

1. **Non-absorbed mode**: All decompression is completed in `MLAAttention.__call__`, then decompressed Q/K/V are handed to `RadixAttention`. The attention layer is unaware of the low-rank origin.
2. **V padding for fused KV cache**: After MLA decompression, K has head_dim=qk_head_dim and V has head_dim=v_head_dim. Since `MHATokenToKVPool` requires K and V to have the same shape, V is zero-padded to qk_head_dim before being passed to `RadixAttention`. After attention, the padding is stripped from the output before the output projection.

### Constraints and Boundaries

MLA has two implementation strategies — the current implementation uses non-absorbed mode:

| | Non-absorbed (current) | Absorbed (not implemented) |
|---|---|---|
| Data flow | Compress -> **decompress to Q/K/V** (V padded to qk_head_dim) -> standard attention | Compress -> **cache compressed state** -> decompress inside attention |
| KV cache content | Decompressed full K/V | Compressed state (kv_lora_rank + qk_rope_head_dim) |
| Per-token KV cache | heads x (qk_head_dim + qk_head_dim) (V padded to qk_head_dim) | kv_lora_rank + qk_rope_head_dim |
| Attention kernel | Standard `RadixAttention` + `MHATokenToKVPool` | Custom kernel needed (`MLATokenToKVPool` defined but not wired in) |
| Memory efficiency | Low (24576/576 ≈ 43x vs absorbed) | High |

The current implementation completes all decompression in `MLAAttention.__call__`, outputting decompressed Q/K/V tensors (V padded to qk_head_dim) before handing off to `RadixAttention`. The attention layer does not need to know the data originates from a low-rank decomposition. The padding is stripped after attention, before `o_proj`. See [Appendix: Absorbed Mode Roadmap](#appendix-absorbed-mode-roadmap) for the absorbed mode roadmap.

### Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Non-absorbed mode uses ~43x more KV cache than absorbed mode | Acceptable for initial implementation; absorbed mode is a future optimization |
| bf16 precision loss in chained matmuls | Q/K/V projection pipeline validated via cosine similarity (>= 0.99) against fp32 reference; attention and o_proj numerical correctness is covered by model-level accuracy tests |

## Design Details

**Module:** `MLAAttention` (Flax NNX)

**Projections:**

| Layer | Shape | Sharding |
|-------|-------|----------|
| `q_a_proj` | (hidden_size, q_lora_rank) | kernel: (None, None) |
| `q_b_proj` | (q_lora_rank, num_heads x qk_head_dim) | kernel: (None, "tensor") |
| `kv_a_proj` | (hidden_size, kv_lora_rank + qk_rope_head_dim) | kernel: (None, None) |
| `kv_b_proj` | (kv_lora_rank, num_heads x (qk_nope_head_dim + v_head_dim)) | kernel: (None, "tensor") |
| `o_proj` | (num_heads x v_head_dim, hidden_size) | kernel: ("tensor", None) |

**Constructor Parameters:** `hidden_size`, `num_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`, `qk_rope_head_dim`, `v_head_dim`, `rope_theta`, `rope_interleave`, etc., provided by model configuration.

**Output Shapes (pre-padding):** Q = [tokens, num_heads, qk_head_dim], K = [tokens, num_heads, qk_head_dim], V = [tokens, num_heads, v_head_dim], where qk_head_dim = qk_nope_head_dim + qk_rope_head_dim. Note: V is zero-padded to qk_head_dim before entering RadixAttention.

**Attention Scaling:** `scaling = qk_head_dim^(-0.5)`. The scaling is applied to the full Q.K^T after concatenation, not to the nope or rope portions individually.

## Testing

### Unit Tests

Tests validate the Q/K/V projection pipeline by stepping through `MLAAttention` submodules individually and comparing against an fp32 numpy reference implementation. The reference follows the standard MLA computation flow, adapted for the sglang-jax weight layout (`weight=(in_features, out_features)`, forward = `x @ weight`).

**Test cases:**

| Test | What it verifies |
|------|-----------------|
| `test_output_shapes` | Q, K, V output tensor shapes match expected dimensions |
| `test_q_path` | Q projection chain: `q_a_proj -> RMSNorm -> q_b_proj -> reshape` |
| `test_kv_path` | KV projection chain: `kv_a_proj -> split -> RMSNorm -> kv_b_proj -> reshape -> split` |
| `test_full_qkv` | Full Q/K/V projection with RoPE applied and tensors assembled |
| `test_k_rope_broadcast` | `k_rope` broadcast from 1 head to all heads (all heads must be identical) |

Tests use production-scale dimensions sourced from the target model configuration.

#### Correctness Metric

For production-scale dimensions with bf16 matmuls, element-wise `allclose` is highly sensitive to tolerance settings, leading to two problems:

- Tolerances too strict cause persistent false positives
- Relaxed tolerances make the assertion itself lose discriminating power

Based on empirical calibration results, we use cosine similarity (threshold >= 0.99) as the primary correctness metric, measuring directional agreement between outputs and the fp32 reference.

> FlashInfer uses the same approach in their absorbed-MLA decode kernel tests ([flashinfer-ai/flashinfer#551](https://github.com/flashinfer-ai/flashinfer/pull/551#discussion_r1826453290)). Maintainer confirmed: `"cosine similarity is okay in this case."`

#### Reference Implementation

Tests use a numpy fp32 reference implementation as the ground-truth oracle, consisting of the following functions:

| Function | Purpose |
|----------|---------|
| `numpy_linear_fp32(x, weight)` | `x @ weight` in fp32 |
| `numpy_rmsnorm_fp32(x, scale, eps)` | RMSNorm |
| `numpy_rotary_emb_fp32(x, cos, sin)` | Interleaved RoPE |
| `numpy_mla_qkv_fp32(hidden, positions, weights, config)` | Complete Q/K/V projection pipeline |

### Integration Test

`test_radix_attention_integration`: Simulates one decode step to verify that MLA output can be correctly accepted and processed by `RadixAttention` + `MHATokenToKVPool`.

**Scenario:** 2 sequences, KV lengths of 4 and 6 (cached history tokens), each sequence currently processing 1 new token (2 query tokens total).

**Flow:**

1. Create `MHATokenToKVPool` (fused K/V buffer), write random prefix KV into the pool to simulate cached historical K/V
2. Build `cache_loc` and `ForwardBatch`, specifying the physical position of each token's KV in the pool
3. 2 new tokens' hidden states enter `MLAAttention.__call__`, completing Q/K/V projection and RoPE
4. Hand off to `RadixAttention`: write new tokens' K/V into the pool, perform attention over all historical K/V

**Assertions:** Output shape is `[2, hidden_size]`, no NaN/Inf.

> The test uses random weights and random prefix KV without verifying numerical correctness — the focus is validating that MLA's output shapes and dtypes are correctly accepted by the downstream attention + KV cache infrastructure. Numerical correctness is covered by unit tests.

### Tests Not Covered in This Document

This document does not define model-level e2e accuracy baselines. Model-level end-to-end accuracy is covered separately by the accuracy test suite; this document focuses on layer-level numerical correctness and MLA's integration with existing attention / KV cache infrastructure.

## Alternatives

**Absorbed mode**: Cache the compressed latent state instead of decompressed K/V. Requires a custom attention kernel that decompresses on-the-fly during attention computation. ~43x memory reduction but significantly more implementation complexity. Deferred to a future iteration.

## Appendix: Absorbed Mode Roadmap

> Not in scope for this iteration. This appendix documents future optimization directions for reference.

`MLATokenToKVPool` is already defined in `srt/mem_cache/memory_pool.py` with compressed buffer layout `[size, 1, kv_lora_rank + qk_rope_head_dim]`, but is not wired into the attention pipeline. The full chain for absorbed mode requires changes across multiple components:

```mermaid
graph TD
    subgraph "MLAAttention.__call__"
        hidden[hidden] --> Q_path["Q path (same as non-absorbed)"]
        Q_path --> Q["Q [tokens, num_heads, qk_head_dim]"]
        hidden --> kv_a_proj["kv_a_proj"]
        kv_a_proj --> compressed["compressed [tokens, 1, latent_dim]<br/>(skip kv_b_proj decompression)"]
    end

    subgraph "RadixAttention"
        Q --> FA["FlashAttention.__call__()"]
        compressed --> FA
        FA --> call_mla["_call_mla() <- new dispatch path"]
    end

    subgraph "MLATokenToKVPool"
        call_mla -- write --> pool["set_kv_buffer(compressed)"]
    end

    subgraph "Custom Pallas Kernel"
        call_mla --> kernel["read compressed -> kv_b_proj decompress -> attention"]
    end
```

Required changes:

| Component | Change | Complexity |
|-----------|--------|-----------|
| `MLAAttention.__call__` | Skip `kv_b_proj` decompression, output compressed state directly | Low |
| `RadixAttention` | Interface adaptation for compressed KV (or keep compatible via `KVCache` base class) | Low |
| `FlashAttention.__call__` | Add `isinstance(pool, MLATokenToKVPool)` dispatch to `_call_mla` | Medium |
| Pallas attention kernel | Fuse `compressed @ kv_b_proj.weight` decompression into block-wise attention loop | High |
