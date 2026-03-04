# Split KV Cache Implementation for Non-Aligned Head Dimensions

This document summarizes the changes implemented to support separate Key (K) and Value (V) cache storage in `sglang-jax`. This optimization addresses the memory inefficiency caused by padding when `k_head_dim` and `v_head_dim` differ (e.g., Qwen2-MoE with 192 and 128 respectively).

## Overview

Previously, K and V caches were fused into a single tensor with interleaved storage (`[K, V, K, V...]`). To support different head dimensions, both had to be padded to the maximum dimension, causing significant memory waste.

The new implementation detects dimension mismatch and automatically switches to a **Split Storage Mode**, where K and V are stored in separate, independently aligned buffers.

## Key Changes

### 1. Memory Pool (`python/sgl_jax/srt/mem_cache/memory_pool.py`)

*   **Logic**: `MHATokenToKVPool` now accepts `v_head_dim`. If `head_dim != v_head_dim`, it sets `is_split=True`.
*   **Storage**:
    *   **Fused Mode (Old)**: Allocates `kv_buffer` with shape `[size, num_heads * 2, head_dim]`.
    *   **Split Mode (New)**: Allocates `k_buffer` `[size, num_heads, head_dim]` and `v_buffer` `[size, num_heads, v_head_dim]`.
*   **Operations**: Updated `set_kv_buffer`, `get_kv_buffer`, `replace_kv_buffer`, etc., to handle both modes transparently.

### 2. Attention Kernel (`python/sgl_jax/srt/kernels/ragged_paged_attention/`)

*   **New Split Kernel (`ragged_paged_attention_split.py`)**:
    *   Created `_ragged_paged_attention_kernel_split`.
    *   Decoupled K and V memory access. Instead of loading interleaved chunks, it fetches K and V from separate HBM references (`k_cache_hbm_ref`, `v_cache_hbm_ref`) into separate VMEM buffers (`bk_x2_ref`, `bv_x2_ref`).
    *   Supports computing `Q @ K.T` using `k_head_dim` and `Score @ V` using `v_head_dim`.

*   **Dispatcher (`ragged_paged_attention.py`)**:
    *   Updated `ragged_paged_attention` signature to accept optional `k_cache` and `v_cache`.
    *   Added dispatch logic: if `k_cache` and `v_cache` are provided, it invokes the split kernel; otherwise, it falls back to the original fused kernel.

### 3. FlashAttention Backend (`python/sgl_jax/srt/layers/attention/flashattention_backend.py`)

*   **Initialization**: Now accepts `v_head_dim`.
*   **Forward Pass (`__call__`)**:
    *   Checks `token_to_kv_pool.is_split`.
    *   **Split Path**:
        *   Retrieves separate caches via `get_split_kv_buffer`.
        *   Independently pads Q, K (to aligned `head_dim`) and V (to aligned `v_head_dim`).
        *   Invokes `ragged_paged_attention` with split arguments.
        *   Slices the output and updated caches back to their original dimensions before returning.
    *   **Fused Path**: Preserves original behavior for backward compatibility.

### 4. Configuration & Runner (`python/sgl_jax/srt/configs/`, `python/sgl_jax/srt/model_executor/`)

*   **`ModelConfig`**: Updated to extract `v_head_dim` from HF config (defaults to `head_dim`).
*   **`ModelRunner`**:
    *   Updated `init_memory_pool` to pass correct dimensions to `MHATokenToKVPool`.
    *   Updated `profile_max_num_token` to accurately calculate memory usage based on whether split storage is used (sum of aligned K and V sizes vs. 2x max size).
*   **`RadixAttention`**: Updated to store `v_head_dim`.

## Verification

A unit test `python/sgl_jax/test/mem_cache/test_split_kv_cache.py` was created (and verified) to cover:
*   Correct initialization of split buffers when dimensions differ.
*   Correct fallback to fused buffers when dimensions match.
*   Correct data routing in `set_kv_buffer` and `get_kv_buffer` (using mocked kernel updates).
*   Accurate memory usage calculation.

## Impact

For a model like Qwen2-MoE (K=192, V=128):
*   **Old**: K->256, V->256. Total per token: 512 elements.
*   **New**: K->256, V->128. Total per token: 384 elements.
*   **Result**: **25% reduction** in KV cache memory usage.
