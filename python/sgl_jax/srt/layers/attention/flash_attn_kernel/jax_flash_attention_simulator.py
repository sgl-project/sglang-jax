"""
Simplified JAX Flash Attention Simulator

This version avoids complex dynamic operations and focuses on simulating core attention computation logic,
making it easier to debug precision issues when page_size > 1.
"""

import functools
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def cdiv(a, b):
    """Ceiling division"""
    return (a + b - 1) // b


def simple_jax_ragged_paged_attention_simulator(
    q: jax.Array,  # [max_num_batched_tokens, num_q_heads, head_dim]
    k_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    v_cache: jax.Array,  # [total_num_pages, page_size, num_kv_heads, head_dim]
    page_indices: jax.Array,  # i32[num_pages]
    cu_q_lens: jax.Array,  # i32[max_num_seqs + 1]
    cu_kv_lens: jax.Array,  # i32[max_num_seqs + 1]
    num_seqs: jax.Array,  # i32[1]
    seq_lens: jax.Array,  # i32[max_num_seqs]
    *,
    sm_scale: float = 1.0,
    sliding_window: Optional[int] = None,
    soft_cap: Optional[float] = None,
    mask_value: Optional[float] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    debug_print: bool = True,
) -> jax.Array:
    """
    Simplified JAX simulator version of ragged paged attention

    This version reconstructs the KV cache into full format, then uses logic similar to ref_ragged_paged_attention,
    but with simulation of page alignment.
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE

    # Get basic shape information
    num_q_tokens, num_q_heads, head_dim = q.shape
    total_pages, page_size, num_kv_heads, _ = k_cache.shape
    num_q_heads_per_kv_head = num_q_heads // num_kv_heads

    if debug_print:
        print(f"\n=== Simple JAX Simulator Debug Info ===")
        print(f"q.shape: {q.shape}")
        print(f"k_cache.shape: {k_cache.shape}")
        print(f"v_cache.shape: {v_cache.shape}")
        print(f"page_size: {page_size}")
        print(f"num_q_heads: {num_q_heads}")
        print(f"num_kv_heads: {num_kv_heads}")
        print(f"num_seqs: {num_seqs[0]}")
        print(f"cu_q_lens: {cu_q_lens}")
        print(f"cu_kv_lens: {cu_kv_lens}")
        print(f"seq_lens: {seq_lens}")
        print(f"page_indices shape: {page_indices.shape}")
        print(f"page_indices: {page_indices}")

    # Reconstruct complete KV cache
    # Calculate total aligned KV length
    max_aligned_kv_len = jnp.max(cu_kv_lens)

    # Create reconstructed K, V arrays
    k_reconstructed = jnp.zeros(
        (max_aligned_kv_len, num_kv_heads, head_dim), dtype=k_cache.dtype
    )
    v_reconstructed = jnp.zeros(
        (max_aligned_kv_len, num_kv_heads, head_dim), dtype=v_cache.dtype
    )

    # Reconstruct KV data for each sequence
    for seq_idx in range(int(num_seqs[0])):
        kv_start = cu_kv_lens[seq_idx]
        kv_end = cu_kv_lens[seq_idx + 1]
        aligned_kv_len = kv_end - kv_start

        if debug_print:
            print(f"\nReconstructing seq {seq_idx}:")
            print(
                f"  kv_start: {kv_start}, kv_end: {kv_end}, aligned_kv_len: {aligned_kv_len}"
            )

        # Calculate page range needed for this sequence - aligned with pallas kernel logic
        start_page_idx = cdiv(cu_kv_lens[seq_idx], page_size)
        end_page_idx = cdiv(cu_kv_lens[seq_idx + 1], page_size)

        if debug_print:
            print(f"  start_page_idx: {start_page_idx}, end_page_idx: {end_page_idx}")

        # Collect page data
        seq_k_pages = []
        seq_v_pages = []

        for page_offset in range(end_page_idx - start_page_idx):
            abs_page_idx = start_page_idx + page_offset
            if abs_page_idx < page_indices.shape[0]:
                page_idx = page_indices[abs_page_idx]
                seq_k_pages.append(k_cache[page_idx])
                seq_v_pages.append(v_cache[page_idx])
                if debug_print:
                    print(
                        f"    page_offset {page_offset}: abs_page_idx {abs_page_idx} -> page_idx {page_idx}"
                    )

        # Concatenate pages
        if seq_k_pages:
            seq_k_full = jnp.concatenate(seq_k_pages, axis=0)
            seq_v_full = jnp.concatenate(seq_v_pages, axis=0)

            # Calculate valid range in page concatenated array
            in_page_start = 0
            in_page_end = aligned_kv_len  # Use aligned length, this is key!
            in_page_end = min(in_page_end, seq_k_full.shape[0])

            if debug_print:
                print(f"  seq_k_full.shape: {seq_k_full.shape}")
                print(f"  in_page_start: {in_page_start}, in_page_end: {in_page_end}")

            # Extract data with aligned length
            seq_k = seq_k_full[in_page_start:in_page_end]
            seq_v = seq_v_full[in_page_start:in_page_end]

            k_reconstructed = k_reconstructed.at[kv_start:kv_end].set(seq_k)
            v_reconstructed = v_reconstructed.at[kv_start:kv_end].set(seq_v)

    if debug_print:
        print(f"\nReconstructed KV shapes:")
        print(f"k_reconstructed.shape: {k_reconstructed.shape}")
        print(f"v_reconstructed.shape: {v_reconstructed.shape}")

    # Now use reconstructed KV to execute logic similar to ref_ragged_paged_attention
    outputs = []
    for i in range(int(num_seqs[0])):
        q_start = cu_q_lens[i]
        q_end = cu_q_lens[i + 1]
        q_len = q_end - q_start

        kv_start = cu_kv_lens[i]
        kv_end = cu_kv_lens[i + 1]
        aligned_kv_len = kv_end - kv_start

        if i < seq_lens.shape[0]:
            actual_kv_len = seq_lens[i]
        else:
            actual_kv_len = aligned_kv_len
        actual_kv_len = max(0, min(actual_kv_len, aligned_kv_len))

        if debug_print:
            print(f"\nProcessing seq {i}:")
            print(f"  q_start: {q_start}, q_end: {q_end}, q_len: {q_len}")
            print(f"  kv_start: {kv_start}, kv_end: {kv_end}")
            print(f"  aligned_kv_len: {aligned_kv_len}, actual_kv_len: {actual_kv_len}")

        seq_q = q[q_start:q_end]
        seq_k = k_reconstructed[kv_start : kv_start + actual_kv_len]
        seq_v = v_reconstructed[kv_start : kv_start + actual_kv_len]

        if debug_print:
            print(f"  seq_q.shape: {seq_q.shape}")
            print(f"  seq_k.shape: {seq_k.shape}")
            print(f"  seq_v.shape: {seq_v.shape}")

        if k_scale is not None:
            seq_k = seq_k.astype(jnp.float32) * k_scale
            seq_k = seq_k.astype(seq_q.dtype)
        if v_scale is not None:
            seq_v = seq_v.astype(jnp.float32) * v_scale
            seq_v = seq_v.astype(seq_q.dtype)

        # Process GQA: repeat KV heads
        seq_k = jnp.repeat(seq_k, num_q_heads_per_kv_head, axis=1)
        seq_v = jnp.repeat(seq_v, num_q_heads_per_kv_head, axis=1)

        attn = jnp.einsum(
            "qhd,khd->hqk", seq_q, seq_k, preferred_element_type=jnp.float32
        )
        attn *= sm_scale

        # Build causal mask - key: match original kernel logic
        q_span = (actual_kv_len - q_len) + jax.lax.broadcasted_iota(
            jnp.int32, attn.shape, 1
        )
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)

        mask = q_span < kv_span

        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)

        if soft_cap is not None:
            attn = soft_cap * jnp.tanh(attn / soft_cap)

        attn += jnp.where(mask, mask_value, 0.0)
        attn = jax.nn.softmax(attn, axis=-1).astype(seq_v.dtype)
        out = jnp.einsum("hqk,khd->qhd", attn, seq_v).astype(seq_q.dtype)

        outputs.append(out)

        if debug_print:
            print(f"  output.shape: {out.shape}")

    final_output = jnp.concatenate(outputs, axis=0)

    if debug_print:
        print(f"\nFinal output shape: {final_output.shape}")
        print("=== End Simple JAX Simulator Debug Info ===\n")

    return final_output
