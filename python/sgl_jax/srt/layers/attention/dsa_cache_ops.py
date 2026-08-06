"""Shared functional cache operations used by GLM-5.2 DSA paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scatter_paged_cache(
    cache3d: jax.Array,
    new_tokens: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
) -> jax.Array:
    """Scatter each query token into its sequence's paged cache tail."""

    page_size = cache3d.shape[1]
    num_tokens = new_tokens.shape[0]
    num_seqs = seq_lens.shape[0]

    token_id = jnp.arange(num_tokens)
    seq_id = jnp.searchsorted(cu_q_lens[1:], token_id, side="right")
    seq_id = jnp.clip(seq_id, 0, num_seqs - 1)
    q_start = cu_q_lens[seq_id]
    q_end = cu_q_lens[seq_id + 1]
    kv_len = seq_lens[seq_id]
    abs_pos = jnp.maximum(kv_len - (q_end - q_start) + (token_id - q_start), 0)
    valid = (token_id >= q_start) & (token_id < q_end) & (kv_len > 0)

    page_local = abs_pos // page_size
    offset = abs_pos % page_size
    page = page_indices[cu_kv_lens[seq_id] // page_size + page_local]

    sentinel = cache3d.shape[0] - 1
    safe_page = jnp.where(valid, page, sentinel)
    safe_offset = jnp.where(valid, offset, 0)
    return cache3d.at[safe_page, safe_offset].set(new_tokens.astype(cache3d.dtype))
