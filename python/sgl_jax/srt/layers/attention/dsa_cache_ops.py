"""Shared functional cache operations used by GLM-5.2 DSA paths."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def scatter_paged_cache(
    cache: jax.Array,
    new_tokens: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
) -> jax.Array:
    """Scatter tokens into a 3D index cache or a 4D packed KV cache.

    Supported layouts are ``[pages, page_size, dim]`` and
    ``[pages, page_size, packing, packed_dim]``. ``new_tokens`` must match the
    cache's trailing feature dimensions exactly; no broadcasting is allowed.
    """

    if cache.ndim not in (3, 4):
        raise ValueError(f"cache must be rank 3 or 4, got shape {cache.shape}.")
    if new_tokens.ndim != cache.ndim - 1:
        raise ValueError(
            f"new_tokens rank must be cache.ndim - 1, got "
            f"new_tokens.shape={new_tokens.shape} and cache.shape={cache.shape}."
        )
    if new_tokens.shape[1:] != cache.shape[2:]:
        raise ValueError(
            f"new_tokens trailing shape {new_tokens.shape[1:]} must match "
            f"cache slot shape {cache.shape[2:]}."
        )

    page_size = cache.shape[1]
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

    sentinel = cache.shape[0] - 1
    safe_page = jnp.where(valid, page, sentinel)
    safe_offset = jnp.where(valid, offset, 0)
    return cache.at[safe_page, safe_offset].set(new_tokens.astype(cache.dtype))
