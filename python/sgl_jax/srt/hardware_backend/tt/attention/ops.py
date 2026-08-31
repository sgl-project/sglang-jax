"""TT attention operations implemented with JAX's typed FFI."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _call(name, *operands, input_output_aliases=None, **attributes):
    result = operands[0]
    return jax.ffi.ffi_call(
        name,
        jax.ShapeDtypeStruct(result.shape, result.dtype),
        vmap_method="sequential",
        input_output_aliases=input_output_aliases,
    )(*operands, **attributes)


def chunked_scaled_dot_product_attention(
    query, key_cache, value_cache, page_table, chunk_start, *, scale=None
):
    attributes = {} if scale is None else {"scale": np.float32(scale)}
    return _call(
        "tt.chunked_scaled_dot_product_attention",
        query,
        key_cache,
        value_cache,
        page_table,
        chunk_start,
        **attributes,
    )


def paged_scaled_dot_product_attention_decode(
    query, key_cache, value_cache, page_table, positions
):
    return _call(
        "tt.paged_scaled_dot_product_attention_decode",
        query,
        key_cache,
        value_cache,
        page_table,
        positions,
        is_causal=True,
        has_attention_mask=False,
        has_cur_pos_tensor=True,
        has_attention_sink=False,
    )


def paged_update_cache(cache, value, positions, page_table):
    return _call(
        "tt.paged_update_cache",
        cache,
        value,
        positions,
        page_table,
        input_output_aliases={0: 0},
    )


def paged_fill_cache(cache, value, page_table, batch_indices):
    return _call(
        "tt.paged_fill_cache",
        cache,
        value,
        page_table,
        batch_indices,
        input_output_aliases={0: 0},
    )


def annotate_weight_dtype(tensor, dtype):
    if dtype not in {"bf16", "bfp_bf8", "bfp_bf4"}:
        raise ValueError(f"Unsupported TT weight dtype: {dtype}")

    original_shape = tensor.shape
    if tensor.ndim < 3:
        tensor = jnp.reshape(tensor, (1,) * (3 - tensor.ndim) + original_shape)
    tensor = _call(
        "tt.weight_dtype_override",
        tensor,
        **{"ttcore.weight_dtype": dtype},
    )
    return jnp.reshape(tensor, original_shape)
