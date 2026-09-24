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


def paged_scaled_dot_product_attention_decode(query, key_cache, value_cache, page_table, positions):
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


def _recurrent_call(name, state, output_shape, *operands):
    return jax.ffi.ffi_call(
        name,
        (jax.ShapeDtypeStruct(state.shape, state.dtype), output_shape),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, *operands)


def causal_conv1d_update(state, value, weight, indices, initial):
    return _recurrent_call(
        "tt.causal_conv1d_update",
        state,
        jax.ShapeDtypeStruct(value.shape, value.dtype),
        value,
        weight,
        indices,
        initial.astype(jnp.bfloat16),
    )


def gated_delta_decode(state, q, k, v, b, a, A_log, dt_bias, indices, initial):
    return _recurrent_call(
        "tt.gated_delta_decode",
        state,
        jax.ShapeDtypeStruct(v.shape, jnp.float32),
        q,
        k,
        v,
        b.astype(jnp.float32),
        a.astype(jnp.float32),
        A_log.astype(jnp.float32),
        dt_bias.astype(jnp.float32),
        indices,
        initial.astype(jnp.bfloat16),
    )


def state_pool_update(state, indices, updates):
    return _call("tt.state_pool_update", state, indices, updates, input_output_aliases={0: 0})


def gated_delta_rule(q, k, v, gate, beta, state):
    return jax.ffi.ffi_call(
        "tt.gated_delta_rule",
        (jax.ShapeDtypeStruct(state.shape, state.dtype), jax.ShapeDtypeStruct(v.shape, v.dtype)),
        vmap_method="sequential",
    )(q, k, v, gate, beta, state)
