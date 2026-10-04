"""TT attention operations implemented with JAX's typed FFI."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _like(value):
    """An FFI result typed like value: its sharding and, inside shard_map, the
    mesh axes it varies over."""
    aval = jax.typeof(value)
    return jax.ShapeDtypeStruct(
        value.shape, value.dtype, sharding=aval.sharding, manual_axis_type=aval.manual_axis_type
    )


def _call(name, *operands, input_output_aliases=None, **attributes):
    return jax.ffi.ffi_call(
        name,
        _like(operands[0]),
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


def paged_update_cache(cache, value, positions, page_table, *, share_cache=False):
    # share_cache serializes updates, so several users may write to one page.
    attributes = {"share_cache": True} if share_cache else {}
    return _call(
        "tt.paged_update_cache",
        cache,
        value,
        positions,
        page_table,
        input_output_aliases={0: 0},
        **attributes,
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


def _recurrent_call(name, state, output_shape, *operands, **attributes):
    return jax.ffi.ffi_call(
        name,
        (_like(state), output_shape),
        input_output_aliases={0: 0},
        vmap_method="sequential",
    )(state, *operands, **attributes)


def causal_conv1d_update(state, value, weight, indices, initial):
    return _recurrent_call(
        "tt.causal_conv1d_update",
        state,
        _like(value).update(dtype=jnp.float32),
        value,
        weight,
        indices,
        initial.astype(jnp.bfloat16),
    )


def gated_delta_decode(state, qkv, b, a, A_log, dt_bias, indices, initial):
    # qkv is the flat convolution output: the query, key and value heads in turn.
    return _recurrent_call(
        "tt.gated_delta_decode",
        state,
        _like(b).update(shape=(*b.shape, state.shape[-1]), dtype=jnp.float32),
        qkv,
        b,
        a,
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
        (_like(state), _like(v)),
        vmap_method="sequential",
    )(q, k, v, gate, beta, state)
