"""Opt-in TPU PLE kernels; the two dense projections stay outside Pallas.

Each program owns whole RMSNorm groups and updates only the selected state
slots. Nonzero state_indices must be unique, as in the recurrent-state pool;
slot zero may repeat and is never written. Donate the pool at the enclosing
jit boundary to avoid a copy of the full pool for functional aliasing.

The public pool is [slots, channels, time]; a transposed view exposes
[slots, time, channels] to Pallas so the 128-lane axis holds channels.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _norm(x, weight, eps):
    x32 = x.astype(jnp.float32)
    variance = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(variance + eps) * (1 + weight)).astype(x.dtype)


def _gate_and_norm(key, query, value, nk, nq, nc, eps):
    # The compiled XLA gate retains FP32 normalized K/Q in the product, even
    # though the Python graph contains a BF16 -> FP32 round trip. Actually
    # rounding here changes near-zero dot signs and the signed-sqrt gate.
    key = _norm(key.astype(jnp.float32), nk, eps)
    query_n = _norm(query.astype(jnp.float32), nq, eps)
    dot = jnp.sum(key * query_n, axis=-1, keepdims=True) / math.sqrt(key.shape[-1])
    gate = jax.nn.sigmoid(jnp.sign(dot) * jnp.sqrt(jnp.maximum(jnp.abs(dot), 1e-6)))
    u = (gate * value.astype(jnp.float32)).astype(query.dtype)
    return u, _norm(u, nc, eps)


def _check_inputs(key, query, value, norms, weight, state, indices, initial, hidden):
    if query.ndim != 2 or key.shape != query.shape:
        raise ValueError("key and query must have matching [tokens, channels] shapes")
    tokens, channels = query.shape
    if hidden <= 0 or hidden % 128 or channels % hidden:
        raise ValueError("Pallas PLE requires whole RMS groups with hidden_size divisible by 128")
    if value.shape != (tokens, hidden) or any(w.shape != (channels,) for w in norms):
        raise ValueError("value or RMSNorm weight shape does not match hidden_size/channels")
    if weight.ndim != 2 or weight.shape[0] != channels:
        raise ValueError("conv_weight must be [channels, kernel]")
    if indices.ndim != 1 or initial.shape != indices.shape:
        raise ValueError("state_indices and has_initial_state must both be [requests]")
    if state.dtype != query.dtype:
        raise ValueError("Pallas PLE currently requires activation and state dtypes to match")


def _decode_kernel(
    slots_ref,
    init_ref,
    key_ref,
    query_ref,
    value_ref,
    nk_ref,
    nq_ref,
    nc_ref,
    weight_ref,
    pool_ref,
    out_ref,
    new_pool_ref,
    state_ref,
    sem_ref,
    *,
    batch,
    tile,
    hidden,
    dilation,
    eps,
):
    first = pl.program_id(0) * tile
    channel = pl.program_id(1) * hidden
    state_ref[...] = jnp.zeros(state_ref.shape, state_ref.dtype)
    for row in range(tile):

        @pl.when((first + row < batch) & (init_ref[first + row] != 0))
        def _start(row=row):
            pltpu.make_async_copy(
                pool_ref.at[slots_ref[first + row], :, pl.ds(channel, hidden)],
                state_ref.at[row],
                sem_ref.at[row],
            ).start()

    for row in range(tile):

        @pl.when((first + row < batch) & (init_ref[first + row] != 0))
        def _wait(row=row):
            dst = state_ref.at[row]
            pltpu.make_async_copy(dst, dst, sem_ref.at[row]).wait()

    u, x = _gate_and_norm(
        key_ref[...],
        query_ref[...],
        value_ref[...],
        nk_ref[...],
        nq_ref[...],
        nc_ref[...],
        eps,
    )
    # Time-major VMEM avoids padding the nine time entries to 128 lanes
    # for every channel throughout the arithmetic.
    state = state_ref[...]
    kernel = weight_ref.shape[0]
    y = x.astype(jnp.float32) * weight_ref[kernel - 1, :].astype(jnp.float32)[None, :]
    for tap in range(kernel - 1):
        y += (
            state[:, tap * dilation, :].astype(jnp.float32)
            * weight_ref[tap, :].astype(jnp.float32)[None, :]
        )
    out_ref[...] = u + jax.nn.silu(y.astype(x.dtype))
    next_state = x[:, None, :]
    if state.shape[1] > 1:
        next_state = jnp.concatenate((state[:, 1:, :], next_state), axis=1)
    state_ref[...] = next_state.astype(state_ref.dtype)
    # All reads complete before any write; distinct live slots have disjoint
    # owners, and the shared padding slot is read-only. No pool-wide scatter.
    for row in range(tile):

        @pl.when((first + row < batch) & (slots_ref[first + row] != 0))
        def _store(row=row):
            pltpu.make_async_copy(
                state_ref.at[row],
                new_pool_ref.at[slots_ref[first + row], :, pl.ds(channel, hidden)],
                sem_ref.at[row],
            ).start()

    for row in range(tile):

        @pl.when((first + row < batch) & (slots_ref[first + row] != 0))
        def _wait_store(row=row):
            src = state_ref.at[row]
            pltpu.make_async_copy(src, src, sem_ref.at[row]).wait()


def ngram_decode_pallas(
    key,
    query,
    value,
    norm_key,
    norm_query,
    norm_conv,
    conv_weight,
    conv_state,
    state_indices,
    has_initial_state,
    *,
    hidden_size: int,
    dilation: int = 3,
    eps: float = 1e-6,
    tile: int = 8,
    interpret: bool = False,
):
    """Fuse three norms, gate, dilated conv, SiLU, delta and slot writeback.

    Inputs are shard-local; channels must contain complete hidden_size groups.
    Returns the PLE delta (not the outer residual) and the full updated pool.
    """
    _check_inputs(
        key,
        query,
        value,
        (norm_key, norm_query, norm_conv),
        conv_weight,
        conv_state,
        state_indices,
        has_initial_state,
        hidden_size,
    )
    batch, channels = query.shape
    if state_indices.shape != (batch,):
        raise ValueError("decode requires one state index per token")
    if dilation < 1 or conv_weight.shape[1] < 2:
        raise ValueError("Pallas PLE requires dilation >= 1 and kernel size >= 2")
    if conv_state.shape[1:] != (channels, (conv_weight.shape[1] - 1) * dilation):
        raise ValueError("conv_state shape does not match channels/kernel/dilation")
    if tile <= 0 or tile % 8:
        raise ValueError("tile must be a positive multiple of 8")
    if batch == 0:
        return jnp.zeros_like(query), conv_state
    groups = channels // hidden_size
    padded = pl.cdiv(batch, tile) * tile

    def pad(x):
        return jnp.pad(x, ((0, padded - batch), (0, 0)))

    act_spec = pl.BlockSpec((tile, hidden_size), lambda b, g, *_: (b, g))
    norm_spec = pl.BlockSpec((1, hidden_size), lambda b, g, *_: (0, g))
    pool_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    output, state = pl.pallas_call(
        functools.partial(
            _decode_kernel,
            batch=batch,
            tile=tile,
            hidden=hidden_size,
            dilation=dilation,
            eps=eps,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(padded // tile, groups),
            in_specs=(
                act_spec,
                act_spec,
                pl.BlockSpec((tile, hidden_size), lambda b, g, *_: (b, 0)),
                norm_spec,
                norm_spec,
                norm_spec,
                pl.BlockSpec((None, conv_weight.shape[1], hidden_size), lambda b, g, *_: (g, 0, 0)),
                pool_spec,
            ),
            out_specs=(act_spec, pool_spec),
            scratch_shapes=(
                pltpu.VMEM((tile, conv_state.shape[2], hidden_size), conv_state.dtype),
                pltpu.SemaphoreType.DMA((tile,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((padded, channels), query.dtype),
            jax.ShapeDtypeStruct(
                (conv_state.shape[0], conv_state.shape[2], channels), conv_state.dtype
            ),
        ),
        input_output_aliases={9: 1},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
        name="ngram_norm_gate_conv_decode",
    )(
        jnp.pad(state_indices.astype(jnp.int32), (0, padded - batch)),
        jnp.pad(has_initial_state.astype(jnp.int32), (0, padded - batch)),
        pad(key),
        pad(query),
        pad(value),
        norm_key.astype(jnp.float32)[None, :],
        norm_query.astype(jnp.float32)[None, :],
        norm_conv.astype(jnp.float32)[None, :],
        conv_weight.astype(query.dtype).reshape(groups, hidden_size, -1).transpose(0, 2, 1),
        conv_state.transpose(0, 2, 1),
    )
    return output[:batch].reshape(batch, channels), state.transpose(0, 2, 1)


def _extend_kernel(
    slots_ref,
    init_ref,
    cu_ref,
    key_ref,
    query_ref,
    value_ref,
    nk_ref,
    nq_ref,
    nc_ref,
    weight_ref,
    pool_ref,
    out_ref,
    new_pool_ref,
    key_buf,
    query_buf,
    value_buf,
    state_buf,
    sem,
    *,
    tile,
    hidden,
    dilation,
    eps,
):
    request, group = pl.program_id(0), pl.program_id(1)
    slot = slots_ref[request]
    start, end = cu_ref[request], cu_ref[request + 1]
    channel = group * hidden
    state_len = state_buf.shape[0]
    state_buf[...] = jnp.zeros(state_buf.shape, state_buf.dtype)

    @pl.when(init_ref[request] != 0)
    def _load_state():
        pltpu.make_async_copy(
            pool_ref.at[slot, :, pl.ds(channel, hidden)], state_buf, sem.at[3]
        ).start()
        pltpu.make_async_copy(state_buf, state_buf, sem.at[3]).wait()

    def step(block, _):
        pos = start + block * tile
        count = jnp.minimum(tile, end - pos)
        key_buf[...] = jnp.zeros(key_buf.shape, key_buf.dtype)
        query_buf[...] = jnp.zeros(query_buf.shape, query_buf.dtype)
        value_buf[...] = jnp.zeros(value_buf.shape, value_buf.dtype)
        # The token axis is untiled in [T, G, 1, H], allowing ragged DMA
        # without reading or writing another request's boundary tokens.
        for src, dst, index in ((key_ref, key_buf, 0), (query_ref, query_buf, 1)):
            pltpu.make_async_copy(
                src.at[pl.ds(pos, count), group, :, :],
                dst.at[pl.ds(0, count), :, :],
                sem.at[index],
            ).start()
        pltpu.make_async_copy(
            value_ref.at[pl.ds(pos, count), :, :],
            value_buf.at[pl.ds(0, count), :, :],
            sem.at[2],
        ).start()
        for dst, index in ((key_buf, 0), (query_buf, 1), (value_buf, 2)):
            part = dst.at[pl.ds(0, count), :, :]
            pltpu.make_async_copy(part, part, sem.at[index]).wait()

        u, x = _gate_and_norm(
            key_buf[...].reshape(tile, hidden),
            query_buf[...].reshape(tile, hidden),
            value_buf[...].reshape(tile, hidden),
            nk_ref[...],
            nq_ref[...],
            nc_ref[...],
            eps,
        )
        window = jnp.concatenate((state_buf[...], x), axis=0)
        y = jnp.zeros((tile, hidden), jnp.float32)
        for tap in range(weight_ref.shape[0]):
            y += window[tap * dilation : tap * dilation + tile].astype(jnp.float32) * (
                weight_ref[tap, :].astype(jnp.float32)[None, :]
            )
        query_buf[...] = (u + jax.nn.silu(y.astype(x.dtype))).reshape(tile, 1, hidden)
        pltpu.make_async_copy(
            query_buf.at[pl.ds(0, count), :, :],
            out_ref.at[pl.ds(pos, count), group, :, :],
            sem.at[0],
        ).start()
        # Mosaic's dynamic gather cannot cross source vregs. Roll across the
        # padded time axis, then take a static slice of the last live window.
        pad = pl.cdiv(state_len + tile, 8) * 8 - (state_len + tile)
        rolled = pltpu.roll(
            jnp.pad(window, ((0, pad), (0, 0))), state_len + tile + pad - count, axis=0
        )
        state_buf[...] = rolled[:state_len].astype(state_buf.dtype)
        src = query_buf.at[pl.ds(0, count), :, :]
        pltpu.make_async_copy(src, src, sem.at[0]).wait()
        return ()

    # One owner per (request, group) carries state through the chunks. In
    # particular a late tile cannot overwrite initial state before an early
    # tile reads it, even with donation enabled.
    jax.lax.fori_loop(0, pl.cdiv(end - start, tile), step, ())

    @pl.when(slot != 0)
    def _store_state():
        pltpu.make_async_copy(
            state_buf, new_pool_ref.at[slot, :, pl.ds(channel, hidden)], sem.at[3]
        ).start()
        pltpu.make_async_copy(state_buf, state_buf, sem.at[3]).wait()


def ngram_extend_pallas(
    key,
    query,
    value,
    norm_key,
    norm_query,
    norm_conv,
    conv_weight,
    conv_state,
    state_indices,
    has_initial_state,
    cu_seqlens,
    *,
    hidden_size: int,
    dilation: int = 3,
    eps: float = 1e-6,
    tile: int = 32,
    interpret: bool = False,
):
    """Ragged/chunked prefill counterpart of :func:`ngram_decode_pallas`.

    A program processes one request/group in token tiles, then writes its
    final state once. The public pool layout is unchanged. All packed tokens
    must be covered by cu_seqlens; empty requests are allowed.
    """
    _check_inputs(
        key,
        query,
        value,
        (norm_key, norm_query, norm_conv),
        conv_weight,
        conv_state,
        state_indices,
        has_initial_state,
        hidden_size,
    )
    tokens, channels = query.shape
    if cu_seqlens.shape != (state_indices.shape[0] + 1,):
        raise ValueError("cu_seqlens must be [requests + 1]")
    kernel = conv_weight.shape[1]
    if dilation < 1 or kernel < 2 or tile <= 0 or tile % 8:
        raise ValueError("Requires dilation >= 1, kernel >= 2 and tile divisible by 8")
    if conv_state.shape[1:] != (channels, (kernel - 1) * dilation):
        raise ValueError("conv_state shape does not match channels/kernel/dilation")
    if tokens == 0:
        raise ValueError("Fused PLE prefill requires at least one packed token")
    groups = channels // hidden_size
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    norm_spec = pl.BlockSpec((1, hidden_size), lambda b, g, *_: (0, g))
    output, state = pl.pallas_call(
        functools.partial(
            _extend_kernel, tile=tile, hidden=hidden_size, dilation=dilation, eps=eps
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(state_indices.shape[0], groups),
            in_specs=(
                hbm,
                hbm,
                hbm,
                norm_spec,
                norm_spec,
                norm_spec,
                pl.BlockSpec((None, kernel, hidden_size), lambda b, g, *_: (g, 0, 0)),
                hbm,
            ),
            out_specs=(hbm, hbm),
            scratch_shapes=(
                pltpu.VMEM((tile, 1, hidden_size), key.dtype),
                pltpu.VMEM((tile, 1, hidden_size), query.dtype),
                pltpu.VMEM((tile, 1, hidden_size), value.dtype),
                pltpu.VMEM((conv_state.shape[2], hidden_size), conv_state.dtype),
                pltpu.SemaphoreType.DMA((4,)),
            ),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((tokens, groups, 1, hidden_size), query.dtype),
            jax.ShapeDtypeStruct(
                (conv_state.shape[0], conv_state.shape[2], channels), conv_state.dtype
            ),
        ),
        input_output_aliases={10: 1},
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")),
        interpret=interpret,
        name="ngram_norm_gate_conv_extend",
    )(
        state_indices.astype(jnp.int32),
        has_initial_state.astype(jnp.int32),
        cu_seqlens.astype(jnp.int32),
        key.reshape(tokens, groups, 1, hidden_size),
        query.reshape(tokens, groups, 1, hidden_size),
        value.reshape(tokens, 1, hidden_size),
        norm_key.astype(jnp.float32)[None, :],
        norm_query.astype(jnp.float32)[None, :],
        norm_conv.astype(jnp.float32)[None, :],
        conv_weight.astype(query.dtype).reshape(groups, hidden_size, kernel).transpose(0, 2, 1),
        conv_state.transpose(0, 2, 1),
    )
    return output.reshape(tokens, channels), state.transpose(0, 2, 1)
