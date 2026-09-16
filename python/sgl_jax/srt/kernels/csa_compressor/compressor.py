"""Ratio-4 overlap compression shared by CSA KV and index keys."""

from __future__ import annotations

import functools
import math
import os
from typing import NamedTuple

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from .tune import (
    CSA_ATTENTION_DIM,
    CSA_CACHE_PACKING,
    CSA_COMPRESSION_RATIO,
    CSA_FP8_AMAX_FLOOR,
    CSA_FP8_BLOCK_SIZE,
    CSA_INDEX_DIM,
    CSA_INDEX_PADDING_BYTES,
    CSA_INDEX_PROJECTED_DIM,
    CSA_INDEX_RECORD_BYTES,
    CSA_MAIN_NOPE_PADDING_BYTES,
    CSA_MAIN_NOPE_RECORD_BYTES,
    CSA_MAIN_NOPE_SCALE_COUNT,
    CSA_MAIN_PROJECTED_DIM,
    CSA_MAIN_RECORD_BYTES,
    CSA_NORM_EPS,
    CSA_ROPE_DIM,
    CSA_ROPE_FREQUENCY_DIM,
    CSA_ROPE_RECORD_BYTES,
    CSA_STATE_SLOTS,
    TPU_V6E,
    CompressorSchedule,
    get_csa_compressor_projection_k_tile,
    get_csa_compressor_query_tile,
)

COMPRESS_RATIO = CSA_COMPRESSION_RATIO
STATE_SLOTS = CSA_STATE_SLOTS


class CompressorBlock(NamedTuple):
    """Caller-built ordered work: [R,Q] token ids and [R] request ids; -1 is padding.

    Q is one token or complete aligned groups. Active rows have no inner padding;
    a request occurs at most once per block. Blocks execute in request order.
    """

    token_indices: jax.Array
    request_indices: jax.Array


class CompressorMetadata(NamedTuple):
    """Device arrays prepared by the caller, independent of serving/pool objects.

    positions/cache_locations: [T]; cu_q_lens: [B+1]; state_indices: [B].
    State and cache locations use -1 for inactive requests or suppressed writes.
    Active requests own distinct state rows; emitted cache locations are unique.
    """

    positions: jax.Array
    cu_q_lens: jax.Array
    state_indices: jax.Array
    cache_locations: jax.Array
    blocks: tuple[CompressorBlock, ...]


def _interpret_pallas() -> bool:
    requested = os.environ.get("PALLAS_INTERPRET", "").strip().lower()
    return requested in ("1", "true") or jax.default_backend() != "tpu"


def _project(x, weight):
    return jax.lax.dot_general(
        x.astype(jnp.bfloat16),
        weight.astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _pool_normalize(window_kv, window_score, norm, *, norm_eps):
    rows, slots, width = window_kv.shape
    shape = (rows, slots, width // TPU_V6E.vector_lanes, TPU_V6E.vector_lanes)
    # Keep time outside the vector tile: its reduction order matters before FP8 rounding.
    values, scores = window_kv.reshape(shape), window_score.reshape(shape)
    pooled = jnp.sum(values * jax.nn.softmax(scores, axis=1), axis=1)
    variance = jnp.mean(jnp.square(pooled), axis=(1, 2), keepdims=True)
    pooled = pooled * jax.lax.rsqrt(variance + norm_eps)
    return (pooled * norm.astype(jnp.float32).reshape(1, *shape[-2:])).reshape(rows, width)


def _csa_state_step_kernel(
    x_ref,
    state_ref,
    weight_ref,
    ape_ref,
    norm_ref,
    cos_ref,
    sin_ref,
    positions_ref,
    value_ref,
    emit_ref,
    state_out_ref,
    projection_ref,
    *,
    head_dim: int,
    k_steps: int,
    norm_eps: float,
    record_kind: str,
):
    """Project, update the overlap state, and pool a completed ratio-4 window."""
    k_step = pl.program_id(1)

    @pl.when(k_step == 0)
    def _zero_projection():
        projection_ref[...] = jnp.zeros_like(projection_ref)

    projection_ref[...] += _project(x_ref[...], weight_ref[...])

    @pl.when(k_step == k_steps - 1)
    def _finish():
        batch = x_ref.shape[0]
        head_tiles = head_dim // TPU_V6E.vector_lanes
        projection_tiles = 2 * head_tiles
        positions = positions_ref[:, 0]
        slots = jnp.mod(positions, COMPRESS_RATIO).astype(jnp.int32)
        projected = projection_ref[...].reshape(
            batch,
            COMPRESS_RATIO,
            head_tiles,
            TPU_V6E.vector_lanes,
        )
        kv = projected[:, :2].reshape(batch, projection_tiles, TPU_V6E.vector_lanes)
        score = projected[:, 2:].reshape(batch, projection_tiles, TPU_V6E.vector_lanes)

        state_out_ref[...] = state_ref[...]
        for row in range(batch):
            destination = slots[row] + COMPRESS_RATIO
            state_out_ref[row, destination, 0, ...] = kv[row]
            state_out_ref[row, destination, 1, ...] = score[row] + ape_ref[row].astype(jnp.float32)

        state = state_out_ref[...]
        emit = (jnp.mod(positions + 1, COMPRESS_RATIO) == 0).astype(jnp.int32)
        value_ref[...] = jnp.zeros_like(value_ref)

        # Non-emitting blocks only update state; no pooled record is consumed.
        @pl.when(jnp.any(emit != 0))
        def _emit_record():
            kv_halves = state[:, :, 0].reshape(batch, STATE_SLOTS, 2, head_dim)
            score_halves = state[:, :, 1].reshape(batch, STATE_SLOTS, 2, head_dim)
            previous_kv = kv_halves[:, :COMPRESS_RATIO, 0]
            current_kv = kv_halves[:, COMPRESS_RATIO:, 1]
            previous_score = score_halves[:, :COMPRESS_RATIO, 0]
            current_score = score_halves[:, COMPRESS_RATIO:, 1]
            window_kv = jnp.concatenate((previous_kv, current_kv), axis=1)
            window_score = jnp.concatenate((previous_score, current_score), axis=1)
            flat = _pool_normalize(window_kv, window_score, norm_ref[...], norm_eps=norm_eps)
            nope = flat[:, : head_dim - CSA_ROPE_DIM]
            pairs = flat[:, head_dim - CSA_ROPE_DIM :].reshape(
                batch,
                CSA_ROPE_FREQUENCY_DIM,
                2,
            )
            real, imag = pairs[..., 0], pairs[..., 1]
            cos = cos_ref[...].astype(jnp.float32)
            sin = sin_ref[...].astype(jnp.float32)
            rope = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1).reshape(
                batch, CSA_ROPE_DIM
            )
            pooled = jnp.concatenate((nope, rope), axis=-1)
            if record_kind == "main":
                nope_record, rope_record = _pack_main_cache_rows(pooled)
                value_ref[...] = jnp.concatenate((nope_record, rope_record[:, None, :]), axis=1)
            else:
                value_ref[...] = _pack_index_cache_rows(pooled)

        emit_ref[:, 0] = emit.astype(jnp.bool_)
        current = state[:, COMPRESS_RATIO:]
        rolled = jnp.concatenate((current, current), axis=1)
        for row in range(batch):
            state_out_ref[row, ...] = jnp.where(emit[row] != 0, rolled[row], state[row])


def _pack_main_cache_rows(pooled):
    batch = pooled.shape[0]
    fp8_max = jnp.float32(jnp.finfo(jnp.float8_e4m3fn).max)
    quantized = []
    scales = []
    for block in range(CSA_MAIN_NOPE_SCALE_COUNT):
        values = pooled[
            :,
            block * CSA_FP8_BLOCK_SIZE : (block + 1) * CSA_FP8_BLOCK_SIZE,
        ]
        amax = jnp.maximum(
            jnp.max(jnp.abs(values), axis=-1, keepdims=True),
            CSA_FP8_AMAX_FLOOR,
        )
        scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
        quantized.append((values / scale).astype(jnp.float8_e4m3fn))
        scales.append(scale)
    values = pltpu.bitcast(jnp.concatenate(quantized, axis=-1), jnp.uint8)
    scale_bits = pltpu.bitcast(jnp.concatenate(scales, axis=-1), jnp.uint32)
    scale_bytes = jnp.right_shift(scale_bits, 23).astype(jnp.uint8)
    nope = jnp.concatenate(
        (
            values,
            scale_bytes,
            jnp.zeros((batch, CSA_MAIN_NOPE_PADDING_BYTES), jnp.uint8),
        ),
        axis=-1,
    )
    rope_bits = pltpu.bitcast(pooled[:, -CSA_ROPE_DIM:].astype(jnp.bfloat16), jnp.uint16).astype(
        jnp.int32
    )
    rope = jnp.concatenate(
        (
            jnp.right_shift(rope_bits, 8).astype(jnp.uint8),
            (rope_bits & 0xFF).astype(jnp.uint8),
        ),
        axis=-1,
    )
    return nope.reshape(batch, CSA_CACHE_PACKING, TPU_V6E.vector_lanes), rope


def _pack_index_cache_rows(pooled):
    """Pack TPU-Inference-compatible FP8 values and one E8M0 scale."""
    batch = pooled.shape[0]
    fp8_max = jnp.float32(jnp.finfo(jnp.float8_e4m3fn).max)
    amax = jnp.maximum(
        jnp.max(jnp.abs(pooled), axis=-1, keepdims=True),
        CSA_FP8_AMAX_FLOOR,
    )
    scale = jnp.exp2(jnp.ceil(jnp.log2(amax / fp8_max)))
    values = pltpu.bitcast((pooled / scale).astype(jnp.float8_e4m3fn), jnp.uint8)
    scale_byte = jnp.right_shift(pltpu.bitcast(scale, jnp.uint32), 23).astype(jnp.uint8)
    return jnp.concatenate(
        (
            values,
            scale_byte,
            jnp.zeros((batch, CSA_INDEX_PADDING_BYTES), jnp.uint8),
        ),
        axis=-1,
    ).reshape(
        batch,
        CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
        TPU_V6E.vector_lanes,
    )


def _pool_uniform_groups(
    initial_state,
    kv,
    score,
    norm_ref,
    cos,
    sin,
    *,
    head_dim: int,
    norm_eps: float,
):
    """Pool aligned ratio-4 groups, carrying the preceding group forward."""
    groups = kv.shape[0]
    initial_kv = initial_state[0, :COMPRESS_RATIO, 0].reshape(COMPRESS_RATIO, 2, head_dim)[:, 0]
    initial_score = initial_state[0, :COMPRESS_RATIO, 1].reshape(COMPRESS_RATIO, 2, head_dim)[:, 0]
    if groups == 1:
        previous_kv = initial_kv[None]
        previous_score = initial_score[None]
    else:
        previous_kv = jnp.concatenate((initial_kv[None], kv[:-1, :, :head_dim]), axis=0)
        previous_score = jnp.concatenate((initial_score[None], score[:-1, :, :head_dim]), axis=0)
    window_kv = jnp.concatenate((previous_kv, kv[:, :, head_dim:]), axis=1)
    window_score = jnp.concatenate((previous_score, score[:, :, head_dim:]), axis=1)
    pooled = _pool_normalize(window_kv, window_score, norm_ref[...], norm_eps=norm_eps)
    pairs = pooled[:, -CSA_ROPE_DIM:].reshape(groups, CSA_ROPE_FREQUENCY_DIM, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rope = jnp.stack(
        (real * cos - imag * sin, real * sin + imag * cos),
        axis=-1,
    ).reshape(groups, CSA_ROPE_DIM)
    return jnp.concatenate((pooled[:, :-CSA_ROPE_DIM], rope), axis=-1)


def _csa_dual_uniform_prefill_kernel(
    x_ref,
    main_state_ref,
    index_state_ref,
    fused_weight_ref,
    main_ape_ref,
    index_ape_ref,
    main_norm_ref,
    index_norm_ref,
    cos_ref,
    sin_ref,
    main_nope_ref,
    main_rope_ref,
    index_ref,
    main_state_out_ref,
    index_state_out_ref,
    projection_ref,
    *,
    k_steps: int,
    query_tiles: int,
    last_group: int,
    norm_eps: float,
):
    """Stream query tiles, keeping overlap state in VMEM across tiles."""
    query_tile, k_step = pl.program_id(1), pl.program_id(2)

    @pl.when((query_tile == 0) & (k_step == 0))
    def _initialize_state():
        main_state_out_ref[...] = main_state_ref[...]
        index_state_out_ref[...] = index_state_ref[...]

    product = _project(x_ref[0], fused_weight_ref[...])
    projection_ref[...] = jnp.where(
        k_step == 0,
        product,
        projection_ref[...] + product,
    )

    @pl.when(k_step == k_steps - 1)
    def _finish():
        groups = x_ref.shape[1] // COMPRESS_RATIO
        projected = projection_ref[...]
        main_kv = projected[:, :CSA_MAIN_PROJECTED_DIM].reshape(
            groups,
            COMPRESS_RATIO,
            CSA_MAIN_PROJECTED_DIM,
        )
        main_score = projected[:, CSA_MAIN_PROJECTED_DIM : 2 * CSA_MAIN_PROJECTED_DIM].reshape(
            groups, COMPRESS_RATIO, CSA_MAIN_PROJECTED_DIM
        )
        main_score += (
            main_ape_ref[...]
            .astype(jnp.float32)
            .reshape(
                1,
                COMPRESS_RATIO,
                CSA_MAIN_PROJECTED_DIM,
            )
        )
        index_start = 2 * CSA_MAIN_PROJECTED_DIM
        index_kv = projected[:, index_start : index_start + CSA_INDEX_PROJECTED_DIM].reshape(
            groups, COMPRESS_RATIO, CSA_INDEX_PROJECTED_DIM
        )
        index_score = projected[:, index_start + CSA_INDEX_PROJECTED_DIM :].reshape(
            groups,
            COMPRESS_RATIO,
            CSA_INDEX_PROJECTED_DIM,
        )
        index_score += (
            index_ape_ref[...]
            .astype(jnp.float32)
            .reshape(
                1,
                COMPRESS_RATIO,
                CSA_INDEX_PROJECTED_DIM,
            )
        )

        main = _pool_uniform_groups(
            main_state_out_ref,
            main_kv,
            main_score,
            main_norm_ref,
            cos_ref[0].astype(jnp.float32),
            sin_ref[0].astype(jnp.float32),
            head_dim=CSA_ATTENTION_DIM,
            norm_eps=norm_eps,
        )
        index = _pool_uniform_groups(
            index_state_out_ref,
            index_kv,
            index_score,
            index_norm_ref,
            cos_ref[0].astype(jnp.float32),
            sin_ref[0].astype(jnp.float32),
            head_dim=CSA_INDEX_DIM,
            norm_eps=norm_eps,
        )
        main_nope, main_rope = _pack_main_cache_rows(main)
        index_record = _pack_index_cache_rows(index)
        main_nope_ref[0] = main_nope
        main_rope_ref[0] = main_rope
        index_ref[0] = index_record

        # Padded groups must neither escape to the cache nor replace the carried state.
        main_last = jnp.where(
            query_tile == query_tiles - 1,
            jnp.stack((main_kv[last_group], main_score[last_group]), axis=1),
            jnp.stack((main_kv[-1], main_score[-1]), axis=1),
        )
        index_last = jnp.where(
            query_tile == query_tiles - 1,
            jnp.stack((index_kv[last_group], index_score[last_group]), axis=1),
            jnp.stack((index_kv[-1], index_score[-1]), axis=1),
        )
        main_state_out_ref[0] = jnp.concatenate((main_last, main_last), axis=0).reshape(
            STATE_SLOTS,
            2,
            CSA_MAIN_PROJECTED_DIM // TPU_V6E.vector_lanes,
            TPU_V6E.vector_lanes,
        )
        index_state_out_ref[0] = jnp.concatenate((index_last, index_last), axis=0).reshape(
            STATE_SLOTS,
            2,
            CSA_INDEX_PROJECTED_DIM // TPU_V6E.vector_lanes,
            TPU_V6E.vector_lanes,
        )


@functools.partial(
    jax.jit,
    static_argnames=("norm_eps", "interpret", "schedule"),
)
def csa_dual_uniform_prefill_pallas(
    x,
    main_state,
    index_state,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    *,
    norm_eps: float = CSA_NORM_EPS,
    interpret: bool | None = None,
    schedule: CompressorSchedule | None = None,
):
    """Compress aligned uniform ratio-4 chunks and return cache-native records."""
    if x.ndim != 3 or x.dtype != jnp.bfloat16:
        raise ValueError("x must be BF16 [batch,sequence,hidden]")
    batch, sequence, hidden = x.shape
    if sequence < COMPRESS_RATIO or sequence % COMPRESS_RATIO:
        raise ValueError("sequence must contain complete ratio-4 groups")
    groups = sequence // COMPRESS_RATIO
    main_projected = 2 * CSA_ATTENTION_DIM
    index_projected = 2 * CSA_INDEX_DIM
    if main_state.shape != (batch, STATE_SLOTS, 2, main_projected):
        raise ValueError("main_state must be [batch,8,2,1024]")
    if index_state.shape != (batch, STATE_SLOTS, 2, index_projected):
        raise ValueError("index_state must be [batch,8,2,256]")
    fused_width = 2 * (main_projected + index_projected)
    if fused_weight.shape != (hidden, fused_width):
        raise ValueError("fused_weight must be [hidden,2560]")
    if main_ape.shape != (COMPRESS_RATIO, main_projected) or index_ape.shape != (
        COMPRESS_RATIO,
        index_projected,
    ):
        raise ValueError("compressor APE shapes are invalid")
    if main_norm.shape != (CSA_ATTENTION_DIM,) or index_norm.shape != (CSA_INDEX_DIM,):
        raise ValueError("compressor norm shapes are invalid")
    if cos.shape != (batch, groups, CSA_ROPE_FREQUENCY_DIM) or sin.shape != cos.shape:
        raise ValueError("cos and sin must be [batch,groups,32]")
    tile_k = (
        get_csa_compressor_projection_k_tile(hidden)
        if schedule is None
        else schedule.projection_k_tile
    )
    k_steps = hidden // tile_k
    query_tile = (
        get_csa_compressor_query_tile(sequence)
        if schedule is None
        else min(sequence, schedule.query_tile)
    )
    query_tiles = pl.cdiv(sequence, query_tile)
    tile_groups = query_tile // COMPRESS_RATIO
    padded_groups = query_tiles * tile_groups
    if padded_groups != groups:
        x = jnp.pad(x, ((0, 0), (0, query_tiles * query_tile - sequence), (0, 0)))
        cos = jnp.pad(cos, ((0, 0), (0, padded_groups - groups), (0, 0)), constant_values=1)
        sin = jnp.pad(sin, ((0, 0), (0, padded_groups - groups), (0, 0)))
    main_tiles = main_projected // TPU_V6E.vector_lanes
    index_tiles = index_projected // TPU_V6E.vector_lanes
    main_norm_tiles = CSA_ATTENTION_DIM // TPU_V6E.vector_lanes
    index_norm_tiles = CSA_INDEX_DIM // TPU_V6E.vector_lanes
    if interpret is None:
        interpret = _interpret_pallas()

    outputs = pl.pallas_call(
        functools.partial(
            _csa_dual_uniform_prefill_kernel,
            k_steps=k_steps,
            query_tiles=query_tiles,
            last_group=(groups - 1) % tile_groups,
            norm_eps=float(norm_eps),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch, query_tiles, k_steps),
            in_specs=(
                pl.BlockSpec((1, query_tile, tile_k), lambda request, q, k: (request, q, k)),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (request, 0, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (request, 0, 0, 0, 0),
                ),
                pl.BlockSpec((tile_k, fused_width), lambda request, q, k: (k, 0)),
                pl.BlockSpec(
                    (COMPRESS_RATIO, main_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (0, 0, 0),
                ),
                pl.BlockSpec(
                    (COMPRESS_RATIO, index_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (0, 0, 0),
                ),
                pl.BlockSpec(
                    (main_norm_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (0, 0),
                ),
                pl.BlockSpec(
                    (index_norm_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (0, 0),
                ),
                pl.BlockSpec(
                    (1, tile_groups, CSA_ROPE_FREQUENCY_DIM),
                    lambda request, q, k: (request, q, 0),
                ),
                pl.BlockSpec(
                    (1, tile_groups, CSA_ROPE_FREQUENCY_DIM),
                    lambda request, q, k: (request, q, 0),
                ),
            ),
            out_specs=(
                pl.BlockSpec(
                    (1, tile_groups, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                    lambda request, q, k: (request, q, 0, 0),
                ),
                pl.BlockSpec(
                    (1, tile_groups, TPU_V6E.vector_lanes),
                    lambda request, q, k: (request, q, 0),
                ),
                pl.BlockSpec(
                    (
                        1,
                        tile_groups,
                        CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
                        TPU_V6E.vector_lanes,
                    ),
                    lambda request, q, k: (request, q, 0, 0),
                ),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (request, 0, 0, 0, 0),
                ),
                pl.BlockSpec(
                    (1, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
                    lambda request, q, k: (request, 0, 0, 0, 0),
                ),
            ),
            scratch_shapes=(pltpu.VMEM((query_tile, fused_width), jnp.float32),),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (batch, padded_groups, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct((batch, padded_groups, TPU_V6E.vector_lanes), jnp.uint8),
            jax.ShapeDtypeStruct(
                (
                    batch,
                    padded_groups,
                    CSA_INDEX_RECORD_BYTES // TPU_V6E.vector_lanes,
                    TPU_V6E.vector_lanes,
                ),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct(
                (batch, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
                jnp.float32,
            ),
            jax.ShapeDtypeStruct(
                (batch, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
                jnp.float32,
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=f"csa-dual-prefill-s{sequence}-k{tile_k}",
    )(
        x,
        main_state.reshape(batch, STATE_SLOTS, 2, main_tiles, TPU_V6E.vector_lanes),
        index_state.reshape(batch, STATE_SLOTS, 2, index_tiles, TPU_V6E.vector_lanes),
        fused_weight.astype(jnp.bfloat16),
        main_ape.reshape(COMPRESS_RATIO, main_tiles, TPU_V6E.vector_lanes),
        index_ape.reshape(COMPRESS_RATIO, index_tiles, TPU_V6E.vector_lanes),
        main_norm.reshape(main_norm_tiles, TPU_V6E.vector_lanes),
        index_norm.reshape(index_norm_tiles, TPU_V6E.vector_lanes),
        cos,
        sin,
    )
    return (
        outputs[0][:, :groups],
        outputs[1][:, :groups],
        outputs[2][:, :groups],
        outputs[3].reshape(batch, STATE_SLOTS, 2, CSA_MAIN_PROJECTED_DIM),
        outputs[4].reshape(batch, STATE_SLOTS, 2, CSA_INDEX_PROJECTED_DIM),
    )


@functools.partial(
    jax.jit,
    donate_argnames=("state_pool",),
    static_argnames=(
        "norm_eps",
        "record_kind",
        "interpret",
        "schedule",
    ),
)
def csa_state_step_fused_pallas(
    x_t,
    state_pool,
    fused_weight,
    ape,
    norm_weight,
    cos,
    sin,
    positions,
    *,
    record_kind: str,
    norm_eps: float = CSA_NORM_EPS,
    interpret: bool | None = None,
    schedule: CompressorSchedule | None = None,
):
    """Fuse decode projection, state update, pooling, norm, RoPE, and packing."""
    if x_t.ndim != 2:
        raise ValueError("x_t must be [batch,hidden]")
    batch, hidden = x_t.shape
    if record_kind not in ("main", "index"):
        raise ValueError("record_kind must be 'main' or 'index'")
    head_dim = CSA_ATTENTION_DIM if record_kind == "main" else CSA_INDEX_DIM
    projected_dim = 2 * head_dim
    if state_pool.shape != (batch, STATE_SLOTS, 2, projected_dim):
        raise ValueError("state_pool must be [batch,8,2,2*head_dim]")
    if state_pool.dtype != jnp.float32:
        raise ValueError("state_pool must use FP32")
    if fused_weight.shape != (hidden, 2 * projected_dim):
        raise ValueError("fused_weight must be [hidden,4*head_dim]")
    if ape.shape != (COMPRESS_RATIO, projected_dim) or ape.dtype != jnp.float32:
        raise ValueError("ape must be FP32 [4,2*head_dim]")
    if norm_weight.shape != (head_dim,):
        raise ValueError("norm_weight must be [head_dim]")
    if positions.shape != (batch,) or not jnp.issubdtype(positions.dtype, jnp.integer):
        raise ValueError("positions must be integer [batch]")
    if cos.ndim != 2 or sin.shape != cos.shape or cos.shape[1] != CSA_ROPE_FREQUENCY_DIM:
        raise ValueError("cos and sin must both be [max_position,32]")

    tile_b = TPU_V6E.sublanes
    padded_batch = (batch + tile_b - 1) // tile_b * tile_b
    pad = padded_batch - batch
    head_tiles = head_dim // TPU_V6E.vector_lanes
    projection_tiles = projected_dim // TPU_V6E.vector_lanes
    tile_k = (
        get_csa_compressor_projection_k_tile(hidden)
        if schedule is None
        else schedule.projection_k_tile
    )
    k_steps = hidden // tile_k
    x_t = jnp.pad(x_t.astype(jnp.bfloat16), ((0, pad), (0, 0)))
    state_pool = jnp.pad(
        state_pool,
        ((0, pad), (0, 0), (0, 0), (0, 0)),
        constant_values=0,
    )
    if pad:
        state_pool = state_pool.at[batch:, :, 1].set(
            -jnp.inf,
            out_sharding=jax.typeof(state_pool).sharding,
        )
    positions = jnp.pad(positions.astype(jnp.int32), (0, pad))
    slots = jnp.mod(positions, COMPRESS_RATIO)
    ape_selected = jnp.take(ape, slots, axis=0).reshape(
        padded_batch, projection_tiles, TPU_V6E.vector_lanes
    )
    source_position = jnp.maximum(positions + 1 - COMPRESS_RATIO, 0)
    cos_selected = jnp.take(cos, source_position, axis=0, mode="clip")
    sin_selected = jnp.take(sin, source_position, axis=0, mode="clip")
    state_tiled = state_pool.reshape(
        padded_batch,
        STATE_SLOTS,
        2,
        projection_tiles,
        TPU_V6E.vector_lanes,
    )
    if interpret is None:
        interpret = _interpret_pallas()

    record_bytes = CSA_MAIN_RECORD_BYTES if record_kind == "main" else CSA_INDEX_RECORD_BYTES
    output_tiles = record_bytes // TPU_V6E.vector_lanes
    outputs = pl.pallas_call(
        functools.partial(
            _csa_state_step_kernel,
            head_dim=head_dim,
            k_steps=k_steps,
            norm_eps=float(norm_eps),
            record_kind=record_kind,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded_batch // tile_b, k_steps),
            in_specs=(
                pl.BlockSpec((tile_b, tile_k), lambda block, k: (block, k)),
                pl.BlockSpec(
                    (tile_b, STATE_SLOTS, 2, projection_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0, 0, 0),
                ),
                pl.BlockSpec((tile_k, 2 * projected_dim), lambda block, k: (k, 0)),
                pl.BlockSpec(
                    (tile_b, projection_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0),
                ),
                pl.BlockSpec((head_tiles, TPU_V6E.vector_lanes), lambda block, k: (0, 0)),
                pl.BlockSpec((tile_b, CSA_ROPE_FREQUENCY_DIM), lambda block, k: (block, 0)),
                pl.BlockSpec((tile_b, CSA_ROPE_FREQUENCY_DIM), lambda block, k: (block, 0)),
                pl.BlockSpec((tile_b, 1), lambda block, k: (block, 0)),
            ),
            out_specs=(
                pl.BlockSpec(
                    (tile_b, output_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0),
                ),
                pl.BlockSpec((tile_b, 1), lambda block, k: (block, 0)),
                pl.BlockSpec(
                    (tile_b, STATE_SLOTS, 2, projection_tiles, TPU_V6E.vector_lanes),
                    lambda block, k: (block, 0, 0, 0, 0),
                ),
            ),
            scratch_shapes=(pltpu.VMEM((tile_b, 2 * projected_dim), jnp.float32),),
        ),
        out_shape=(
            jax.ShapeDtypeStruct(
                (padded_batch, output_tiles, TPU_V6E.vector_lanes),
                jnp.uint8,
            ),
            jax.ShapeDtypeStruct((padded_batch, 1), jnp.bool_),
            jax.ShapeDtypeStruct(state_tiled.shape, jnp.float32),
        ),
        input_output_aliases={1: 2},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "arbitrary"),
            disable_bounds_checks=True,
        ),
        interpret=interpret,
        name=(f"csa-compressor-decode-b{tile_b}-k{tile_k}-d{head_dim}-{record_kind}"),
    )(
        x_t,
        state_tiled,
        fused_weight.astype(jnp.bfloat16),
        ape_selected,
        norm_weight.reshape(head_tiles, TPU_V6E.vector_lanes),
        cos_selected,
        sin_selected,
        positions[:, None],
    )
    value, emit, state_out = outputs
    return (
        value[:batch].reshape(batch, record_bytes),
        emit[:batch, 0],
        state_out[:batch].reshape(batch, STATE_SLOTS, 2, projected_dim),
    )


def _scatter_fp8_cache_rows(main_nope, main_rope, index_records, locations, caches, *, page_size):
    page, slot = locations // page_size, locations % page_size
    group, lane = slot // CSA_CACHE_PACKING, slot % CSA_CACHE_PACKING
    return (
        caches[0].at[page, slot].set(main_nope, mode="drop"),
        caches[1].at[page, group, lane].set(main_rope, mode="drop"),
        caches[2].at[page, group, lane].set(index_records, mode="drop"),
    )


def _write_fp8_cache_blocks_kernel(
    locations_ref,
    main_nope_ref,
    main_rope_ref,
    index_ref,
    main_nope_cache_ref,
    main_rope_cache_ref,
    index_cache_ref,
    _,
    __,
    ___,
    dma_semaphores,
    *,
    page_size: int,
    tile_n: int,
    capacity: int,
):
    location = locations_ref[pl.program_id(0)]

    @pl.when(location < capacity)
    def _write():
        page, slot = location // page_size, location % page_size
        group, groups = slot // CSA_CACHE_PACKING, tile_n // CSA_CACHE_PACKING
        copies = (
            pltpu.make_async_copy(
                main_nope_ref.at[0],
                main_nope_cache_ref.at[page, pl.ds(slot, tile_n)],
                dma_semaphores.at[0],
            ),
            pltpu.make_async_copy(
                main_rope_ref.at[0],
                main_rope_cache_ref.at[page, pl.ds(group, groups)],
                dma_semaphores.at[1],
            ),
            pltpu.make_async_copy(
                index_ref.at[0],
                index_cache_ref.at[page, pl.ds(group, groups)],
                dma_semaphores.at[2],
            ),
        )
        for copy in copies:
            copy.start()
        for copy in copies:
            copy.wait()


@functools.partial(jax.jit, static_argnames=("page_size", "interpret"))
def _write_fp8_cache_rows(
    main_nope,
    main_rope,
    index_records,
    emit,
    locations,
    main_nope_cache,
    main_rope_cache,
    index_cache,
    *,
    page_size: int,
    interpret: bool | None = None,
):
    """Keep native cache layout; batch complete same-page runs without reading old groups."""
    tokens = main_nope.shape[0]
    capacity = main_nope_cache.shape[0] * page_size
    valid = emit & (locations >= 0) & (locations < capacity)
    locations = jnp.where(valid, locations, capacity)
    caches = (main_nope_cache, main_rope_cache, index_cache)

    def scatter(caches):
        return _scatter_fp8_cache_rows(
            main_nope, main_rope, index_records, locations, caches, page_size=page_size
        )

    # Largest page-local block that fits the input and the uint8 vector row tile.
    row_tile = TPU_V6E.uint8_row_tile
    tile_n = min(page_size, tokens) // row_tile * row_tile
    if not tile_n:
        return scatter(caches)
    pad = (-tokens) % tile_n
    block_locations = jnp.pad(locations, (0, pad), constant_values=capacity).reshape(-1, tile_n)
    starts = block_locations[:, 0]
    contiguous = (starts % CSA_CACHE_PACKING == 0) & (starts % page_size + tile_n <= page_size)
    contiguous &= jnp.all(block_locations == starts[:, None] + jnp.arange(tile_n), axis=1)
    empty = jnp.all(block_locations == capacity, axis=1)
    if interpret is None:
        interpret = _interpret_pallas()

    def write_blocks(caches, block_starts=starts):
        records = tuple(
            jnp.pad(value, ((0, pad), *((0, 0) for _ in value.shape[1:])))
            for value in (main_nope, main_rope, index_records)
        )
        shapes = (
            (tile_n, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
            (tile_n // CSA_CACHE_PACKING, CSA_CACHE_PACKING, CSA_ROPE_RECORD_BYTES),
            (tile_n // CSA_CACHE_PACKING, CSA_CACHE_PACKING, CSA_INDEX_RECORD_BYTES),
        )
        return pl.pallas_call(
            functools.partial(
                _write_fp8_cache_blocks_kernel,
                page_size=page_size,
                tile_n=tile_n,
                capacity=capacity,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(starts.shape[0],),
                in_specs=(
                    *(
                        pl.BlockSpec((1, *shape), lambda block, *_: (block, 0, 0, 0))
                        for shape in shapes
                    ),
                    *(pl.BlockSpec(memory_space=pltpu.HBM) for _ in caches),
                ),
                out_specs=tuple(pl.BlockSpec(memory_space=pltpu.HBM) for _ in caches),
                scratch_shapes=(pltpu.SemaphoreType.DMA((len(caches),)),),
            ),
            out_shape=tuple(jax.ShapeDtypeStruct(cache.shape, cache.dtype) for cache in caches),
            input_output_aliases={4: 0, 5: 1, 6: 2},
            compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
            interpret=interpret,
            name="csa-fp8-cache-write-blocks",
        )(
            block_starts,
            # Keep writeback staging from competing with projection weights for VMEM.
            *(
                (
                    pltpu.with_memory_space_constraint(record.reshape(-1, *shape), pltpu.HBM)
                    if starts.shape[0] > 1
                    else record.reshape(-1, *shape)
                )
                for record, shape in zip(records, shapes, strict=True)
            ),
            *caches,
        )

    # One block cannot need both writeback paths.
    if starts.shape[0] == 1:
        return jax.lax.cond(jnp.all(contiguous | empty), write_blocks, scatter, caches)

    def write_mixed(caches):
        caches = write_blocks(caches, jnp.where(contiguous, starts, capacity))
        # Complete DMA blocks own every lane of their packed groups. The caller's
        # unique-location contract makes the remaining scatter writes disjoint.
        records = tuple(
            jnp.pad(value, ((0, pad), *((0, 0) for _ in value.shape[1:])))
            for value in (main_nope, main_rope, index_records)
        )

        def step(block, caches):
            def scatter_block(caches):
                rows = tuple(
                    jax.lax.dynamic_slice_in_dim(record, block * tile_n, tile_n)
                    for record in records
                )
                return _scatter_fp8_cache_rows(
                    *rows, block_locations[block], caches, page_size=page_size
                )

            return jax.lax.cond(
                contiguous[block] | empty[block], lambda c: c, scatter_block, caches
            )

        return jax.lax.fori_loop(0, starts.shape[0], step, caches)

    def write_irregular(caches):
        return jax.lax.cond(jnp.any(contiguous), write_mixed, scatter, caches)

    return jax.lax.cond(jnp.all(contiguous | empty), write_blocks, write_irregular, caches)


@functools.partial(
    jax.jit,
    static_argnames=("schedule", "norm_eps", "interpret"),
    donate_argnames=(
        "main_state",
        "index_state",
        "main_nope_cache",
        "main_rope_cache",
        "index_cache",
    ),
)
def csa_compressor(
    x,
    fused_weight,
    main_ape,
    index_ape,
    main_norm,
    index_norm,
    cos,
    sin,
    main_state,
    index_state,
    main_nope_cache,
    main_rope_cache,
    index_cache,
    metadata: CompressorMetadata,
    *,
    schedule: CompressorSchedule,
    norm_eps: float = CSA_NORM_EPS,
    interpret: bool = False,
):
    """Dual CSA compression into caller-owned state and packed FP8 caches.

    Blocks are prepared by the backend: single tokens or aligned complete groups,
    ordered per request, with every real token covered exactly once. Metadata
    values are dynamic; block capacities/topology specialize the compiled program.
    Returns (main_state, index_state, main_nope, main_rope, index_cache).
    State/cache inputs are donated; replace them with the returned buffers.
    """
    if x.ndim != 2 or x.dtype != jnp.bfloat16:
        raise ValueError("x must be BF16 [T,hidden]")
    tokens, hidden = x.shape
    if fused_weight.shape != (hidden, 2 * (CSA_MAIN_PROJECTED_DIM + CSA_INDEX_PROJECTED_DIM)):
        raise ValueError("fused_weight must be [hidden,2560]")
    if fused_weight.dtype != jnp.bfloat16:
        raise ValueError("fused_weight must use BF16")
    if schedule.projection_k_tile <= 0 or hidden % schedule.projection_k_tile:
        raise ValueError("projection_k_tile must be a positive divisor of hidden")
    if schedule.projection_k_tile % TPU_V6E.vector_lanes:
        raise ValueError("projection_k_tile must be lane aligned")
    if schedule.query_tile <= 0 or schedule.query_tile % COMPRESS_RATIO:
        raise ValueError("query_tile must contain complete compression groups")
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("norm_eps must be finite and positive")
    for state, width in (
        (main_state, CSA_MAIN_PROJECTED_DIM),
        (index_state, CSA_INDEX_PROJECTED_DIM),
    ):
        if (
            state.ndim != 4
            or state.shape[1:] != (STATE_SLOTS, 2, width)
            or state.dtype != jnp.float32
        ):
            raise ValueError("state pools must be FP32 [slots,8,2,projected_dim]")
    if main_state.shape[0] != index_state.shape[0] or main_state.shape[0] == 0:
        raise ValueError("state pools must have the same nonzero slot capacity")
    for value, shape in (
        (main_ape, (COMPRESS_RATIO, CSA_MAIN_PROJECTED_DIM)),
        (index_ape, (COMPRESS_RATIO, CSA_INDEX_PROJECTED_DIM)),
        (main_norm, (CSA_ATTENTION_DIM,)),
        (index_norm, (CSA_INDEX_DIM,)),
    ):
        if value.shape != shape or value.dtype != jnp.float32:
            raise ValueError(f"APE/norm must be FP32 with shape {shape}")
    if (
        cos.ndim != 2
        or cos.shape[0] == 0
        or cos.shape[1] != CSA_ROPE_FREQUENCY_DIM
        or sin.shape != cos.shape
    ):
        raise ValueError("cos and sin must be [max_position,32]")
    if cos.dtype != jnp.float32 or sin.dtype != jnp.float32:
        raise ValueError("cos and sin must use FP32")
    if main_nope_cache.ndim != 4:
        raise ValueError("main_nope_cache must be [pages,page_size,4,128]")
    pages, page_size = main_nope_cache.shape[:2]
    if pages == 0 or page_size <= 0 or page_size % CSA_CACHE_PACKING:
        raise ValueError("cache page size must be positive and divisible by four")
    for cache, shape in (
        (main_nope_cache, (pages, page_size, CSA_CACHE_PACKING, TPU_V6E.vector_lanes)),
        (
            main_rope_cache,
            (pages, page_size // CSA_CACHE_PACKING, CSA_CACHE_PACKING, CSA_ROPE_RECORD_BYTES),
        ),
        (
            index_cache,
            (pages, page_size // CSA_CACHE_PACKING, CSA_CACHE_PACKING, CSA_INDEX_RECORD_BYTES),
        ),
    ):
        if cache.shape != shape or cache.dtype != jnp.uint8:
            raise ValueError(f"packed cache must be uint8 {shape}")
    requests = metadata.state_indices.shape[0]
    for value, shape in (
        (metadata.positions, (tokens,)),
        (metadata.cache_locations, (tokens,)),
        (metadata.cu_q_lens, (requests + 1,)),
        (metadata.state_indices, (requests,)),
    ):
        if value.shape != shape or value.dtype != jnp.int32:
            raise ValueError(f"compressor metadata must be int32 {shape}")
    for block in metadata.blocks:
        if block.token_indices.ndim != 2 or block.token_indices.dtype != jnp.int32:
            raise ValueError("block token_indices must be int32 [R,Q]")
        rows, length = block.token_indices.shape
        if rows <= 0 or length <= 0 or (length != 1 and length % COMPRESS_RATIO):
            raise ValueError("blocks need positive rows and one token or complete groups")
        if block.request_indices.shape != (rows,) or block.request_indices.dtype != jnp.int32:
            raise ValueError("block request_indices must be int32 [R]")
    if not tokens or not requests:
        return main_state, index_state, main_nope_cache, main_rope_cache, index_cache

    for block in metadata.blocks:
        ids = block.token_indices
        request = jnp.clip(block.request_indices, 0, requests - 1)
        safe_ids = jnp.clip(ids, 0, tokens - 1)
        state_ids = metadata.state_indices[request]
        active = (block.request_indices >= 0) & (block.request_indices < requests)
        active &= (state_ids >= 0) & (state_ids < main_state.shape[0])
        active &= jnp.all(
            (ids >= metadata.cu_q_lens[request, None])
            & (ids < metadata.cu_q_lens[request + 1, None])
            & (ids < tokens),
            axis=1,
        )
        positions = metadata.positions[safe_ids]
        active &= jnp.all((positions >= 0) & (positions < cos.shape[0]), axis=1)
        safe_state = jnp.clip(state_ids, 0, main_state.shape[0] - 1)
        selected_main = jnp.where(active[:, None, None, None], main_state[safe_state], 0)
        selected_index = jnp.where(active[:, None, None, None], index_state[safe_state], 0)
        inputs = jnp.where(active[:, None, None], x[safe_ids], 0)
        positions = jnp.where(active[:, None], positions, 0)
        if ids.shape[1] == 1:
            main_record, emit, selected_main = csa_state_step_fused_pallas(
                inputs[:, 0],
                selected_main,
                fused_weight[:, : 2 * CSA_MAIN_PROJECTED_DIM],
                main_ape,
                main_norm,
                cos,
                sin,
                positions[:, 0],
                record_kind="main",
                norm_eps=norm_eps,
                interpret=interpret,
                schedule=schedule,
            )
            index_record, _, selected_index = csa_state_step_fused_pallas(
                inputs[:, 0],
                selected_index,
                fused_weight[:, 2 * CSA_MAIN_PROJECTED_DIM :],
                index_ape,
                index_norm,
                cos,
                sin,
                positions[:, 0],
                record_kind="index",
                norm_eps=norm_eps,
                interpret=interpret,
                schedule=schedule,
            )
            nope = main_record[:, :CSA_MAIN_NOPE_RECORD_BYTES]
            rope = main_record[:, CSA_MAIN_NOPE_RECORD_BYTES:]
            locations = metadata.cache_locations[safe_ids[:, 0]]
            valid = active & emit
        else:
            group_positions = positions[:, ::COMPRESS_RATIO]
            nope, rope, index_record, selected_main, selected_index = (
                csa_dual_uniform_prefill_pallas(
                    inputs,
                    selected_main,
                    selected_index,
                    fused_weight,
                    main_ape,
                    index_ape,
                    main_norm,
                    index_norm,
                    cos[group_positions],
                    sin[group_positions],
                    norm_eps=norm_eps,
                    interpret=interpret,
                    schedule=schedule,
                )
            )
            locations = metadata.cache_locations[safe_ids[:, COMPRESS_RATIO - 1 :: COMPRESS_RATIO]]
            valid = jnp.broadcast_to(active[:, None], locations.shape).reshape(-1)
        locations = locations.reshape(-1)
        capacity = pages * page_size
        destinations = jnp.where(
            valid & (locations >= 0) & (locations < capacity), locations, capacity
        )
        if ids.shape[1] == 1:
            # Sparse decode writes retain the native page/group/lane layout.
            main_nope_cache, main_rope_cache, index_cache = _scatter_fp8_cache_rows(
                nope.reshape(-1, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                rope.reshape(-1, CSA_ROPE_RECORD_BYTES),
                index_record.reshape(-1, CSA_INDEX_RECORD_BYTES),
                destinations,
                (main_nope_cache, main_rope_cache, index_cache),
                page_size=page_size,
            )
        else:
            main_nope_cache, main_rope_cache, index_cache = _write_fp8_cache_rows(
                nope.reshape(-1, CSA_CACHE_PACKING, TPU_V6E.vector_lanes),
                rope.reshape(-1, CSA_ROPE_RECORD_BYTES),
                index_record.reshape(-1, CSA_INDEX_RECORD_BYTES),
                valid & (locations < capacity),
                locations,
                main_nope_cache,
                main_rope_cache,
                index_cache,
                page_size=page_size,
                interpret=interpret,
            )
        state_destinations = jnp.where(active, state_ids, main_state.shape[0])
        main_state = main_state.at[state_destinations].set(selected_main, mode="drop")
        index_state = index_state.at[state_destinations].set(selected_index, mode="drop")
    return main_state, index_state, main_nope_cache, main_rope_cache, index_cache


__all__ = [
    "CompressorBlock",
    "CompressorMetadata",
    "csa_compressor",
]
