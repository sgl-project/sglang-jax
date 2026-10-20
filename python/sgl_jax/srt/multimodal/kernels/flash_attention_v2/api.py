# Copyright 2026 The SGLang-JAX Authors.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from .config import (
    BF16_BYTES,
    DEFAULT_SM_SCALE,
    HEAD_DIM,
    KV_ROWS,
    KV_STREAMS,
    MASK_EXTRA_TILES,
    MAX_HEAD_DIM,
    NATIVE_CHANNELS,
    NATIVE_HEAD_ALIGNMENT,
    NATIVE_TOKEN_ALIGNMENT,
    QKV_INPUTS,
    SCALAR_PREFETCH_COUNT,
    STATE_COLS,
    VMEM_LIMIT_BYTES,
    AttentionOptions,
    HeadValue,
    IOPlan,
    TuningConfig,
    select_tuning,
)
from .io import TokenMajorBlockIO, _align, _pack_sequences
from .kernel import _kernel
from .pipeline import Pipeline


def _head_vector(
    value: HeadValue | None,
    heads: int,
    padded: int,
    *,
    default: float,
    name: str,
    gqa_ratio: int = 1,
) -> jax.Array:
    if value is None:
        value = jnp.full((heads,), default, jnp.float32)
    else:
        value = jnp.asarray(value, dtype=jnp.float32)
        if value.ndim == 0:
            value = jnp.broadcast_to(value, (heads,))
        elif value.shape == (heads // gqa_ratio,) and gqa_ratio != 1:
            value = jnp.repeat(value, gqa_ratio)
        elif value.shape != (heads,):
            raise ValueError(f"{name} must be scalar or have one value per head")
    return jnp.pad(value, (0, padded - heads), constant_values=default)


def _flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_kv: jax.Array,
    *,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    sm_scale: float | None = None,
    logits_soft_cap: float = 0.0,
    mask_value: float = -math.inf,
    q_scale: HeadValue | None = None,
    k_scale: HeadValue | None = None,
    v_scale: HeadValue | None = None,
    attention_sink: HeadValue | None = None,
    force_recovery: bool | jax.Array = False,
    tuning: TuningConfig | None = None,
) -> jax.Array:
    """Pack sequences into aligned head-major blocks before kernel dispatch."""
    query_heads, kv_heads = q.shape[1], k.shape[1]
    gqa_ratio = query_heads // kv_heads
    group_heads, q_rows = tuning.head_group, tuning.q_residency
    padded_heads = pl.cdiv(query_heads, group_heads) * group_heads
    # A group may straddle a GQA boundary.
    kv_group_heads = min(
        kv_heads,
        max(
            pl.cdiv((g * group_heads) % gqa_ratio + group_heads, gqa_ratio)
            for g in range(padded_heads // group_heads)
        ),
    )
    options = AttentionOptions(
        causal=causal,
        window_left=window_size[0],
        window_right=window_size[1],
        sm_scale=DEFAULT_SM_SCALE if sm_scale is None else sm_scale,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        identity_qk_scales=q_scale is None and k_scale is None,
        identity_v_scale=v_scale is None,
        has_sink=attention_sink is not None,
    )
    q_scale_values = _head_vector(q_scale, query_heads, padded_heads, default=1, name="q_scale")
    k_scale_values = _head_vector(
        k_scale,
        query_heads,
        padded_heads,
        default=1,
        name="k_scale",
        gqa_ratio=gqa_ratio,
    )
    v_scale_values = _head_vector(
        v_scale,
        query_heads,
        padded_heads,
        default=1,
        name="v_scale",
        gqa_ratio=gqa_ratio,
    )
    sink = _head_vector(
        attention_sink,
        query_heads,
        padded_heads,
        default=-math.inf,
        name="attention_sink",
    )
    force = _head_vector(
        force_recovery, query_heads, padded_heads, default=0, name="force_recovery"
    ).astype(jnp.int32)

    packed_q, q_metadata, restore = _pack_sequences(q, cu_seqlens_q, padded_heads, q_rows)
    packed_k, kv_metadata, _ = _pack_sequences(k, cu_seqlens_kv, kv_heads + group_heads, KV_ROWS)
    packed_v, _, _ = _pack_sequences(v, cu_seqlens_kv, kv_heads + group_heads, KV_ROWS)
    io_plan = IOPlan.create(k.shape[0], kv_group_heads, tuning)
    kv_slots, output_slots = io_plan.kv.slot_count, io_plan.output.slot_count
    q_shape = (group_heads, q_rows, HEAD_DIM)
    if io_plan.q.slot_count > 1:
        q_shape = (io_plan.q.slot_count, *q_shape)
    scratch = (
        pltpu.VMEM(q_shape, jnp.bfloat16),
        pltpu.VMEM((kv_slots, kv_group_heads, KV_ROWS, HEAD_DIM), jnp.bfloat16),
        pltpu.VMEM((kv_slots, kv_group_heads, KV_ROWS, HEAD_DIM), jnp.bfloat16),
        pltpu.VMEM((group_heads, q_rows, STATE_COLS), jnp.float32),
        pltpu.VMEM((group_heads, q_rows, STATE_COLS), jnp.float32),
        pltpu.VMEM((group_heads, q_rows, HEAD_DIM), jnp.float32),
        pltpu.VMEM((output_slots, group_heads, q_rows, HEAD_DIM), jnp.bfloat16),
        pltpu.VMEM((q_rows // KV_ROWS + MASK_EXTRA_TILES, q_rows, KV_ROWS), jnp.int32),
        (
            pltpu.SemaphoreType.DMA
            if io_plan.q.slot_count == 1
            else pltpu.SemaphoreType.DMA((io_plan.q.slot_count,))
        ),
        pltpu.SemaphoreType.DMA((kv_slots,)),
        pltpu.SemaphoreType.DMA((kv_slots,)),
        pltpu.SemaphoreType.DMA((output_slots,)),
    )

    def kernel(*refs) -> None:
        _kernel(
            *refs,
            options=options,
            gqa_ratio=gqa_ratio,
            kv_group_heads=kv_group_heads,
            pipeline=Pipeline(),
            query_heads=query_heads,
            io_plan=io_plan,
            kv_lookahead=tuning.kv_lookahead,
        )

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(packed_q.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=SCALAR_PREFETCH_COUNT,
            grid=(cu_seqlens_q.shape[0] - 1, padded_heads // group_heads),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * QKV_INPUTS,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="flash_attention_v2",
    )(
        q_metadata,
        kv_metadata,
        q_scale_values,
        k_scale_values,
        v_scale_values,
        sink,
        force,
        packed_q,
        packed_k,
        packed_v,
    )
    output = output.transpose(1, 0, 2, 3).reshape(padded_heads, -1, HEAD_DIM)
    return output[:query_heads, restore, :].transpose(1, 0, 2)


def _flash_attention_native(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_kv: jax.Array,
    *,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    sm_scale: float | None = None,
    logits_soft_cap: float = 0.0,
    mask_value: float = -math.inf,
    q_scale: HeadValue | None = None,
    k_scale: HeadValue | None = None,
    v_scale: HeadValue | None = None,
    attention_sink: HeadValue | None = None,
    force_recovery: bool | jax.Array = False,
    tuning: TuningConfig | None = None,
) -> jax.Array:
    """Read token-major blocks directly into native VMEM buffers."""
    tokens, heads, dim = q.shape
    kv_tokens, kv_heads, _ = k.shape
    group, q_rows = tuning.head_group, tuning.q_residency
    ratio = heads // kv_heads
    padded_heads = _align(heads, group)
    kv_group = min(
        kv_heads,
        max(pl.cdiv((g * group) % ratio + group, ratio) for g in range(padded_heads // group)),
    )
    raw_kv_heads = max(
        _align(
            (g * group // ratio) % NATIVE_HEAD_ALIGNMENT + kv_group,
            NATIVE_HEAD_ALIGNMENT,
        )
        for g in range(padded_heads // group)
    )
    padded_kv_heads = max(
        g * group // ratio // NATIVE_HEAD_ALIGNMENT * NATIVE_HEAD_ALIGNMENT + raw_kv_heads
        for g in range(padded_heads // group)
    )
    physical_dim = _align(dim, NATIVE_CHANNELS)
    folds = physical_dim // NATIVE_CHANNELS
    # The D128 transport can afford bank padding while preserving full 2K KV
    # residency. D256 leaves the raw stride compact to fit the same VMEM limit.
    q_stride = group + (NATIVE_HEAD_ALIGNMENT if folds == 1 else 0)
    kv_stride = raw_kv_heads + (NATIVE_HEAD_ALIGNMENT if folds == 1 else 0)
    io_plan = IOPlan.create(
        kv_tokens,
        kv_group,
        tuning,
        q_staging_bytes=q_rows * q_stride * physical_dim * BF16_BYTES,
        kv_tile_bytes=KV_STREAMS * KV_ROWS * kv_stride * physical_dim * BF16_BYTES,
        output_staging_bytes=q_rows * group * physical_dim * BF16_BYTES,
    )
    q_slots, kv_slots, out_slots = (
        io_plan.q.slot_count,
        io_plan.kv.slot_count,
        io_plan.output.slot_count,
    )

    def pad(x, target_heads):
        return jnp.pad(
            x,
            (
                (
                    0,
                    max(
                        NATIVE_TOKEN_ALIGNMENT,
                        _align(x.shape[0], NATIVE_TOKEN_ALIGNMENT),
                    )
                    - x.shape[0],
                ),
                (0, target_heads - x.shape[1]),
                (0, physical_dim - dim),
            ),
        )

    q_hbm, k_hbm, v_hbm = (
        pad(q, padded_heads),
        pad(k, padded_kv_heads),
        pad(v, padded_kv_heads),
    )
    q_metadata = cu_seqlens_q
    kv_metadata = cu_seqlens_kv
    options = AttentionOptions(
        causal=causal,
        window_left=window_size[0],
        window_right=window_size[1],
        sm_scale=dim**-0.5 if sm_scale is None else sm_scale,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        identity_qk_scales=q_scale is None and k_scale is None,
        identity_v_scale=v_scale is None,
        has_sink=attention_sink is not None,
    )
    vectors = tuple(
        _head_vector(
            value,
            heads,
            padded_heads,
            default=default,
            name=name,
            gqa_ratio=group_ratio,
        )
        for value, default, name, group_ratio in (
            (q_scale, 1, "q_scale", 1),
            (k_scale, 1, "k_scale", ratio),
            (v_scale, 1, "v_scale", ratio),
            (attention_sink, -math.inf, "attention_sink", 1),
            (force_recovery, 0, "force_recovery", 1),
        )
    )
    vectors = (*vectors[:-1], vectors[-1].astype(jnp.int32))
    q_shape = (
        (group, q_rows, physical_dim) if q_slots == 1 else (q_slots, group, q_rows, physical_dim)
    )
    scratch = (
        pltpu.VMEM(q_shape, jnp.bfloat16),
        pltpu.VMEM((kv_slots, folds, KV_ROWS, kv_stride, NATIVE_CHANNELS), jnp.bfloat16),
        pltpu.VMEM((kv_slots, folds, KV_ROWS, kv_stride, NATIVE_CHANNELS), jnp.bfloat16),
        pltpu.VMEM((group, q_rows, STATE_COLS), jnp.float32),
        pltpu.VMEM((group, q_rows, STATE_COLS), jnp.float32),
        pltpu.VMEM((group, q_rows, HEAD_DIM), jnp.float32),
        pltpu.VMEM((out_slots, group, q_rows, physical_dim), jnp.bfloat16),
        pltpu.VMEM((q_rows // KV_ROWS + MASK_EXTRA_TILES, q_rows, KV_ROWS), jnp.int32),
        pltpu.SemaphoreType.DMA((q_slots, folds)),
        pltpu.SemaphoreType.DMA((kv_slots, folds)),
        pltpu.SemaphoreType.DMA((kv_slots, folds)),
        pltpu.SemaphoreType.DMA((out_slots, folds)),
        pltpu.VMEM((q_slots, folds, q_rows, q_stride, NATIVE_CHANNELS), jnp.bfloat16),
        pltpu.VMEM((out_slots, folds, q_rows, group, NATIVE_CHANNELS), jnp.bfloat16),
    )

    def run_sequence(sequence, *refs):
        q_raw, out_raw = refs[-2:]
        core_refs = refs[:-2]
        q_input, _k_input, _v_input, output = core_refs[
            SCALAR_PREFETCH_COUNT : SCALAR_PREFETCH_COUNT + QKV_INPUTS + 1
        ]
        seq = pl.program_id(0)

        def make_io(plan, tensors, token_base, head_start, head_count, loaded_tiles=0):
            hbm = tensors[0][0]
            if hbm is q_input:
                raw, role, raw_heads = (q_raw,), "query", group
                metadata = core_refs[0]
            elif hbm is output:
                raw, role, raw_heads = (out_raw,), "output", group
                metadata = core_refs[0]
            else:
                raw, role, raw_heads = tuple(t[1] for t in tensors), "kv", raw_kv_heads
                metadata = core_refs[1]
            return TokenMajorBlockIO(
                plan,
                tensors,
                token_base,
                head_start,
                head_count,
                loaded_tiles,
                raw,
                role,
                raw_heads,
                metadata[sequence + (seq + 1)] - metadata[sequence + seq],
            )

        _kernel(
            *core_refs,
            options=options,
            gqa_ratio=ratio,
            kv_group_heads=kv_group,
            pipeline=Pipeline(),
            query_heads=heads,
            io_plan=io_plan,
            kv_lookahead=tuning.kv_lookahead,
            io_factory=make_io,
            native_layout=True,
            sequence_offset=sequence,
        )

    def kernel(*refs):
        metadata = refs[0]
        valid_end = metadata[metadata.shape[0] - 1]

        def has_query(sequence):
            return (sequence < metadata.shape[0] - 1) & (metadata[sequence] < valid_end)

        def next_sequence(sequence):
            @pl.when(metadata[sequence + 1] > metadata[sequence])
            def nonempty():
                # Each core invocation drains its DMA before reusing scratch.
                run_sequence(sequence, *refs)

            return sequence + 1

        # Repeated tail boundaries terminate the loop without launching empty
        # grid programs. Metadata capacity stays fixed across runtime batches.
        lax.while_loop(has_query, next_sequence, jnp.int32(0))

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(q_hbm.shape, jnp.bfloat16),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=SCALAR_PREFETCH_COUNT,
            grid=(1, padded_heads // group),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)] * QKV_INPUTS,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch,
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=VMEM_LIMIT_BYTES,
        ),
        name="flash_attention_v2_native",
    )(q_metadata, kv_metadata, *vectors, q_hbm, k_hbm, v_hbm)
    return output[:tokens, :heads, :dim]


def flash_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    cu_seqlens_q: jax.Array,
    cu_seqlens_kv: jax.Array,
    *,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    sm_scale: float | None = None,
    logits_soft_cap: float = 0.0,
    mask_value: float = -math.inf,
    q_scale: HeadValue | None = None,
    k_scale: HeadValue | None = None,
    v_scale: HeadValue | None = None,
    attention_sink: HeadValue | None = None,
    force_recovery: bool | jax.Array = False,
    tuning: TuningConfig | None = None,
) -> jax.Array:
    """Packed BF16 attention with GQA, windows, scales and sink logits.

    Empty or fully masked queries return zero. force_recovery accepts a scalar
    or per-head vector. Bind tuning statically when using jax.jit.
    Logical head dimensions are 1..256, padded to physical width 128 or 256.
    Larger dimensions are rejected before tracing the attention computation.
    IO is internal: auto retains blocked D256 IO and uses native IO for D<256.
    Set tuning.io_layout to select either implementation explicitly.
    Only TPU v7x targets are supported, including explicit abstract meshes.
    """
    if not pltpu.is_tpu_device() or pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        raise ValueError("Flash Attention v2 requires a TPU v7x target (TPU7x)")
    if q.ndim != 3 or k.ndim != 3 or v.shape != k.shape:
        raise ValueError("Q/K/V must be rank 3; K and V must have identical shapes")
    if not 0 < q.shape[2] <= MAX_HEAD_DIM:
        raise ValueError(f"head_dim must be in [1, {MAX_HEAD_DIM}]; got {q.shape[2]}")
    if k.shape[2] != q.shape[2]:
        raise ValueError("Q/K/V head dimensions must match")
    if any(x.dtype != jnp.bfloat16 for x in (q, k, v)):
        raise TypeError("Q/K/V must be BF16")
    query_heads, kv_heads = q.shape[1], k.shape[1]
    if not kv_heads or not query_heads or query_heads % kv_heads:
        raise ValueError("Query heads must be a positive multiple of KV heads")
    if cu_seqlens_q.shape != cu_seqlens_kv.shape or cu_seqlens_q.ndim != 1:
        raise ValueError("Q/KV cumulative lengths must have the same rank-1 shape")
    if cu_seqlens_q.shape[0] < 2:
        raise ValueError("Provide at least one sequence")
    if cu_seqlens_q.dtype != jnp.int32 or cu_seqlens_kv.dtype != jnp.int32:
        raise TypeError("Cumulative lengths must be int32")
    if tuning is None:
        tuning = select_tuning(
            q.shape[0],
            k.shape[0],
            cu_seqlens_q.shape[0] - 1,
            query_heads,
            kv_heads,
            causal=causal,
            window_size=window_size,
            mask_value=mask_value,
        )
    native_io = tuning.io_layout == "native" or (
        tuning.io_layout == "auto" and q.shape[2] != HEAD_DIM
    )
    if not native_io and q.shape[2] != HEAD_DIM:
        raise ValueError("Blocked IO requires head dimension 256")
    if q.shape[0] == 0:
        return jnp.zeros_like(q)
    implementation = _flash_attention_native if native_io else _flash_attention
    return implementation(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        causal=causal,
        window_size=window_size,
        sm_scale=sm_scale,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        attention_sink=attention_sink,
        force_recovery=force_recovery,
        tuning=tuning,
    )
