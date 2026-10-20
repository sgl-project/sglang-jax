"""gmm_v2_fused_glu: gate_up (interleaved SwiGLU epilogue) -> h (VMEM) -> down in ONE pallas_call.

Opt-in replacement for the two decode-time gmm_v2 calls of an EP MoE layer when both calls would run
with one grid step per gm tile (tile_k == size_k and tile_n == size_n, the GLM-5.2 v7x int4 W4A8 decode
case). Same metadata scan, same lhs/out row tiling, same quantized tile matmul arithmetic as gmm_v2,
so the output is bit-identical to ``gmm_v2(glu_chunk=c)`` followed by ``gmm_v2`` on the activated
intermediate. What changes: one pallas_call and one grid step per expert instead of two, and the
[M, I] bf16 intermediate never leaves VMEM (no HBM write + read, no second metadata scan, no second
lhs row DMA). Per-expert bookkeeping (~0.65 us/step on v7x) + the h round trip is the whole payoff.

Restrictions (the wrapper returns None so callers fall back to the two-call path): per-channel rhs
scale (k_blocks == 1), no bias, quantized lhs path, full-K/full-N tiles for both halves.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel import gmm_v2 as _gmm_v2_mod
from sgl_jax.srt.kernels.gmm.megablox_gmm_kernel.gmm_v2 import (
    GmmConfigs,
    MetadataRef,
    TileSizes,
    calculate_tiling,
    fill_metadata,
    glu_from_interleaved,
    make_gmm_configs,
    zero_out_end,
    zero_out_start,
)


def _quant_tile_matmul(tiled_lhs: jax.Array, tiled_rhs: jax.Array, cfgs: GmmConfigs) -> jax.Array:
    """Quantized [tile_m, tile_k] @ [tile_k, tile_n] exactly as gmm_v2.inner_kernel._matmul does it
    (per-lhs-block absmax fp8 quantization, int4 rhs upcast on the fp8 MXU, block rescale, N split
    by MXU column size). Returns acc in cfgs.acc_dtype; the per-channel rhs scale is applied by the caller.
    """
    lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
    q_block_size = cfgs.lhs_cfgs.quant_block_size
    assert lhs_q_dtype is not None and jnp.issubdtype(lhs_q_dtype, jnp.floating)
    dtype_max = float(jnp.finfo(lhs_q_dtype).max)
    tile_m, tile_k, tile_n = cfgs.tiles.tile_m, cfgs.tiles.tile_k, cfgs.tiles.tile_n
    acc_dtype = cfgs.acc_dtype
    acc_list = []
    mxu_size = pltpu.get_tpu_info().mxu_column_size
    for start_n in range(0, tile_n, mxu_size):
        end_n = min(tile_n, start_n + mxu_size)
        acc_n = jnp.zeros((tile_m, end_n - start_n), dtype=acc_dtype)
        for start_k in range(0, tile_k, q_block_size):
            end_k = min(tile_k, start_k + q_block_size)
            block_lhs = tiled_lhs[:, start_k:end_k]
            block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]
            block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
            block_abs_max = jnp.maximum(
                block_abs_max, dtype_max * jnp.finfo(block_abs_max.dtype).tiny
            )
            block_scale = block_abs_max / dtype_max
            block_lhs_q = (block_lhs / block_scale).astype(lhs_q_dtype)
            if not jnp.issubdtype(block_rhs.dtype, jnp.floating):
                block_rhs = block_rhs.astype(lhs_q_dtype)
            block_acc = jnp.matmul(
                block_lhs_q, block_rhs, preferred_element_type=jnp.float32
            ).astype(acc_dtype)
            acc_n += block_acc * block_scale.astype(acc_dtype)
        acc_list.append(acc_n)
    return jnp.concatenate(acc_list, axis=1)


def _row_mask(acc: jax.Array, metadata_ref: MetadataRef, gm_id: jax.Array, sublane: int):
    m_start = metadata_ref.gm_id_to_m_offset[gm_id]
    m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
    m_offset = m_start - m_start % sublane
    m_start_local = m_start - m_offset
    m_end_local = m_end - m_offset
    iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
    mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
    return jnp.where(mask, acc, 0), m_end_local


def _fused_inner(
    lhs_ref,  # [rows, sublane, K] bf16 (bounded slice of the gm tile)
    w1_ref,  # [K, N1] int4 / fp8 (interleaved gate|up chunks)
    s1_ref,  # [1, N1] f32
    w2_ref,  # [I, K]
    s2_ref,  # [1, K] f32
    out_ref,  # [rows, sublane, K] (bounded slice of the gm tile)
    partial_out_ref,
    metadata_ref,
    glu_pre_ref,
    *,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    glu_chunk: int,
):
    """One grid step = one gm tile (one expert's rows); rows go to out[M, K] like gmm_v2."""
    sublane = cfgs1.dims.size_lhs_sublane
    gm_id = pl.program_id(0)

    # ---- gate_up + SwiGLU epilogue (== gmm_v2 glu_chunk path, minus the partial-row carry, which
    # only re-adds other groups' rows: those rows are masked to zero here and never consumed) ----
    tiled_lhs = lhs_ref[...].reshape(-1, cfgs1.tiles.tile_k)
    acc = _quant_tile_matmul(tiled_lhs, w1_ref[...], cfgs1)
    acc *= s1_ref[...].astype(acc.dtype)
    acc_masked, _ = _row_mask(acc, metadata_ref, gm_id, sublane)
    glu_pre_ref[...] = acc_masked.reshape(glu_pre_ref.shape).astype(glu_pre_ref.dtype)
    h = glu_from_interleaved(glu_pre_ref[...], glu_chunk).astype(
        cfgs2.out_dtype
    )  # == gmm1 output dtype
    h = h.reshape(-1, cfgs2.tiles.tile_k)  # [tile_m, I]

    # ---- down (== gmm_v2 single-k-step epilogue incl. the partial-row carry) ----
    acc2 = _quant_tile_matmul(h, w2_ref[...], cfgs2)
    acc2 *= s2_ref[...].astype(acc2.dtype)
    acc2_masked, m_end_local = _row_mask(acc2, metadata_ref, gm_id, sublane)
    out_ref[...] = acc2_masked.reshape(out_ref.shape).astype(out_ref.dtype)
    partial_out_zeros = jnp.zeros_like(partial_out_ref)
    out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])
    last_row = m_end_local // sublane
    partial_out_ref[...] = jnp.where(
        m_end_local % sublane == 0, partial_out_zeros, out_ref[last_row]
    )


def _kernel_main(
    group_sizes_ref,
    group_offset_ref,
    *rest,
    cfgs1: GmmConfigs,
    cfgs2: GmmConfigs,
    glu_chunk: int,
):
    if cfgs1.skip_prefix:
        m_start_ref, rest = rest[0], rest[1:]
    else:
        m_start_ref = None
    (
        lhs_ref,
        w1_ref,
        s1_ref,
        w2_ref,
        s2_ref,
        out_ref,
        partial_out_ref,
        metadata_ref,
        zero_ref,
        semaphore_ref,
        glu_pre_ref,
    ) = rest
    dims = cfgs1.dims
    sublane = dims.size_lhs_sublane
    num_gm = fill_metadata(
        group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs1, m_start_ref=m_start_ref
    )

    def group_of(gm):
        return metadata_ref.gm_id_to_group_id[gm]

    def lhs_index_map(gm):
        m_start = metadata_ref.gm_id_to_m_offset[gm]
        m_end = metadata_ref.gm_id_to_m_offset[gm + 1]
        row_start = m_start // sublane
        row_size = pl.cdiv(m_end, sublane) - row_start
        return (pl.ds(row_start, row_size), 0, 0)

    def out_index_map(gm):
        is_last_gm = gm == (pl.num_programs(0) - 1)
        m_start = metadata_ref.gm_id_to_m_offset[gm]
        m_end = metadata_ref.gm_id_to_m_offset[gm + 1]
        row_start = m_start // sublane
        row_end = jnp.where(is_last_gm, pl.cdiv(m_end, sublane), m_end // sublane)
        return (pl.ds(row_start, row_end - row_start), 0, 0)

    bounded = pl.BoundedSlice(cfgs1.tiles.tile_m // sublane)
    in_specs = [
        pl.BlockSpec((bounded, sublane, cfgs1.tiles.tile_k), lhs_index_map),
        pl.BlockSpec(
            (None, cfgs1.tiles.tile_k, cfgs1.tiles.tile_n), lambda gm: (group_of(gm), 0, 0)
        ),
        pl.BlockSpec((None, None, 1, cfgs1.tiles.tile_n), lambda gm: (group_of(gm), 0, 0, 0)),
        pl.BlockSpec(
            (None, cfgs2.tiles.tile_k, cfgs2.tiles.tile_n), lambda gm: (group_of(gm), 0, 0)
        ),
        pl.BlockSpec((None, None, 1, cfgs2.tiles.tile_n), lambda gm: (group_of(gm), 0, 0, 0)),
    ]
    out_spec = pl.BlockSpec((bounded, sublane, cfgs2.tiles.tile_n), out_index_map)

    if cfgs2.zero_init:
        zero_size = zero_out_start(
            out_ref, zero_ref, semaphore_ref, metadata_ref, num_gm, dims=cfgs2.dims
        )

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(_fused_inner, cfgs1=cfgs1, cfgs2=cfgs2, glu_chunk=glu_chunk),
        grid=(num_gm,),
        in_specs=in_specs,
        out_specs=out_spec,
    )
    lhs_in = lhs_ref.reshape(-1, sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, sublane, out_ref.shape[-1])
    pipeline_fn(
        lhs_in,
        w1_ref,
        s1_ref,
        w2_ref,
        s2_ref,
        out_in,
        scratches=[partial_out_ref, metadata_ref, glu_pre_ref],
    )

    if cfgs2.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs2.dims)


def fused_glu_down_configs(
    lhs,
    w1,
    w1_scale,
    w2,
    w2_scale,
    group_sizes,
    group_offset,
    *,
    glu_chunk,
    preferred_element_type,
    acc_dtype,
    zero_initialize,
    vmem_limit_bytes,
    skip_prefix,
    tile_m=None,
):
    """Build the two gmm_v2 configs the fused kernel mirrors; None when the shapes do not qualify."""
    if w1_scale is None or w2_scale is None or w1_scale.shape[1] != 1 or w2_scale.shape[1] != 1:
        return None
    size_m, size_k = lhs.shape
    _, _, n1 = w1.shape
    _, size_i, k2 = w2.shape
    if k2 != size_k or n1 != 2 * size_i or not glu_chunk:
        return None
    # gmm_v2 would tile each call with vmem_limit/3 per rhs buffer; mirror its tile_m and require full tiles.
    cfgs1 = make_gmm_configs(
        lhs,
        w1,
        w1_scale,
        None,
        group_sizes,
        group_offset,
        tile_info=(TileSizes(tile_m, size_k, n1) if tile_m else calculate_tiling),
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=True,
        zero_initialize=False,
        skip_prefix=skip_prefix,
        glu_chunk=glu_chunk,
    )
    h_proxy = jax.ShapeDtypeStruct((size_m, size_i), cfgs1.out_dtype)
    cfgs2 = make_gmm_configs(
        h_proxy,
        w2,
        w2_scale,
        None,
        group_sizes,
        group_offset,
        tile_info=(TileSizes(tile_m, size_i, size_k) if tile_m else calculate_tiling),
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=True,
        zero_initialize=zero_initialize,
        skip_prefix=skip_prefix,
    )
    t1, t2 = cfgs1.tiles, cfgs2.tiles
    # _quant_tile_matmul is the fp8-MXU form (per-block absmax float quantization); chips with an
    # int8 MXU pick an int8 lhs for int4 / int8 weights and keep the two-call path.
    for q in (cfgs1.lhs_cfgs.quant_dtype, cfgs2.lhs_cfgs.quant_dtype):
        if q is None or not jnp.issubdtype(q, jnp.floating):
            return None
    if (t1.tile_k, t1.tile_n) != (size_k, n1) or (t2.tile_k, t2.tile_n) != (size_i, size_k):
        return None
    if t1.tile_m != t2.tile_m:
        return None
    if cfgs1.rhs_cfgs.quant_block_size not in (
        None,
        size_k,
    ) or cfgs2.rhs_cfgs.quant_block_size not in (None, size_i):
        return None
    return cfgs1, cfgs2


def gmm_v2_fused_glu(
    lhs: jax.Array,  # [M, K] bf16
    w1: jax.Array,  # [G, K, 2I] interleaved gate|up chunks
    w1_scale: jax.Array,  # [G, 1, 1, 2I]
    w2: jax.Array,  # [G, I, K]
    w2_scale: jax.Array,  # [G, 1, 1, K]
    group_sizes: jax.Array,
    group_offset: jax.Array | None = None,
    *,
    glu_chunk: int,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    zero_initialize: bool = True,
    vmem_limit_bytes: int | None = None,
    skip_prefix_groups: (
        bool | None
    ) = None,  # None -> gmm_v2's default (start the metadata scan at the shard's first expert)
    name: str | None = None,
    tile_m: int | None = None,  # override gmm_v2's tile_m (tile_k / tile_n stay full-K / full-N)
) -> jax.Array | None:
    """Returns out [M, K] (== gmm_v2(glu) then gmm_v2(down)) or None if the shapes do not qualify."""
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    elif jnp.isscalar(group_offset):
        group_offset = group_offset[None]
    if skip_prefix_groups is None:
        skip_prefix_groups = _gmm_v2_mod._SKIP_PREFIX_GROUPS
    tiling_vmem = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)  # what gmm_v2 tiles against
    r = fused_glu_down_configs(
        lhs,
        w1,
        w1_scale,
        w2,
        w2_scale,
        group_sizes,
        group_offset,
        glu_chunk=glu_chunk,
        preferred_element_type=preferred_element_type,
        acc_dtype=acc_dtype,
        zero_initialize=zero_initialize,
        vmem_limit_bytes=tiling_vmem,
        skip_prefix=skip_prefix_groups,
        tile_m=tile_m,
    )
    if r is None:
        return None
    cfgs1, cfgs2 = r
    dims, t1, t2 = cfgs1.dims, cfgs1.tiles, cfgs2.tiles
    if vmem_limit_bytes is None:
        vmem_limit_bytes = tiling_vmem
    w1_scale = w1_scale.astype(jnp.float32)
    w2_scale = w2_scale.astype(jnp.float32)
    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, t1.tile_m) - 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    scratch_shapes = [
        pltpu.VMEM((dims.size_lhs_sublane, t2.tile_n), cfgs2.out_dtype),  # partial_out (down)
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]
    if cfgs2.zero_init:
        out_bytes = jnp.dtype(cfgs2.out_dtype).itemsize
        tile_zero_m = min(2 * 1024 * 1024 // num_lanes // out_bytes, dims.size_m)
        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs2.out_dtype),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    else:
        scratch_shapes += [None, None]
    scratch_shapes.append(
        pltpu.VMEM(
            (t1.tile_m // dims.size_lhs_sublane, dims.size_lhs_sublane, t1.tile_n), cfgs1.out_dtype
        )
    )
    out_init = jax.ShapeDtypeStruct((dims.size_m, t2.tile_n), cfgs2.out_dtype)
    prefetch = [group_sizes, group_offset]
    if cfgs1.skip_prefix:
        gidx = jnp.arange(group_sizes.shape[0], dtype=jnp.int32)
        m_start = jnp.sum(jnp.where(gidx < group_offset[0], group_sizes, 0)).astype(jnp.int32)[None]
        prefetch.append(m_start)
    flops = 2 * dims.size_m * dims.size_k * t1.tile_n + 2 * dims.size_m * t2.tile_k * t2.tile_n
    rhs_bits = jax.dtypes.itemsize_bits(w1.dtype)
    bytes_accessed = (
        dims.size_m * dims.size_k * lhs.dtype.itemsize
        + dims.size_group * (dims.size_k * t1.tile_n + t2.tile_k * t2.tile_n) * rhs_bits // 8
        + dims.size_m * t2.tile_n * jnp.dtype(cfgs2.out_dtype).itemsize
    )
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    operands = [lhs, w1, w1_scale, w2, w2_scale]
    return pl.pallas_call(
        functools.partial(_kernel_main, cfgs1=cfgs1, cfgs2=cfgs2, glu_chunk=int(glu_chunk)),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            in_specs=[hbm] * len(operands),
            out_specs=hbm,
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes, disable_bounds_checks=True
        ),
        name=name
        or (
            f"gmm_v2_fused_glu-g_{dims.size_group}-m_{dims.size_m}"
            f"-k_{dims.size_k}-i_{t2.tile_k}-tm_{t1.tile_m}"
        ),
        cost_estimate=pl.CostEstimate(
            flops=flops, bytes_accessed=bytes_accessed, transcendentals=dims.size_m * t2.tile_k
        ),
    )(*prefetch, *operands)
