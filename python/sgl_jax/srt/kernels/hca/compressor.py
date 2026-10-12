"""HCA projection, recurrent-state updates, and boundary emission.

State updates alias the donated pool and write only the current row. The full
128-row state is read only when a compression boundary emits a cache record.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp
from jax.experimental.pallas import tpu as pltpu

from sgl_jax.srt.kernels.hca.common import searchsorted_right
from sgl_jax.srt.kernels.hca.tuned_block_sizes import HCAKernelSchedule
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime


def _interpret_pallas() -> bool:
    return (
        os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")
        or not is_tpu_runtime()
    )


def _pool_normalize_rotate(kv, score, norm_weight, cos, sin, *, norm_eps: float):
    """Turn one 128-row group per entry into an HCA record.

    Softmax-pool ``kv`` with per-feature weights from ``score`` (axis 1 is the
    within-group row), RMS-normalize, scale by ``norm_weight``, and rotate the
    final 64 features as 32 interleaved complex pairs.  All inputs are FP32;
    ``kv``/``score`` are ``[entries, ratio, head_tiles, 128]``, ``norm_weight``
    broadcasts over entries, and ``cos``/``sin`` are ``[entries, 32]``.  Returns
    flat FP32 ``[entries, head_dim]``.
    """
    entries = kv.shape[0]
    if kv.ndim == 3:
        # ``[entries, ratio, head_dim]`` rows straight from the state pool's layout.
        head_dim = kv.shape[2]
        pooled = jnp.sum(kv * jax.nn.softmax(score, axis=1), axis=1)
        pooled *= jax.lax.rsqrt(jnp.mean(jnp.square(pooled), axis=1, keepdims=True) + norm_eps)
        pooled = pooled * norm_weight.reshape(1, head_dim)
    else:
        head_dim = kv.shape[2] * kv.shape[3]
        pooled = jnp.sum(kv * jax.nn.softmax(score, axis=1), axis=1)
        pooled *= jax.lax.rsqrt(jnp.mean(jnp.square(pooled), axis=(1, 2), keepdims=True) + norm_eps)
        pooled = (pooled * norm_weight).reshape(entries, head_dim)

    rope = pooled[:, head_dim - 64 :]
    pairs = rope.reshape(entries, 32, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = jnp.stack((real * cos - imag * sin, real * sin + imag * cos), axis=-1).reshape(
        entries, 64
    )
    return jnp.concatenate((pooled[:, : head_dim - 64], rotated), axis=-1)


def _compress_prefill_kernel(
    x_ref,
    wkv_ref,
    wgate_ref,
    ape_ref,
    norm_ref,
    cos_ref,
    sin_ref,
    out_ref,
    kv_acc_ref,
    score_acc_ref,
    *,
    ratio: int,
    head_dim: int,
    entries_per_step: int,
    norm_eps: float,
    k_steps: int,
):
    """Reduce projection K tiles in a fixed order and emit HCA records."""
    k_step = pl.program_id(2)

    @pl.when(k_step == 0)
    def _zero_accumulators():
        kv_acc_ref[...] = jnp.zeros_like(kv_acc_ref)
        score_acc_ref[...] = jnp.zeros_like(score_acc_ref)

    x_tile = x_ref[0].astype(jnp.bfloat16)
    kv_acc_ref[...] += jax.lax.dot_general(
        x_tile,
        wkv_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    score_acc_ref[...] += jax.lax.dot_general(
        x_tile,
        wgate_ref[...],
        (((1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )

    @pl.when(k_step == k_steps - 1)
    def _epilogue():
        head_tiles = head_dim // 128
        kv = kv_acc_ref[...].reshape(entries_per_step, ratio, head_tiles, 128)
        score = score_acc_ref[...].reshape(entries_per_step, ratio, head_tiles, 128)
        score += ape_ref[...].astype(jnp.float32).reshape(1, ratio, head_tiles, 128)
        out_ref[0] = _pool_normalize_rotate(
            kv,
            score,
            norm_ref[0].astype(jnp.float32).reshape(1, head_tiles, 128),
            cos_ref[...].astype(jnp.float32),
            sin_ref[...].astype(jnp.float32),
            norm_eps=norm_eps,
        ).astype(out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "compress_ratio",
        "head_dim",
        "norm_eps",
        "interpret",
        "out_dtype",
        "schedule",
    ),
)
def token_compress_prefill_pallas(
    x,
    wkv,
    wgate,
    ape,
    norm_weight,
    cos_strided,
    sin_strided,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
    norm_eps: float = 1e-6,
    interpret: bool | None = None,
    out_dtype=jnp.bfloat16,
):
    """Compress complete uniform-prefill groups without materializing projections."""
    batch, sequence, hidden = x.shape
    if ape.shape != (compress_ratio, head_dim):
        raise ValueError("ape must be [128,512]")

    num_entries = sequence // compress_ratio
    entries_per_step = min(schedule.prefill_entries_per_step, num_entries)
    if entries_per_step <= 0 or sequence % compress_ratio:
        raise ValueError("sequence must contain complete HCA compression groups")
    if cos_strided.shape != (num_entries, 32) or sin_strided.shape != cos_strided.shape:
        raise ValueError("RoPE tables must both be [entries,32]")

    tile_k = schedule.projection_tile_k(hidden)
    k_steps = hidden // tile_k
    token_tile = entries_per_step * compress_ratio
    # ``entries_per_step`` need not divide ``num_entries``: reductions are
    # entry-local, so the masked trailing step's stray reads reach no live output.
    num_steps = (num_entries + entries_per_step - 1) // entries_per_step
    kernel = functools.partial(
        _compress_prefill_kernel,
        ratio=compress_ratio,
        head_dim=head_dim,
        entries_per_step=entries_per_step,
        norm_eps=float(norm_eps),
        k_steps=k_steps,
    )
    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=0,
        grid=(batch, num_steps, k_steps),
        in_specs=(
            pl.BlockSpec((1, token_tile, tile_k), lambda bi, i, k: (bi, i, k)),
            pl.BlockSpec((head_dim, tile_k), lambda bi, i, k: (0, k)),
            pl.BlockSpec((head_dim, tile_k), lambda bi, i, k: (0, k)),
            pl.BlockSpec((compress_ratio, head_dim), lambda bi, i, k: (0, 0)),
            pl.BlockSpec((1, head_dim), lambda bi, i, k: (0, 0)),
            pl.BlockSpec((entries_per_step, 32), lambda bi, i, k: (i, 0)),
            pl.BlockSpec((entries_per_step, 32), lambda bi, i, k: (i, 0)),
        ),
        out_specs=pl.BlockSpec((1, entries_per_step, head_dim), lambda bi, i, k: (bi, i, 0)),
        scratch_shapes=(
            pltpu.VMEM((token_tile, head_dim), jnp.float32),
            pltpu.VMEM((token_tile, head_dim), jnp.float32),
        ),
    )
    return pl.pallas_call(
        kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((batch, num_entries, head_dim), out_dtype),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=_interpret_pallas() if interpret is None else interpret,
        name=f"hca-prefill-r128-m{token_tile}-k{tile_k}-n512",
    )(
        x.astype(jnp.bfloat16),
        wkv.astype(jnp.bfloat16),
        wgate.astype(jnp.bfloat16),
        ape,
        norm_weight.reshape(1, head_dim),
        cos_strided,
        sin_strided,
    )


def init_hca_state_pool(
    num_requests: int,
    *,
    compress_ratio: int = 128,
    head_dim: int = 512,
):
    """Create ``[request,ratio,(kv,score),dim]`` FP32 HCA state.

    Score rows start at ``-inf``.  The separate state-kind axis makes the two rows
    written by one token contiguous without byte-packing or reducing precision.
    """
    if num_requests < 1 or compress_ratio < 1 or head_dim < 1:
        raise ValueError("num_requests, compress_ratio, and head_dim must be positive")
    shape = (num_requests, compress_ratio, head_dim)
    kv = jnp.zeros(shape, jnp.float32)
    score = jnp.full(shape, -jnp.inf, jnp.float32)
    return jnp.stack((kv, score), axis=2)


def _hca_projection_fused_kernel(
    x_ref,
    fused_weight_ref,
    ape_selected_ref,
    projected_ref,
    acc_ref,
    *,
    k_steps: int,
    native_output: bool = False,
):
    """Reduce hidden tiles in order; assemble native output inside VMEM."""
    k_step = pl.program_id(2)
    output_half = pl.program_id(1)
    product = jax.lax.dot_general(
        x_ref[...].astype(jnp.bfloat16),
        fused_weight_ref[...].astype(jnp.bfloat16),
        (((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    acc_ref[...] = jnp.where(k_step == 0, product, acc_ref[...] + product)

    @pl.when(k_step == k_steps - 1)
    def _finish():
        if native_output:
            kv, score = jnp.split(acc_ref[...], 2, axis=1)
            projected_ref[...] = jnp.stack(
                (kv, score + ape_selected_ref[...].astype(jnp.float32)), axis=1
            )
            return
        projected_ref[...] = acc_ref[...]

        @pl.when(output_half == 1)
        def _add_ape():
            projected_ref[...] = acc_ref[...] + ape_selected_ref[...].astype(jnp.float32)


@functools.partial(
    jax.jit,
    static_argnames=("compress_ratio", "head_dim", "schedule"),
)
def hca_project_fused_pallas(
    x_t,
    fused_weight,
    ape,
    positions,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
):
    """Return FP32 ``[KV, score + APE]`` rows for each token as ``[T, 2, head_dim]``.
    This function does not read or update recurrent state."""
    return _hca_project_fused(
        x_t,
        fused_weight,
        ape,
        positions,
        schedule=schedule,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        native_output=True,
    )


@functools.partial(
    jax.jit,
    static_argnames=("compress_ratio", "head_dim", "schedule"),
)
def hca_project_fused_halves(
    x_t,
    fused_weight,
    ape,
    positions,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
):
    """``(kv [T, head_dim], score + APE [T, head_dim])`` in FP32.

    Keep lane-aligned halves for callers that do not need the native state layout.
    """
    return _hca_project_fused(
        x_t,
        fused_weight,
        ape,
        positions,
        schedule=schedule,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        native_output=False,
    )


def _hca_project_fused(
    x_t, fused_weight, ape, positions, *, schedule, compress_ratio, head_dim, native_output
):
    # Weight/APE shapes and the r128/d512 constants are the backend's
    # contract; what varies per call is the token count, checked below.
    if x_t.ndim != 2:
        raise ValueError(f"x_t must be [B,hidden], got {x_t.shape}")
    batch, hidden = x_t.shape
    if ape.shape != (compress_ratio, head_dim) or ape.dtype != jnp.float32:
        raise ValueError("ape must be FP32 [128,512]")
    if positions.shape != (batch,):
        raise ValueError("positions must be [T]")
    tile_b = min(
        schedule.projection_batch_tile_max,
        ((batch + schedule.sublanes - 1) // schedule.sublanes) * schedule.sublanes,
    )
    padded_batch = ((batch + tile_b - 1) // tile_b) * tile_b
    batch_steps = padded_batch // tile_b
    tile_k = schedule.projection_tile_k(hidden)
    k_steps = hidden // tile_k
    pad = padded_batch - batch
    x_padded = jnp.pad(x_t.astype(jnp.bfloat16), ((0, pad), (0, 0)))
    positions_padded = jnp.pad(positions.astype(jnp.int32), (0, pad))
    ape_selected = jnp.take(ape, jnp.mod(positions_padded, compress_ratio), axis=0).astype(
        jnp.float32
    )

    # A native-output program owns both halves, including their shared physical tiles.
    output_width = 2 * head_dim if native_output else head_dim
    output_shape = (padded_batch, 2, head_dim) if native_output else (padded_batch, 2 * head_dim)
    output_spec = (
        pl.BlockSpec((tile_b, 2, head_dim), lambda bi, m, k: (bi, 0, 0))
        if native_output
        else pl.BlockSpec((tile_b, head_dim), lambda bi, m, k: (bi, m))
    )
    projection_weight = fused_weight.astype(jnp.bfloat16)
    if native_output and not _interpret_pallas():
        # Reuse weights across token tiles instead of reloading them from HBM.
        projection_weight = pltpu.with_memory_space_constraint(projection_weight, pltpu.VMEM)
    projected = pl.pallas_call(
        functools.partial(
            _hca_projection_fused_kernel, k_steps=k_steps, native_output=native_output
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(batch_steps, 1 if native_output else 2, k_steps),
            in_specs=(
                pl.BlockSpec((tile_b, tile_k), lambda bi, m, k: (bi, k)),
                pl.BlockSpec((tile_k, output_width), lambda bi, m, k: (k, m)),
                pl.BlockSpec((tile_b, head_dim), lambda bi, m, k: (bi, 0)),
            ),
            out_specs=output_spec,
            scratch_shapes=(pltpu.VMEM((tile_b, output_width), jnp.float32),),
        ),
        out_shape=jax.ShapeDtypeStruct(output_shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary")
        ),
        interpret=_interpret_pallas(),
        name=f"hca-state-project-b{tile_b}-k{tile_k}-n{output_width}",
    )(x_padded, projection_weight, ape_selected)
    if native_output:
        return projected[:batch]
    return projected[:batch, :head_dim], projected[:batch, head_dim:]


@functools.partial(
    jax.jit,
    donate_argnames=("state_pool",),
    static_argnames=("compress_ratio", "head_dim", "schedule"),
)
def hca_state_pool_update_fused_pallas(
    x_t,
    state_pool,
    fused_weight,
    ape,
    positions,
    request_slots,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
    valid_mask=None,
):
    """Project decode tokens, then scatter the rows into the donated state pool at
    ``[request_slot, position % 128]``.

    ``valid_mask`` marks real tokens in a padded batch; padded rows are dropped
    without touching any state row."""
    # state_pool geometry is fixed by init_hca_state_pool; only the
    # per-call token vectors can disagree.
    if request_slots.shape != positions.shape:
        raise ValueError("positions and request_slots must have the same shape")
    if valid_mask is not None and valid_mask.shape != positions.shape:
        raise ValueError("valid_mask must be [T]")
    projected = hca_project_fused_pallas(
        x_t,
        fused_weight,
        ape,
        positions,
        schedule=schedule,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
    )

    slot = jnp.mod(positions, compress_ratio).astype(jnp.int32)
    rows = request_slots.astype(jnp.int32)
    mode = "promise_in_bounds"
    if valid_mask is not None:
        # Padded tokens share the dummy row and could collide; give each its own
        # dropped out-of-bounds row so ``unique_indices`` stays literally true.
        rows = jnp.where(
            valid_mask, rows, state_pool.shape[0] + jnp.arange(rows.shape[0], dtype=jnp.int32)
        )
        slot = jnp.where(valid_mask, slot, 0)
        mode = "drop"
    return state_pool.at[rows, slot, :, :].set(
        projected,
        mode=mode,
        indices_are_sorted=False,
        unique_indices=True,
    )


def _hca_emit_values(selected, valid, norm_weight, cos_sin, *, norm_eps: float):
    """Pool native ``[tile_n, 128, 2, head_dim]`` state rows."""
    live = valid[:, None, None]
    kv = jnp.where(live, selected[:, :, 0, :].astype(jnp.float32), 0.0)
    score = jnp.where(live, selected[:, :, 1, :].astype(jnp.float32), -jnp.inf)
    cos_sin = cos_sin.astype(jnp.float32)
    normed = _pool_normalize_rotate(
        kv,
        score,
        norm_weight.astype(jnp.float32),
        cos_sin[:, :32],
        cos_sin[:, 32:64],
        norm_eps=norm_eps,
    )
    return jnp.where(valid[:, None], normed, 0.0).astype(jnp.bfloat16)


def _hca_emit_boundary_native_kernel(
    slot_ref,
    start_ref,
    shift_ref,
    ncur_ref,
    ring_ref,
    valid_ref,
    state_pool_hbm_ref,
    projected_hbm_ref,
    norm_weight_ref,
    cos_sin_ref,
    output_ref,
    state_slab,
    cur_slab,
    dma_semaphores,
    *,
    tile_n: int,
    norm_eps: float,
):
    """Boundary snapshots without the gathers: pool over the 128 ring rows plus the
    128 chunk rows of each window with per-row validity masks.

    The softmax pool is permutation-invariant over the window, so the ring rows can
    stay in slot order and the chunk rows in token order; a row is live when its
    position falls on the right side of the request's prefix (ring row ``s`` holds
    position ``end - ((r - s) mod 128)``, chunk slab row ``i`` holds window index
    ``i + shift``).
    """
    block = pl.program_id(0)
    first = block * tile_n
    state_slab[...] = jnp.zeros(state_slab.shape, state_slab.dtype)
    cur_slab[...] = jnp.zeros(cur_slab.shape, cur_slab.dtype)
    for row in range(tile_n):
        live = valid_ref[first + row] != 0

        @pl.when(live)
        def _start(row=row):
            pltpu.make_async_copy(
                state_pool_hbm_ref.at[slot_ref[first + row]],
                state_slab.at[row],
                dma_semaphores.at[0, row],
            ).start()
            start = pl.multiple_of(start_ref[first + row], 1)
            pltpu.make_async_copy(
                projected_hbm_ref.at[pl.ds(start, 128)],
                cur_slab.at[row],
                dma_semaphores.at[1, row],
            ).start()

    for row in range(tile_n):
        live = valid_ref[first + row] != 0

        @pl.when(live)
        def _wait(row=row):
            d0 = state_slab.at[row]
            pltpu.make_async_copy(d0, d0, dma_semaphores.at[0, row]).wait()
            d1 = cur_slab.at[row]
            pltpu.make_async_copy(d1, d1, dma_semaphores.at[1, row]).wait()

    # Masks are built in the full [tile_n, 256, D] shape: Mosaic rejects reshaping a
    # bool (or lane-major int) vector into a trailing unit dim, so per-boundary
    # scalars are spread with iota selects instead of [:, None] broadcasts.
    head_dim = state_slab.shape[-1]
    shape3 = (tile_n, 256, head_dim)
    row3 = jax.lax.broadcasted_iota(jnp.int32, shape3, 0)
    win3 = jax.lax.broadcasted_iota(jnp.int32, shape3, 1)
    ring3 = jnp.zeros(shape3, jnp.int32)
    ncur3 = jnp.zeros(shape3, jnp.int32)
    shift3 = jnp.zeros(shape3, jnp.int32)
    valid3 = jnp.zeros(shape3, jnp.int32)
    for row in range(tile_n):
        sel = row3 == row
        ring3 = jnp.where(sel, ring_ref[first + row], ring3)
        ncur3 = jnp.where(sel, ncur_ref[first + row], ncur3)
        shift3 = jnp.where(sel, shift_ref[first + row], shift3)
        valid3 = jnp.where(sel, valid_ref[first + row], valid3)
    is_state = win3 < 128
    state_live = is_state & (((ring3 - win3) & 127) >= ncur3)
    widx = win3 - 128 + shift3
    cur_live = (~is_state) & (widx >= 128 - ncur3) & (widx < 128)
    live = (state_live | cur_live) & (valid3 != 0)
    kv = jnp.concatenate(
        [state_slab[:, :, 0, :].astype(jnp.float32), cur_slab[:, :, 0, :].astype(jnp.float32)],
        axis=1,
    )
    score = jnp.concatenate(
        [state_slab[:, :, 1, :].astype(jnp.float32), cur_slab[:, :, 1, :].astype(jnp.float32)],
        axis=1,
    )
    kv = jnp.where(live, kv, 0.0)
    score = jnp.where(live, score, -jnp.inf)
    cos_sin = cos_sin_ref[...].astype(jnp.float32)
    normed = _pool_normalize_rotate(
        kv,
        score,
        norm_weight_ref[...].astype(jnp.float32),
        cos_sin[:, :32],
        cos_sin[:, 32:64],
        norm_eps=norm_eps,
    )
    out_row = jax.lax.broadcasted_iota(jnp.int32, normed.shape, 0)
    out_valid = jnp.zeros(normed.shape, jnp.int32)
    for row in range(tile_n):
        out_valid = jnp.where(out_row == row, valid_ref[first + row], out_valid)
    output_ref[...] = jnp.where(out_valid != 0, normed, 0.0).astype(output_ref.dtype)


@functools.partial(jax.jit, static_argnames=("norm_eps", "schedule"))
def _hca_emit_boundary_native_pallas(
    state_pool,
    projected,
    slot,
    start,
    shift,
    ncur,
    ring,
    valid_mask,
    norm_weight,
    cos_sin_selected,
    *,
    norm_eps: float,
    schedule: HCAKernelSchedule,
):
    """``_hca_emit_selected_pallas`` fed straight from the pool and the chunk rows."""
    packed = slot.shape[0]
    tile_n = schedule.boundary_rows(state_pool.shape[1], state_pool.shape[-1])
    padded = ((packed + tile_n - 1) // tile_n) * tile_n
    pad = padded - packed
    i32 = lambda a: jnp.pad(jnp.asarray(a, jnp.int32), (0, pad))
    valid_i = i32(jnp.asarray(valid_mask, jnp.bool_).astype(jnp.int32))
    cos_sin = jnp.pad(jnp.asarray(cos_sin_selected, jnp.float32), ((0, pad), (0, 64)))
    head_dim = state_pool.shape[-1]
    out = pl.pallas_call(
        functools.partial(
            _hca_emit_boundary_native_kernel, tile_n=tile_n, norm_eps=float(norm_eps)
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=6,
            grid=(padded // tile_n,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((1, head_dim), lambda block, *_: (0, 0)),
                pl.BlockSpec((tile_n, 128), lambda block, *_: (block, 0)),
            ),
            out_specs=pl.BlockSpec((tile_n, head_dim), lambda block, *_: (block, 0)),
            scratch_shapes=(
                pltpu.VMEM((tile_n, 128, 2, head_dim), state_pool.dtype),
                pltpu.VMEM((tile_n, 128, 2, head_dim), projected.dtype),
                pltpu.SemaphoreType.DMA((2, tile_n)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((padded, head_dim), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=schedule.vmem_budget_bytes,
            disable_bounds_checks=True,
        ),
        interpret=_interpret_pallas(),
        name=f"hca-boundary-native-n{tile_n}-r128-d{head_dim}",
    )(
        i32(slot),
        i32(start),
        i32(shift),
        i32(ncur),
        i32(ring),
        valid_i,
        state_pool,
        projected,
        jnp.asarray(norm_weight, jnp.float32).reshape(1, head_dim),
        cos_sin,
    )
    return out[:packed]


def _hca_emit_pool_native_kernel(
    request_slots_ref,
    valid_scalar_ref,
    state_pool_hbm_ref,
    valid_storage_ref,
    norm_weight_ref,
    cos_sin_ref,
    output_ref,
    selected_ref,
    dma_semaphores,
    *,
    tile_n: int,
    norm_eps: float,
):
    """Read native state slabs; skip DMA and pooling for tiles without boundaries."""
    block = pl.program_id(0)
    first = block * tile_n
    valid_rows = [valid_scalar_ref[first + row] for row in range(tile_n)]
    any_valid = functools.reduce(jnp.logical_or, [v != 0 for v in valid_rows])

    @pl.when(jnp.logical_not(any_valid))
    def _empty_tile():
        output_ref[...] = jnp.zeros(output_ref.shape, output_ref.dtype)

    @pl.when(any_valid)
    def _pool_tile():
        selected_ref[...] = jnp.zeros(selected_ref.shape, selected_ref.dtype)
        for row in range(tile_n):

            @pl.when(valid_rows[row])
            def _start_state_row_dma(row=row):
                request = request_slots_ref[first + row]
                pltpu.make_async_copy(
                    state_pool_hbm_ref.at[request],
                    selected_ref.at[row],
                    dma_semaphores.at[row],
                ).start()

        for row in range(tile_n):

            @pl.when(valid_rows[row])
            def _wait_state_row_dma(row=row):
                destination = selected_ref.at[row]
                pltpu.make_async_copy(destination, destination, dma_semaphores.at[row]).wait()

        output_ref[...] = _hca_emit_values(
            selected_ref[...],
            valid_storage_ref[:, 0, 0],
            norm_weight_ref[...],
            cos_sin_ref[...],
            norm_eps=norm_eps,
        )


def _hca_emit_selected_kernel(
    selected_ref,
    valid_ref,
    norm_weight_ref,
    cos_sin_ref,
    output_ref,
    *,
    norm_eps: float,
):
    """Emit from boundary snapshots that are already contiguous."""
    output_ref[...] = _hca_emit_values(
        selected_ref[...],
        valid_ref[:, 0, 0],
        norm_weight_ref[...],
        cos_sin_ref[...],
        norm_eps=norm_eps,
    )


def _boundary_launch(packed, valid_mask, cos_sin, schedule):
    """Shared tile selection and padding for the two boundary-emit launchers.

    Rows are independent, so the small tile serves small packed batches and the
    large tile serves serving-sized ones; only the outer scheduling changes.
    """
    tile_n = (
        schedule.boundary_small_tile
        if packed <= schedule.boundary_small_tile
        else schedule.boundary_large_tile
    )
    padded = ((packed + tile_n - 1) // tile_n) * tile_n
    pad = padded - packed
    valid_mask = jnp.pad(valid_mask, (0, pad))
    valid_storage = jnp.broadcast_to(
        valid_mask[:, None, None], (padded, schedule.sublanes, schedule.mxu_lanes)
    )
    return tile_n, padded, pad, valid_mask, valid_storage, jnp.pad(cos_sin, ((0, pad), (0, 64)))


@functools.partial(jax.jit, static_argnames=("norm_eps", "schedule"))
def _hca_emit_selected_pallas(
    selected,
    valid_mask,
    norm_weight,
    cos_sin_selected,
    *,
    schedule: HCAKernelSchedule,
    norm_eps: float,
):
    packed = selected.shape[0]
    tile_n, padded, pad, _, valid_storage, cos_sin_selected = _boundary_launch(
        packed, valid_mask, cos_sin_selected, schedule
    )
    selected = jnp.pad(selected, ((0, pad), (0, 0), (0, 0), (0, 0)))
    output = pl.pallas_call(
        functools.partial(_hca_emit_selected_kernel, norm_eps=float(norm_eps)),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=(padded // tile_n,),
            in_specs=(
                pl.BlockSpec(
                    (tile_n, 128, 2, 512),
                    lambda block: (block, 0, 0, 0),
                ),
                pl.BlockSpec((tile_n, 8, 128), lambda block: (block, 0, 0)),
                pl.BlockSpec((1, 512), lambda block: (0, 0)),
                pl.BlockSpec((tile_n, 128), lambda block: (block, 0)),
            ),
            out_specs=pl.BlockSpec((tile_n, 512), lambda block: (block, 0)),
        ),
        out_shape=jax.ShapeDtypeStruct((padded, 512), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), disable_bounds_checks=True
        ),
        interpret=_interpret_pallas(),
        name=f"hca-boundary-snapshot-n{tile_n}-r128-d512",
    )(
        selected,
        valid_storage,
        norm_weight.reshape(1, 512),
        cos_sin_selected,
    )
    return output[:packed]


@functools.partial(jax.jit, static_argnames=("norm_eps", "schedule"))
def _hca_emit_pool_pallas(
    state_pool,
    request_slots,
    valid_mask,
    norm_weight,
    cos_sin_selected,
    *,
    schedule: HCAKernelSchedule,
    norm_eps: float,
):
    packed = request_slots.shape[0]
    tile_n, padded, pad, valid_mask, valid_storage, cos_sin_selected = _boundary_launch(
        packed, valid_mask, cos_sin_selected, schedule
    )
    request_slots = jnp.pad(request_slots.astype(jnp.int32), (0, pad))
    output = pl.pallas_call(
        functools.partial(
            _hca_emit_pool_native_kernel,
            tile_n=tile_n,
            norm_eps=float(norm_eps),
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(padded // tile_n,),
            in_specs=(
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec((tile_n, 8, 128), lambda block, *_: (block, 0, 0)),
                pl.BlockSpec((1, 512), lambda block, *_: (0, 0)),
                pl.BlockSpec((tile_n, 128), lambda block, *_: (block, 0)),
            ),
            out_specs=pl.BlockSpec((tile_n, 512), lambda block, *_: (block, 0)),
            scratch_shapes=(
                pltpu.VMEM((tile_n, 128, 2, 512), jnp.float32),
                pltpu.SemaphoreType.DMA((tile_n,)),
            ),
        ),
        out_shape=jax.ShapeDtypeStruct((padded, 512), jnp.bfloat16),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",), disable_bounds_checks=True
        ),
        interpret=_interpret_pallas(),
        name=f"hca-boundary-pool-native-n{tile_n}-r128-d512",
    )(
        request_slots,
        valid_mask,
        state_pool,
        valid_storage,
        norm_weight.reshape(1, 512),
        cos_sin_selected,
    )
    return output[:packed]


@functools.partial(
    jax.jit,
    static_argnames=(
        "head_dim",
        "norm_eps",
        "output_dtype",
        "schedule",
    ),
)
def hca_state_pool_emit_pallas(
    state_pool,
    request_slots,
    valid_mask,
    norm_weight,
    cos_selected,
    sin_selected,
    *,
    schedule: HCAKernelSchedule,
    head_dim: int = 512,
    norm_eps: float = 1e-6,
    output_dtype=jnp.bfloat16,
):
    """Pool completed HCA windows with the production r128/d512 Pallas kernel."""
    packed = request_slots.shape[0]
    if request_slots.shape != (packed,) or valid_mask.shape != (packed,):
        raise ValueError("request_slots and valid_mask must both be [P]")
    if cos_selected.shape != (packed, 32) or sin_selected.shape != cos_selected.shape:
        raise ValueError("production HCA RoPE tables must both be [P,32]")

    safe_slots = jnp.where(valid_mask, request_slots, 0).astype(jnp.int32)
    return _hca_emit_pool_pallas(
        state_pool,
        safe_slots,
        valid_mask,
        norm_weight,
        jnp.concatenate((cos_selected, sin_selected), axis=-1),
        schedule=schedule,
        norm_eps=norm_eps,
    )


@functools.partial(
    jax.jit,
    donate_argnames=("state_pool",),
    static_argnames=(
        "compress_ratio",
        "head_dim",
        "norm_eps",
        "output_dtype",
        "schedule",
    ),
)
def hca_state_pool_update_ragged_fused_pallas(
    x,
    state_pool,
    fused_weight,
    ape,
    norm_weight,
    cos,
    sin,
    positions,
    request_slots,
    query_starts,
    prefix_lens,
    seq_lens,
    boundary_token_indices,
    *,
    schedule: HCAKernelSchedule,
    compress_ratio: int = 128,
    head_dim: int = 512,
    norm_eps: float = 1e-6,
    output_dtype=jnp.bfloat16,
):
    """Bit-aligned vectorized state update for request-major ragged chunks.

    Boundary token indices are framework metadata, so only real compression
    boundaries materialize a 128-row snapshot.  Projection and pooling reuse the
    same Pallas geometries as decode/fresh-prefill; the final state scatter keeps
    only each request's last 128 absolute positions, whose modulo slots are unique.
    """
    tokens, hidden = x.shape
    batch = seq_lens.shape[0]
    if state_pool.ndim != 4 or state_pool.shape[1:] != (
        compress_ratio,
        2,
        head_dim,
    ):
        raise ValueError("state_pool must be [requests,128,2,512]")
    # state_pool geometry and weight shapes are guaranteed upstream; these
    # catch a bucketing or page-table mistake in the batch metadata.
    if positions.shape != (tokens,) or request_slots.shape != (tokens,):
        raise ValueError("positions and request_slots must be [T]")
    if query_starts.shape != (batch,) or prefix_lens.shape != (batch,):
        raise ValueError("query_starts and prefix_lens must be [B]")

    # Resolve each request's physical recurrent row once: after arbitrary alloc,
    # free and reuse, slots are neither dense nor request-major in the pool.
    active_requests = seq_lens > prefix_lens
    safe_starts = jnp.minimum(query_starts, tokens - 1)
    request_rows = jnp.where(
        active_requests,
        request_slots[safe_starts],
        state_pool.shape[0] + jnp.arange(batch, dtype=jnp.int32),
    ).astype(jnp.int32)
    selected_state = state_pool.at[request_rows].get(mode="fill", fill_value=0.0)

    projected = hca_project_fused_pallas(
        x,
        fused_weight,
        ape,
        positions,
        schedule=schedule,
        compress_ratio=compress_ratio,
        head_dim=head_dim,
    )
    boundary_count = boundary_token_indices.shape[0]
    emitted = jnp.zeros((tokens, head_dim), jnp.dtype(output_dtype))
    emit_mask = jnp.zeros((tokens,), jnp.bool_)
    if boundary_count:
        boundary_tokens = boundary_token_indices.astype(jnp.int32)
        # Entries padded with the sentinel ``tokens`` read clamped-but-real data,
        # emit zeros through ``boundary_valid``, and drop their scatters.
        boundary_valid = boundary_tokens < tokens
        safe_boundary_tokens = jnp.minimum(boundary_tokens, tokens - 1)
        boundary_request_ids = (searchsorted_right(query_starts, safe_boundary_tokens) - 1).astype(
            jnp.int32
        )
        boundary_rows = request_slots[safe_boundary_tokens].astype(jnp.int32)
        boundary_positions = positions[safe_boundary_tokens].astype(jnp.int32)
        group_positions = (
            boundary_positions[:, None]
            - jnp.arange(compress_ratio - 1, -1, -1, dtype=jnp.int32)[None, :]
        )
        group_slots = jnp.mod(group_positions, compress_ratio)
        current_indices = (
            query_starts[boundary_request_ids, None]
            + group_positions
            - prefix_lens[boundary_request_ids, None]
        )
        current_valid = group_positions >= prefix_lens[boundary_request_ids, None]
        group_starts = jnp.clip(boundary_positions + 1 - compress_ratio, 0, cos.shape[0] - 1)
        cos_selected = cos.at[group_starts].get(mode="promise_in_bounds")
        sin_selected = sin.at[group_starts].get(mode="promise_in_bounds")
        if compress_ratio == 128 and state_pool.shape[1:] == (128, 2, head_dim) and tokens >= 128:
            # No [N, 128, 2, D] gathers: the kernel DMAs each boundary's ring slab and
            # its 128 chunk rows and masks by position (see the kernel docstring).
            prefix_b = prefix_lens[boundary_request_ids]
            n_cur = jnp.clip(boundary_positions + 1 - prefix_b, 0, compress_ratio)
            first_cur = (
                query_starts[boundary_request_ids]
                + (boundary_positions - (compress_ratio - 1))
                - prefix_b
            )
            start = jnp.clip(first_cur, 0, tokens - compress_ratio)
            pooled = _hca_emit_boundary_native_pallas(
                state_pool,
                projected,
                boundary_rows,
                start,
                start - first_cur,
                n_cur,
                jnp.mod(boundary_positions, compress_ratio),
                boundary_valid,
                norm_weight,
                jnp.concatenate((cos_selected, sin_selected), axis=-1),
                norm_eps=norm_eps,
                schedule=schedule,
            )
        else:
            historical = state_pool.at[boundary_rows[:, None], group_slots].get(
                mode="promise_in_bounds"
            )
            safe_current = jnp.clip(current_indices, 0, tokens - 1)
            current = projected[safe_current]
            snapshots = jnp.where(current_valid[:, :, None, None], current, historical)
            pooled = _hca_emit_selected_pallas(
                snapshots,
                boundary_valid,
                norm_weight,
                jnp.concatenate((cos_selected, sin_selected), axis=-1),
                schedule=schedule,
                norm_eps=norm_eps,
            )
        emitted = emitted.at[boundary_tokens].set(pooled, mode="drop")
        emit_mask = emit_mask.at[boundary_tokens].set(True, mode="drop")

    slots = jnp.arange(compress_ratio, dtype=jnp.int32)[None, :]
    final_positions = seq_lens[:, None] - 1
    latest_positions = final_positions - jnp.mod(final_positions - slots, compress_ratio)
    current_valid = latest_positions >= prefix_lens[:, None]
    current_indices = query_starts[:, None] + latest_positions - prefix_lens[:, None]
    safe_current = jnp.clip(current_indices, 0, tokens - 1)
    current = projected[safe_current]
    updated_selected = jnp.where(current_valid[:, :, None, None], current, selected_state)
    # Empty request rows can have query_start == tokens. Never let a clamped
    # gather resolve these to the last live slot and overwrite its new state.
    # Distinct dropped rows also keep unique_indices true with several pads.
    updated_pool = state_pool.at[request_rows].set(
        updated_selected,
        mode="drop",
        indices_are_sorted=False,
        unique_indices=True,
    )
    return emitted, emit_mask, updated_pool


__all__ = [
    "hca_project_fused_pallas",
    "hca_state_pool_emit_pallas",
    "hca_state_pool_update_fused_pallas",
    "hca_state_pool_update_ragged_fused_pallas",
    "init_hca_state_pool",
    "token_compress_prefill_pallas",
]
