"""TPU SparseCore + TensorCore implementation of exact DSA attention.

Adapted from ``primatrix/pallas-kernel`` commit ``63898542``
(``tops/ops/dsa/dsa.py``). The only source-level adaptation removes the
optional jaxtyping dependency so this kernel follows sgl-jax's array API.
"""

from __future__ import annotations

import functools
import inspect

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.pallas import tpu_sc as plsc


_SC_GATHER_WINDOW = 128


def _align_to(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _sparse_core_info():
    """Return SparseCore target information while tracing on a TPU."""
    info = pltpu.get_tpu_info().sparse_core
    if info is None:
        raise RuntimeError("The current TPU target does not expose SparseCore support.")
    return info


def _make_sparse_core_mesh(info):
    return plsc.VectorSubcoreMesh(
        num_cores=info.num_cores,
        num_subcores=info.num_subcores,
        core_axis_name="sc_core",
        subcore_axis_name="sc_subcore",
    )


def _sparse_core_gather(
    table: jax.Array,
    indices: jax.Array,
) -> jax.Array:
    """Gather complete BF16 rows using SparseCore hardware row packing."""
    if table.dtype != jnp.bfloat16 or table.ndim != 2:
        raise TypeError("SparseCore table must be a rank-2 bfloat16 array.")
    if table.shape[0] % 2:
        raise ValueError("SparseCore BF16 row packing requires an even row count.")
    if table.shape[1] % 128:
        raise ValueError("SparseCore indirect DMA requires cache dimension C divisible by 128.")
    if indices.dtype != jnp.int32 or indices.ndim != 1:
        raise TypeError("SparseCore indices must be a rank-1 int32 array.")

    info = _sparse_core_info()
    num_indices = indices.shape[0]
    indices_per_step = _SC_GATHER_WINDOW * info.num_cores * info.num_subcores
    if num_indices % indices_per_step:
        raise ValueError(
            f"the number of gather indices must be divisible by "
            f"{indices_per_step}, got {num_indices}"
        )

    value_dim = table.shape[-1]
    mesh = _make_sparse_core_mesh(info)

    def kernel(table_hbm, indices_hbm, output_hbm):
        workers = lax.axis_size((mesh.core_axis_name, mesh.subcore_axis_name))
        row_wave_size = _SC_GATHER_WINDOW * workers
        num_row_chunks = num_indices // row_wave_size
        subcore_first_row_chunk = (
            lax.axis_index((mesh.core_axis_name, mesh.subcore_axis_name)) * num_row_chunks
        )

        @functools.partial(
            pltpu.emit_pipeline,
            grid=(num_row_chunks,),
            in_specs=pl.BlockSpec(
                (_SC_GATHER_WINDOW,),
                lambda chunk: (subcore_first_row_chunk + chunk,),
            ),
        )
        def index_pipeline(indices_vmem):
            row_chunk = subcore_first_row_chunk + pl.program_id(0)
            row_subchunk_size = info.num_lanes
            num_row_subchunks = _SC_GATHER_WINDOW // row_subchunk_size
            packed_output_rows = row_subchunk_size // 2

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, 1),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), value_dim),
                    lambda row_subchunk, _col: (
                        lax.div(
                            indices_vmem[
                                pl.ds(
                                    row_subchunk * row_subchunk_size,
                                    row_subchunk_size,
                                )
                            ],
                            2,
                        ),
                        0,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (packed_output_rows, value_dim),
                    lambda row_subchunk, _col: (
                        row_chunk * num_row_subchunks + row_subchunk,
                        0,
                    ),
                ),
            )
            def data_pipeline(gather_vmem, output_vmem):
                gather_bf16 = gather_vmem.bitcast(jnp.bfloat16)
                output_bf16 = output_vmem.bitcast(jnp.bfloat16)
                row_subchunk = pl.program_id(0)
                index_slice = indices_vmem[
                    pl.ds(
                        row_subchunk * row_subchunk_size,
                        row_subchunk_size,
                    )
                ]

                @plsc.parallel_loop(0, value_dim, step=32)
                def copy_columns(column):
                    rows = []
                    for row in range(row_subchunk_size):
                        packed_rows = gather_bf16[
                            pl.ds(row * 2, 2),
                            pl.ds(column, 32),
                        ].astype(jnp.float32)
                        rows.append(
                            jnp.where(
                                lax.bitwise_and(index_slice[row], 1) == 0,
                                packed_rows[0],
                                packed_rows[1],
                            )
                        )
                    output_bf16[
                        pl.ds(0, row_subchunk_size),
                        pl.ds(column, 32),
                    ] = jnp.stack(rows, axis=0).astype(jnp.bfloat16)

            data_pipeline(
                table_hbm.bitcast(jnp.int32),
                output_hbm.bitcast(jnp.int32),
            )

        index_pipeline(indices_hbm)

    output = jax.ShapeDtypeStruct(
        (num_indices, value_dim),
        jnp.bfloat16,
    )
    compiler_param_kwargs = {"use_tc_tiling_on_sc": True}
    if "needs_layout_passes" in inspect.signature(pltpu.CompilerParams).parameters:
        compiler_param_kwargs["needs_layout_passes"] = True
    kwargs = {
        "mesh": mesh,
        "scratch_types": [],
        "compiler_params": pltpu.CompilerParams(**compiler_param_kwargs),
        "name": "dsa_sparse_core_gather",
    }
    # JAX 0.10 called this argument out_shape; JAX 0.11 calls it out_type.
    if "out_type" in inspect.signature(pl.kernel).parameters:
        kwargs["out_type"] = output
    else:
        kwargs["out_shape"] = output
        kwargs["scratch_shapes"] = kwargs.pop("scratch_types")
    return pl.kernel(kernel, **kwargs)(table, indices)


def _gather_cache_microbatch(
    cache: jax.Array,
    safe_slots: jax.Array,
) -> jax.Array:
    """Gather one cache microbatch from prevalidated physical slots."""
    batch_size, topk = safe_slots.shape
    gathered = _sparse_core_gather(cache, safe_slots.reshape(-1))
    return gathered.reshape(batch_size, topk, cache.shape[-1])


def _dsa_tensor_core_kernel(
    sm_scale_ref,  # FP32 scalar, SMEM
    selected_counts_ref,  # INT32 [B], SMEM
    q_hbm_ref,  # [B, H, align(V + R, 128)], HBM
    cache_hbm_ref,  # [B, K, C_pad], HBM
    output_hbm_ref,  # [B, H, V_pad], HBM
    q_x2_ref,  # [2, bq, H, align(V + R, 128)], VMEM
    cache_x2_ref,  # [2, bq, b_topk, C_pad], VMEM
    output_x2_ref,  # [2, bq, H, V_pad], VMEM
    dma_sems,  # DMA semaphore [3, 2]: q, cache, output
    m_ref,  # FP32 [bq, H, 128], VMEM online-softmax maxima
    l_ref,  # FP32 [bq, H, 128], VMEM online-softmax denominators
    accumulator_ref,  # FP32 [bq, H, V_pad], VMEM value accumulator
    *,
    num_q_steps: int,
    num_k_steps: int,
    bq: int,
    b_topk: int,
    padded_latent_dim: int,
    padded_q_dim: int,
):
    """Single-program online-softmax kernel with explicit HBM/VMEM DMA."""

    def async_copy(src, dst, sem, *, wait: bool):
        copy = pltpu.make_async_copy(src, dst, sem)
        if wait:
            copy.wait()
        else:
            copy.start()

    def fetch_q(q_step, buffer, *, wait: bool):
        q_start = q_step * bq
        q_dst = q_x2_ref.at[buffer]
        sem = dma_sems.at[0, buffer]
        if wait:
            async_copy(q_dst, q_dst, sem, wait=True)
        else:
            async_copy(
                q_hbm_ref.at[pl.ds(q_start, bq)],
                q_dst,
                sem,
                wait=False,
            )

    def fetch_cache(q_step, k_step, buffer, *, wait: bool):
        cache_dst = cache_x2_ref.at[buffer]
        sem = dma_sems.at[1, buffer]
        if wait:
            async_copy(cache_dst, cache_dst, sem, wait=True)
        else:
            async_copy(
                cache_hbm_ref.at[
                    pl.ds(q_step * bq, bq),
                    pl.ds(k_step * b_topk, b_topk),
                ],
                cache_dst,
                sem,
                wait=False,
            )

    def send_output(q_step, buffer, *, wait: bool):
        output_src = output_x2_ref.at[buffer]
        sem = dma_sems.at[2, buffer]
        output_dst = output_hbm_ref.at[pl.ds(q_step * bq, bq)]
        async_copy(output_src, output_dst, sem, wait=wait)

    # Prologue: seed the first query and cache buffers.
    fetch_q(0, 0, wait=False)
    fetch_cache(0, 0, 0, wait=False)

    def compute_q(q_step, _):
        q_buffer = q_step % 2
        next_q_buffer = 1 - q_buffer

        @pl.when(q_step + 1 < num_q_steps)
        def prefetch_next_q():
            fetch_q(q_step + 1, next_q_buffer, wait=False)

        fetch_q(q_step, q_buffer, wait=True)
        m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
        l_ref[...] = jnp.zeros_like(l_ref)
        accumulator_ref[...] = jnp.zeros_like(accumulator_ref)

        def compute_k(k_step, _):
            cache_buffer = k_step % 2
            next_cache_buffer = 1 - cache_buffer

            @pl.when(k_step + 1 < num_k_steps)
            def prefetch_next_cache():
                fetch_cache(q_step, k_step + 1, next_cache_buffer, wait=False)

            fetch_cache(q_step, k_step, cache_buffer, wait=True)
            q = q_x2_ref[q_buffer]
            # SMEM supports scalar loads only. Static unrolling constructs the VMEM
            # vector without asking Mosaic to lower a vector-valued SMEM load.
            selected_counts = jnp.stack([selected_counts_ref[q_step * bq + i] for i in range(bq)])
            cache = cache_x2_ref[cache_buffer]

            scores = lax.dot_general(
                q,
                cache[..., :padded_q_dim],
                dimension_numbers=(((2,), (2,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )
            scores *= sm_scale_ref[()]

            k_offsets = k_step * b_topk + jnp.arange(b_topk, dtype=jnp.int32)
            valid = k_offsets[None, None, :] < selected_counts[:, None, None]
            masked_scores = jnp.where(valid, scores, jnp.float32(-1.0e30))

            old_m = m_ref[..., 0]
            old_l = l_ref[..., 0]
            tile_m = jnp.max(masked_scores, axis=-1)
            new_m = jnp.maximum(old_m, tile_m)
            alpha = jnp.exp(old_m - new_m)
            probabilities = jnp.exp(masked_scores - new_m[..., None]) * valid
            new_l = old_l * alpha + jnp.sum(probabilities, axis=-1)
            value_update = lax.dot_general(
                probabilities.astype(cache.dtype),
                cache[..., :padded_latent_dim],
                dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                preferred_element_type=jnp.float32,
            )

            accumulator_ref[...] = accumulator_ref[...] * alpha[..., None] + value_update
            m_ref[...] = jnp.broadcast_to(new_m[..., None], m_ref.shape)
            l_ref[...] = jnp.broadcast_to(new_l[..., None], l_ref.shape)
            return None

        lax.fori_loop(0, num_k_steps, compute_k, None, unroll=False)

        output_buffer = q_step % 2

        @pl.when(q_step >= 2)
        def wait_old_output():
            send_output(q_step - 2, output_buffer, wait=True)

        denominator = jnp.maximum(l_ref[..., 0], jnp.float32(1.0e-30))
        output_x2_ref[output_buffer] = (accumulator_ref[...] / denominator[..., None]).astype(
            output_x2_ref.dtype
        )
        send_output(q_step, output_buffer, wait=False)

        @pl.when(q_step + 1 < num_q_steps)
        def seed_next_query_cache():
            fetch_cache(q_step + 1, 0, 0, wait=False)

        return None

    lax.fori_loop(0, num_q_steps, compute_q, None, unroll=False)

    # Epilogue: drain the at most two outstanding output transfers.
    for q_step in range(max(0, num_q_steps - 2), num_q_steps):
        send_output(q_step, q_step % 2, wait=True)


def _tensor_core_attention(
    q_latent: jax.Array,
    q_rope: jax.Array,
    gathered_cache: jax.Array,
    selected_counts: jax.Array,
    *,
    sm_scale: float,
    bq: int,
    b_topk: int,
) -> jax.Array:
    """Run the TensorCore online-softmax stage for one SparseCore microbatch."""
    batch_size, num_heads, _ = q_latent.shape
    latent_dim = q_latent.shape[-1]
    rope_dim = q_rope.shape[-1]
    q_dim = latent_dim + rope_dim
    topk = gathered_cache.shape[1]
    num_q_steps = batch_size // bq
    num_k_steps = topk // b_topk
    padded_latent_dim = _align_to(latent_dim, 128)
    padded_q_dim = _align_to(q_dim, 128)
    q = jnp.concatenate(
        (
            q_latent,
            q_rope,
            jnp.zeros(
                (*q_latent.shape[:-1], padded_q_dim - q_dim),
                dtype=q_latent.dtype,
            ),
        ),
        axis=-1,
    )

    padded_output = pl.pallas_call(
        functools.partial(
            _dsa_tensor_core_kernel,
            num_q_steps=num_q_steps,
            num_k_steps=num_k_steps,
            bq=bq,
            b_topk=b_topk,
            padded_latent_dim=padded_latent_dim,
            padded_q_dim=padded_q_dim,
        ),
        out_shape=jax.ShapeDtypeStruct(
            (batch_size, num_heads, padded_latent_dim),
            q_latent.dtype,
        ),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, bq, num_heads, padded_q_dim), q_latent.dtype),
                pltpu.VMEM(
                    (2, bq, b_topk, gathered_cache.shape[-1]),
                    gathered_cache.dtype,
                ),
                pltpu.VMEM(
                    (2, bq, num_heads, padded_latent_dim),
                    q_latent.dtype,
                ),
                pltpu.SemaphoreType.DMA((3, 2)),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, 128), jnp.float32),
                pltpu.VMEM((bq, num_heads, padded_latent_dim), jnp.float32),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(),
        ),
        name="dsa_tensor_core_attention",
    )(
        jnp.asarray(sm_scale, dtype=jnp.float32),
        selected_counts,
        q,
        gathered_cache,
    )
    return padded_output[..., :latent_dim]


@functools.partial(
    jax.jit,
    static_argnames=("bq_sparse", "bq", "b_topk"),
)
def sparse_core_tensor_core_dsa(
    q_latent: jax.Array,
    q_rope: jax.Array,
    cache: jax.Array,
    physical_slots: jax.Array,
    selected_counts: jax.Array,
    sm_scale: jax.Array | float,
    *,
    bq_sparse: int = 128,
    bq: int = 32,
    b_topk: int = 128,
) -> jax.Array:
    """Compute sparse DSA attention with overlapped SparseCore and TensorCore work.

    Args:
      q_latent: BF16 latent queries, shape ``[Q, H, V]``. ``Q`` is the
        number of query tokens and ``H`` is the number of query heads.
      q_rope: Rotary query components, shape ``[Q, H, R]`` and the same dtype.
      cache: Physical-slot cache, shape ``[S, C]`` and the same dtype. ``S`` is
        the number of directly addressable cache rows. The first ``V`` elements
        of each row are both latent key and output value, the following ``R`` are
        the rotary key, and any remainder is padding.
      physical_slots: INT32 cache-row indices in ``[0, S)``, shape ``[Q, K]``.
        Only entries before the corresponding ``selected_counts`` value are read.
      selected_counts: INT32 valid top-k counts, shape ``[Q]``. Values are
        clamped to ``[0, K]``; a zero count produces an all-zero output row.
      sm_scale: FP32 scalar applied to attention logits.
      bq_sparse: Query microbatch consumed by each SparseCore gather.
      bq: TensorCore query tile. Must divide ``bq_sparse``.
      b_topk: TensorCore online-softmax tile. Must divide ``K``.

    Returns:
      Sparse-attention output with shape ``[Q, H, V]`` and query dtype.

    The outer ``lax.fori_loop`` carries one gathered microbatch. In each
    iteration SparseCore gathers the next cache microbatch while TensorCore
    consumes the pending one, so at most two gathered buffers are live. All
    launch parameters are static, making this API safe to call either directly
    or from another ``jax.jit``-compiled function.
    """
    if q_latent.dtype != jnp.bfloat16:
        raise TypeError("q_latent must have dtype bfloat16.")
    if q_rope.dtype != q_latent.dtype or cache.dtype != q_latent.dtype:
        raise TypeError("q_latent, q_rope, and cache must have the same dtype.")
    if physical_slots.dtype != jnp.int32 or selected_counts.dtype != jnp.int32:
        raise TypeError("physical_slots and selected_counts must have dtype int32.")
    if bq_sparse <= 0 or bq <= 0 or b_topk <= 0:
        raise ValueError("bq_sparse, bq, and b_topk must be positive.")

    num_queries, _, latent_dim = q_latent.shape
    rope_dim = q_rope.shape[-1]
    cache_dim = cache.shape[-1]
    topk = physical_slots.shape[1]
    if cache_dim < latent_dim + rope_dim:
        raise ValueError(
            f"cache dimension C={cache_dim} must be at least V + R = {latent_dim + rope_dim}."
        )
    if topk > 2048:
        raise ValueError(f"K must be at most 2048, got {topk}.")
    if num_queries % bq_sparse:
        raise ValueError(f"Q={num_queries} must be divisible by bq_sparse={bq_sparse}.")
    if bq_sparse % bq:
        raise ValueError(f"bq_sparse={bq_sparse} must be divisible by bq={bq}.")
    if topk % b_topk:
        raise ValueError(f"K={topk} must be divisible by b_topk={b_topk}.")

    info = _sparse_core_info()
    gather_size = bq_sparse * topk
    gather_multiple = _SC_GATHER_WINDOW * info.num_cores * info.num_subcores
    if gather_size % gather_multiple:
        raise ValueError(
            f"bq_sparse * K must be divisible by {gather_multiple}, got {gather_size}."
        )

    num_microbatches = num_queries // bq_sparse
    clipped_counts = jnp.clip(selected_counts, 0, topk)
    valid_slots = jnp.arange(topk, dtype=jnp.int32)[None, :] < clipped_counts[:, None]
    safe_slots = jnp.where(valid_slots, physical_slots, jnp.int32(0))
    safe_slot_batches = safe_slots.reshape(num_microbatches, bq_sparse, topk)
    count_batches = clipped_counts.reshape(num_microbatches, bq_sparse)
    q_latent_batches = q_latent.reshape(
        num_microbatches,
        bq_sparse,
        q_latent.shape[1],
        latent_dim,
    )
    q_rope_batches = q_rope.reshape(
        num_microbatches,
        bq_sparse,
        q_rope.shape[1],
        rope_dim,
    )

    def gather(batch_id):
        return _gather_cache_microbatch(
            cache,
            lax.dynamic_index_in_dim(safe_slot_batches, batch_id, keepdims=False),
        )

    cache_buffer_0 = gather(0)
    cache_buffer_1 = jnp.empty_like(cache_buffer_0)
    output = jnp.empty_like(q_latent)

    def pipeline_step(batch_id, carry):
        def run_step(current_cache, standby_cache, output_buffer):
            del standby_cache
            next_cache = gather(batch_id + 1)
            current_output = _tensor_core_attention(
                lax.dynamic_index_in_dim(q_latent_batches, batch_id, keepdims=False),
                lax.dynamic_index_in_dim(q_rope_batches, batch_id, keepdims=False),
                current_cache,
                lax.dynamic_index_in_dim(count_batches, batch_id, keepdims=False),
                sm_scale=sm_scale,
                bq=bq,
                b_topk=b_topk,
            )
            output_buffer = lax.dynamic_update_slice_in_dim(
                output_buffer,
                current_output,
                batch_id * bq_sparse,
                axis=0,
            )
            return next_cache, output_buffer

        def even_step(buffers):
            buffer_0, buffer_1, output_buffer = buffers
            next_cache, output_buffer = run_step(
                buffer_0,
                buffer_1,
                output_buffer,
            )
            return buffer_0, next_cache, output_buffer

        def odd_step(buffers):
            buffer_0, buffer_1, output_buffer = buffers
            next_cache, output_buffer = run_step(
                buffer_1,
                buffer_0,
                output_buffer,
            )
            return next_cache, buffer_1, output_buffer

        return lax.cond(
            lax.bitwise_and(batch_id, 1) == 0,
            even_step,
            odd_step,
            carry,
        )

    cache_buffer_0, cache_buffer_1, output = lax.fori_loop(
        0,
        num_microbatches - 1,
        pipeline_step,
        (cache_buffer_0, cache_buffer_1, output),
    )
    final_batch = num_microbatches - 1
    final_cache = cache_buffer_0 if final_batch % 2 == 0 else cache_buffer_1
    final_output = _tensor_core_attention(
        lax.dynamic_index_in_dim(q_latent_batches, final_batch, keepdims=False),
        lax.dynamic_index_in_dim(q_rope_batches, final_batch, keepdims=False),
        final_cache,
        lax.dynamic_index_in_dim(count_batches, final_batch, keepdims=False),
        sm_scale=sm_scale,
        bq=bq,
        b_topk=b_topk,
    )
    return lax.dynamic_update_slice_in_dim(
        output,
        final_output,
        final_batch * bq_sparse,
        axis=0,
    )
