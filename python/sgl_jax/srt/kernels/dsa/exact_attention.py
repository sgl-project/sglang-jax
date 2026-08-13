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


def _sparse_core_gather_cost_estimate(
    num_indices: int,
    value_dim: int,
    *,
    table_dtype: jnp.dtype,
    indices_dtype: jnp.dtype,
    output_dtype: jnp.dtype,
) -> pl.CostEstimate:
    """Estimate the HBM traffic for one SparseCore gather launch.

    The table is indexed indirectly, so only the selected rows are read.  A
    gather has no arithmetic work; the value bytes are counted once for the
    HBM read and once for the HBM write.
    """
    value_elements = num_indices * 2 * value_dim
    bytes_accessed = (
        num_indices * jnp.dtype(indices_dtype).itemsize
        + value_elements * jnp.dtype(table_dtype).itemsize
        + value_elements * jnp.dtype(output_dtype).itemsize
    )
    return pl.CostEstimate(
        flops=0,
        bytes_accessed=bytes_accessed,
        transcendentals=0,
    )


def _sparse_core_gather(
    table: jax.Array,
    indices: jax.Array,
    output_buffer: jax.Array | None = None,
    output_buffer_index: jax.Array | int | None = None,
) -> jax.Array:
    """Gather complete ``[2, D]`` BF16 cache pairs by slot index.

    If ``output_buffer`` is supplied, it is a two-slot pool and the SparseCore
    kernel writes into the selected slot through a stateful Ref. This keeps
    the pool in one fixed loop-state position.
    """
    if table.dtype != jnp.bfloat16 or table.ndim != 3 or table.shape[1] != 2:
        raise TypeError("SparseCore table must be a BF16 array with shape [S, 2, D].")
    if table.shape[-1] % 128:
        raise ValueError("SparseCore cache pair dimension D must be divisible by 128.")
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

            @functools.partial(
                pltpu.emit_pipeline,
                grid=(num_row_subchunks, 1),
                in_specs=pl.BlockSpec(
                    (pl.Indirect(row_subchunk_size), 1, value_dim),
                    lambda row_subchunk, _col: (
                        indices_vmem[
                            pl.ds(
                                row_subchunk * row_subchunk_size,
                                row_subchunk_size,
                            )
                        ],
                        0,
                        0,
                    ),
                ),
                out_specs=pl.BlockSpec(
                    (row_subchunk_size, 1, value_dim),
                    lambda row_subchunk, _col: (
                        row_chunk * num_row_subchunks + row_subchunk,
                        0,
                        0,
                    ),
                ),
            )
            def data_pipeline(gather_vmem, output_vmem):
                @plsc.parallel_loop(0, value_dim, step=32)
                def copy_columns(column):
                    output_vmem[
                        pl.ds(0, row_subchunk_size),
                        pl.ds(0, 1),
                        pl.ds(column, 32),
                    ] = gather_vmem[
                        pl.ds(0, row_subchunk_size),
                        pl.ds(0, 1),
                        pl.ds(column, 32),
                    ]

            data_pipeline(
                table_hbm.bitcast(jnp.int32),
                output_hbm.bitcast(jnp.int32),
            )

        index_pipeline(indices_hbm)

    output = jax.ShapeDtypeStruct(
        (num_indices, 2, value_dim),
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
    kwargs["cost_estimate"] = _sparse_core_gather_cost_estimate(
        num_indices,
        value_dim,
        table_dtype=table.dtype,
        indices_dtype=indices.dtype,
        output_dtype=output.dtype,
    )
    if output_buffer is None:
        if output_buffer_index is not None:
            raise ValueError("output_buffer_index requires output_buffer.")
        return pl.kernel(kernel, **kwargs)(table, indices)

    if (
        output_buffer.dtype != output.dtype
        or output_buffer.ndim < 2
        or output_buffer.shape[0] != 2
        or output_buffer.size != 2 * num_indices * 2 * value_dim
    ):
        raise TypeError(
            "SparseCore gather output pool must have leading dimension 2 and "
            f"two slots of size {output.shape}, got "
            f"{output_buffer.shape} and {output_buffer.dtype}."
        )
    if output_buffer_index is None:
        raise ValueError("output_buffer_index is required for an output pool.")

    compiler_params = kwargs["compiler_params"]
    cost_estimate = kwargs["cost_estimate"]

    def run_stateful_gather(output_slot):
        def stateful_gather(refs):
            table_ref, indices_ref, output_ref = refs

            def run_kernel():
                kernel(
                    table_ref,
                    indices_ref,
                    output_ref.at[output_slot].reshape(-1, 2, value_dim),
                )

            pl.core_map(
                mesh,
                compiler_params=compiler_params,
                cost_estimate=cost_estimate,
                name="dsa_sparse_core_gather_into",
            )(run_kernel)

        _, _, updated_buffer = pl.run_state(stateful_gather)(
            (table, indices, output_buffer)
        )
        return updated_buffer

    return lax.cond(
        output_buffer_index == 0,
        lambda _: run_stateful_gather(0),
        lambda _: run_stateful_gather(1),
        operand=None,
    )



def _gather_cache_microbatch_into(
    cache: jax.Array,
    safe_slots: jax.Array,
    output_pool: jax.Array,
    output_buffer_index: jax.Array | int,
) -> jax.Array:
    """Gather one cache microbatch into one slot of ``output_pool``."""
    return _sparse_core_gather(
        cache,
        safe_slots.reshape(-1),
        output_buffer=output_pool,
        output_buffer_index=output_buffer_index,
    )


def _dsa_tensor_core_kernel(
    sm_scale_ref,  # FP32 scalar, SMEM
    selected_counts_ref,  # INT32 [B], SMEM
    cache_buffer_index_ref,  # INT32 scalar, SMEM
    q_hbm_ref,  # [B, H, align(V + R, 128)], HBM
    cache_hbm_ref,  # [2, B, K, 2, D], HBM; first dim is the ping-pong pool
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
                    cache_buffer_index_ref[()],
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
            cache = cache_x2_ref[cache_buffer].reshape(
                (bq, b_topk, cache_x2_ref.shape[-2] * cache_x2_ref.shape[-1])
            )

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
    cache_buffers: jax.Array,
    selected_counts: jax.Array,
    cache_buffer_index: jax.Array,
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
    topk = cache_buffers.shape[2]
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

    # The cache is streamed once for every query tile.  Count the padded
    # dimensions because those are the dimensions consumed by the actual
    # TensorCore dot operations and DMA buffers.
    flops = 2 * batch_size * num_heads * topk * (
        padded_q_dim + padded_latent_dim
    )
    transcendentals = batch_size * num_heads * (num_k_steps + topk)
    bytes_accessed = (
        q.size * q.dtype.itemsize
        + (num_q_steps * cache_buffers.size // 2) * cache_buffers.dtype.itemsize
        + batch_size * num_heads * padded_latent_dim * q_latent.dtype.itemsize
        + jnp.dtype(jnp.float32).itemsize
        + selected_counts.size * selected_counts.dtype.itemsize
    )
    cost_estimate = pl.CostEstimate(
        flops=flops,
        bytes_accessed=bytes_accessed,
        transcendentals=transcendentals,
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
            num_scalar_prefetch=3,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((2, bq, num_heads, padded_q_dim), q_latent.dtype),
                pltpu.VMEM(
                    (2, bq, b_topk, 2, cache_buffers.shape[-1]),
                    cache_buffers.dtype,
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
        cost_estimate=cost_estimate,
        name="dsa_tensor_core_attention",
    )(
        jnp.asarray(sm_scale, dtype=jnp.float32),
        selected_counts,
        cache_buffer_index,
        q,
        cache_buffers,
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
    bq: int = 16,
    b_topk: int = 128,
) -> jax.Array:
    """Compute sparse DSA attention with overlapped SparseCore and TensorCore work.

    Args:
      q_latent: BF16 latent queries, shape ``[Q, H, V]``. ``Q`` is the
        number of query tokens and ``H`` is the number of query heads.
      q_rope: Rotary query components, shape ``[Q, H, R]`` and the same dtype.
      cache: Paired physical-slot cache with shape ``[S, 2, D]`` and BF16
        dtype. ``cache[s].reshape(2 * D)`` is one logical cache row.
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

    The outer ``lax.fori_loop`` carries one cache pool whose leading dimension
    is two. The low bit of ``batch_id`` selects the current slice and the
    opposite slice is used as the gather destination. The pool itself never
    changes position in the loop state. All launch parameters are static,
    making this API safe to call either directly or from another
    ``jax.jit``-compiled function.
    """
    if q_latent.dtype != jnp.bfloat16:
        raise TypeError("q_latent must have dtype bfloat16.")
    if q_rope.dtype != q_latent.dtype or cache.dtype != q_latent.dtype:
        raise TypeError("q_latent, q_rope, and cache must have the same dtype.")
    if cache.ndim != 3 or cache.shape[1] != 2:
        raise ValueError(f"cache must have shape [S, 2, D], got {cache.shape}.")
    if cache.shape[-1] % 128:
        raise ValueError("cache pair dimension D must be divisible by 128.")
    if physical_slots.dtype != jnp.int32 or selected_counts.dtype != jnp.int32:
        raise TypeError("physical_slots and selected_counts must have dtype int32.")
    if bq_sparse <= 0 or bq <= 0 or b_topk <= 0:
        raise ValueError("bq_sparse, bq, and b_topk must be positive.")

    num_queries, _, latent_dim = q_latent.shape
    rope_dim = q_rope.shape[-1]
    cache_dim = cache.shape[1] * cache.shape[2]
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

    def gather_into(batch_id, output_pool, output_buffer_index):
        return _gather_cache_microbatch_into(
            cache,
            lax.dynamic_index_in_dim(safe_slot_batches, batch_id, keepdims=False),
            output_pool,
            output_buffer_index,
        )

    cache_buffers = jnp.empty(
        (2, bq_sparse, topk, 2, cache.shape[-1]),
        dtype=cache.dtype,
    )
    cache_buffers = gather_into(0, cache_buffers, jnp.int32(0))
    output = jnp.empty_like(q_latent)

    def pipeline_step(batch_id, carry):
        cache_buffers, output_buffer = carry
        current_buffer_index = lax.bitwise_and(batch_id, jnp.int32(1))
        next_buffer_index = lax.bitwise_xor(current_buffer_index, jnp.int32(1))
        cache_buffers = gather_into(
            batch_id + 1,
            cache_buffers,
            next_buffer_index,
        )
        current_output = _tensor_core_attention(
            lax.dynamic_index_in_dim(q_latent_batches, batch_id, keepdims=False),
            lax.dynamic_index_in_dim(q_rope_batches, batch_id, keepdims=False),
            cache_buffers,
            lax.dynamic_index_in_dim(count_batches, batch_id, keepdims=False),
            current_buffer_index,
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
        return cache_buffers, output_buffer

    cache_buffers, output = lax.fori_loop(
        0,
        num_microbatches - 1,
        pipeline_step,
        (cache_buffers, output),
    )
    final_batch = num_microbatches - 1
    final_output = _tensor_core_attention(
        lax.dynamic_index_in_dim(q_latent_batches, final_batch, keepdims=False),
        lax.dynamic_index_in_dim(q_rope_batches, final_batch, keepdims=False),
        cache_buffers,
        lax.dynamic_index_in_dim(count_batches, final_batch, keepdims=False),
        jnp.int32(final_batch & 1),
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
