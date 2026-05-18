"""Fused DECODE Pallas kernel for the GLA recurrent attention.

Replaces the JAX ``fused_recurrent_simple_gla`` scan + the explicit JAX
gather/scatter shard_maps around the recurrent state buffer with a single
Pallas kernel that does in-kernel async DMA gather/scatter.

Design:
  * Single Pallas program per (k_block, v_block); the token loop is
    inside the kernel so the output block can be the full ``(N, H, BV)``
    tensor (avoids the "two programs writing the same output block"
    constraint that would force per-head Python serialization).
  * Token-level async double-buffer: 2 VMEM banks (``h_in_buf``,
    ``h_out_buf``) and 2 DMA semaphores each. Each DMA carries all H
    heads of one token in a single contiguous ``(H, BK, BV)`` chunk.
  * Per-head 2D compute inside the per-token step. Mosaic TPU does not
    currently support 1D→3D vector layout casts (``vector<H>`` →
    ``vector<H×1×1>`` is rejected by ``infer-vector-layout``), so the
    H-vectorised compute body cannot be expressed cleanly; per-head 2D
    tiles avoid the issue while keeping the DMA H-batched.
  * Default-precision dot, matching the JAX reference ``_scan_segment``
    body.
  * Conditional logic: gather is always issued (slot 0 is dummy zeros
    by convention); compute is masked with ``(has_init AND pool_idx != 0)``;
    scatter is always issued but the staged value is masked to 0 when
    ``pool_idx == 0`` (preserves dummy-slot-is-zero invariant). Keeps
    semaphore counts balanced across all branches.
  * ``disable_semaphore_checks=True`` (pattern from RPA v3) skips the
    Mosaic kernel-exit semaphore-balance assertion; we still drain the
    last two scatters inside the kernel for HBM coherence vs downstream
    readers of the aliased buffer.
"""

from __future__ import annotations

import functools
import inspect as _inspect

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.tpu as pltpu
import jax.numpy as jnp

from sgl_jax.srt.kernels.simple_gla.simple_gla import get_interpret

_COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS = (
    "disable_semaphore_checks" in _inspect.signature(pltpu.CompilerParams).parameters
)


def _semaphore_kwargs(disable_semaphore_checks: bool) -> dict:
    """Forward `disable_semaphore_checks` only if the running jaxlib supports it."""
    if _COMPILER_PARAMS_SUPPORTS_SEMAPHORE_CHECKS:
        return {"disable_semaphore_checks": disable_semaphore_checks}
    return {}


def _decode_simple_gla_kernel(
    # BlockSpec inputs (full-tensor blocks; grid is (1, 1) outside)
    q_ref,  # (N, H, BK)
    k_ref,  # (N, H, BK)
    v_ref,  # (N, H, BV)
    # SMEM inputs
    g_gamma_ref,  # [H]
    recurrent_indices_ref,  # [N]
    has_initial_state_ref,  # [N]
    # ANY-memory inputs (HBM-resident, kernel manages DMA)
    recurrent_buffer_ref,  # [total_slots*H, K, V] — pre-flattened
    # Outputs
    o_ref,  # (N, H, BV)
    updated_recurrent_buffer_ref,  # [total_slots*H, K, V] — aliased
    # Scratch
    h_in_buf,  # VMEM (2, H, BK, BV) buffer.dtype
    h_out_buf,  # VMEM (2, H, BK, BV) buffer.dtype
    sem_gather,  # SemaphoreType.DMA((2,))
    sem_scatter,  # SemaphoreType.DMA((2,))
    *,
    BK: int,
    BV: int,
    scale: float,
    N: int,
    H: int,
):
    """Single Pallas program per (k_block, v_block); ALL tokens + ALL heads inside.

    Pipeline (token-level async double-buffer):
        prologue:
            start gather(token 0) → h_in_buf[0]   via sem_gather[0]
            if N >= 2: start gather(token 1) → h_in_buf[1] via sem_gather[1]

        for n in range(N):
            bank = n % 2
            wait gather(token n) on sem_gather[bank]
            for each head h in 0..H-1:
                materialise h_in_buf[bank, h] as fp32 scratch (masked by
                    has_init AND pool_idx != 0)
                h_new = h_old * exp(g_gamma[h]) + outer(k, v)
                o_ref[n, h, :] = sum(q * h_new, axis=K) * scale
                stage h_to_scatter (masked to 0 when pool_idx == 0) into
                    h_out_buf[bank, h]
            if n >= 2: wait scatter(token n-2) on sem_scatter[bank]
            start scatter(token n) → buf  via sem_scatter[bank]
            if n+2 < N: start gather(token n+2) → h_in_buf[bank]

        drain:
            wait scatter(N-1) on sem_scatter[(N-1)%2]
            if N >= 2: wait scatter(N-2) on sem_scatter[(N-2)%2]

    Notes:
      * ``recurrent_buffer_ref`` is pre-flattened to (total_slots*H, K, V)
        by the launcher (workaround for a JAX 0.8.x interpret-mode
        RefReshaper bug). Slot indexing uses ``pl.ds(pool_idx * H, H)`` to
        pull all H consecutive heads for one token in a single DMA.
      * The ``h_to_scatter`` mask protects the dummy slot 0 invariant:
        when pool_idx[n] == 0, we still scatter (keeping sem balance),
        but the scattered value is zeros, so slot 0 stays zero-valued.
    """
    i_k = pl.program_id(0)
    i_v = pl.program_id(1)

    def _buf_in_slice(n_token: int):
        pool_idx = recurrent_indices_ref[n_token]
        return recurrent_buffer_ref.at[
            pl.ds(pool_idx * H, H),
            pl.ds(i_k * BK, BK),
            pl.ds(i_v * BV, BV),
        ]

    def _buf_out_slice(n_token: int):
        pool_idx = recurrent_indices_ref[n_token]
        return updated_recurrent_buffer_ref.at[
            pl.ds(pool_idx * H, H),
            pl.ds(i_k * BK, BK),
            pl.ds(i_v * BV, BV),
        ]

    # ───── Prologue: kick off gathers for tokens 0 and 1 ─────
    pltpu.make_async_copy(_buf_in_slice(0), h_in_buf.at[0], sem_gather.at[0]).start()
    if N >= 2:
        pltpu.make_async_copy(_buf_in_slice(1), h_in_buf.at[1], sem_gather.at[1]).start()

    # Per-head decay scalars are loaded inside the per-head loop below.
    # Mosaic TPU's infer-vector-layout rejects 1D→3D reshapes
    # (``vector<H>`` → ``vector<H×1×1>``), so we can't broadcast a (H,)
    # decay vector against a (H, BK, BV) scratch. Per-head 2D tiles avoid
    # the issue; the DMA stays H-batched.

    # ───── Steady-state ping-pong loop over tokens ─────
    for n in range(N):
        bank = n % 2
        pool_idx = recurrent_indices_ref[n]
        has_init = has_initial_state_ref[n]
        # pool_idx == 0 means dummy slot ⇒ no prior state.
        # has_init == False means new sequence ⇒ no prior state.
        use_state = jnp.logical_and(has_init, pool_idx != 0)
        scatter_mask = pool_idx != 0

        # 1) Wait for this token's gather to land.
        pltpu.make_async_copy(_buf_in_slice(n), h_in_buf.at[bank], sem_gather.at[bank]).wait()

        # 2) Per-head 2D compute.
        for h in range(H):
            decay_h = jnp.exp(g_gamma_ref[h].astype(jnp.float32))  # scalar

            gathered_h = h_in_buf.at[bank, h][...].astype(jnp.float32)  # (BK, BV)
            h_old_h = jnp.where(use_state, gathered_h, 0.0)

            q_h = q_ref[n, h, :].astype(jnp.float32)  # (BK,)
            k_h = k_ref[n, h, :].astype(jnp.float32)  # (BK,)
            v_h = v_ref[n, h, :].astype(jnp.float32)  # (BV,)

            h_new_h = h_old_h * decay_h + k_h[:, None] * v_h[None, :]  # (BK, BV)

            # Dot mirrors ``_scan_segment``'s default-precision sum.
            o_h = jnp.sum(q_h[:, None] * h_new_h, axis=0) * scale  # (BV,)
            o_ref[n, h, :] = o_h.astype(o_ref.dtype)

            # Mask to zeros when pool_idx == 0 so writing back to dummy
            # slot 0 leaves it unchanged.
            h_to_scatter_h = jnp.where(scatter_mask, h_new_h, 0.0)
            h_out_buf.at[bank, h][...] = h_to_scatter_h.astype(h_out_buf.dtype)

        # 4) Wait for prior scatter on this bank (token n-2) to drain.
        if n >= 2:
            pltpu.make_async_copy(
                h_out_buf.at[bank], _buf_out_slice(n - 2), sem_scatter.at[bank]
            ).wait()

        # 5) Start scatter for this token (H heads at once — single DMA).
        pltpu.make_async_copy(h_out_buf.at[bank], _buf_out_slice(n), sem_scatter.at[bank]).start()

        # 6) Pre-issue gather for token (n + 2) using h_in_buf[bank]
        #    (consumed by step 2 above).
        if n + 2 < N:
            pltpu.make_async_copy(
                _buf_in_slice(n + 2), h_in_buf.at[bank], sem_gather.at[bank]
            ).start()

    # ───── Drain: wait for the two trailing scatters ─────
    last1_bank = (N - 1) % 2
    pltpu.make_async_copy(
        h_out_buf.at[last1_bank], _buf_out_slice(N - 1), sem_scatter.at[last1_bank]
    ).wait()
    if N >= 2:
        last2_bank = (N - 2) % 2
        pltpu.make_async_copy(
            h_out_buf.at[last2_bank], _buf_out_slice(N - 2), sem_scatter.at[last2_bank]
        ).wait()


@functools.partial(
    jax.jit,
    static_argnames=["scale"],
    donate_argnames=["recurrent_buffer"],
)
def _launch_decode_simple_gla(
    q: jax.Array,  # [N, H, K]
    k: jax.Array,  # [N, H, K]
    v: jax.Array,  # [N, H, V]
    g_gamma: jax.Array,  # [H]
    recurrent_buffer: jax.Array,  # [total_slots, H, K, V]
    recurrent_indices: jax.Array,  # [N]
    has_initial_state: jax.Array,  # [N]
    *,
    scale: float,
) -> tuple[jax.Array, jax.Array]:
    """Launch the DECODE fused Pallas kernel.

    Grid is (cdiv(K, BK), cdiv(V, BV)) — the token dim N is iterated
    inside the kernel. Block specs are full-tensor blocks; each program
    covers one (BK, BV) tile across ALL tokens and ALL heads.

    Notes:
      I-1: ``recurrent_buffer`` is pre-flattened to (total_slots*H, K, V)
           so the kernel can use ``pl.ds(pool_idx * H, H)`` to gather all
           H heads of one token in a single contiguous DMA.
      I-2: ``input_output_aliases={6: 1}`` — input position 6 is the flat
           recurrent buffer (after q, k, v, g_gamma, indices, has_init);
           output position 1 is the flat updated buffer.
      I-3: ``disable_semaphore_checks=True`` is required because the
           kernel uses raw ``make_async_copy`` for the async double-buffer
           pipeline. Pattern matches ``ragged_paged_attention_v3.py``.
    """
    interpret = get_interpret()

    N, H, K = q.shape
    V = v.shape[-1]
    BK = min(K, 128)
    BV = min(V, 128)
    assert K % BK == 0
    assert V % BV == 0

    # In-launcher pre-flatten — see I-1 above.
    buf_total_slots = recurrent_buffer.shape[0]
    buf_flat_shape = (buf_total_slots * H, K, V)
    recurrent_buffer_flat = jnp.reshape(recurrent_buffer, buf_flat_shape)

    # Grid: K and V blocking only. N is iterated inside the kernel so
    # the output block can be the full (N, H, BV) tensor.
    grid = (pl.cdiv(K, BK), pl.cdiv(V, BV))

    def q_index_map(k_i, _v_i):
        return 0, 0, k_i

    def k_index_map(k_i, _v_i):
        return 0, 0, k_i

    def v_index_map(_k_i, v_i):
        return 0, 0, v_i

    def o_index_map(_k_i, v_i):
        return 0, 0, v_i

    in_specs = [
        pl.BlockSpec((N, H, BK), q_index_map),
        pl.BlockSpec((N, H, BK), k_index_map),
        pl.BlockSpec((N, H, BV), v_index_map),
        pl.BlockSpec(memory_space=pltpu.SMEM),  # g_gamma
        pl.BlockSpec(memory_space=pltpu.SMEM),  # recurrent_indices
        pl.BlockSpec(memory_space=pltpu.SMEM),  # has_initial_state
        pl.BlockSpec(memory_space=pltpu.ANY),  # recurrent_buffer (HBM)
    ]
    out_specs = [
        pl.BlockSpec((N, H, BV), o_index_map),
        pl.BlockSpec(memory_space=pltpu.ANY),  # updated_recurrent_buffer
    ]
    out_shape = [
        jax.ShapeDtypeStruct((N, H, V), q.dtype),
        # IMPORTANT: input AND output pre-flattened to the same shape,
        # required by input_output_aliases.
        jax.ShapeDtypeStruct(buf_flat_shape, recurrent_buffer.dtype),
    ]

    scratch_shapes = [
        # Two banks for inbound gather staging (kept in buffer dtype).
        # Each bank holds (H, BK, BV) — one token's worth of all heads.
        pltpu.VMEM((2, H, BK, BV), recurrent_buffer.dtype),
        # Two banks for outbound scatter staging (kept in buffer dtype).
        pltpu.VMEM((2, H, BK, BV), recurrent_buffer.dtype),
        # 2-element DMA semaphore arrays for gather and scatter.
        pltpu.SemaphoreType.DMA((2,)),
        pltpu.SemaphoreType.DMA((2,)),
    ]

    kernel = functools.partial(_decode_simple_gla_kernel, BK=BK, BV=BV, scale=scale, N=N, H=H)

    o, updated_buffer_flat = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel"),
            vmem_limit_bytes=128 * 1024 * 1024,
            **(_semaphore_kwargs(True)),  # see launcher I-3
        ),
        input_output_aliases={6: 1},  # see launcher I-2
    )(
        q,
        k,
        v,
        g_gamma,
        recurrent_indices,
        has_initial_state,
        recurrent_buffer_flat,
    )

    # Restore the original (total_slots, H, K, V) view for the caller.
    updated_buffer = jnp.reshape(updated_buffer_flat, recurrent_buffer.shape)
    return o, updated_buffer


def decode_simple_gla_fused(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    recurrent_buffer: jax.Array,
    recurrent_indices: jax.Array,
    has_initial_state: jax.Array,
    *,
    g_gamma: jax.Array,
    scale: float | None = None,
) -> tuple[jax.Array, jax.Array]:
    """DECODE fused entry point. Returns (o, updated_recurrent_buffer)."""
    K = q.shape[-1]
    if scale is None:
        scale = K**-0.5
    return _launch_decode_simple_gla(
        q,
        k,
        v,
        g_gamma,
        recurrent_buffer,
        recurrent_indices,
        has_initial_state,
        scale=float(scale),
    )


__all__ = ["decode_simple_gla_fused"]
