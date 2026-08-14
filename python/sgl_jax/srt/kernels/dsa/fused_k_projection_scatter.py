"""Page-aligned GLM DSA index-key projection fused with paged-cache writes.

The kernel is specialized for page-aligned Extend batches.  It keeps the
narrow index-key projection weights resident in VMEM, streams hidden-state
pages through one persistent program, applies the index-key epilogue, and
writes BF16 pages directly to their physical index-cache locations.

Q and index-weight projection deliberately remain on the native XLA path.
Decode and ragged/page-unaligned Extend use the general projection/scatter
fallback in ``dsa_indexer_ops``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _fused_k_projection_scatter_kernel(
    target_pages_ref,  # i32[num_token_pages], SMEM
    hidden_hbm_ref,  # bf16[T, hidden_dim], HBM
    wk_vmem_ref,  # bf16[hidden_dim, 128], VMEM
    norm_weight_vmem_ref,  # bf16[128], VMEM
    norm_bias_vmem_ref,  # bf16[128], VMEM
    rope_cos_vmem_ref,  # bf16[T, 32], VMEM
    rope_sin_vmem_ref,  # bf16[T, 32], VMEM
    hadamard_vmem_ref,  # f32[128, 128], VMEM
    _cache_hbm_ref,  # bf16[P, page_size, 128], HBM (aliased input)
    updated_cache_hbm_ref,  # aliased output
    hidden_x2_vmem_ref,  # bf16[2, page_size, hidden_dim]
    cache_page_x2_vmem_ref,  # bf16[2, page_size, 128]
    hidden_dma_sems,  # DMA semaphore[2]
    cache_dma_sems,  # DMA semaphore[2]
    *,
    num_token_pages: int,
    page_size: int,
):
    """Project one token page at a time and asynchronously write cache pages."""

    def fetch_hidden(page_id, bank, *, wait: bool):
        copy = pltpu.make_async_copy(
            hidden_hbm_ref.at[pl.ds(page_id * page_size, page_size)],
            hidden_x2_vmem_ref.at[bank],
            hidden_dma_sems.at[bank],
        )
        if wait:
            copy.wait()
        else:
            copy.start()

    def write_cache(page_id, bank, *, wait: bool):
        physical_page = target_pages_ref[page_id]
        copy = pltpu.make_async_copy(
            cache_page_x2_vmem_ref.at[bank],
            updated_cache_hbm_ref.at[physical_page],
            cache_dma_sems.at[bank],
        )
        if wait:
            copy.wait()
        else:
            copy.start()

    def project_page(page_id, bank):
        hidden = hidden_x2_vmem_ref[bank]
        key = lax.dot_general(
            hidden,
            wk_vmem_ref[...],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        ).astype(jnp.bfloat16)

        mean = jnp.mean(key, axis=-1, keepdims=True)
        centered = key - mean
        variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
        key = centered * lax.rsqrt(variance + jnp.asarray(1e-5, key.dtype))
        key = key * norm_weight_vmem_ref[...] + norm_bias_vmem_ref[...]

        token_start = page_id * page_size
        cos = rope_cos_vmem_ref[pl.ds(token_start, page_size)]
        sin = rope_sin_vmem_ref[pl.ds(token_start, page_size)]
        key_rope = key[:, :64]
        # Mosaic rejects both the strided ``[:, ::2]`` gather and a
        # [page, 64] -> [page, 32, 2] shape cast, so expand the GPT-J pairs at
        # compile time into unit-width slices.
        rotated_columns = []
        for pair_id in range(32):
            even = lax.slice_in_dim(key_rope, 2 * pair_id, 2 * pair_id + 1, axis=1)
            odd = lax.slice_in_dim(key_rope, 2 * pair_id + 1, 2 * pair_id + 2, axis=1)
            cos_i = lax.slice_in_dim(cos, pair_id, pair_id + 1, axis=1)
            sin_i = lax.slice_in_dim(sin, pair_id, pair_id + 1, axis=1)
            rotated_columns.extend((even * cos_i - odd * sin_i, odd * cos_i + even * sin_i))
        rotated = jnp.concatenate(rotated_columns, axis=1)
        key = jnp.concatenate((rotated, key[:, 64:]), axis=-1)

        # Match the existing dense FP32 Hadamard einsum before the cache cast.
        key = lax.dot_general(
            key.astype(jnp.float32),
            hadamard_vmem_ref[...],
            dimension_numbers=(((1,), (0,)), ((), ())),
            preferred_element_type=jnp.float32,
        )
        cache_page_x2_vmem_ref[bank] = key.astype(updated_cache_hbm_ref.dtype)

    fetch_hidden(0, 0, wait=False)

    def page_body(page_id, _):
        bank = page_id & 1
        fetch_hidden(page_id, bank, wait=True)

        @pl.when(page_id + 1 < num_token_pages)
        def prefetch_next_hidden():
            fetch_hidden(page_id + 1, 1 - bank, wait=False)

        @pl.when(page_id >= 2)
        def wait_old_cache_write():
            write_cache(page_id - 2, bank, wait=True)

        project_page(page_id, bank)
        write_cache(page_id, bank, wait=False)
        return None

    lax.fori_loop(0, num_token_pages, page_body, None, unroll=False)

    @pl.when(num_token_pages >= 2)
    def drain_penultimate():
        page_id = num_token_pages - 2
        write_cache(page_id, page_id & 1, wait=True)

    page_id = num_token_pages - 1
    write_cache(page_id, page_id & 1, wait=True)


@functools.partial(jax.jit, static_argnames=("page_size", "interpret"))
def fused_k_projection_scatter_pallas(
    hidden_states: jax.Array,
    wk: jax.Array,
    norm_weight: jax.Array,
    norm_bias: jax.Array,
    rope_cos: jax.Array,
    rope_sin: jax.Array,
    hadamard: jax.Array,
    index_key_cache: jax.Array,
    target_pages: jax.Array,
    *,
    page_size: int = 64,
    interpret: bool = False,
) -> jax.Array:
    """Project and scatter one or more complete token pages."""

    if hidden_states.ndim != 2:
        raise ValueError(f"hidden_states must be rank 2, got {hidden_states.shape}")
    num_tokens, hidden_dim = hidden_states.shape
    if num_tokens < page_size or num_tokens % page_size:
        raise ValueError(
            f"num_tokens={num_tokens} must be a positive multiple of page_size={page_size}"
        )
    if index_key_cache.ndim != 3 or index_key_cache.shape[1:] != (page_size, 128):
        raise ValueError(
            "index_key_cache must have shape [num_pages, page_size, 128], "
            f"got {index_key_cache.shape}"
        )
    if wk.shape != (hidden_dim, 128):
        raise ValueError(f"wk must have shape {(hidden_dim, 128)}, got {wk.shape}")
    if norm_weight.shape != (128,) or norm_bias.shape != (128,):
        raise ValueError("index-key norm parameters must both have shape [128]")
    if rope_cos.shape != (num_tokens, 32) or rope_sin.shape != (num_tokens, 32):
        raise ValueError("rope_cos and rope_sin must both have shape [T, 32]")
    if hadamard.shape != (128, 128) or hadamard.dtype != jnp.float32:
        raise ValueError("hadamard must be float32[128, 128]")

    num_token_pages = num_tokens // page_size
    if target_pages.shape != (num_token_pages,) or target_pages.dtype != jnp.int32:
        raise ValueError(
            f"target_pages must be int32[{num_token_pages}], got {target_pages.shape} "
            f"and {target_pages.dtype}"
        )

    inputs = (
        hidden_states,
        wk,
        norm_weight,
        norm_bias,
        rope_cos,
        rope_sin,
        hadamard,
        index_key_cache,
    )
    in_specs = (
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(wk.shape, lambda *_: (0, 0)),
        pl.BlockSpec(norm_weight.shape, lambda *_: (0,)),
        pl.BlockSpec(norm_bias.shape, lambda *_: (0,)),
        pl.BlockSpec(rope_cos.shape, lambda *_: (0, 0)),
        pl.BlockSpec(rope_sin.shape, lambda *_: (0, 0)),
        pl.BlockSpec(hadamard.shape, lambda *_: (0, 0)),
        pl.BlockSpec(memory_space=pltpu.HBM),
    )
    cache_input_index = len(inputs)
    return pl.pallas_call(
        functools.partial(
            _fused_k_projection_scatter_kernel,
            num_token_pages=num_token_pages,
            page_size=page_size,
        ),
        out_shape=jax.ShapeDtypeStruct(index_key_cache.shape, index_key_cache.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=in_specs,
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(
                pltpu.VMEM((2, page_size, hidden_dim), hidden_states.dtype),
                pltpu.VMEM((2, page_size, 128), index_key_cache.dtype),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SemaphoreType.DMA((2,)),
            ),
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            vmem_limit_bytes=128 * 1024 * 1024,
            disable_bounds_checks=True,
        ),
        input_output_aliases={cache_input_index: 0},
        interpret=interpret,
        name="dsa_fused_k_projection_scatter_aligned",
    )(target_pages, *inputs)


__all__ = ["fused_k_projection_scatter_pallas"]
