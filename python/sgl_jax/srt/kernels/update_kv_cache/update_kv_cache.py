# Adapted from https://github.com/vllm-project/tpu-inference/releases/tag/v0.11.1
# Copyright 2025 The tpu-inference Authors. All rights reserved.
from functools import partial

import jax
import jax.numpy as jnp
from jax._src import dtypes
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils import cdiv


def get_slot_mapping(
    num_slices_per_block: int,
    kv_cache_start_loc: jax.Array,
    new_kv_start_loc: jax.Array,
    slice_lens: jax.Array,
):
    slot_mapping = jnp.stack([kv_cache_start_loc, new_kv_start_loc, slice_lens], axis=1)
    padded_size = (
        (slot_mapping.shape[0] + num_slices_per_block - 1)
        // num_slices_per_block
        * num_slices_per_block
    )
    slot_mapping = jnp.pad(
        slot_mapping,
        [[0, padded_size - slot_mapping.shape[0]], [0, 0]],
        constant_values=0,
    )
    slot_mapping = jnp.transpose(slot_mapping)
    return slot_mapping.astype(jnp.int32)


VMEM_SIZE = 64 * 1024 * 1024  # 32MB


def get_num_slices_per_block(new_kv: jax.Array, kv_cache: jax.Array, page_size=128):
    """
    new_kv: [total_num_token, num_combined_kv_heads, head_dim]
    kv_cache: [max_num_tokens, num_combined_kv_heads, head_dim]
    """
    assert (
        new_kv.dtype == kv_cache.dtype
    ), f"new_kv.dtype={new_kv.dtype} is not equal to kv_cache.dtype={kv_cache.dtype}"
    assert new_kv.dtype != jnp.float16, f"new_kv.dtype={new_kv.dtype} is not supported"

    bits = dtypes.itemsize_bits(kv_cache.dtype)
    assert bits % 8 == 0, f"bits={bits} is not divisible by 8"

    bytes_per_element = bits // 8

    total_num_token = new_kv.shape[0]
    kv_head_num = new_kv.shape[1]
    head_dim = new_kv.shape[2]

    max_num_slices_per_block = VMEM_SIZE // (bytes_per_element * page_size * kv_head_num * head_dim)
    assert (
        max_num_slices_per_block > 0
    ), f"max_num_slices_per_block={max_num_slices_per_block} is not greater than 0"

    return (
        total_num_token if total_num_token < max_num_slices_per_block else max_num_slices_per_block
    )


def kv_cache_update_kernel(
    # Prefetch
    # [3, padded_num_slices], list of (kv_cache_start, new_kv_start, slice_len)
    slices_ref,
    # Input
    new_kv_hbm_ref,  # [num_tokens, num_combined_kv_heads, head_dim]
    kv_cache_hbm_ref,  # [total_num_pages * page_size, num_combined_kv_heads,
    # head_dim]
    # Output
    _,  # [total_num_pages * page_size, num_combined_kv_heads, head_dim]
    # Scratch
    scratch,  # [num_slices_per_block, page_size, num_combined_kv_heads,
    # head_dim]
    sem,
):
    async_copies = []
    block_idx = pl.program_id(0)
    num_slices_per_block = scratch.shape[0]
    # Copy from new_kv_hbm_ref to scratch
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        new_kv_start = slices_ref[1, offset_i]
        length = slices_ref[2, offset_i]
        async_copy = pltpu.make_async_copy(
            new_kv_hbm_ref.at[pl.ds(new_kv_start, length), ...],
            scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)

    for async_copy in async_copies:
        async_copy.wait()

    # Copy from scratch to kv_cache_hbm_ref
    async_copies.clear()
    for i in range(num_slices_per_block):
        offset_i = i + block_idx * num_slices_per_block
        kv_cache_start = slices_ref[0, offset_i]
        length = slices_ref[2, offset_i]
        async_copy = pltpu.make_async_copy(
            scratch.at[jnp.uint32(i), pl.ds(0, length), ...],
            kv_cache_hbm_ref.at[pl.ds(kv_cache_start, length), ...],
            sem,
        )
        async_copy.start()
        async_copies.append(async_copy)
    for async_copy in async_copies:
        async_copy.wait()


@partial(
    jax.jit,
    static_argnames=["page_size", "num_slices_per_block", "kv_partition_axis"],
)
def kv_cache_update(
    new_kv: jax.Array,  # [total_num_token, num_kv_heads, head_dim]
    # [3, slices], list of (kv_cache_start, new_kv_start, slice_len)
    slices: jax.Array,
    # [max_num_tokens, num_kv_heads, head_dim]
    kv_cache: jax.Array,
    num_kv_update_slices: jax.Array,  # [1]
    *,
    page_size: int = 1,  # because we treat each token as an independent query
    num_slices_per_block: int = 8,
    kv_partition_axis: str = "tensor",
):
    @jax.shard_map(
        in_specs=(
            # new_kv - consistent with KV cache sharding
            P(None, kv_partition_axis, None),
            P(None, None),  # slices
            # kv_cache - consistent with KV cache sharding
            P(None, kv_partition_axis, None),
            P(None),  # num_kv_update_slices
        ),
        out_specs=P(
            None, kv_partition_axis, None
        ),  # output also maintains KV cache sharding consistency
        check_vma=False,
    )
    def _kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices):
        assert (
            slices.shape[1] % num_slices_per_block == 0
        ), f"slices.shape[1]={slices.shape[1]} is not divisible by num_slices_per_block={num_slices_per_block}"
        _, num_combined_kv_heads, head_dim = new_kv.shape

        assert num_combined_kv_heads % 2 == 0, (
            f"num_combined_kv_heads={num_combined_kv_heads} should be even after pre-padding. "
            "This indicates a configuration issue with kv heads padding."
        )

        assert (
            kv_cache.shape[1] == num_combined_kv_heads
        ), f"kv_cache.shape[1]={kv_cache.shape[1]} is not equal to num_combined_kv_heads={num_combined_kv_heads}"
        assert (
            kv_cache.shape[2] == head_dim
        ), f"kv_cache.shape[2]={kv_cache.shape[2]} is not equal to head_dim={head_dim}"
        assert head_dim % 128 == 0, f"head_dim={head_dim} is not divisible by 128"
        # smaller or equal to page_size

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ]

        out_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)]
        out_shape = [jax.ShapeDtypeStruct(kv_cache.shape, dtype=kv_cache.dtype)]

        scalar_prefetches = [slices]
        scratch = pltpu.VMEM(
            (num_slices_per_block, page_size, num_combined_kv_heads, head_dim),
            new_kv.dtype,
        )

        scratch_shapes = [
            scratch,
            pltpu.SemaphoreType.DMA,
        ]

        kernel = pl.pallas_call(
            kv_cache_update_kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=(cdiv(num_kv_update_slices[0], num_slices_per_block),),
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(
                vmem_limit_bytes=VMEM_SIZE,
            ),
            out_shape=out_shape,
            input_output_aliases={len(scalar_prefetches) + 1: 0},
        )

        result = kernel(*scalar_prefetches, new_kv, kv_cache)[0]

        return result

    return _kv_cache_update_wrapper(new_kv, slices, kv_cache, num_kv_update_slices)
