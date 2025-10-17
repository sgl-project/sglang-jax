import functools
import time

import jax
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from utils import create_bench_data, create_input_params

from sgl_jax.srt.mem_cache.memory_pool import (
    get_num_slices_per_block,
    get_slot_mapping,
    kv_cache_update_kernel,
)
from sgl_jax.srt.utils import cdiv


def benchmark_backend(
    new_value,
    cache,
    kv_cache_start_loc,
    new_kv_start_loc,
    slice_lens,
    update_slices_num,
    num_slices_per_block,
    page_size=1,
):
    assert len(new_value.shape) == 3 and len(cache.shape) == 3

    slices = get_slot_mapping(
        num_slices_per_block, kv_cache_start_loc, new_kv_start_loc, slice_lens
    )

    @functools.partial(
        jax.jit,
        static_argnames=["page_size", "num_slices_per_block"],
    )
    def wrap_kv_cache_update(
        new_value, cache, slices, page_size, num_kv_update_slices, num_slices_per_block
    ):
        num_kv_heads = new_value.shape[1]
        head_dim = new_value.shape[2]
        in_specs = [
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
            pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY),
        ]

        out_specs = [pl.BlockSpec(memory_space=pltpu.MemorySpace.ANY)]
        out_shape = [jax.ShapeDtypeStruct(cache.shape, dtype=cache.dtype)]

        scalar_prefetches = [slices]
        scratch = pltpu.VMEM(
            (num_slices_per_block, page_size, num_kv_heads, head_dim),
            new_value.dtype,
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
                grid=(cdiv(num_kv_update_slices, num_slices_per_block),),
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shape,
            input_output_aliases={len(scalar_prefetches) + 1: 0},
        )
        return kernel(*scalar_prefetches, new_value, cache)[0]

    # warmup
    out = wrap_kv_cache_update(
        new_value, cache, slices, page_size, update_slices_num, num_slices_per_block
    )
    jax.block_until_ready(out)

    # run
    times = []
    with jax.profiler.trace("/home/gcpuser/aolemila/profile_update/"):
        for i in range(3):
            start = time.perf_counter()
            out = wrap_kv_cache_update(
                new_value,
                cache,
                slices,
                page_size,
                update_slices_num,
                num_slices_per_block,
            )
            jax.block_until_ready(out)
            times.append(time.perf_counter() - start)

    return np.mean(times)


def main():
    head_num_config = [8, 16, 32]
    max_cache_len_config = [80000, 160000, 320000, 640000, 1280000]
    new_kv_len_config = [1024, 2048, 4096, 9182, 16384]
    head_dim_config = [128]
    page_sizes = [64, 128, 256]

    configs = []
    for head_num in head_num_config:
        for max_cache_len in max_cache_len_config:
            for new_value_len in new_kv_len_config:
                for head_dim in head_dim_config:
                    for page_size in page_sizes:
                        configs.append(
                            (
                                head_num,
                                max_cache_len,
                                new_value_len,
                                head_dim,
                                page_size,
                            )
                        )

    num_slices_per_block_config = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    for config in configs:
        head_num, max_cache_len, new_value_len, head_dim, page_size = config
        new_value, cache = create_bench_data(
            max_cache_len,
            new_value_len,
            head_num,
            head_dim,
        )
        max_num_slices_per_block_config = get_num_slices_per_block(
            new_value, cache, page_size
        )
        random_cache_loc, slice_lens, new_value_start_loc, update_slices_num = (
            create_input_params(max_cache_len, new_value_len, page_size=page_size)
        )

        print(
            f"###### HEAD_NUM: {head_num}, MAX_CACHE_LEN: {max_cache_len}, NEW_KV_LEN: {new_value_len}, HEAD_DIM: {head_dim} , PAGE_SIZE: {page_size} ######"
        )

        min_cost = 1 << 30
        fastest_num_slices_per_block = 0
        for num_slices_per_block in num_slices_per_block_config:
            nspb = min(max_num_slices_per_block_config, num_slices_per_block)
            cost = benchmark_backend(
                new_value,
                cache,
                random_cache_loc,
                new_value_start_loc,
                slice_lens,
                update_slices_num,
                nspb,
                page_size=page_size,
            )

            if cost < min_cost:
                min_cost = cost
                fastest_num_slices_per_block = num_slices_per_block
            print(
                f"[num_slices_per_block={num_slices_per_block}] avg cost: {cost * 1000} ms"
            )

        print(
            f"Fastest [num_slices_per_block={fastest_num_slices_per_block}] costs: {min_cost * 1000} ms"
        )


if __name__ == "__main__":
    main()
