import functools
import time
from math import inf

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    ragged_paged_attention,
)


def benchmark_backend(
    max_context_len,
    max_kv_cache_tokens,
    max_num_batched_tokens,
    q_head_num,
    kv_head_num,
    head_dim,
    num_kv_pages_per_block,
    num_queries_per_block,
    page_size,
):
    scale = head_dim**-0.5

    if max_num_batched_tokens > 256:
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            _,
            distribution,
        ) = create_prefill_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )
    elif max_num_batched_tokens <= 256:
        (
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            _,
            distribution,
        ) = create_decode_uniform_data(
            max_context_len,
            max_kv_cache_tokens,
            max_num_batched_tokens,
            q_head_num,
            kv_head_num,
            head_dim,
            page_size=page_size,
        )

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "num_kv_pages_per_block", "num_queries_per_block"],
    )
    def jitted_attn(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        sm_scale,
        num_kv_pages_per_block,
        num_queries_per_block,
    ):
        return ragged_paged_attention(
            q,
            k,
            v,
            kv_cache,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
            vmem_limit_bytes=64 * 1024 * 1024,
        )

    attn = functools.partial(
        jitted_attn,
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        distribution,
        scale,
        num_kv_pages_per_block,
        num_queries_per_block,
    )

    # Warmup
    output = attn()
    jax.block_until_ready(output)

    # Benchmark
    times = []
    for i in range(3):
        start = time.perf_counter()
        output = attn()
        jax.block_until_ready(output)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)

    # cal num_q_heads_per_blk, num_kv_heads_per_blk
    return (
        avg_time,
        q.dtype,
        kv_cache.dtype,
    )


def main():
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()

    page_size_config = [64, 128, 256]
    max_num_batched_tokens_config = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    q_head_num_config = [2, 4, 8, 16, 32, 64]
    kv_head_num_config = [2, 4, 8, 16, 32, 64]
    head_dim_config = [128]
    max_kv_cache_tokens_config = [600000]
    all_combinations = []
    max_context_len = 40960
    for q_head_num in q_head_num_config:
        for kv_head_num in kv_head_num_config:
            for head_dim in head_dim_config:
                for page_size in page_size_config:
                    for max_kv_cache_tokens in max_kv_cache_tokens_config:
                        for max_num_batched_tokens in max_num_batched_tokens_config:
                            if q_head_num < kv_head_num or q_head_num % kv_head_num != 0:
                                continue
                            all_combinations.append(
                                (
                                    page_size,
                                    max_kv_cache_tokens,
                                    max_num_batched_tokens,
                                    q_head_num,
                                    kv_head_num,
                                    head_dim,
                                )
                            )

    num_kv_pages_per_blk_config = [1, 2, 4, 8, 16, 32, 64]
    num_queries_per_block_config = [1, 2, 4, 8, 16, 32, 64, 128]

    block_spec_configs = []
    for num_kv_pages_per_blk in num_kv_pages_per_blk_config:
        for num_queries_per_block in num_queries_per_block_config:
            block_spec_configs.append((num_kv_pages_per_blk, num_queries_per_block))

    print(
        "(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens): (num_kv_pages_per_block, num_queries_per_block)"
    )

    for i, (
        page_size,
        max_kv_cache_tokens,
        max_num_batched_tokens,
        q_head_num,
        kv_head_num,
        head_dim,
    ) in enumerate(all_combinations):
        best_output = inf
        best_config = None
        for i, (num_kv_pages_per_blk, num_queries_per_block) in enumerate(block_spec_configs):
            try:
                (
                    flash_time,
                    q_dtype,
                    k_dtype,
                ) = benchmark_backend(
                    max_context_len,
                    max_kv_cache_tokens,
                    max_num_batched_tokens,
                    q_head_num,
                    kv_head_num,
                    head_dim,
                    num_kv_pages_per_blk,
                    num_queries_per_block,
                    page_size,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = (num_kv_pages_per_blk, num_queries_per_block)
            except Exception:
                pass

        if best_config is not None:
            print(
                f"('{q_dtype}', '{k_dtype}', {q_head_num}, {kv_head_num}, {head_dim}, {page_size}, {max_num_batched_tokens}): ({best_config[0]}, {best_config[1]}),"
            )
        else:
            print(
                f"Skipping configuration: {q_head_num} heads, {page_size} page size, {max_num_batched_tokens} tokens (All block sizes failed)"
            )


if __name__ == "__main__":
    main()
