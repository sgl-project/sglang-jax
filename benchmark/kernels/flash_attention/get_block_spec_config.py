import functools
from math import inf

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    get_kernel_scope_name,
    ragged_paged_attention,
)
from sgl_jax.srt.kernels.utils.perf import multiple_iteration_timeit_from_trace


def is_decode_only(max_num_batched_tokens):
    return max_num_batched_tokens <= 256


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

    if not is_decode_only(max_num_batched_tokens):
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
    else:
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
            decode_mode=0,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
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
    times = multiple_iteration_timeit_from_trace(
        compute_func=lambda: attn(),
        data_generator=lambda: (),
        task=get_kernel_scope_name(num_queries_per_block, num_kv_pages_per_block, page_size),
        tries=3,
    )
    avg_time = float(np.mean(times)) if times else float("nan")

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

    page_size_config = [128, 256]
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
    q_head_num_config = [1, 2, 4, 8, 16, 32]
    kv_head_num_config = [1, 2, 4, 8, 16, 32]
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

    num_kv_pages_per_blk_config = [1, 2, 4, 8, 16, 32]
    num_queries_per_block_config = [1, 2, 4, 8, 16, 32, 64, 128]

    prefill_only_block_spec_configs = []
    for num_kv_pages_per_blk in num_kv_pages_per_blk_config:
        for num_queries_per_block in num_queries_per_block_config:
            prefill_only_block_spec_configs.append((num_kv_pages_per_blk, num_queries_per_block))

    decode_only_block_spec_configs = []
    for num_kv_pages_per_blk in num_kv_pages_per_blk_config:
        decode_only_block_spec_configs.append((num_kv_pages_per_blk, 1))

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
        if is_decode_only(max_num_batched_tokens):
            # decode only
            block_spec_configs = decode_only_block_spec_configs
        else:
            block_spec_configs = prefill_only_block_spec_configs

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
        if best_config:
            print(
                f"('{q_dtype}', '{k_dtype}', {q_head_num}, {kv_head_num}, {head_dim}, {page_size}, {max_num_batched_tokens}): ({best_config[0]}, {best_config[1]}),"
            )


if __name__ == "__main__":
    main()
