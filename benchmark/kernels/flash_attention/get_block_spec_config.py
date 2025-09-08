# \!/usr/bin/env python3

import argparse
import functools
import time
from math import inf

import jax
import jax.numpy as jnp
import numpy as np
from utils import (
    check_vmem_oom,
    create_decode_uniform_data,
    create_prefill_uniform_data,
)

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    get_min_heads_per_blk,
    ragged_paged_attention,
)


def benchmark_backend(
    mode,
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

    if mode == "prefill":
        q, k, v, kv_lens, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
            create_prefill_uniform_data(
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size=page_size,
            )
        )
    elif mode == "decode":
        q, k, v, kv_lens, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
            create_decode_uniform_data(
                max_context_len,
                max_kv_cache_tokens,
                max_num_batched_tokens,
                q_head_num,
                kv_head_num,
                head_dim,
                page_size=page_size,
            )
        )

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "num_kv_pages_per_block", "num_queries_per_block"],
    )
    def jitted_attn(
        q,
        k,
        v,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
        sm_scale,
        num_kv_pages_per_block,
        num_queries_per_block,
    ):
        return ragged_paged_attention(
            q,
            k,
            v,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            seq_lens,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )

    attn = functools.partial(
        jitted_attn,
        q,
        k,
        v,
        page_indices,
        cu_q_lens,
        cu_kv_lens,
        num_seqs,
        seq_lens,
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
        k.dtype,
        get_min_heads_per_blk(q_head_num, kv_head_num, q.dtype, k.dtype),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Flash Attention Block Spec Config Benchmark"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["prefill", "decode"],
        required=True,
        help="Benchmark mode: prefill or decode",
    )
    args = parser.parse_args()
    mode = args.mode

    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()

    page_size_config = [64, 128, 256]
    max_num_batched_tokens_config_for_prefill = [
        512,
        1024,
        2048,
        4096,
        8192,
    ]
    max_num_batched_tokens_config_for_decode = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
    ]
    q_head_num_config = [2, 4, 8, 16]
    kv_head_num_config = [2, 4, 8, 16]
    head_dim_config = [128]
    max_kv_cache_tokens_config = [600000]
    all_prefill_combinations = []
    all_decode_combinations = []
    max_context_len = 40960
    for q_head_num in q_head_num_config:
        for kv_head_num in kv_head_num_config:
            for head_dim in head_dim_config:
                for page_size in page_size_config:
                    for max_kv_cache_tokens in max_kv_cache_tokens_config:
                        for (
                            max_num_batched_tokens
                        ) in max_num_batched_tokens_config_for_prefill:
                            if (
                                q_head_num < kv_head_num
                                or q_head_num % kv_head_num != 0
                            ):
                                continue
                            all_prefill_combinations.append(
                                (
                                    page_size,
                                    max_kv_cache_tokens,
                                    max_num_batched_tokens,
                                    q_head_num,
                                    kv_head_num,
                                    head_dim,
                                )
                            )
                        for (
                            max_num_batched_tokens
                        ) in max_num_batched_tokens_config_for_decode:
                            if (
                                q_head_num < kv_head_num
                                or q_head_num % kv_head_num != 0
                            ):
                                continue
                            all_decode_combinations.append(
                                (
                                    page_size,
                                    max_kv_cache_tokens,
                                    max_num_batched_tokens,
                                    q_head_num,
                                    kv_head_num,
                                    head_dim,
                                )
                            )

    num_kv_pages_per_blk_config = [2, 4, 8, 16]
    num_queries_per_block_config = [2, 4, 8, 16, 32, 64, 128]

    block_spec_configs = []
    for num_kv_pages_per_blk in num_kv_pages_per_blk_config:
        for num_queries_per_block in num_queries_per_block_config:
            block_spec_configs.append((num_kv_pages_per_blk, num_queries_per_block))

    print(
        f"###########################################################################################################################################################"
    )
    print(
        f"############################################################ {mode.upper()} ###############################################################################"
    )
    print(
        f"###########################################################################################################################################################"
    )
    print(
        f"(q_dtype, kv_dtype, num_q_heads_per_blk, num_kv_heads_per_blk, head_dim, page_size, max_num_batched_tokens): (num_kv_pages_per_block, num_queries_per_block)"
    )
    if mode == "prefill":
        all_combinations = all_prefill_combinations
    else:
        all_combinations = all_decode_combinations

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
        (num_q_heads_per_blk, num_kv_heads_per_blk) = get_min_heads_per_blk(
            q_head_num, kv_head_num, jnp.bfloat16, jnp.bfloat16
        )
        for i, (num_kv_pages_per_blk, num_queries_per_block) in enumerate(
            block_spec_configs
        ):
            if not check_vmem_oom(
                num_q_heads_per_blk,
                num_kv_heads_per_blk,
                num_kv_pages_per_blk,
                num_queries_per_block,
                page_size,
                head_dim,
                jnp.bfloat16,
                jnp.bfloat16,
            ):
                continue
            try:
                (
                    flash_time,
                    q_dtype,
                    k_dtype,
                    (num_q_heads_per_blk, num_kv_heads_per_blk),
                ) = benchmark_backend(
                    mode,
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
            except Exception as e:
                pass

        print(
            f"('{q_dtype}', '{k_dtype}', {num_q_heads_per_blk}, {num_kv_heads_per_blk}, {head_dim}, {page_size}, {max_num_batched_tokens}): ({best_config[0]}, {best_config[1]}),"
        )


if __name__ == "__main__":
    main()
