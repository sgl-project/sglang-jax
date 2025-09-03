# \!/usr/bin/env python3

import functools
import time
from math import inf

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)


def benchmark_backend(
    mode,
    batch_size,
    seq_len,
    max_kv_cache_tokens,
    num_heads,
    head_dim,
    num_kv_pages_per_block,
    num_queries_per_block,
    page_size,
):
    scale = head_dim**-0.5

    if mode == "prefill":
        q, k, v, kv_lens, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
            create_prefill_uniform_data(
                batch_size,
                seq_len,
                seq_len,
                max_kv_cache_tokens,
                num_heads,
                head_dim,
                page_size=page_size,
            )
        )
    elif mode == "decode":
        q, k, v, kv_lens, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
            create_decode_uniform_data(
                batch_size,
                seq_len,
                max_kv_cache_tokens,
                num_heads,
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
    return avg_time


def main():
    print("JAX devices:", jax.devices())
    print("Device count:", jax.device_count())
    print()

    page_size = 128
    batch_size_config = [8, 16, 32]
    seq_len_config = [1024, 2048, 4096]
    head_num_config = [2, 4, 8]
    head_dim_config = [128]
    max_kv_cache_tokens_config = [160000]
    all_combinations = []
    for max_kv_cache_tokens in max_kv_cache_tokens_config:
        for batch_size in batch_size_config:
            for seq_len in seq_len_config:
                for head_num in head_num_config:
                    for head_dim in head_dim_config:
                        all_combinations.append(
                            (
                                max_kv_cache_tokens,
                                batch_size,
                                seq_len,
                                head_num,
                                head_dim,
                            )
                        )

    block_spec_configs = [
        # [num_kv_pages_per_blk, num_queries_per_block ]
        [4, 4],
        [4, 8],
        [4, 16],
        [4, 32],
        [8, 4],
        [8, 8],
        [8, 16],
        [8, 32],
        [16, 4],
        [16, 8],
        [16, 16],
        [16, 32],
    ]

    for mode in ["prefill", "decode"]:
        print(f"###########################################################")
        print(f"######################### {mode.upper()} ##########################")
        for i, (
            max_kv_cache_tokens,
            batch_size,
            seq_len,
            num_heads,
            head_dim,
        ) in enumerate(all_combinations):
            best_output = inf
            best_config = None
            for i, (num_kv_pages_per_blk, num_queries_per_block) in enumerate(
                block_spec_configs
            ):
                flash_time = benchmark_backend(
                    mode,
                    batch_size,
                    seq_len,
                    max_kv_cache_tokens,
                    num_heads,
                    head_dim,
                    num_kv_pages_per_blk,
                    num_queries_per_block,
                    page_size,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = (num_kv_pages_per_blk, num_queries_per_block)

            print(
                f"########## MAX_KVCACHE_LEN:{max_kv_cache_tokens}, BATCH_SIZE:{batch_size}, SEQ_LEN:{seq_len}, HEAD_NUM:{num_heads}, HEAD_DIM:{head_dim} ##########"
            )
            print(
                f"[Best Config] num_kv_pages_per_blk:{best_config[0]}, num_queries_per_block:{best_config[1]}"
            )
            print(f"[Best Time] {best_output*1000:.2f}ms")
            print()


if __name__ == "__main__":
    main()
