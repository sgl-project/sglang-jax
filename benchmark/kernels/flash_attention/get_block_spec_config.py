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
    num_heads,
    head_dim,
    num_kv_pages_per_block,
    num_queries_per_block,
):
    scale = head_dim**-0.5

    if mode == "prefill":
        q, k, v, kv_lens, page_indices, cu_q_lens, num_seqs, _, _ = (
            create_prefill_uniform_data(
                batch_size, seq_len, seq_len, num_heads, head_dim
            )
        )
    elif mode == "decode":
        q, k, v, kv_lens, page_indices, cu_q_lens, num_seqs, _, _ = (
            create_decode_uniform_data(batch_size, seq_len, num_heads, head_dim)
        )

    @functools.partial(
        jax.jit,
        static_argnames=["sm_scale", "num_kv_pages_per_block", "num_queries_per_block"],
    )
    def jitted_attn(
        q,
        k,
        v,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
        sm_scale,
        num_kv_pages_per_block,
        num_queries_per_block,
    ):
        return ragged_paged_attention(
            q,
            k,
            v,
            kv_lens,
            page_indices,
            cu_q_lens,
            num_seqs,
            sm_scale=sm_scale,
            num_kv_pages_per_block=num_kv_pages_per_block,
            num_queries_per_block=num_queries_per_block,
        )

    attn = functools.partial(
        jitted_attn,
        q,
        k,
        v,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_seqs,
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

    batch_size_config = [8, 16, 32]
    seq_len_config = [1024, 2048]
    head_num_config = [2, 4, 8, 16, 32, 64, 128]
    head_dim_config = [128]
    all_combinations = []
    for batch_size in batch_size_config:
        for seq_len in seq_len_config:
            for head_num in head_num_config:
                for head_dim in head_dim_config:
                    all_combinations.append((batch_size, seq_len, head_num, head_dim))

    block_spec_configs = [
        # [num_kv_pages_per_blk, num_queries_per_block ]
        [32, 512],
        [64, 512],
        [128, 512],
        [256, 128],
        [256, 256],
        [512, 64],
        [512, 128],
        [1024, 32],
        [1024, 64],
    ]

    for mode in ["prefill", "decode"]:
        print(f"###########################################################")
        print(f"######################### {mode.upper()} ##########################")
        for i, (batch_size, seq_len, num_heads, head_dim) in enumerate(
            all_combinations
        ):
            best_output = inf
            best_config = None
            for i, (num_kv_pages_per_blk, num_queries_per_block) in enumerate(
                block_spec_configs
            ):
                flash_time = benchmark_backend(
                    mode,
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim,
                    num_kv_pages_per_blk,
                    num_queries_per_block,
                )
                if flash_time < best_output:
                    best_output = flash_time
                    best_config = (num_kv_pages_per_blk, num_queries_per_block)

            print(
                f"########## BATCH_SIZE:{batch_size}, SEQ_LEN:{seq_len}, HEAD_NUM:{num_heads}, HEAD_DIM:{head_dim} ##########"
            )
            print(
                f"[Best Config] num_kv_pages_per_blk:{best_config[0]}, num_queries_per_block:{best_config[1]}"
            )
            print(f"[Best Time] {best_output*1000:.2f}ms")
            print()


if __name__ == "__main__":
    main()
