import functools
import time

import jax
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)


def benchmark_backend(
    mode,
    backend_type,
    batch_size,
    seq_len,
    num_heads,
    head_dim=128,
    max_kv_cache_tokens_num=120000,
):
    if backend_type == "flash":
        if mode == "prefill":
            q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, _, _ = (
                create_prefill_uniform_data(
                    batch_size,
                    seq_len,
                    seq_len,
                    max_kv_cache_tokens_num,
                    num_heads,
                    head_dim,
                )
            )
        elif mode == "decode":
            q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, _, _ = (
                create_decode_uniform_data(
                    batch_size, seq_len, max_kv_cache_tokens_num, num_heads, head_dim
                )
            )

        @functools.partial(
            jax.jit,
            static_argnames=[
                "sm_scale",
            ],
        )
        def jitted_attn(
            q,
            k,
            v,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            num_seqs,
            sm_scale,
        ):
            return ragged_paged_attention(
                q,
                k,
                v,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                num_seqs,
                sm_scale=sm_scale,
                num_kv_pages_per_block=8,
                num_queries_per_block=32,
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
            head_dim**-0.5,
        )
    else:
        raise ValueError(f"Invalid backend type: {backend_type}")

    # Benchmark
    # warm up
    out = attn()
    jax.block_until_ready(out)
    # start benchmark
    times = []
    for i in range(3):
        start = time.perf_counter()
        output = attn()
        jax.block_until_ready(output)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    return avg_time


def main():
    bench_modes = ["prefill", "decode"]
    num_head_config = [2, 4, 8, 16]
    seq_len_config = [1024, 2048, 4096]
    batch_size_config = [1, 2, 4, 8, 10]
    head_dim_config = [128]
    all_combined_config = []
    for batch_size in batch_size_config:
        for seq_len in seq_len_config:
            for num_heads in num_head_config:
                for head_dim in head_dim_config:
                    all_combined_config.append(
                        (batch_size, seq_len, num_heads, head_dim)
                    )

    results = []
    for mode in bench_modes:
        print(f"[{mode.upper()}] BENCHMARK RESULTS SUMMARY")
        for i, (batch_size, seq_len, num_heads, head_dim) in enumerate(
            all_combined_config
        ):
            print(f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}")

            flash_time = benchmark_backend(
                mode,
                "flash",
                batch_size,
                seq_len,
                num_heads,
                head_dim=head_dim,
            )

            results.append(
                {
                    "config": f"B{batch_size}_S{seq_len}_H{num_heads}",
                    "flash_ms": flash_time * 1000,
                }
            )
            print()

        print("=" * 80)
        print("-" * 80)

        for r in results:
            print(f"{r['config']:<15} {r['flash_ms']:<11.2f}")


if __name__ == "__main__":
    main()
