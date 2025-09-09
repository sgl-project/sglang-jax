import argparse
import functools
import time

import jax
import numpy as np
from jax import profiler
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
    page_size=128,
):
    if backend_type == "flash":
        if mode == "prefill":
            q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
                create_prefill_uniform_data(
                    batch_size,
                    seq_len,
                    seq_len,
                    max_kv_cache_tokens_num,
                    num_heads,
                    head_dim,
                    page_size=page_size,
                )
            )
        elif mode == "decode":
            q, k, v, _, page_indices, cu_q_lens, cu_kv_lens, num_seqs, seq_lens, _ = (
                create_decode_uniform_data(
                    batch_size,
                    seq_len,
                    max_kv_cache_tokens_num,
                    num_heads,
                    head_dim,
                    page_size=page_size,
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
            seq_lens,
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
                seq_lens,
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
            seq_lens,
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
    parser = argparse.ArgumentParser(description="Flash Attention Benchmark")
    parser.add_argument(
        "--profiler",
        action="store_true",
        help="Enable JAX profiler with reduced configs",
    )
    args = parser.parse_args()

    page_size = 128
    bench_modes = ["prefill", "decode"]

    if args.profiler:
        # Reduced configs for profiling - just a few representative cases
        all_combined_config = [
            (1, 1024, 4, 128),  # Small: B1_S1024_H4
            (2, 2048, 8, 128),  # Medium: B2_S2048_H8
            (4, 4096, 16, 128),  # Large: B4_S4096_H16
        ]
        print("Starting JAX profiling for Flash Attention...")
        # Enable detailed TPU profiling to see inside pallas kernels
        profiler.start_trace("./flash_attention_profile", create_perfetto_link=True)
    else:
        # Original full benchmark configs
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

    try:
        for mode in bench_modes:
            if args.profiler:
                print(f"[{mode.upper()}] PROFILING BENCHMARK")
            else:
                print(f"[{mode.upper()}] BENCHMARK RESULTS SUMMARY")

            for i, (batch_size, seq_len, num_heads, head_dim) in enumerate(
                all_combined_config
            ):
                print(
                    f"Config: batch={batch_size}, seq_len={seq_len}, heads={num_heads}"
                )

                flash_time = benchmark_backend(
                    mode,
                    "flash",
                    batch_size,
                    seq_len,
                    num_heads,
                    head_dim=head_dim,
                    page_size=page_size,
                )

                if args.profiler:
                    results.append(
                        {
                            "config": f"B{batch_size}_S{seq_len}_H{num_heads}_{mode}",
                            "flash_ms": flash_time * 1000,
                        }
                    )
                    print(f"Time: {flash_time * 1000:.2f}ms")
                else:
                    results.append(
                        {
                            "config": f"B{batch_size}_S{seq_len}_H{num_heads}",
                            "flash_ms": flash_time * 1000,
                        }
                    )
                print()

            if not args.profiler:
                print("=" * 80)
                print("-" * 80)
                for r in results:
                    print(f"{r['config']:<15} {r['flash_ms']:<11.2f}")
                results = []  # Clear for next mode

    finally:
        if args.profiler:
            print("=" * 80)
            print("PROFILING RESULTS SUMMARY")
            print("-" * 80)
            for r in results:
                print(f"{r['config']:<20} {r['flash_ms']:<11.2f}ms")

            # Stop profiling and save results
            print("\nStopping profiler...")
            profiler.stop_trace()
            print("Profiling data saved to ./flash_attention_profile")
            print("View with: tensorboard --logdir=./flash_attention_profile")


if __name__ == "__main__":
    main()
