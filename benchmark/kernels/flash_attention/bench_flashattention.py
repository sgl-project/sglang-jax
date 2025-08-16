import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
from utils import create_decode_uniform_data, create_prefill_uniform_data

from sgl_jax.srt.layers.attention.flash_attn_kernel.flash_attention import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.native_backend import forward_attention


def benchmark_backend(
    mode, backend_type, batch_size, seq_len, num_heads, num_kv_heads, head_dim=128
):
    if backend_type == "flash":
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
            static_argnames=[
                "sm_scale",
                "num_kv_pages_per_block",
                "num_queries_per_block",
            ],
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
            head_dim**-0.5,
            num_kv_pages_per_block,
            num_queries_per_block,
        )
    elif backend_type == "native":
        from sgl_jax.srt.model_executor.forward_batch import ForwardMode

        if mode == "prefill":
            q, k, v, _, page_indices, _, _, seq_lens, cache_loc = (
                create_prefill_uniform_data(
                    batch_size, seq_len, seq_len, num_heads, head_dim
                )
            )
            extend_prefix_lens = jnp.zeros(batch_size, dtype=jnp.int32)
            extend_seq_lens = seq_lens.copy()
            mode = ForwardMode.PREFILL
        elif mode == "decode":
            q, k, v, _, page_indices, _, _, seq_lens, cache_loc = (
                create_decode_uniform_data(batch_size, seq_len, num_heads, head_dim)
            )
            extend_prefix_lens = None
            extend_seq_lens = None
            mode = ForwardMode.DECODE

        @functools.partial(
            jax.jit,
            static_argnames=["num_heads", "num_kv_heads", "scale", "is_causal", "mode"],
        )
        def jitted_attn(
            q,
            k,
            v,
            seq_len,
            loc,
            extend_prefix_lens,
            extend_seq_lens,
            num_heads,
            num_kv_heads,
            scale,
            is_causal,
            mode,
        ):
            return forward_attention(
                q,
                k,
                v,
                seq_len,
                loc,
                extend_prefix_lens,
                extend_seq_lens,
                num_heads,
                num_kv_heads,
                scale=scale,
                is_causal=is_causal,
                mode=mode,
            )

        attn = functools.partial(
            jitted_attn,
            q,
            k,
            v,
            seq_lens,
            cache_loc,
            extend_prefix_lens,
            extend_seq_lens,
            num_heads,
            num_kv_heads,
            scale=head_dim**-0.5,
            is_causal=True,
            mode=mode,
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
        output, _, _ = attn()
        jax.block_until_ready(output)
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times)
    return avg_time


def main():
    bench_modes = ["prefill", "decode"]
    num_head_config = [2, 4, 8, 16, 32, 64]
    seq_len_config = [128, 256, 512, 1024]
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
                num_heads,
                head_dim=head_dim,
            )
            native_time = benchmark_backend(
                mode,
                "native",
                batch_size,
                seq_len,
                num_heads,
                num_heads,
                head_dim=head_dim,
            )

            speedup = native_time / flash_time if flash_time > 0 else float("inf")
            improvement = (
                ((native_time - flash_time) / native_time) * 100
                if native_time > 0
                else 0
            )

            results.append(
                {
                    "config": f"B{batch_size}_S{seq_len}_H{num_heads}",
                    "native_ms": native_time * 1000,
                    "flash_ms": flash_time * 1000,
                    "speedup": speedup,
                    "improvement": improvement,
                }
            )

            print(
                f"  Results: Native={native_time*1000:.2f}ms, Flash={flash_time*1000:.2f}ms, Speedup={speedup:.2f}x, Improvement={improvement:.1f}%"
            )

            print()

        print("=" * 80)
        print(f"[{mode.upper()}] BENCHMARK RESULTS SUMMARY")
        print("=" * 80)
        print(
            f"{'Config':<15} {'Native(ms)':<12} {'Flash(ms)':<11} {'Speedup':<10} {'Improve(%)':<10}"
        )
        print("-" * 80)

        for r in results:
            speedup_str = (
                f"{r['speedup']:.2f}x" if r["speedup"] != float("inf") else "N/A"
            )
            print(
                f"{r['config']:<15} {r['native_ms']:<12.2f} {r['flash_ms']:<11.2f} {speedup_str:<10} {r['improvement']:<10.1f}"
            )

        if results:
            valid_speedups = [
                r["speedup"] for r in results if r["speedup"] != float("inf")
            ]
            improvements = [r["improvement"] for r in results]
            print("-" * 80)
            print(f"Average speedup: {np.mean(valid_speedups):.2f}x")
            print(f"Average improvement: {np.mean(improvements):.1f}%")


if __name__ == "__main__":
    main()
