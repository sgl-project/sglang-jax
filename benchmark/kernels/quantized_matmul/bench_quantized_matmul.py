# SPDX-License-Identifier: Apache-2.0
"""Benchmark and auto-tune quantized matmul kernel block sizes for Grok FP8.

Usage:
    # Tune FP8 for Grok:
    python -m benchmark.kernels.quantized_matmul.bench_quantized_matmul --tune

    # Benchmark with existing tuned sizes (no sweep):
    python -m benchmark.kernels.quantized_matmul.bench_quantized_matmul
"""

from __future__ import annotations

import argparse
import itertools
import time

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.quantized_matmul.kernel import quantized_matmul_kernel
from sgl_jax.srt.kernels.quantized_matmul.tuned_block_sizes import (
    TunedKey,
    TunedValue,
    get_device_vmem_limit,
    get_tpu_version,
    get_tuned_block_sizes,
)

DEFAULT_N_BATCH = [1, 2, 4, 8, 16, 32, 64, 8192, 16384]

# Grok model per-device shapes (TP=8):
#   hidden_size=8192, intermediate_size=32768
#   num_attention_heads=64, num_key_value_heads=8, head_dim=128
GROK_SHAPES_FP8 = [
    # Attention (column-parallel: n_out sharded by TP)
    (1024, 8192),  # q_proj: 8192/8=1024 x 8192
    (128, 8192),  # k_proj: 1024/8=128 x 8192
    (128, 8192),  # v_proj: 1024/8=128 x 8192
    # Attention (row-parallel: n_in sharded by TP)
    (8192, 1024),  # o_proj: 8192 x 8192/8=1024
    # Residual MLP (column-parallel: n_out sharded by TP)
    (4096, 8192),  # gate_proj: 32768/8=4096 x 8192
    (4096, 8192),  # up_proj: 32768/8=4096 x 8192
    # Residual MLP (row-parallel: n_in sharded by TP)
    (8192, 4096),  # down_proj: 8192 x 32768/8=4096
]


def create_test_data(
    n_batch: int,
    n_out: int,
    n_in: int,
    w_q_dtype: jnp.dtype,
    seed: int = 42,
):
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)

    x = jax.random.normal(k1, (n_batch, n_in), dtype=jnp.bfloat16)
    w_q = jax.random.normal(k2, (n_out, n_in), dtype=jnp.bfloat16).astype(w_q_dtype)
    w_scale = jnp.ones((n_out,), dtype=jnp.float32)

    return x, w_q, w_scale


def benchmark_kernel(
    n_batch: int,
    n_out: int,
    n_in: int,
    x_q_dtype: jnp.dtype = jnp.float8_e4m3fn,
    w_q_dtype: jnp.dtype = jnp.float8_e4m3fn,
    tuned_value: TunedValue | None = None,
    iters: int = 3,
):
    x, w_q, w_scale = create_test_data(n_batch, n_out, n_in, w_q_dtype)

    def fn():
        return quantized_matmul_kernel(
            x, w_q, w_scale, x_q_dtype=x_q_dtype, tuned_value=tuned_value
        )

    # Warmup
    out = fn()
    jax.block_until_ready(out)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)

    return np.mean(times)


def get_block_candidates(n_batch: int, n_out: int, n_in: int) -> list[TunedValue]:
    """Generate block size candidates as power-of-2 divisors of each dimension."""

    def _pow2_divisors(n: int, minimum: int) -> list[int]:
        result = []
        p = minimum
        while p <= n:
            if n % p == 0:
                result.append(p)
            p *= 2
        return result if result else [n]

    batch_cands = _pow2_divisors(n_batch, minimum=min(n_batch, 128))
    out_cands = _pow2_divisors(n_out, minimum=min(n_out, 256))
    in_cands = _pow2_divisors(n_in, minimum=min(n_in, 256))

    candidates = []
    for bb, ob, ib in itertools.product(batch_cands, out_cands, in_cands):
        candidates.append(TunedValue(bb, ob, ib))
    return candidates


def tune_benchmark(
    shapes: list[tuple[int, int]],
    n_batch_list: list[int],
    iters: int = 3,
):
    tpu_version = get_tpu_version()
    results: dict[TunedKey, tuple[TunedValue, float]] = {}

    total = len(shapes) * len(n_batch_list)
    idx = 0

    for n_out, n_in in shapes:
        for n_batch in n_batch_list:
            idx += 1
            print(f"\n[{idx}/{total}] n_batch={n_batch}, n_out={n_out}, n_in={n_in}")

            candidates = get_block_candidates(n_batch, n_out, n_in)
            print(f"  {len(candidates)} block configs to try...")

            best_tv: TunedValue | None = None
            best_ms = float("inf")

            for i, tv in enumerate(candidates):
                try:
                    avg_s = benchmark_kernel(n_batch, n_out, n_in, tuned_value=tv, iters=iters)
                except Exception as e:
                    print(f"    [{i+1}/{len(candidates)}] {tv}: SKIP ({e})")
                    continue

                ms = avg_s * 1000
                marker = " <-- best" if ms < best_ms else ""
                print(
                    f"    [{i+1}/{len(candidates)}] "
                    f"({tv.batch_block_size}, {tv.out_block_size}, {tv.in_block_size}): "
                    f"{ms:.3f} ms{marker}"
                )
                if ms < best_ms:
                    best_ms = ms
                    best_tv = tv

            if best_tv is not None:
                key = TunedKey(
                    tpu_version,
                    n_batch,
                    n_out,
                    n_in,
                    "float8_e4m3fn",
                    "float8_e4m3fn",
                )
                results[key] = (best_tv, best_ms)
                print(
                    f"  BEST: ({best_tv.batch_block_size}, {best_tv.out_block_size}, "
                    f"{best_tv.in_block_size}) = {best_ms:.3f} ms"
                )

    # Write results to file and print
    output_file = "tuned_block_sizes_results.txt"
    lines = []
    if results:
        lines.append("# --- Copy/paste into TUNED_BLOCK_SIZES_RAW in tuned_block_sizes.py ---")
        lines.append("# go/keep-sorted start")
        for key in sorted(results.keys()):
            tv, ms = results[key]
            raw_key = (
                key.tpu_version,
                key.n_batch,
                key.n_out,
                key.n_in,
                key.x_q_dtype,
                key.w_q_dtype,
            )
            raw_val = (tv.batch_block_size, tv.out_block_size, tv.in_block_size)
            lines.append(f"    {raw_key}: {raw_val},  # {ms:.3f} ms")
        lines.append("# go/keep-sorted end")

    text = "\n".join(lines)
    print(f"\n{text}")

    with open(output_file, "w") as f:
        f.write(text + "\n")
    print(f"\nResults written to {output_file}")

    return results


def full_benchmark(
    shapes: list[tuple[int, int]],
    n_batch_list: list[int],
    iters: int = 3,
):
    print("QUANTIZED MATMUL BENCHMARK (FP8)")
    print("=" * 90)
    print(
        f"{'n_batch':>8} {'n_out':>8} {'n_in':>8} "
        f"{'batch_blk':>10} {'out_blk':>10} {'in_blk':>10} {'time_ms':>10}"
    )
    print("-" * 90)

    for n_out, n_in in shapes:
        for n_batch in n_batch_list:
            tv = get_tuned_block_sizes(n_batch, n_out, n_in, "float8_e4m3fn", "float8_e4m3fn")
            try:
                avg_s = benchmark_kernel(n_batch, n_out, n_in, iters=iters)
                ms = avg_s * 1000
            except Exception as e:
                ms = float("nan")
                print(f"  ERROR for n_batch={n_batch}, n_out={n_out}, n_in={n_in}: {e}")
                continue

            print(
                f"{n_batch:>8} {n_out:>8} {n_in:>8} "
                f"{tv.batch_block_size:>10} {tv.out_block_size:>10} {tv.in_block_size:>10} "
                f"{ms:>10.3f}"
            )

    print("=" * 90)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark / auto-tune quantized matmul for Grok FP8."
    )
    parser.add_argument("--tune", action="store_true", help="Sweep and find optimal block sizes.")
    parser.add_argument("--iters", type=int, default=3, help="Benchmark iterations.")
    parser.add_argument("--n-batch", type=int, nargs="+", default=None, help="Batch sizes.")
    parser.add_argument("--n-out", type=int, nargs="+", default=None, help="Output feature sizes.")
    parser.add_argument("--n-in", type=int, nargs="+", default=None, help="Input feature sizes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_batch_list = args.n_batch if args.n_batch else DEFAULT_N_BATCH

    if args.n_out and args.n_in:
        shapes = list(itertools.product(args.n_out, args.n_in))
    elif args.n_out or args.n_in:
        raise ValueError("--n-out and --n-in must both be specified together.")
    else:
        shapes = GROK_SHAPES_FP8

    tpu_ver = get_tpu_version()
    device_kind = jax.devices()[0].device_kind
    print(f"Device: {device_kind} (TPU v{tpu_ver})")
    print(f"VMEM limit: {get_device_vmem_limit() / (1024*1024):.0f} MiB")
    print(f"Batch sizes: {n_batch_list}")
    print(f"Shapes: {shapes}")

    if args.tune:
        tune_benchmark(shapes, n_batch_list, iters=args.iters)
    else:
        full_benchmark(shapes, n_batch_list, iters=args.iters)


if __name__ == "__main__":
    main()
