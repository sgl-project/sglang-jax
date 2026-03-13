# SPDX-License-Identifier: Apache-2.0
"""Benchmark per-channel quantized matmul: JAX fallback vs 3rd-party TPU kernel.

Compares two per-channel quantized matmul implementations:
  1. JAX fallback: dequant + lax.dot_general (always available)
  2. 3rd-party TPU Pallas kernel: int32 accumulation across full K (TPU only)

Usage:
    # Run benchmark with default shapes (Qwen3-MoE / Grok style):
    python -m benchmark.kernels.quantized_matmul.bench_perchannel_kernel

    # Custom shapes:
    python -m benchmark.kernels.quantized_matmul.bench_perchannel_kernel \
        --n-batch 1 4 16 64 --n-out 1024 4096 --n-in 1024 4096

    # Change weight dtype:
    python -m benchmark.kernels.quantized_matmul.bench_perchannel_kernel --w-dtype float8_e4m3fn
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from sgl_jax.srt.kernels.quantized_matmul.blockwise_3rd_utils import (
    get_perchannel_3rd_kernel,
)
from sgl_jax.srt.utils.quantization.quantization_utils import quantize_tensor_simple

# ──────────────────────────────────────────────────────────────────────────────
# Default benchmark shapes
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_N_BATCH = [1, 2, 4, 8, 16, 32, 64, 128, 8192, 16384]

# Representative per-device shapes (TP=8) across common models:
#   (n_out, n_in)
DEFAULT_SHAPES = [
    # Attention projections
    (1024, 8192),  # q_proj column-parallel
    (128, 8192),  # k/v_proj column-parallel (Grok)
    (512, 4096),  # q_proj column-parallel (smaller model)
    # Row-parallel / output projections
    (8192, 1024),  # o_proj row-parallel
    (4096, 512),  # o_proj row-parallel (smaller)
    # MLP projections
    (4096, 8192),  # gate/up_proj column-parallel
    (8192, 4096),  # down_proj row-parallel
    # Qwen3-MoE style (smaller per-expert)
    (256, 2048),  # MoE expert column
    (2048, 256),  # MoE expert row
]

DTYPE_MAP = {
    "int8": jnp.int8,
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e5m2": jnp.float8_e5m2,
}


# ──────────────────────────────────────────────────────────────────────────────
# Test data creation
# ──────────────────────────────────────────────────────────────────────────────


def create_test_data(
    n_batch: int,
    n_out: int,
    n_in: int,
    w_q_dtype: jnp.dtype,
    seed: int = 42,
):
    """Create test inputs for per-channel quantized matmul."""
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key, 2)

    x = jax.random.normal(k1, (n_batch, n_in), dtype=jnp.bfloat16)
    w_fp = jax.random.normal(k2, (n_out, n_in), dtype=jnp.bfloat16)

    # Quantize weights per-channel
    w_q = w_fp.astype(w_q_dtype)
    w_scale = jnp.max(jnp.abs(w_fp), axis=1).astype(jnp.float32)
    # Avoid division by zero
    dtype_info = (
        jnp.iinfo(w_q_dtype)
        if jnp.issubdtype(w_q_dtype, jnp.integer)
        else jnp.finfo(w_q_dtype)
    )
    w_scale = w_scale / float(dtype_info.max)
    w_scale = jnp.where(w_scale == 0, 1.0, w_scale)

    return x, w_q, w_scale, w_fp


# ──────────────────────────────────────────────────────────────────────────────
# Kernel wrappers
# ──────────────────────────────────────────────────────────────────────────────


def run_jax_perchannel(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool,
    act_quant_dtype: jnp.dtype,
) -> jax.Array:
    """JAX fallback per-channel matmul: dequant weights + dot_general."""
    compute_dtype = jnp.float32
    if quantize_activation:
        x_q, x_scale = quantize_tensor_simple(x, act_quant_dtype, dim=-1)
        out = lax.dot_general(
            x_q,
            w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        out = (
            out.astype(compute_dtype)
            * x_scale.astype(compute_dtype)
            * jnp.expand_dims(w_scale, 0).astype(compute_dtype)
        )
    else:
        out = lax.dot_general(
            x,
            w_q,
            dimension_numbers=(((1,), (1,)), ((), ())),
            preferred_element_type=compute_dtype,
        )
        out = out.astype(compute_dtype) * jnp.expand_dims(w_scale, 0).astype(
            compute_dtype
        )
    return out.astype(x.dtype)


def run_3rd_perchannel(
    x: jax.Array,
    w_q: jax.Array,
    w_scale: jax.Array,
    quantize_activation: bool,
    act_quant_dtype: jnp.dtype,
    kernel_fn,
) -> jax.Array:
    """3rd-party TPU per-channel kernel wrapper."""
    x_q_dtype = act_quant_dtype if quantize_activation else x.dtype
    out = kernel_fn(
        x=x,
        w_q=w_q,
        w_scale=w_scale,
        x_q_dtype=x_q_dtype,
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────


def benchmark_fn(fn, warmup: int = 3, iters: int = 10):
    """Run fn with warmup and return (mean_ms, std_ms)."""
    # Warmup
    for _ in range(warmup):
        out = fn()
        jax.block_until_ready(out)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        times.append((time.perf_counter() - start) * 1000)

    return float(np.mean(times)), float(np.std(times))


def compute_accuracy(out: jax.Array, ref: jax.Array):
    """Compute MAE and relative error."""
    diff = jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))
    mae = float(jnp.mean(diff))
    ref_mean = float(jnp.mean(jnp.abs(ref.astype(jnp.float32))))
    rel_error = mae / max(ref_mean, 1e-10)
    max_diff = float(jnp.max(diff))
    return mae, rel_error, max_diff


def run_benchmark(
    shapes: list[tuple[int, int]],
    n_batch_list: list[int],
    w_q_dtype: jnp.dtype,
    quantize_activation: bool,
    iters: int,
):
    """Run the full benchmark comparison."""
    perchannel_kernel = get_perchannel_3rd_kernel()
    has_3rd = perchannel_kernel is not None and jax.default_backend() == "tpu"

    act_quant_dtype = w_q_dtype

    print("=" * 120)
    print("PER-CHANNEL QUANTIZED MATMUL BENCHMARK")
    print(f"  Device: {jax.devices()[0].device_kind}")
    print(f"  Weight dtype: {jnp.dtype(w_q_dtype).name}")
    print(
        f"  Activation quant: {'enabled (' + jnp.dtype(act_quant_dtype).name + ')' if quantize_activation else 'disabled'}"
    )
    print(
        f"  3rd-party kernel: {'available' if has_3rd else 'NOT available (JAX only)'}"
    )
    print(f"  Iterations: {iters}")
    print("=" * 120)

    if has_3rd:
        header = (
            f"{'n_batch':>8} {'n_out':>8} {'n_in':>8} │ "
            f"{'JAX(ms)':>10} {'±':>6} │ "
            f"{'3rd(ms)':>10} {'±':>6} │ "
            f"{'speedup':>8} │ "
            f"{'MAE':>10} {'rel_err':>10} {'max_diff':>10}"
        )
    else:
        header = (
            f"{'n_batch':>8} {'n_out':>8} {'n_in':>8} │ " f"{'JAX(ms)':>10} {'±':>6}"
        )
    print(header)
    print("─" * len(header))

    for n_out, n_in in shapes:
        for n_batch in n_batch_list:
            x, w_q, w_scale, _ = create_test_data(n_batch, n_out, n_in, w_q_dtype)

            # Benchmark JAX fallback
            try:
                jax_mean, jax_std = benchmark_fn(
                    lambda: run_jax_perchannel(
                        x, w_q, w_scale, quantize_activation, act_quant_dtype
                    ),
                    iters=iters,
                )
            except Exception as e:
                print(f"{n_batch:>8} {n_out:>8} {n_in:>8} │ JAX ERROR: {e}")
                continue

            if not has_3rd:
                print(
                    f"{n_batch:>8} {n_out:>8} {n_in:>8} │ {jax_mean:>10.3f} {jax_std:>6.3f}"
                )
                continue

            # Benchmark 3rd-party kernel
            try:
                third_mean, third_std = benchmark_fn(
                    lambda: run_3rd_perchannel(
                        x,
                        w_q,
                        w_scale,
                        quantize_activation,
                        act_quant_dtype,
                        perchannel_kernel,
                    ),
                    iters=iters,
                )

                # Accuracy comparison (3rd vs JAX)
                jax_out = run_jax_perchannel(
                    x, w_q, w_scale, quantize_activation, act_quant_dtype
                )
                third_out = run_3rd_perchannel(
                    x,
                    w_q,
                    w_scale,
                    quantize_activation,
                    act_quant_dtype,
                    perchannel_kernel,
                )
                mae, rel_err, max_diff = compute_accuracy(third_out, jax_out)

                speedup = jax_mean / third_mean if third_mean > 0 else float("inf")
                print(
                    f"{n_batch:>8} {n_out:>8} {n_in:>8} │ "
                    f"{jax_mean:>10.3f} {jax_std:>6.3f} │ "
                    f"{third_mean:>10.3f} {third_std:>6.3f} │ "
                    f"{speedup:>7.2f}x │ "
                    f"{mae:>10.6f} {rel_err:>10.6f} {max_diff:>10.6f}"
                )
            except Exception as e:
                print(
                    f"{n_batch:>8} {n_out:>8} {n_in:>8} │ "
                    f"{jax_mean:>10.3f} {jax_std:>6.3f} │ "
                    f"3rd ERROR: {e}"
                )

        # Separator between shape groups
        print()

    print("=" * 120)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark per-channel quantized matmul: JAX vs 3rd-party TPU kernel."
    )
    parser.add_argument(
        "--n-batch", type=int, nargs="+", default=None, help="Batch sizes to benchmark."
    )
    parser.add_argument(
        "--n-out", type=int, nargs="+", default=None, help="Output feature sizes."
    )
    parser.add_argument(
        "--n-in", type=int, nargs="+", default=None, help="Input feature sizes."
    )
    parser.add_argument(
        "--w-dtype",
        type=str,
        default="int8",
        choices=list(DTYPE_MAP.keys()),
        help="Weight quantization dtype (default: int8).",
    )
    parser.add_argument(
        "--no-act-quant",
        action="store_true",
        help="Disable activation quantization.",
    )
    parser.add_argument(
        "--iters", type=int, default=10, help="Benchmark iterations (default: 10)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_batch_list = args.n_batch if args.n_batch else DEFAULT_N_BATCH
    w_q_dtype = DTYPE_MAP[args.w_dtype]

    if args.n_out and args.n_in:
        import itertools

        shapes = list(itertools.product(args.n_out, args.n_in))
    elif args.n_out or args.n_in:
        raise ValueError("--n-out and --n-in must both be specified together.")
    else:
        shapes = DEFAULT_SHAPES

    run_benchmark(
        shapes=shapes,
        n_batch_list=n_batch_list,
        w_q_dtype=w_q_dtype,
        quantize_activation=not args.no_act_quant,
        iters=args.iters,
    )


if __name__ == "__main__":
    main()
