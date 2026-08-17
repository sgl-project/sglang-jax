"""Benchmark GDN prefill recurrence kernels: chunked-JAX vs sequential scan.

Compares:
  - `ragged_gated_delta_rule_ref`: sequential O(T) token-by-token lax.scan (Phase 0)
  - `chunked_gated_delta_rule_jax`: chunkwise-parallel pure-JAX recurrence (Phase 1)

Measures latency (ms), prefill throughput (tokens/s), speedup ratio, and verifies
numerical parity on multi-k sequence lengths on TPU/GPU/CPU.

Usage:
  python -m benchmark.kernels.gdn.bench_gdn
  python -m benchmark.kernels.gdn.bench_gdn --seq-lens 512,1024,2048,4096,8192,16384 --batch-sizes 1,2,4
  python -m benchmark.kernels.gdn.bench_gdn --profile --profile-dir /tmp/gdn_profile
"""

from __future__ import annotations

import argparse
import functools
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.gdn import (
    chunked_gated_delta_rule_jax,
    ragged_gated_delta_rule_ref,
)


def create_gdn_input_data(
    seq_lens: list[int],
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    dtype=jnp.bfloat16,
    seed: int = 42,
) -> tuple[dict[str, Any], int]:
    """Generate reproducible dummy input tensors for GDN linear attention forward."""
    rng = np.random.default_rng(seed)
    B = len(seq_lens)
    total_tokens = sum(seq_lens)
    key_dim = n_kq * d_k
    value_dim = n_v * d_v
    conv_dim = 2 * key_dim + value_dim

    mixed_qkv = jnp.asarray(rng.standard_normal((total_tokens, conv_dim)) * 0.1, dtype=dtype)
    b = jnp.asarray(rng.standard_normal((total_tokens, n_v)) * 0.1, dtype=jnp.float32)
    a = jnp.asarray(rng.standard_normal((total_tokens, n_v)) * 0.1, dtype=jnp.float32)

    recurrent_state = jnp.zeros((B + 1, n_v, d_k, d_v), dtype=jnp.float32)
    A_log = jnp.asarray(rng.standard_normal((n_v,)), dtype=jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal((n_v,)), dtype=jnp.float32)

    cu_seqlens = jnp.asarray([0] + np.cumsum(seq_lens).tolist(), dtype=jnp.int32)
    state_indices = jnp.arange(1, B + 1, dtype=jnp.int32)
    has_initial_state = jnp.zeros(B, dtype=bool)

    data = {
        "mixed_qkv": mixed_qkv,
        "b": b,
        "a": a,
        "recurrent_state": recurrent_state,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
        "state_indices": state_indices,
        "has_initial_state": has_initial_state,
    }
    return data, total_tokens


def benchmark_kernel(
    fn,
    args: tuple,
    kwargs: dict | None = None,
    warmup: int = 3,
    iters: int = 10,
) -> tuple[float, Any]:
    """JIT-compile, warmup, and time kernel execution using jax.block_until_ready."""
    kw = kwargs or {}
    jitted_fn = jax.jit(functools.partial(fn, **kw))

    # Warmup
    for _ in range(warmup):
        out = jitted_fn(*args)
        jax.block_until_ready(out)

    # Timing
    t0 = time.perf_counter()
    for _ in range(iters):
        out = jitted_fn(*args)
        jax.block_until_ready(out)
    avg_latency_s = (time.perf_counter() - t0) / iters

    return avg_latency_s, out


def run_benchmark_grid(
    batch_sizes: list[int],
    seq_lens_per_batch: list[int],
    n_kq: int,
    n_v: int,
    d_k: int,
    d_v: int,
    chunk_size: int,
    warmup: int,
    iters: int,
    profile_dir: str | None = None,
):
    print("=" * 105)
    print(
        f"GDN Linear Attention Kernel Benchmark (heads: n_kq={n_kq}, n_v={n_v}, d_k={d_k}, d_v={d_v}, BT={chunk_size})"
    )
    print("=" * 105)
    header = (
        f"{'B':>3s} | {'T_total':>8s} | {'T_per_seq':>10s} | "
        f"{'Ref Lat (ms)':>13s} | {'Ref (tok/s)':>13s} | "
        f"{'Chk Lat (ms)':>13s} | {'Chk (tok/s)':>13s} | "
        f"{'Speedup':>9s} | {'MaxDiff (Out/Rec)':>18s}"
    )
    print(header)
    print("-" * 105)

    kw_ref = dict(n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v)
    kw_chk = dict(n_kq=n_kq, n_v=n_v, d_k=d_k, d_v=d_v, chunk_size=chunk_size)

    for B in batch_sizes:
        for L in seq_lens_per_batch:
            seq_lens = [L] * B
            data, total_tokens = create_gdn_input_data(
                seq_lens=seq_lens,
                n_kq=n_kq,
                n_v=n_v,
                d_k=d_k,
                d_v=d_v,
            )

            args = (
                data["mixed_qkv"],
                data["b"],
                data["a"],
                data["recurrent_state"],
                data["A_log"],
                data["dt_bias"],
                data["cu_seqlens"],
                data["state_indices"],
                data["has_initial_state"],
            )

            # 1. Run Reference (sequential scan)
            ref_lat_s, (ref_rec, ref_out) = benchmark_kernel(
                ragged_gated_delta_rule_ref, args, kw_ref, warmup=warmup, iters=iters
            )
            ref_lat_ms = ref_lat_s * 1e3
            ref_tps = total_tokens / ref_lat_s

            # 2. Run Chunked-JAX
            chk_lat_s, (chk_rec, chk_out) = benchmark_kernel(
                chunked_gated_delta_rule_jax, args, kw_chk, warmup=warmup, iters=iters
            )
            chk_lat_ms = chk_lat_s * 1e3
            chk_tps = total_tokens / chk_lat_s

            # 3. Compute speedup and numerical diffs
            speedup = ref_lat_s / chk_lat_s if chk_lat_s > 0 else float("inf")
            out_diff = float(
                np.max(
                    np.abs(
                        np.asarray(chk_out, dtype=np.float32)
                        - np.asarray(ref_out, dtype=np.float32)
                    )
                )
            )
            rec_diff = float(
                np.max(
                    np.abs(
                        np.asarray(chk_rec, dtype=np.float32)
                        - np.asarray(ref_rec, dtype=np.float32)
                    )
                )
            )
            diff_str = f"{out_diff:.1e} / {rec_diff:.1e}"

            row = (
                f"{B:3d} | {total_tokens:8d} | {L:10d} | "
                f"{ref_lat_ms:13.2f} | {ref_tps:13.1f} | "
                f"{chk_lat_ms:13.2f} | {chk_tps:13.1f} | "
                f"{speedup:8.2f}x | {diff_str:>18s}"
            )
            print(row)

    print("=" * 105)

    if profile_dir:
        os.makedirs(profile_dir, exist_ok=True)
        print(f"\nRecording JAX profile trace to {profile_dir}...")
        data, _ = create_gdn_input_data([2048] * 2, n_kq, n_v, d_k, d_v)
        args = (
            data["mixed_qkv"],
            data["b"],
            data["a"],
            data["recurrent_state"],
            data["A_log"],
            data["dt_bias"],
            data["cu_seqlens"],
            data["state_indices"],
            data["has_initial_state"],
        )
        jitted_chk = jax.jit(lambda *a: chunked_gated_delta_rule_jax(*a, **kw_chk))
        # Warmup
        jax.block_until_ready(jitted_chk(*args))
        with jax.profiler.trace(profile_dir):
            for i in range(5):
                with jax.profiler.StepTraceAnnotation("gdn_chunked_step", step_num=i):
                    jax.block_until_ready(jitted_chk(*args))
        print("Profile saved. View with Perfetto (https://ui.perfetto.dev) or TensorBoard/XProf.")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GDN Prefill Kernels (Chunked vs Reference)"
    )
    parser.add_argument(
        "--seq-lens",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated sequence lengths to benchmark (per request)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2",
        help="Comma-separated batch sizes",
    )
    parser.add_argument(
        "--n-kq",
        type=int,
        default=4,
        help="Number of Q/K heads per shard (default: 4)",
    )
    parser.add_argument(
        "--n-v",
        type=int,
        default=8,
        help="Number of V heads per shard (default: 8)",
    )
    parser.add_argument(
        "--d-k",
        type=int,
        default=128,
        help="Head dimension for Q/K (default: 128)",
    )
    parser.add_argument(
        "--d-v",
        type=int,
        default=128,
        help="Head dimension for V (default: 128)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Chunk size for chunked recurrence (default: 64)",
    )
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations (default: 3)")
    parser.add_argument(
        "--iters",
        type=int,
        default=10,
        help="Benchmark timing iterations (default: 10)",
    )
    parser.add_argument("--profile", action="store_true", help="Record JAX trace profile")
    parser.add_argument(
        "--profile-dir",
        type=str,
        default="/tmp/gdn_profile",
        help="Trace output directory",
    )
    args = parser.parse_args()

    seq_lens = [int(x.strip()) for x in args.seq_lens.split(",") if x.strip()]
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",") if x.strip()]

    run_benchmark_grid(
        batch_sizes=batch_sizes,
        seq_lens_per_batch=seq_lens,
        n_kq=args.n_kq,
        n_v=args.n_v,
        d_k=args.d_k,
        d_v=args.d_v,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        iters=args.iters,
        profile_dir=args.profile_dir if args.profile else None,
    )


if __name__ == "__main__":
    main()
