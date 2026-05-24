"""
Tune block sizes for the block-wise quantized matmul kernel.

Sweeps (batch_block_size, out_block_size, in_block_size) for each
(n_batch, n_out, n_in, x_q_dtype, w_q_dtype) and prints the best config in the
TUNED_BLOCK_SIZES_RAW dict format used by
sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.tuned_block_sizes.

Usage (single TPU host):
    python -m benchmark.kernels.quantized_matmul.tune_blockwise_block_sizes \
        --shapes-file shapes.txt --block-k 128

shapes.txt format: one "n_out n_in" per line, # for comments.
"""

from __future__ import annotations

import argparse
from math import inf

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.utils import multiple_iteration_timeit_from_trace
from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.blockwise_kernel import (
    quantized_matmul_kernel,
)
from sgl_jax.srt.kernels.quantized_matmul.quantized_matmul_kernels.tuned_block_sizes import (
    TunedValue,
    get_tpu_version,
)

DEFAULT_N_BATCH = (1, 8, 32, 64, 2048)

OUTPUT_PATH = "tuned_blockwise_block_sizes.txt"


def _make_candidates(
    n_batch: int, n_out: int, n_in: int, block_k: int
) -> list[tuple[int, int, int]]:
    """Generate (batch_block, out_block, in_block) candidates."""
    bb_options = sorted({v for v in (8, 16, 32, 64, 128, 256, n_batch) if v <= max(n_batch, 8)})

    ob_options: set[int] = set()
    for v in (128, 256, 512, 1024, 2048, 4096):
        if v <= n_out:
            ob_options.add(v)
    if not ob_options:
        ob_options.add(min(128, n_out))
    ob_options = sorted(ob_options)

    ib_options: set[int] = set()
    for v in (256, 512, 1024, 2048, 4096):
        if v <= n_in and v % block_k == 0:
            ib_options.add(v)
    ib_options.add(block_k)
    ib_options = sorted(ib_options)

    return [(bb, ob, ib) for bb in bb_options for ob in ob_options for ib in ib_options]


def _create_inputs(
    n_batch: int,
    n_out: int,
    n_in: int,
    block_k: int,
    x_dtype: jnp.dtype,
    w_dtype: jnp.dtype,
):
    in_blocks = n_in // block_k
    x = jnp.ones((n_batch, n_in), dtype=x_dtype)
    w_q = jnp.ones((n_out, n_in), dtype=w_dtype)
    w_scale = jnp.ones((in_blocks, 1, n_out), dtype=jnp.float32)
    return x, w_q, w_scale


def benchmark_one(
    n_batch: int,
    n_out: int,
    n_in: int,
    block_k: int,
    x_dtype: jnp.dtype,
    w_dtype: jnp.dtype,
    tuned_value: TunedValue,
    tries: int = 3,
) -> float:
    """Benchmark one candidate. Returns mean time in ms, or inf on failure."""
    x, w_q, w_scale = _create_inputs(n_batch, n_out, n_in, block_k, x_dtype, w_dtype)

    def fn():
        return quantized_matmul_kernel(
            x,
            w_q,
            w_scale,
            block_size=block_k,
            x_q_dtype=x_dtype,
            tuned_value=tuned_value,
        )

    try:
        jax.block_until_ready(fn())
    except Exception as e:  # noqa: BLE001
        print(f"      compile failed for {tuned_value}: {type(e).__name__}")
        return inf

    bb, ob, ib, _ = tuned_value
    task = f"qmm-b_{n_batch}-o_{n_out}-i_{n_in}-bb_{bb}-ob_{ob}-ib_{ib}"
    try:
        times = multiple_iteration_timeit_from_trace(
            compute_func=lambda: fn(),
            data_generator=lambda: (),
            task=task,
            tries=tries,
        )
        return float(np.mean(times)) if times else inf
    except Exception:  # noqa: BLE001
        return inf


def _load_shapes(path: str) -> list[tuple[int, int]]:
    shapes = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            shapes.append((int(parts[0]), int(parts[1])))
    return shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes-file", required=True)
    parser.add_argument("--block-k", type=int, default=128)
    parser.add_argument("--n-batch", type=int, nargs="+", default=list(DEFAULT_N_BATCH))
    parser.add_argument("--x-dtype", default="bfloat16")
    parser.add_argument("--w-dtype", default="float8_e4m3fn")
    parser.add_argument("--output", default=OUTPUT_PATH)
    args = parser.parse_args()

    x_dtype = jnp.dtype(args.x_dtype)
    w_dtype = jnp.dtype(args.w_dtype)
    tpu_ver = get_tpu_version()

    shapes = _load_shapes(args.shapes_file)
    print(f"JAX devices: {jax.devices()}")
    print(f"TPU version: {tpu_ver}")
    print(f"Shapes: {len(shapes)}, n_batch: {args.n_batch}, block_k: {args.block_k}")
    print(f"Dtype: x={x_dtype.name}, w={w_dtype.name}")
    print(f"Results -> {args.output}")
    print("=" * 100)

    with open(args.output, "a") as fout:
        for n_out, n_in in shapes:
            print(f"\n--- (n_out={n_out}, n_in={n_in}) ---")
            for n_batch in args.n_batch:
                candidates = _make_candidates(n_batch, n_out, n_in, args.block_k)
                best_time = inf
                best: tuple[int, int, int] | None = None
                for bb, ob, ib in candidates:
                    tv = TunedValue(bb, ob, ib, 1)
                    t = benchmark_one(
                        n_batch, n_out, n_in, args.block_k, x_dtype, w_dtype, tv
                    )
                    if t < best_time:
                        best_time = t
                        best = (bb, ob, ib)
                if best is None:
                    print(f"  n_batch={n_batch:>5}: NO VALID TILING")
                    continue
                bb, ob, ib = best
                entry = (
                    f'    ({tpu_ver}, {n_batch}, {n_out}, {n_in}, '
                    f'"{x_dtype.name}", "{w_dtype.name}"): ({bb}, {ob}, {ib}),'
                )
                fout.write(entry + "\n")
                fout.flush()
                print(f"  n_batch={n_batch:>5}: best=({bb},{ob},{ib})  {best_time:.4f} ms")

    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
