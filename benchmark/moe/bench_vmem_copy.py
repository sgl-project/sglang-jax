"""Benchmark VMEM-to-VMEM vs HBM-to-VMEM async copy latency.

Tests three copy paths with varying sizes:
1. HBM → HBM
2. HBM → VMEM
3. VMEM → VMEM

Usage:
    python -m benchmark.moe.bench_vmem_copy
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "tpu")

import functools
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _kernel_hbm_to_hbm(src_hbm, dst_hbm, _out, sem, *, num_tokens, num_repeats):
    def _repeat(_, __):
        pltpu.make_async_copy(
            src_ref=src_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).start()
        pltpu.make_async_copy(
            src_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def _kernel_hbm_to_vmem(src_hbm, _dst_hbm, _out, sem, vmem_buf, *, num_tokens, num_repeats):
    def _repeat(_, __):
        pltpu.make_async_copy(
            src_ref=src_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=vmem_buf.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).start()
        pltpu.make_async_copy(
            src_ref=vmem_buf.at[pl.ds(0, num_tokens)],
            dst_ref=vmem_buf.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def _kernel_vmem_to_vmem(src_hbm, _dst_hbm, _out, sem, vmem_a, vmem_b, *, num_tokens, num_repeats):
    # Load data into vmem_a first
    pltpu.make_async_copy(
        src_ref=src_hbm.at[pl.ds(0, num_tokens)],
        dst_ref=vmem_a.at[pl.ds(0, num_tokens)],
        sem=sem,
    ).start()
    pltpu.make_async_copy(
        src_ref=vmem_a.at[pl.ds(0, num_tokens)],
        dst_ref=vmem_a.at[pl.ds(0, num_tokens)],
        sem=sem,
    ).wait()

    # Benchmark: VMEM → VMEM
    def _repeat(_, __):
        pltpu.make_async_copy(
            src_ref=vmem_a.at[pl.ds(0, num_tokens)],
            dst_ref=vmem_b.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).start()
        pltpu.make_async_copy(
            src_ref=vmem_b.at[pl.ds(0, num_tokens)],
            dst_ref=vmem_b.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).wait()
        return None
    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def build_benchmark(mode, num_tokens, hidden_size, num_repeats):
    t_packing = 2
    hidden_per_pack = hidden_size // t_packing
    dtype = jnp.bfloat16
    buf_shape = (num_tokens, t_packing, hidden_per_pack)
    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    if mode == "hbm_to_hbm":
        kernel_fn = functools.partial(
            _kernel_hbm_to_hbm, num_tokens=num_tokens, num_repeats=num_repeats
        )
        scratch = (pltpu.SemaphoreType.DMA,)
    elif mode == "hbm_to_vmem":
        kernel_fn = functools.partial(
            _kernel_hbm_to_vmem, num_tokens=num_tokens, num_repeats=num_repeats
        )
        scratch = (pltpu.SemaphoreType.DMA, pltpu.VMEM(buf_shape, dtype))
    elif mode == "vmem_to_vmem":
        kernel_fn = functools.partial(
            _kernel_vmem_to_vmem, num_tokens=num_tokens, num_repeats=num_repeats
        )
        scratch = (pltpu.SemaphoreType.DMA, pltpu.VMEM(buf_shape, dtype), pltpu.VMEM(buf_shape, dtype))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    @jax.jit
    def run(src, dst):
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct(buf_shape, dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm_spec, hbm_spec],
                out_specs=hbm_spec,
                scratch_shapes=scratch,
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        )(src, dst)

    src = jnp.ones(buf_shape, dtype=dtype)
    dst = jnp.zeros(buf_shape, dtype=dtype)
    return run, src, dst


def timeit(run_fn, src, dst, *, num_repeats, warmup=3, iters=5):
    for _ in range(warmup):
        out = run_fn(src, dst)
        jax.block_until_ready(out)
    times_us = []
    for _ in range(iters):
        start = time.perf_counter()
        out = run_fn(src, dst)
        jax.block_until_ready(out)
        elapsed_us = (time.perf_counter() - start) * 1e6
        times_us.append(elapsed_us / num_repeats)
    return times_us


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=6144)
    parser.add_argument("--num-repeats", type=int, default=100)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16, 64, 128])
    args = parser.parse_args()

    modes = ["hbm_to_hbm", "hbm_to_vmem", "vmem_to_vmem"]
    print(f"{'tokens':>8}", end="")
    for m in modes:
        print(f"  {m + '(µs)':>18}", end="")
    print()
    print("-" * 66)

    for num_tokens in args.sizes:
        row = f"{num_tokens:>8}"
        for mode in modes:
            try:
                run_fn, src, dst = build_benchmark(
                    mode, num_tokens, args.hidden_size, args.num_repeats
                )
                times = timeit(run_fn, src, dst, num_repeats=args.num_repeats)
                med = float(np.median(times[1:] if len(times) > 1 else times))
                row += f"  {med:>18.2f}"
            except Exception as e:
                row += f"  {'ERROR':>18}"
        print(row)

    print("\nDone.")


if __name__ == "__main__":
    main()
