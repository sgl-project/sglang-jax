"""Micro-benchmark: DMA latency vs transfer size (local and remote).

Compares per-token DMA (many small DMAs) vs batch DMA (one large DMA) for
the same total data volume. Tests both local (HBM-to-HBM same device) and
remote (ICI cross-device) transfers.

Usage:
    # Local DMA only (single device)
    python -m benchmark.moe.bench_dma_size --hidden-size 7168

    # Local + remote DMA (needs EP >= 2)
    python -m benchmark.moe.bench_dma_size --hidden-size 7168 --test-remote

    # Custom token sizes
    python -m benchmark.moe.bench_dma_size --hidden-size 7168 --sizes 1 2 4 8 16 32 64 128 256
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from benchmark.utils import multiple_iteration_timeit_from_trace

P = jax.sharding.PartitionSpec


def _local_small_dma_kernel(
    src_hbm,
    dst_hbm,
    _output_hbm,
    sem,
    *,
    num_tokens: int,
    num_repeats: int,
):
    """Issue num_tokens individual DMAs of size 1, repeated num_repeats times."""

    def _repeat(_, __):
        def _one_token(t_id, ___):
            pltpu.make_async_copy(
                src_ref=src_hbm.at[pl.ds(t_id, 1)],
                dst_ref=dst_hbm.at[pl.ds(t_id, 1)],
                sem=sem,
            ).start()
            return None

        lax.fori_loop(0, num_tokens, _one_token, None, unroll=False)
        ref = dst_hbm.at[pl.ds(0, num_tokens)]
        pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=sem).wait()
        return None

    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def _local_batch_dma_kernel(
    src_hbm,
    dst_hbm,
    _output_hbm,
    sem,
    *,
    num_tokens: int,
    num_repeats: int,
):
    """Issue 1 DMA of size num_tokens, repeated num_repeats times."""

    def _repeat(_, __):
        pltpu.make_async_copy(
            src_ref=src_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            sem=sem,
        ).start()
        ref = dst_hbm.at[pl.ds(0, num_tokens)]
        pltpu.make_async_copy(src_ref=ref, dst_ref=ref, sem=sem).wait()
        return None

    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def _remote_small_dma_kernel(
    src_hbm,
    dst_hbm,
    _output_hbm,
    send_sem,
    recv_sem,
    *,
    num_tokens: int,
    num_repeats: int,
    tp_axis_name: str,
    dp_axis_name: str,
):
    """Issue num_tokens individual remote DMAs of size 1, repeated num_repeats times."""
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_rank = lax.axis_index(dp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = tp_size * dp_size
    target_id = (my_id + 1) % num_devices

    target_mesh_id = (target_id // tp_size, target_id % tp_size)

    def _repeat(_, __):
        def _one_token(t_id, ___):
            pltpu.make_async_remote_copy(
                src_ref=src_hbm.at[pl.ds(t_id, 1)],
                dst_ref=dst_hbm.at[pl.ds(t_id, 1)],
                send_sem=send_sem,
                recv_sem=recv_sem,
                device_id=target_mesh_id,
                device_id_type=pltpu.DeviceIdType.MESH,
            ).start()
            return None

        lax.fori_loop(0, num_tokens, _one_token, None, unroll=False)

        send_ref = src_hbm.at[pl.ds(0, num_tokens)]
        pltpu.make_async_copy(
            src_ref=send_ref, dst_ref=send_ref, sem=send_sem
        ).wait()

        recv_ref = dst_hbm.at[pl.ds(0, num_tokens)]
        pltpu.make_async_copy(
            src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem
        ).wait()
        return None

    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def _remote_batch_dma_kernel(
    src_hbm,
    dst_hbm,
    _output_hbm,
    send_sem,
    recv_sem,
    *,
    num_tokens: int,
    num_repeats: int,
    tp_axis_name: str,
    dp_axis_name: str,
):
    """Issue 1 remote DMA of size num_tokens, repeated num_repeats times."""
    tp_rank = lax.axis_index(tp_axis_name)
    tp_size = lax.axis_size(tp_axis_name)
    dp_rank = lax.axis_index(dp_axis_name)
    dp_size = lax.axis_size(dp_axis_name)
    my_id = dp_rank * tp_size + tp_rank
    num_devices = tp_size * dp_size
    target_id = (my_id + 1) % num_devices

    target_mesh_id = (target_id // tp_size, target_id % tp_size)

    def _repeat(_, __):
        pltpu.make_async_remote_copy(
            src_ref=src_hbm.at[pl.ds(0, num_tokens)],
            dst_ref=dst_hbm.at[pl.ds(0, num_tokens)],
            send_sem=send_sem,
            recv_sem=recv_sem,
            device_id=target_mesh_id,
            device_id_type=pltpu.DeviceIdType.MESH,
        ).start()

        send_ref = src_hbm.at[pl.ds(0, num_tokens)]
        pltpu.make_async_copy(
            src_ref=send_ref, dst_ref=send_ref, sem=send_sem
        ).wait()

        recv_ref = dst_hbm.at[pl.ds(0, num_tokens)]
        pltpu.make_async_copy(
            src_ref=recv_ref, dst_ref=recv_ref, sem=recv_sem
        ).wait()
        return None

    lax.fori_loop(0, num_repeats, _repeat, None, unroll=False)


def build_local_benchmark(num_tokens: int, hidden_size: int, num_repeats: int, mode: str):
    """Build a local DMA benchmark function."""
    t_packing = 2
    hidden_per_pack = hidden_size // t_packing
    dtype = jnp.bfloat16

    if mode == "small":
        kernel_fn = functools.partial(
            _local_small_dma_kernel,
            num_tokens=num_tokens,
            num_repeats=num_repeats,
        )
    else:
        kernel_fn = functools.partial(
            _local_batch_dma_kernel,
            num_tokens=num_tokens,
            num_repeats=num_repeats,
        )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    @jax.jit
    def run(src, dst):
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((1,), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm_spec, hbm_spec],
                out_specs=hbm_spec,
                scratch_shapes=(pltpu.SemaphoreType.DMA,),
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        )(src, dst)

    buf_shape = (num_tokens, t_packing, hidden_per_pack)
    src = jnp.zeros(buf_shape, dtype=dtype)
    dst = jnp.zeros(buf_shape, dtype=dtype)

    return run, src, dst


def build_remote_benchmark(
    num_tokens: int, hidden_size: int, num_repeats: int, mode: str, mesh
):
    """Build a remote DMA benchmark function."""
    t_packing = 2
    hidden_per_pack = hidden_size // t_packing
    dtype = jnp.bfloat16
    dp_axis_name = "data"
    tp_axis_name = "tensor"

    if mode == "small":
        kernel_fn = functools.partial(
            _remote_small_dma_kernel,
            num_tokens=num_tokens,
            num_repeats=num_repeats,
            tp_axis_name=tp_axis_name,
            dp_axis_name=dp_axis_name,
        )
    else:
        kernel_fn = functools.partial(
            _remote_batch_dma_kernel,
            num_tokens=num_tokens,
            num_repeats=num_repeats,
            tp_axis_name=tp_axis_name,
            dp_axis_name=dp_axis_name,
        )

    hbm_spec = pl.BlockSpec(memory_space=pltpu.MemorySpace.HBM)

    def _pallas_body(src_hbm, dst_hbm):
        return pl.pallas_call(
            kernel_fn,
            out_shape=jax.ShapeDtypeStruct((1,), dtype),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                in_specs=[hbm_spec, hbm_spec],
                out_specs=hbm_spec,
                scratch_shapes=(
                    pltpu.SemaphoreType.DMA,
                    pltpu.SemaphoreType.DMA,
                ),
            ),
            compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
        )(src_hbm, dst_hbm)

    @jax.jit
    def run(src, dst):
        return jax.experimental.shard_map.shard_map(
            _pallas_body,
            mesh=mesh,
            in_specs=(P("tensor"), P("tensor")),
            out_specs=P("tensor"),
            check_rep=False,
        )(src, dst)

    ep_size = mesh.shape["tensor"]
    buf_shape = (num_tokens * ep_size, t_packing, hidden_per_pack)
    src = jnp.zeros(buf_shape, dtype=dtype)
    dst = jnp.zeros(buf_shape, dtype=dtype)
    src = jax.device_put(src, jax.sharding.NamedSharding(mesh, P("tensor")))
    dst = jax.device_put(dst, jax.sharding.NamedSharding(mesh, P("tensor")))

    return run, src, dst


def run_benchmark(args):
    from jax._src.mesh_utils import create_device_mesh

    sizes = args.sizes
    hidden_size = args.hidden_size
    num_repeats = args.num_repeats
    warmup = args.warmup
    iters = args.iters

    results = []

    # Local DMA benchmarks
    print("=" * 70)
    print(f"LOCAL DMA benchmark (hidden_size={hidden_size})")
    print("=" * 70)
    print(f"{'tokens':>8} {'small_dma(ms)':>14} {'batch_dma(ms)':>14} {'speedup':>10}")
    print("-" * 50)

    for num_tokens in sizes:
        timings = {}
        for mode in ["small", "batch"]:
            run_fn, src, dst = build_local_benchmark(
                num_tokens, hidden_size, num_repeats, mode
            )
            task = f"local-{mode}-{num_tokens}"
            times = multiple_iteration_timeit_from_trace(
                compute_func=lambda s=src, d=dst: run_fn(s, d),
                data_generator=lambda: (),
                task=task,
                tries=iters,
                warmup=warmup,
            )
            if len(times) > 1:
                times = times[1:]
            timings[mode] = float(np.median(times)) if times else float("nan")

        speedup = (timings["small"] / timings["batch"] - 1) * 100 if timings["batch"] > 0 else 0
        print(
            f"{num_tokens:>8} {timings['small']:>14.4f} {timings['batch']:>14.4f} {speedup:>+9.1f}%"
        )
        results.append({
            "type": "local",
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "data_bytes": num_tokens * hidden_size * 2,
            "num_repeats": num_repeats,
            "small_dma_ms": timings["small"],
            "batch_dma_ms": timings["batch"],
            "speedup_pct": speedup,
        })

    # Remote DMA benchmarks
    if args.test_remote:
        ep_size = jax.device_count()
        if ep_size < 2:
            print("\nSkipping remote DMA: need at least 2 devices")
        else:
            mesh = create_device_mesh(
                ici_parallelism=[1, ep_size],
                dcn_parallelism=[1, 1],
                devices=jax.devices()[:ep_size],
                mesh_axes=("data", "tensor"),
            )
            print(f"\n{'=' * 70}")
            print(f"REMOTE DMA benchmark (hidden_size={hidden_size}, ep_size={ep_size})")
            print("=" * 70)
            print(f"{'tokens':>8} {'small_dma(ms)':>14} {'batch_dma(ms)':>14} {'speedup':>10}")
            print("-" * 50)

            for num_tokens in sizes:
                timings = {}
                for mode in ["small", "batch"]:
                    run_fn, src, dst = build_remote_benchmark(
                        num_tokens, hidden_size, num_repeats, mode, mesh
                    )
                    task = f"remote-{mode}-{num_tokens}"
                    times = multiple_iteration_timeit_from_trace(
                        compute_func=lambda s=src, d=dst: run_fn(s, d),
                        data_generator=lambda: (),
                        task=task,
                        tries=iters,
                        warmup=warmup,
                    )
                    if len(times) > 1:
                        times = times[1:]
                    timings[mode] = float(np.median(times)) if times else float("nan")

                speedup = (
                    (timings["small"] / timings["batch"] - 1) * 100
                    if timings["batch"] > 0
                    else 0
                )
                print(
                    f"{num_tokens:>8} {timings['small']:>14.4f} {timings['batch']:>14.4f} {speedup:>+9.1f}%"
                )
                results.append({
                    "type": "remote",
                    "num_tokens": num_tokens,
                    "hidden_size": hidden_size,
                    "ep_size": ep_size,
                    "data_bytes": num_tokens * hidden_size * 2,
                    "num_repeats": num_repeats,
                    "small_dma_ms": timings["small"],
                    "batch_dma_ms": timings["batch"],
                    "speedup_pct": speedup,
                })

    if args.output:
        with open(args.output, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"\nResults saved to {args.output}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="DMA size micro-benchmark")
    parser.add_argument("--hidden-size", type=int, default=7168)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    )
    parser.add_argument("--num-repeats", type=int, default=100,
                        help="Repeats within kernel to amortize launch overhead")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--test-remote", action="store_true",
                        help="Also test remote (ICI) DMAs")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL file path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
