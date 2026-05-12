"""Run only the remote DMA micro-benchmark (skip local)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from benchmark.moe.bench_dma_size import build_remote_benchmark, wallclock_timeit

P = jax.sharding.PartitionSpec

ep_size = jax.device_count()
mesh = create_device_mesh(
    ici_parallelism=[1, ep_size],
    dcn_parallelism=[1, 1],
    devices=jax.devices()[:ep_size],
    mesh_axes=("data", "tensor"),
)

hidden_size = 6144
num_repeats = 100
sizes = [1, 4, 16, 64, 128]

print("=" * 70)
print(f"REMOTE DMA benchmark (hidden_size={hidden_size}, ep_size={ep_size}, wallclock)")
print("=" * 70)
print(f"{'tokens':>8} {'small_dma(ms)':>14} {'batch_dma(ms)':>14} {'speedup':>10}")
print("-" * 50)

for num_tokens in sizes:
    timings = {}
    for mode in ["small", "batch"]:
        run_fn, src, dst = build_remote_benchmark(
            num_tokens, hidden_size, num_repeats, mode, mesh
        )
        times = wallclock_timeit(
            run_fn, src, dst, num_repeats=num_repeats, warmup=3, iters=5
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

print("\n=== REMOTE DMA BENCHMARK DONE ===")
