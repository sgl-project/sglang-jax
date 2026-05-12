"""Run only the remote DMA micro-benchmark (skip local).

Runs each (mode, num_tokens) pair independently to avoid TPU semaphore
conflicts between different pallas_call compilations.
"""
import subprocess
import sys
import os
import json
import re
import numpy as np


def run_single(num_tokens, hidden_size, num_repeats, mode, ep_size):
    """Run a single benchmark in a subprocess and parse the result."""
    script = f"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname("{__file__}"), "..", ".."))
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from benchmark.moe.bench_dma_size import build_remote_benchmark, wallclock_timeit
P = jax.sharding.PartitionSpec
mesh = create_device_mesh(
    ici_parallelism=[1, {ep_size}], dcn_parallelism=[1, 1],
    devices=jax.devices()[:{ep_size}], mesh_axes=("data", "tensor"),
)
run_fn, src, dst = build_remote_benchmark({num_tokens}, {hidden_size}, {num_repeats}, "{mode}", mesh)
times = wallclock_timeit(run_fn, src, dst, num_repeats={num_repeats}, warmup=3, iters=5)
if len(times) > 1:
    times = times[1:]
med = float(np.median(times))
print(f"RESULT={med:.6f}")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        stderr_tail = result.stderr.strip().split("\n")[-5:]
        print(f"  ERROR [{mode}]: {'; '.join(stderr_tail)}", file=sys.stderr)
        return float("nan")

    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT="):
            return float(line.split("=")[1])
    return float("nan")


def main():
    import jax
    ep_size = jax.device_count()
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
            timings[mode] = run_single(
                num_tokens, hidden_size, num_repeats, mode, ep_size
            )

        speedup = (
            (timings["small"] / timings["batch"] - 1) * 100
            if timings["batch"] > 0 and not np.isnan(timings["batch"])
            else 0
        )
        print(
            f"{num_tokens:>8} {timings['small']:>14.4f} {timings['batch']:>14.4f} {speedup:>+9.1f}%"
        )

    print("\n=== REMOTE DMA BENCHMARK DONE ===")


if __name__ == "__main__":
    main()

