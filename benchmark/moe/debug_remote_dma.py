"""Minimal reproducer for remote DMA Anomalies crash."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp
import numpy as np
import time

from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from benchmark.moe.bench_dma_size import build_remote_benchmark

P = jax.sharding.PartitionSpec

ep_size = jax.device_count()
print(f"ep_size={ep_size}")

mesh = create_device_mesh(
    ici_parallelism=[1, ep_size],
    dcn_parallelism=[1, 1],
    devices=jax.devices()[:ep_size],
    mesh_axes=("data", "tensor"),
)

num_tokens = 1
num_repeats = 100

print(f"Building small DMA benchmark for {num_tokens} tokens...")
run_fn, src, dst = build_remote_benchmark(num_tokens, 6144, num_repeats, "small", mesh)
print(f"src shape={src.shape}, dst shape={dst.shape}")
print(f"src sharding={src.sharding}")

print("Warmup 1...")
out = run_fn(src, dst)
jax.block_until_ready(out)
print("Warmup 1 OK")

print("Warmup 2...")
out = run_fn(src, dst)
jax.block_until_ready(out)
print("Warmup 2 OK")

print("Timing...")
start = time.perf_counter()
out = run_fn(src, dst)
jax.block_until_ready(out)
elapsed = (time.perf_counter() - start) * 1000 / num_repeats
print(f"small_dma: {elapsed:.4f} ms/repeat")

print("Building batch DMA benchmark...")
run_fn2, src2, dst2 = build_remote_benchmark(num_tokens, 6144, num_repeats, "batch", mesh)

print("Warmup...")
out = run_fn2(src2, dst2)
jax.block_until_ready(out)
print("Warmup OK")

start = time.perf_counter()
out = run_fn2(src2, dst2)
jax.block_until_ready(out)
elapsed = (time.perf_counter() - start) * 1000 / num_repeats
print(f"batch_dma: {elapsed:.4f} ms/repeat")

print("=== DONE ===")
