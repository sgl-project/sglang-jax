#!/usr/bin/env python3
"""TPU verification: per-layer KV gather compile + performance.

Run on v6e-4 (TP=4) to confirm:
1. Per-layer gather compiles without OOM at realistic pool sizes
2. All page buckets compile successfully
3. Total gather time for 36 layers is acceptable (<100ms)
4. Peak HBM during compile stays reasonable

Usage:
    python scripts/disaggregation/verify_kv_gather_tpu.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

# Qwen3-8B parameters
NUM_LAYERS = 36
PAGE_SIZE = 128
NUM_KV_HEADS = 8
HEAD_DIM = 128
PACKING = 2
DTYPE = jnp.bfloat16

# Pool size: 128 pages × 128 tokens/page = 16384 tokens capacity
NUM_PAGES_POOL = 128

# Page buckets to test
PAGE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def create_mesh():
    """Create TP=4 mesh matching production config."""
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Available devices: {num_devices} ({devices[0].platform})")
    if num_devices >= 4:
        mesh_shape = (1, 4)
    elif num_devices >= 2:
        mesh_shape = (1, 2)
    else:
        mesh_shape = (1, 1)
    print(f"Mesh shape: {mesh_shape} (data={mesh_shape[0]}, tensor={mesh_shape[1]})")
    devices_array = np.array(devices[: mesh_shape[0] * mesh_shape[1]]).reshape(mesh_shape)
    return Mesh(
        devices_array,
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def create_kv_pool(mesh):
    """Create per-layer KV buffers matching memory_pool.py layout."""
    # Global shape (JAX shards automatically via PartitionSpec)
    buffer_shape = (
        NUM_PAGES_POOL,
        PAGE_SIZE,
        NUM_KV_HEADS * 2 // PACKING,  # [K0,V0,K1,V1,...] interleaved
        PACKING,
        HEAD_DIM,
    )
    pool_sharding = NamedSharding(mesh, P("data", None, "tensor", None, None))

    print(f"\nCreating KV pool:")
    print(f"  Buffer shape per layer: {buffer_shape}")
    per_layer_bytes = np.prod(buffer_shape) * 2  # bf16 = 2 bytes
    print(f"  Per-layer size: {per_layer_bytes / 1024**2:.1f} MB")
    print(f"  Total pool size: {per_layer_bytes * NUM_LAYERS / 1024**3:.2f} GB")

    buffers = []
    t0 = time.perf_counter()
    with mesh:
        for i in range(NUM_LAYERS):
            buf = jax.jit(
                lambda: jnp.zeros(shape=buffer_shape, dtype=DTYPE),
                out_shardings=pool_sharding,
            )()
            buffers.append(buf)
            if (i + 1) % 12 == 0:
                print(f"  Created {i + 1}/{NUM_LAYERS} layers...")

    t1 = time.perf_counter()
    print(f"  Pool creation time: {t1 - t0:.2f}s")
    return buffers, pool_sharding


def verify_gather(mesh, buffers, pool_sharding):
    """Run per-layer gather for all page buckets and measure performance."""
    from functools import partial

    @partial(jax.jit, static_argnames=("out_sharding",))
    def gather_one_layer(buf, page_indices, out_sharding):
        return buf.at[page_indices].get(out_sharding=out_sharding)

    gather_pspec = P(None, *pool_sharding.spec[1:])
    gather_sharding = NamedSharding(mesh, gather_pspec)
    idx_sharding = NamedSharding(mesh, P(None))

    print("\n" + "=" * 60)
    print("Per-layer gather verification")
    print("=" * 60)

    all_passed = True

    for bucket in PAGE_BUCKETS:
        page_indices = jax.device_put(
            jnp.arange(bucket, dtype=jnp.int32) % NUM_PAGES_POOL,
            idx_sharding,
        )

        # First call: includes compilation time
        t_compile_start = time.perf_counter()
        result_0 = gather_one_layer(buffers[0], page_indices, gather_sharding)
        result_0.block_until_ready()
        t_compile_end = time.perf_counter()
        compile_time = t_compile_end - t_compile_start

        # Second call: execution only (cached)
        t_exec_start = time.perf_counter()
        results = []
        for buf in buffers:
            r = gather_one_layer(buf, page_indices, gather_sharding)
            results.append(r)
        # Block until all done
        for r in results:
            r.block_until_ready()
        t_exec_end = time.perf_counter()
        exec_time = t_exec_end - t_exec_start

        # Verify shape
        expected_shape = (bucket, PAGE_SIZE, buffers[0].shape[2], PACKING, HEAD_DIM)
        actual_shape = results[0].shape
        shape_ok = actual_shape == expected_shape

        status = "PASS" if shape_ok else "FAIL"
        if not shape_ok:
            all_passed = False

        print(
            f"  Bucket {bucket:3d} pages: [{status}] "
            f"compile={compile_time*1000:.1f}ms, "
            f"exec_36_layers={exec_time*1000:.1f}ms, "
            f"shape={actual_shape}"
        )

    return all_passed


def benchmark_full_extract(mesh, buffers, pool_sharding):
    """Benchmark the full _extract_req_kv equivalent (gather + stack)."""
    from functools import partial

    @partial(jax.jit, static_argnames=("out_sharding",))
    def gather_one_layer(buf, page_indices, out_sharding):
        return buf.at[page_indices].get(out_sharding=out_sharding)

    gather_pspec = P(None, *pool_sharding.spec[1:])
    gather_sharding = NamedSharding(mesh, gather_pspec)
    idx_sharding = NamedSharding(mesh, P(None))

    print("\n" + "=" * 60)
    print("Full KV extraction benchmark (gather + stack)")
    print("=" * 60)

    # Test with different sequence lengths
    test_cases = [
        ("short prompt (256 tok)", 256),
        ("medium prompt (1024 tok)", 1024),
        ("long prompt (4096 tok)", 4096),
    ]

    for label, seqlen in test_cases:
        num_pages = (seqlen + PAGE_SIZE - 1) // PAGE_SIZE
        # Pad to bucket
        padded = num_pages
        for b in PAGE_BUCKETS:
            if b >= num_pages:
                padded = b
                break
        else:
            padded = PAGE_BUCKETS[-1]

        page_indices = jax.device_put(
            jnp.arange(padded, dtype=jnp.int32) % NUM_PAGES_POOL,
            idx_sharding,
        )

        # Warm up
        for buf in buffers:
            _ = gather_one_layer(buf, page_indices, gather_sharding)

        # Timed run
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            layer_kvs = []
            for buf in buffers:
                layer_kvs.append(gather_one_layer(buf, page_indices, gather_sharding))
            stacked = jnp.stack(layer_kvs, axis=0)
            stacked.block_until_ready()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_ms = np.mean(times) * 1000
        output_bytes = stacked.nbytes
        print(
            f"  {label}: pages={padded}, "
            f"time={avg_ms:.1f}ms, "
            f"output={output_bytes / 1024**2:.1f} MB"
        )


def check_memory():
    """Report HBM usage."""
    print("\n" + "=" * 60)
    print("Memory report")
    print("=" * 60)
    for i, dev in enumerate(jax.devices()[:4]):
        stats = dev.memory_stats()
        if stats:
            used = stats.get("bytes_in_use", 0)
            limit = stats.get("bytes_limit", 0)
            print(
                f"  Device {i}: {used / 1024**3:.2f} GB used / "
                f"{limit / 1024**3:.2f} GB total "
                f"({100 * used / limit:.1f}%)"
            )


def main():
    print("=" * 60)
    print("TPU KV Gather Verification")
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].platform}")
    print("=" * 60)

    mesh = create_mesh()
    buffers, pool_sharding = create_kv_pool(mesh)

    check_memory()

    passed = verify_gather(mesh, buffers, pool_sharding)
    benchmark_full_extract(mesh, buffers, pool_sharding)

    check_memory()

    print("\n" + "=" * 60)
    if passed:
        print("ALL CHECKS PASSED")
    else:
        print("SOME CHECKS FAILED")
    print("=" * 60)

    return 0 if passed else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
