"""Benchmark recurrent-state CoW clone against the former JAX scatter path.

Run this on a TPU host. The roofline is a lower bound: each clone must read one
slot and write one slot, so the minimum transfer is ``2 * state_bytes``.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.h0_clone import clone_slots_inplace


def slow_clone(buffer, src, dst):
    payload_dims = (1,) * (buffer.ndim - 1)
    values = jnp.where((src == 0).reshape((-1,) + payload_dims), buffer[dst], buffer[src])
    return buffer.at[dst].set(values)


def _time_us(fn, tries: int) -> float:
    for _ in range(3):
        jax.block_until_ready(fn())
    elapsed = []
    for _ in range(tries):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        elapsed.append((time.perf_counter() - start) * 1e6)
    return min(elapsed)


def _dtype_bytes(dtype) -> int:
    return jnp.dtype(dtype).itemsize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--slots", type=int, default=128)
    parser.add_argument("--clones", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--conv-proj", type=int, default=8192)
    parser.add_argument("--conv-history", type=int, default=3)
    parser.add_argument("--peak-hbm-gbps", type=float, required=True)
    parser.add_argument("--tries", type=int, default=20)
    args = parser.parse_args()

    if not (0 < args.clones and 2 * args.clones < args.slots):
        raise ValueError("--clones must be positive and leave distinct source/destination slots")
    if args.conv_history < 1:
        raise ValueError("--conv-history must be positive")

    temporal = jnp.zeros(
        (args.slots, args.heads, args.head_dim, args.head_dim), dtype=jnp.float32
    )
    conv = jnp.zeros((args.slots, args.conv_proj, args.conv_history), dtype=jnp.bfloat16)
    src = jnp.arange(1, args.clones + 1, dtype=jnp.int32)
    dst = jnp.arange(args.clones + 1, 2 * args.clones + 1, dtype=jnp.int32)

    fast_temporal = jax.jit(clone_slots_inplace)
    fast_conv = jax.jit(clone_slots_inplace)
    slow_temporal = jax.jit(slow_clone)
    slow_conv = jax.jit(slow_clone)

    def fast():
        temporal_out = fast_temporal(temporal, src, dst)
        return fast_conv(conv, src, dst), temporal_out

    def slow():
        temporal_out = slow_temporal(temporal, src, dst)
        return slow_conv(conv, src, dst), temporal_out

    fast_us = _time_us(fast, args.tries)
    slow_us = _time_us(slow, args.tries)
    slot_bytes = (
        args.heads * args.head_dim * args.head_dim * _dtype_bytes(jnp.float32)
        + args.conv_proj * args.conv_history * _dtype_bytes(jnp.bfloat16)
    )
    state_bytes = args.layers * args.clones * slot_bytes
    roofline_us = 2 * state_bytes / (args.peak_hbm_gbps * 1e9) * 1e6
    fast_total_us = fast_us * args.layers
    slow_total_us = slow_us * args.layers
    fast_gbps = (2 * args.clones * slot_bytes) / fast_us / 1e3

    print(f"device={jax.devices()[0]}")
    print(f"layers={args.layers} cloned_slots={args.clones} slot_bytes={slot_bytes:,}")
    print(f"state_bytes={state_bytes:,} roofline_us={roofline_us:.2f}")
    print(
        f"per_layer: slow_us={slow_us:.2f} fast_us={fast_us:.2f} speedup={slow_us / fast_us:.2f}x"
    )
    print(f"estimated_all_layers: slow_us={slow_total_us:.2f} fast_us={fast_total_us:.2f}")
    print(f"fast_effective_bandwidth_gbps={fast_gbps:.1f}")
    if fast_us >= slow_us:
        raise SystemExit("FAIL: Pallas clone did not beat the slow full-buffer scatter path")


if __name__ == "__main__":
    main()
