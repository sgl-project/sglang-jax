#!/usr/bin/env python3
"""TPU mock for speculative overlap device-dependency profiling.

This intentionally does not model sglang request scheduling. It isolates the
device submission shape we care about:

    verify_i -> draft_extend_i -> verify_{i+1}

Run it on a TPU pod with JAX available and inspect the trace in xprof.
"""

from __future__ import annotations

import argparse
import os
import time

import jax
import jax.numpy as jnp


@jax.jit
def mock_verify(x, w):
    with jax.named_scope("mock_verify_body"):
        y = x
        for _ in range(4):
            y = jnp.tanh(y @ w)
        return y


@jax.jit
def mock_draft_extend(x, w):
    with jax.named_scope("mock_draft_extend_body"):
        y = x @ w
        return jnp.tanh(y)


@jax.jit
def mock_draft_then_verify(x, draft_w, verify_w):
    with jax.named_scope("mock_draft_then_verify_body"):
        y = jnp.tanh(x @ draft_w)
        for _ in range(4):
            y = jnp.tanh(y @ verify_w)
        return y


def _make_inputs(size: int):
    x = jnp.ones((size, size), dtype=jnp.bfloat16)
    w = jnp.eye(size, dtype=jnp.bfloat16)
    return x, w


def _block_until_ready(x):
    jax.block_until_ready(x)
    return x


def run_split(steps: int, size: int):
    state, w = _make_inputs(size)
    with jax.profiler.TraceAnnotation("mock_warmup_split"):
        for _ in range(3):
            state = mock_draft_extend(mock_verify(state, w), w)
        _block_until_ready(state)

    for step in range(steps):
        with jax.profiler.TraceAnnotation(f"mock_submit_verify {step}"):
            verify_out = mock_verify(state, w)
        with jax.profiler.TraceAnnotation(f"mock_submit_draft_extend {step}"):
            state = mock_draft_extend(verify_out, w)
    _block_until_ready(state)


def run_fused_chain(steps: int, size: int):
    state, w = _make_inputs(size)
    with jax.profiler.TraceAnnotation("mock_warmup_fused_chain"):
        for _ in range(3):
            state = mock_draft_then_verify(state, w, w)
        _block_until_ready(state)

    for step in range(steps):
        with jax.profiler.TraceAnnotation(f"mock_submit_draft_then_verify {step}"):
            state = mock_draft_then_verify(state, w, w)
    _block_until_ready(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--mode", choices=("split", "fused-chain"), default="split")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--coordinator-address")
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--process-id", type=int, default=0)
    args = parser.parse_args()

    if args.num_processes > 1:
        if not args.coordinator_address:
            raise ValueError("--coordinator-address is required for distributed mock")
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
        )

    os.makedirs(args.profile_dir, exist_ok=True)
    print(
        f"process_id={args.process_id}/{args.num_processes} devices={jax.devices()}",
        flush=True,
    )
    print(
        f"mode={args.mode} steps={args.steps} size={args.size} profile_dir={args.profile_dir}",
        flush=True,
    )

    if args.process_id == 0:
        jax.profiler.start_trace(args.profile_dir)
    start = time.time()
    if args.mode == "split":
        run_split(args.steps, args.size)
    else:
        run_fused_chain(args.steps, args.size)
    elapsed = time.time() - start
    if args.process_id == 0:
        jax.profiler.stop_trace()
    print(f"done elapsed={elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
