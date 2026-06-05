"""TPU/JAX profile harness for spec-overlap scheduling bubbles.

This is intentionally not a server benchmark. It initializes the same
multi-process JAX runtime used by the 4-rank TPU pod, runs tiny synthetic JAX
kernels named like verify/draft_extend, and profiles whether host scheduling
work leaves the TPU lane idle between verify launches.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

Strategy = Literal["current", "target_future_relay", "device_chain_scheduler_thread"]


@jax.jit
def mock_verify_kernel(x, w, loops: int):
    def body(_, y):
        return jnp.tanh(y @ w)

    y = jax.lax.fori_loop(0, loops, body, x)
    accept = (jnp.sum(y, axis=1) > 0).astype(jnp.int32) + 1
    return y, accept


@jax.jit
def mock_draft_extend_kernel(x, accept, w, loops: int):
    scale = 1.0 + accept.astype(x.dtype)[:, None] * jnp.asarray(0.001, x.dtype)
    y = x * scale

    def body(_, cur):
        return jnp.tanh(cur @ w)

    return jax.lax.fori_loop(0, loops, body, y)


@jax.jit
def mock_metadata_device_put_kernel(x):
    return jnp.clip(x + jnp.asarray(1, x.dtype), 0, 4096)


def _host_delay(ms: float) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def _make_inputs(global_batch: int, width: int):
    key = jax.random.PRNGKey(0)
    x_key, w_key = jax.random.split(key)
    x = jax.random.normal(x_key, (global_batch, width), dtype=jnp.bfloat16)
    w = jax.random.normal(w_key, (width, width), dtype=jnp.bfloat16) / jnp.sqrt(
        jnp.asarray(width, dtype=jnp.bfloat16)
    )
    return x, w


def _warmup(x, w, verify_loops: int, draft_loops: int):
    with jax.profiler.TraceAnnotation("mock_warmup"):
        x, accept = mock_verify_kernel(x, w, verify_loops)
        x = mock_draft_extend_kernel(x, accept, w, draft_loops)
        x.block_until_ready()
    return x


def _run_current(
    x,
    w,
    *,
    steps: int,
    verify_loops: int,
    draft_loops: int,
    phase_a_ms: float,
    scheduler_ms: float,
    metadata_ms: float,
):
    for step in range(steps):
        with jax.profiler.TraceAnnotation(f"mock_current_verify_step_{step}"):
            x, accept = mock_verify_kernel(x, w, verify_loops)

        with jax.profiler.TraceAnnotation(f"mock_current_phase_a_d2h_step_{step}"):
            np.asarray(accept)

        with jax.profiler.TraceAnnotation(f"mock_current_prepare_draft_extend_step_{step}"):
            _host_delay(phase_a_ms)

        with jax.profiler.TraceAnnotation(f"mock_current_draft_extend_step_{step}"):
            x = mock_draft_extend_kernel(x, accept, w, draft_loops)

        with jax.profiler.TraceAnnotation(f"mock_current_materialize_phase_b_step_{step}"):
            x.block_until_ready()

        with jax.profiler.TraceAnnotation(f"mock_current_scheduler_run_batch_step_{step}"):
            _host_delay(scheduler_ms)

        with jax.profiler.TraceAnnotation(f"mock_current_metadata_device_put_step_{step}"):
            meta = mock_metadata_device_put_kernel(accept)
            meta.block_until_ready()
            _host_delay(metadata_ms)
    return x


def _run_target_future_relay(
    x,
    w,
    *,
    steps: int,
    verify_loops: int,
    draft_loops: int,
    scheduler_ms: float,
    metadata_ms: float,
):
    with jax.profiler.TraceAnnotation("mock_target_initial_verify"):
        x, accept = mock_verify_kernel(x, w, verify_loops)

    for step in range(steps):
        with jax.profiler.TraceAnnotation(f"mock_target_draft_extend_step_{step}"):
            x = mock_draft_extend_kernel(x, accept, w, draft_loops)

        with jax.profiler.TraceAnnotation(f"mock_target_next_verify_step_{step}"):
            next_x, next_accept = mock_verify_kernel(x, w, verify_loops)

        with jax.profiler.TraceAnnotation(f"mock_target_scheduler_phase_a_step_{step}"):
            jax.copy_to_host_async(accept)
            _host_delay(scheduler_ms)

        with jax.profiler.TraceAnnotation(f"mock_target_future_metadata_step_{step}"):
            mock_metadata_device_put_kernel(accept)
            _host_delay(metadata_ms)

        x, accept = next_x, next_accept
    return x


def _run_device_chain_scheduler_thread(
    x,
    w,
    *,
    steps: int,
    verify_loops: int,
    draft_loops: int,
    scheduler_ms: float,
    metadata_ms: float,
):
    phase_a_queue: queue.Queue[tuple[int, jax.Array] | None] = queue.Queue()

    def scheduler_loop() -> None:
        while True:
            item = phase_a_queue.get()
            if item is None:
                phase_a_queue.task_done()
                return
            step, accept = item
            with jax.profiler.TraceAnnotation(f"mock_device_chain_scheduler_phase_a_step_{step}"):
                jax.copy_to_host_async(accept)
                _host_delay(scheduler_ms)
            with jax.profiler.TraceAnnotation(f"mock_device_chain_future_metadata_step_{step}"):
                mock_metadata_device_put_kernel(accept)
                _host_delay(metadata_ms)
            phase_a_queue.task_done()

    scheduler_thread = threading.Thread(
        target=scheduler_loop,
        name="mock-device-chain-scheduler",
    )
    scheduler_thread.start()

    try:
        with jax.profiler.TraceAnnotation("mock_device_chain_initial_verify"):
            x, accept = mock_verify_kernel(x, w, verify_loops)

        for step in range(steps):
            phase_a_queue.put((step, accept))
            with jax.profiler.TraceAnnotation(f"mock_device_chain_draft_extend_step_{step}"):
                x = mock_draft_extend_kernel(x, accept, w, draft_loops)
            with jax.profiler.TraceAnnotation(f"mock_device_chain_next_verify_step_{step}"):
                x, accept = mock_verify_kernel(x, w, verify_loops)
    finally:
        phase_a_queue.put(None)
        phase_a_queue.join()
        scheduler_thread.join()
    return x


def _initialize_distributed(dist_init_addr: str | None, nnodes: int, node_rank: int) -> None:
    if nnodes <= 1:
        return
    if not dist_init_addr:
        raise ValueError("--dist-init-addr is required when --nnodes > 1")
    if not jax.distributed.is_initialized():
        jax.distributed.initialize(dist_init_addr, nnodes, node_rank)


def run_mock_tpu_overlap(
    *,
    strategy: Strategy,
    steps: int,
    global_batch: int,
    width: int,
    verify_loops: int,
    draft_loops: int,
    scheduler_ms: float,
    metadata_ms: float,
    profile_dir: str | None,
    initialize_distributed: bool,
    dist_init_addr: str | None = None,
    nnodes: int = 1,
    node_rank: int = 0,
    phase_a_ms: float = 2.0,
) -> dict:
    if initialize_distributed:
        _initialize_distributed(dist_init_addr, nnodes, node_rank)

    process_index = jax.process_index()
    x, w = _make_inputs(global_batch, width)
    x = _warmup(x, w, verify_loops, draft_loops)

    if profile_dir and process_index == 0:
        Path(profile_dir).mkdir(parents=True, exist_ok=True)
        jax.profiler.start_trace(profile_dir)

    started = time.perf_counter()
    if strategy == "current":
        x = _run_current(
            x,
            w,
            steps=steps,
            verify_loops=verify_loops,
            draft_loops=draft_loops,
            phase_a_ms=phase_a_ms,
            scheduler_ms=scheduler_ms,
            metadata_ms=metadata_ms,
        )
    elif strategy == "target_future_relay":
        x = _run_target_future_relay(
            x,
            w,
            steps=steps,
            verify_loops=verify_loops,
            draft_loops=draft_loops,
            scheduler_ms=scheduler_ms,
            metadata_ms=metadata_ms,
        )
    elif strategy == "device_chain_scheduler_thread":
        x = _run_device_chain_scheduler_thread(
            x,
            w,
            steps=steps,
            verify_loops=verify_loops,
            draft_loops=draft_loops,
            scheduler_ms=scheduler_ms,
            metadata_ms=metadata_ms,
        )
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    x.block_until_ready()
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if profile_dir and process_index == 0:
        jax.profiler.stop_trace()

    result = {
        "strategy": strategy,
        "steps": steps,
        "process_index": process_index,
        "process_count": jax.process_count(),
        "local_device_count": jax.local_device_count(),
        "device_count": jax.device_count(),
        "elapsed_ms": elapsed_ms,
        "profile_dir": profile_dir,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("current", "target_future_relay", "device_chain_scheduler_thread"),
        required=True,
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--global-batch", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--verify-loops", type=int, default=8)
    parser.add_argument("--draft-loops", type=int, default=1)
    parser.add_argument("--phase-a-ms", type=float, default=2.0)
    parser.add_argument("--scheduler-ms", type=float, default=4.0)
    parser.add_argument("--metadata-ms", type=float, default=2.0)
    parser.add_argument("--profile-dir", type=str)
    parser.add_argument("--dist-init-addr", type=str)
    parser.add_argument("--nnodes", type=int, default=int(os.getenv("NNODES", "1")))
    parser.add_argument("--node-rank", type=int, default=int(os.getenv("NODE_RANK", "0")))
    parser.add_argument("--no-distributed", action="store_true")
    args = parser.parse_args()

    run_mock_tpu_overlap(
        strategy=args.strategy,
        steps=args.steps,
        global_batch=args.global_batch,
        width=args.width,
        verify_loops=args.verify_loops,
        draft_loops=args.draft_loops,
        phase_a_ms=args.phase_a_ms,
        scheduler_ms=args.scheduler_ms,
        metadata_ms=args.metadata_ms,
        profile_dir=args.profile_dir,
        initialize_distributed=not args.no_distributed,
        dist_init_addr=args.dist_init_addr,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
    )


if __name__ == "__main__":
    main()
