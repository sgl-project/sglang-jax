"""Kernel-only GDN prefill A/B benchmark for the Qwen3.5 TP16 shard."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import Mesh

from sgl_jax.srt.kernels.gdn import (
    ragged_gated_delta_rule_chunkwise,
    ragged_gated_delta_rule_ref,
)
from sgl_jax.srt.layers.attention.linear.gdn_backend import GDNAttnBackend


SCHEMA_VERSION = 1
SELECTOR_ENV = "SGLANG_JAX_GDN_PREFILL_IMPL"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
FIXTURE = {"seed": 324, "n_kq": 1, "n_v": 2, "d_k": 128, "d_v": 128}
QWEN_GLOBAL = {"n_kq": 16, "n_v": 32, "d_k": 128, "d_v": 128, "conv_kernel_size": 4}
HARDWARE = {
    "accelerator": "TPU v6e-16",
    "topology": "4x4",
    "nodes": 4,
    "chips_per_node": 4,
    "jax_devices": 16,
    "dtype": "bfloat16",
}
SERVE_ARGS = [
    "--trust-remote-code",
    "--device=tpu",
    "--dtype=bfloat16",
    "--random-seed=3",
    "--tp-size=16",
    "--data-parallel-size=4",
    "--ep-size=16",
    "--dp-schedule-policy=min_running_queue",
    "--nnodes=4",
    "--dist-init-addr=perf-16-0.perf-16-headless-svc:5000",
    "--mem-fraction-static=0.90",
    "--chunked-prefill-size=512",
    "--page-size=64",
    "--max-running-requests=64",
    "--disable-radix-cache",
    "--disable-overlap-schedule",
    "--precompile-bs-paddings=16",
    "--precompile-token-paddings=16,512,1024",
    "--skip-server-warmup",
    "--host=0.0.0.0",
    "--port=30000",
]


def _parse_lengths(value: str) -> list[int]:
    try:
        lengths = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("--lengths must be comma-separated integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("--lengths must contain positive integers")
    return lengths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark real GDN reference and chunkwise prefill recurrence kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--impl", choices=("reference", "chunkwise"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--lengths", type=_parse_lengths, default=_parse_lengths("4096,32768"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--code-revision", default=None)
    parser.add_argument("--coordinator-address", default=None)
    parser.add_argument("--num-processes", type=int, default=None)
    parser.add_argument("--process-id", type=int, default=None)
    args = parser.parse_args()
    if args.rank < 0:
        parser.error("--rank must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    distributed = (args.coordinator_address, args.num_processes, args.process_id)
    if any(value is not None for value in distributed) and not all(value is not None for value in distributed):
        parser.error("--coordinator-address, --num-processes, and --process-id must be supplied together")
    return args


def _revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=Path(__file__).parents[3]
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _resolve_code_revision(
    supplied: str | None, git_revision: str, *, performance: bool
) -> tuple[str, str]:
    git_known = bool(re.fullmatch(r"[0-9a-f]{40}", git_revision))
    if supplied is None:
        if performance:
            raise ValueError("performance mode requires --code-revision with exactly 40 hex characters")
        return (git_revision, "git") if git_known else ("unknown", "unknown")
    if not re.fullmatch(r"[0-9a-f]{40}", supplied):
        raise ValueError("code revision must contain exactly 40 hex characters")
    if git_known and supplied != git_revision:
        raise ValueError("supplied code revision must match local git HEAD")
    return supplied, "supplied+git" if git_known else "supplied"


def _initialize_distributed(args: argparse.Namespace) -> None:
    if args.coordinator_address is not None:
        jax.distributed.initialize(
            coordinator_address=args.coordinator_address,
            num_processes=args.num_processes,
            process_id=args.process_id,
        )


def _runtime_hardware() -> dict[str, Any]:
    devices = jax.devices()
    return {
        "process_count": jax.process_count(),
        "device_count": jax.device_count(),
        "local_device_count": jax.local_device_count(),
        "process_index": jax.process_index(),
        "device_kind": devices[0].device_kind if devices else "unknown",
        "device_platforms": sorted({device.platform.lower() for device in devices}),
    }


def _validate_rank_permutations(
    requested_ids: list[int], pjrt_process_indices: list[int], process_count: int
) -> dict[str, list[int]]:
    expected = list(range(process_count))

    def validate(field: str, values: list[int]) -> list[int]:
        observed = [int(value) for value in values]
        if len(observed) != process_count or sorted(observed) != expected:
            raise RuntimeError(
                f"{field} must be the complete unique permutation {expected}, got {observed}"
            )
        return sorted(observed)

    return {
        "requested_ids": validate("requested IDs", requested_ids),
        "pjrt_process_indices": validate("PJRT process indices", pjrt_process_indices),
    }


def _gather_rank_identities(requested_id: int, pjrt_process_index: int) -> np.ndarray:
    local_identity = jnp.asarray([requested_id, pjrt_process_index], dtype=jnp.int32)
    gathered = np.asarray(multihost_utils.process_allgather(local_identity, tiled=False))
    if gathered.size % 2:
        raise RuntimeError(f"rank identity collective returned invalid shape {gathered.shape}")
    return gathered.reshape(-1, 2)


def _validate_performance_runtime(runtime: dict[str, Any], rank: int) -> dict[str, Any]:
    if (
        runtime["process_count"] != 4
        or runtime["device_count"] != 16
        or runtime["local_device_count"] != 4
        or runtime["device_platforms"] != ["tpu"]
        or "tpu v6" not in runtime["device_kind"].lower()
    ):
        raise RuntimeError(f"performance runtime does not match TPU v6e-16 contract: {runtime}")

    gathered = _gather_rank_identities(rank, runtime["process_index"])
    expected_shape = (runtime["process_count"], 2)
    if gathered.shape != expected_shape:
        raise RuntimeError(
            f"rank identity collective must return shape {expected_shape}, got {gathered.shape}"
        )
    permutations = _validate_rank_permutations(
        requested_ids=gathered[:, 0].tolist(),
        pjrt_process_indices=gathered[:, 1].tolist(),
        process_count=runtime["process_count"],
    )
    return {
        "requested_id": int(rank),
        "pjrt_process_index": int(runtime["process_index"]),
        **permutations,
        "identity_pairs": [
            {"requested_id": int(requested), "pjrt_process_index": int(pjrt)}
            for requested, pjrt in gathered
        ],
    }


def _variant(impl: str) -> str:
    return "reference" if impl == "reference" else "optimized"


def _paths(impl: str, rank: int) -> tuple[str, str]:
    variant = _variant(impl)
    return (
        f"/tmp/beaver-324/jax-cache/i5/{variant}/rank-{rank}",
        f"/tmp/beaver-324/profiler/i5/{variant}/rank-{rank}",
    )


def _runtime_mesh() -> Mesh:
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(devices.reshape(1, devices.size), ("data", "tensor"))


def _selector(impl: str) -> dict[str, Any]:
    if os.environ.get(SELECTOR_ENV) != impl:
        raise RuntimeError(f"{SELECTOR_ENV} must equal --impl={impl!r}")
    backend = GDNAttnBackend(
        num_k_heads=QWEN_GLOBAL["n_kq"],
        num_v_heads=QWEN_GLOBAL["n_v"],
        head_k_dim=QWEN_GLOBAL["d_k"],
        head_v_dim=QWEN_GLOBAL["d_v"],
        conv_kernel_size=QWEN_GLOBAL["conv_kernel_size"],
        mesh=_runtime_mesh(),
    )
    expected_callable = (
        ragged_gated_delta_rule_ref if impl == "reference" else ragged_gated_delta_rule_chunkwise
    )
    if (
        backend.requested_impl != impl
        or backend.effective_impl != impl
        or backend.fallback_reason is not None
        or backend._prefill_callable is not expected_callable
    ):
        raise RuntimeError(
            "GDNAttnBackend selector triple/callable does not match requested variant: "
            f"requested={backend.requested_impl!r}, effective={backend.effective_impl!r}, "
            f"fallback={backend.fallback_reason!r}"
        )
    return {
        "requested_impl": backend.requested_impl,
        "effective_impl": backend.effective_impl,
        "fallback_reason": backend.fallback_reason,
        "selector": {
            "environment": SELECTOR_ENV,
            "source": "GDNAttnBackend",
            "selected_callable": expected_callable.__name__,
        },
    }


def _fixture(length: int) -> dict[str, jax.Array]:
    key = jax.random.PRNGKey(FIXTURE["seed"] + length)
    keys = jax.random.split(key, 6)
    n_kq, n_v, d_k, d_v = (FIXTURE[name] for name in ("n_kq", "n_v", "d_k", "d_v"))
    return {
        "mixed_qkv": jax.random.normal(keys[0], (length, 2 * n_kq * d_k + n_v * d_v), jnp.bfloat16),
        "b": jax.random.normal(keys[1], (length, n_v), jnp.bfloat16),
        "a": jax.random.normal(keys[2], (length, n_v), jnp.bfloat16),
        "recurrent_state": jax.random.normal(keys[3], (2, n_v, d_k, d_v), jnp.float32),
        "A_log": jax.random.normal(keys[4], (n_v,), jnp.float32) * 0.1,
        "dt_bias": jax.random.normal(keys[5], (n_v,), jnp.float32) * 0.1,
        "cu_seqlens": jnp.asarray([0, length], dtype=jnp.int32),
        "state_indices": jnp.asarray([1], dtype=jnp.int32),
        "has_initial_state": jnp.asarray([False]),
    }


def _kernel(function: Callable[..., tuple[jax.Array, jax.Array]]) -> Callable[[dict[str, jax.Array]], tuple[jax.Array, jax.Array]]:
    return jax.jit(
        lambda values: function(
            values["mixed_qkv"], values["b"], values["a"], values["recurrent_state"],
            values["A_log"], values["dt_bias"], values["cu_seqlens"],
            values["state_indices"], values["has_initial_state"],
            n_kq=FIXTURE["n_kq"], n_v=FIXTURE["n_v"], d_k=FIXTURE["d_k"], d_v=FIXTURE["d_v"],
        )
    )


def _ready(outputs: tuple[jax.Array, jax.Array]) -> None:
    jax.block_until_ready(outputs)


def _correctness(reference, chunkwise, values: dict[str, jax.Array]) -> dict[str, bool]:
    reference_result, chunkwise_result = reference(values), chunkwise(values)
    _ready(reference_result)
    _ready(chunkwise_result)
    reference_state, reference_output = (np.asarray(item) for item in reference_result)
    chunkwise_state, chunkwise_output = (np.asarray(item) for item in chunkwise_result)
    correctness = {
        "identical_inputs": True,
        "outputs_allclose": bool(np.allclose(reference_output, chunkwise_output, rtol=2e-2, atol=1e-2)),
        "states_allclose": bool(np.allclose(reference_state, chunkwise_state, rtol=2e-2, atol=1e-2)),
        "finite": bool(
            np.isfinite(reference_output).all() and np.isfinite(chunkwise_output).all()
            and np.isfinite(reference_state).all() and np.isfinite(chunkwise_state).all()
        ),
    }
    if not all(correctness.values()):
        raise RuntimeError("reference and chunkwise GDN results failed the correctness contract")
    return correctness


def _time_kernel(function, values: dict[str, jax.Array], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        _ready(function(values))
    timings = []
    for _ in range(iterations):
        started = time.perf_counter()
        _ready(function(values))
        timings.append((time.perf_counter() - started) * 1000.0)
    return timings


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite existing output: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = _parse_args()
    _initialize_distributed(args)
    runtime_hardware = _runtime_hardware()
    device_platform = runtime_hardware["device_platforms"][0]
    interpret = os.environ.get("PALLAS_INTERPRET", "").lower() == "true"
    performance = device_platform == "tpu" and not interpret
    rank_identity = {
        "requested_id": args.rank,
        "pjrt_process_index": runtime_hardware["process_index"],
        "validation": "not-run-correctness-only",
    }
    if performance:
        rank_identity = _validate_performance_runtime(runtime_hardware, args.rank)
    code_revision, code_revision_source = _resolve_code_revision(
        args.code_revision, _revision(), performance=performance
    )
    selector = _selector(args.impl)
    if performance and args.lengths != [4096, 32768]:
        raise RuntimeError("performance records require exactly --lengths 4096,32768")
    cache_directory, profiler_directory = _paths(args.impl, args.rank)
    runtime_env = {
        "JAX_COMPILATION_CACHE_DIR": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "SGLANG_JAX_PROFILER_DIR": os.environ.get("SGLANG_JAX_PROFILER_DIR"),
    }
    if performance and runtime_env != {
        "JAX_COMPILATION_CACHE_DIR": cache_directory,
        "SGLANG_JAX_PROFILER_DIR": profiler_directory,
    }:
        raise RuntimeError("performance timing requires exact cache and profiler environment paths")

    reference = _kernel(ragged_gated_delta_rule_ref)
    chunkwise = _kernel(ragged_gated_delta_rule_chunkwise)
    selected = reference if args.impl == "reference" else chunkwise
    per_length = []
    for length in args.lengths:
        values = _fixture(length)
        correctness = _correctness(reference, chunkwise, values)
        timings = _time_kernel(selected, values, args.warmup, args.iterations)
        if not all(math.isfinite(timing) and timing > 0 for timing in timings):
            raise RuntimeError("non-positive or non-finite timing observed")
        per_length.append({
            "length": length, "correctness": correctness, "raw_iteration_ms": timings,
            "median_ms": statistics.median(timings), "finite": True,
        })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "rank": args.rank,
        "requested_id": args.rank,
        "pjrt_process_index": runtime_hardware["process_index"],
        "rank_identity": rank_identity,
        **selector,
        "device_platform": device_platform,
        "hardware": HARDWARE,
        "evidence": {"classification": "performance" if performance else "correctness-only"},
        "deterministic_fixture": {
            **FIXTURE, "mixed_qkv_shape": [None, 512], "a_shape": [None, 2],
            "b_shape": [None, 2], "state_slot_shape": [2, 128, 128],
        },
        "code_revision": code_revision,
        "code_revision_source": code_revision_source,
        "model_revision": MODEL_REVISION,
        "serve_args": SERVE_ARGS,
        "cache_directory": cache_directory,
        "profiler_directory": profiler_directory,
        "runtime_env": runtime_env,
        "runtime_hardware": runtime_hardware,
        "lengths": per_length,
        "command": ["python", *os.sys.argv],
    }
    _atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
