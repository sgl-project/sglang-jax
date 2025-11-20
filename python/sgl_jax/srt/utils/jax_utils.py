# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

GBYTES = 1024 * 1024 * 1024
TPU_HEAD_SIZE_ALIGNMENT = 128
TPU_SECOND_LAST_MINOR = 8


def get_device_name(num_devices: int | None = None):
    kind = jax.devices()[0].device_kind
    if "TPU" not in kind:
        raise RuntimeError("Expected TPU devices")
    suffix = ""
    if kind.endswith(" lite"):
        kind = kind[: -len(" lite")]
        suffix = "e"
    elif kind.endswith("e"):
        kind = kind[:-1]
        suffix = "e"
    elif kind.endswith("p"):
        kind = kind[:-1]
        suffix = "p"
    elif kind == "TPU7x":
        kind = "TPU v7"
    assert kind[:-1] == "TPU v", kind
    kind += suffix
    if num_devices is not None:
        kind += f"-{num_devices}"
    return kind


def get_device_hbm_limit() -> int:

    device_kind = get_device_name()
    if device_kind == "TPU v5p" or device_kind == "TPU v5":
        return 95 * GBYTES
    elif device_kind == "TPU v5e":
        return 16 * GBYTES
    elif device_kind == "TPU v6e" or device_kind == "TPU v4":
        return 32 * GBYTES
    elif device_kind == "TPU v7":
        # 192 * GBYTES / 2 because each JAX device (v7x core) has
        # 1/2 of the total chip HBM
        return 96 * GBYTES
    else:
        raise ValueError(f"Unknown device kind: {device_kind}")


def pathways_hbm_usage_gb(live_arrays, devices: Any) -> list[tuple[float, float]]:
    hbm_used = defaultdict(int)
    hbm_limit = get_device_hbm_limit()
    for array in live_arrays:
        for buffer in array.addressable_shards:
            hbm_used[buffer.data.device] += buffer.data.nbytes
    return [(hbm_used[device], hbm_limit) for device in devices]


def get_num_kv_heads_by_tp(total_num_kv_heads: int, tp_size: int) -> int:
    """
    Calculate the number of KV heads per device for tensor parallelism.
    Args:
        total_num_kv_heads: Total number of KV heads in the model
        tp_size: Tensor parallel size (number of devices)
    Returns:
        Number of KV heads per device
    """
    if tp_size >= total_num_kv_heads:
        # When tp_size >= total_kv_heads, each device gets 1 KV head
        # Multiple devices will replicate the same original KV head
        return 1
    else:
        # Normal case: divide KV heads across devices
        return (total_num_kv_heads + tp_size - 1) // tp_size


def get_original_kv_head_id(tp_rank: int, total_num_kv_heads: int, tp_size: int) -> int:
    """
    Determine which original KV head this device should replicate.

    Args:
        tp_rank: Current device rank (0-based)
        total_num_kv_heads: Total number of KV heads in the model
        tp_size: Tensor parallel size

    Returns:
        ID of the original KV head to replicate (0-based)
    """
    if tp_size > total_num_kv_heads:
        # KV head replication case: multiple devices share the same original KV head
        num_kv_head_replicas = (tp_size + total_num_kv_heads - 1) // total_num_kv_heads
        return tp_rank // num_kv_head_replicas
    else:
        # Normal case: each device gets a different range of KV heads
        kv_heads_per_device = get_num_kv_heads_by_tp(total_num_kv_heads, tp_size)
        return (tp_rank * kv_heads_per_device) % total_num_kv_heads


def get_available_device_memory(device, distributed=False, empty_cache=True):
    """
    Get available memory for device:device_id.
    When distributed is True, the available memory is the minimum available memory of all devices.
    """
    if device in ("tpu", "proxy"):
        devices = jax.local_devices()
        if empty_cache:
            jax.clear_caches()
        avail_mem = []
        for dev in devices:
            stats = dev.memory_stats()
            avail_mem.append(stats["bytes_limit"] - stats["bytes_in_use"])
        avail_mem = jnp.array([min(avail_mem) / (1 << 10)], dtype=jnp.float32)
    elif device in ("proxy"):
        live_arrays = jax.live_arrays()
        pathways_hbm_used_mem = pathways_hbm_usage_gb(live_arrays, devices)
        avail_mem = [(hbm_limit - hbm_used) for hbm_used, hbm_limit in pathways_hbm_used_mem]
    elif device in ("gpu", "cuda"):
        if empty_cache:
            jax.clear_caches()
        devices = [d for d in jax.local_devices() if getattr(d, "platform", None) == "gpu"]
        if not devices:
            raise RuntimeError("No GPU devices found by JAX")
        avail = []
        for dev in devices:
            stats = dev.memory_stats()
            avail.append(stats["bytes_limit"] - stats["bytes_in_use"])
        avail_mem = jnp.array([min(avail) / (1 << 10)], dtype=jnp.float32)
    elif device == "cpu":
        import psutil

        memory = psutil.virtual_memory()
        free_gpu_memory = memory.available
        avail_mem = jnp.array([free_gpu_memory / (1 << 10)], dtype=jnp.float32)
    else:
        raise ValueError(f"Invalid device: {device}")

    if distributed:
        # Use pmap to find the minimum available memory across all devices.
        mesh = jax.make_mesh((jax.device_count(),), ("device"))

        @jax.shard_map(mesh=mesh, in_specs=PartitionSpec(None), out_specs=PartitionSpec(None))
        def _get_available_memory_distributed(a):
            return jax.lax.pmin(a, axis_name="device")

        # We broadcast the local min memory to all devices and then find the global min.
        # i64 dtype cannot be all-reduce min
        assert (
            avail_mem.dtype != jnp.float64 and avail_mem.dtype != jnp.int64
        ), "avail_mem must be i32 dtype"
        global_min_mem = _get_available_memory_distributed(avail_mem)[0]
        free_gpu_memory = global_min_mem.item()
    else:
        free_gpu_memory = avail_mem.min().item()

    return int(free_gpu_memory * (1 << 10))


def device_array(*data, sharding=None, **kwargs) -> jax.Array:
    return jax.device_put(*data, device=sharding, **kwargs)


def is_tpu_runtime() -> bool:
    """Return True if the current JAX runtime is on TPU devices.

    Prefer checking actual devices; fall back to default backend if necessary.
    """
    try:
        devs = jax.devices()
        return len(devs) > 0 and all(d.platform == "tpu" for d in devs)
    except Exception:
        return jax.default_backend() == "tpu"


def print_memory(stage_name):
    """Print current memory usage"""
    memory = get_memory_usage()
    print(f"\n[{stage_name}] Memory usage:")
    for device, usage in memory.items():
        print(f"  {device}: {usage}GB" if isinstance(usage, float) else f"  {device}: {usage}")
    return memory


def get_memory_usage():
    """Get actual memory usage if available"""
    try:
        stats = {}
        for i, device in enumerate(jax.devices()):
            try:
                device_stats = device.memory_stats()
                stats[f"device_{i}"] = device_stats.get("bytes_in_use", 0) / (1024**3)
            except Exception:
                stats[f"device_{i}"] = "N/A"
        return stats
    except Exception:
        return {f"device_{i}": "N/A" for i in range(len(jax.devices()))}
