# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


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
    if device == "tpu":
        devices = jax.local_devices()
        if empty_cache:
            jax.clear_caches()
        avail_mem = []
        for dev in devices:
            stats = dev.memory_stats()
            avail_mem.append(stats["bytes_limit"] - stats["bytes_in_use"])
        avail_mem = jnp.array([min(avail_mem) / (1 << 10)], dtype=jnp.float32)
    elif device == "cpu":
        import psutil

        memory = psutil.virtual_memory()
        free_gpu_memory = memory.available
        avail_mem = jnp.array([free_gpu_memory / (1 << 10)], dtype=jnp.float32)
    else:
        raise ValueError(f"Invalid device: {device}")

    if distributed:

        # Use pmap to find the minimum available memory across all devices.
        mesh = jax.make_mesh((jax.process_count(), 4), ("node", "device"))

        @jax.shard_map(
            mesh=mesh, in_specs=PartitionSpec(None), out_specs=PartitionSpec(None)
        )
        def _get_available_memory_distributed(a):
            return jax.lax.pmin(a, axis_name="node")

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
