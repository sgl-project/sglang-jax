# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


def get_num_kv_heads_by_tp(num_kv_heads: int, tp_size: int) -> int:
    if tp_size <= num_kv_heads:
        assert num_kv_heads % tp_size == 0
        return num_kv_heads
    else:
        assert tp_size % num_kv_heads == 0
        return tp_size


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

        with jax.sharding.set_mesh(mesh=mesh):

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


def device_array(mesh, data, sharding=None, **kwargs) -> jax.Array:
    if sharding is None:
        sharding = NamedSharding(mesh, PartitionSpec(None))
    return jax.device_put(data, device=sharding, **kwargs)
