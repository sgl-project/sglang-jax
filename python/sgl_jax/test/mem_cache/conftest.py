"""CPU test shim for the fused-KV in-place write.

``update_fused_kv_cache_vectorized`` lowers to a Mosaic/Pallas kernel that only
exists on TPU. Production code keeps the TPU-only path; here we monkeypatch a
pure-JAX scatter with identical semantics so the mem_cache unit tests stay
CPU-runnable. On TPU we leave the real kernel in place.

Patched at conftest import time (before pytest imports the test modules), so
both ``test_kv_cache`` (direct ``from ... import``) and ``test_host_kv_pool``
(via the jitted ``write_kv_layer``) pick up the shim.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache import memory_pool


def _cpu_fused_kv_scatter(
    fused_kv,
    loc,
    kv_cache,
    page_size,
    kv_partition_axis="tensor",
    data_partition_axis="data",
    mesh=None,
):
    flat_spec = P(data_partition_axis, kv_partition_axis, None, None)
    cache_spec = P(data_partition_axis, None, kv_partition_axis, None, None)
    if mesh is not None:
        flat_out = NamedSharding(mesh, flat_spec)
        cache_out = NamedSharding(mesh, cache_spec)
    else:
        flat_out, cache_out = flat_spec, cache_spec
    flat = jax.lax.reshape(
        kv_cache,
        (kv_cache.shape[0] * kv_cache.shape[1],) + tuple(kv_cache.shape[2:]),
        out_sharding=flat_out,
    )
    loc_i = loc.astype(jnp.int32)
    safe = jnp.where(loc_i == -1, flat.shape[0], loc_i)
    flat = flat.at[safe].set(fused_kv[:, 0], mode="drop", out_sharding=flat_out)
    return jax.lax.reshape(flat, kv_cache.shape, out_sharding=cache_out)


if jax.default_backend() != "tpu":
    memory_pool.update_fused_kv_cache_vectorized = _cpu_fused_kv_scatter
