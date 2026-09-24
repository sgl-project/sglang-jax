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
    # Preserve the production shard_map boundary: loc contains rank-local
    # indices. A global scatter aliases rank 1 writes into rank 0 on CPU.
    spec = P(data_partition_axis, None, kv_partition_axis, None, None)

    @jax.shard_map(
        in_specs=(spec, P(data_partition_axis), spec),
        out_specs=spec,
        mesh=mesh,
        check_vma=False,
    )
    def scatter(local_kv, local_loc, local_cache):
        flat = local_cache.reshape(
            (local_cache.shape[0] * local_cache.shape[1],) + local_cache.shape[2:]
        )
        safe = jnp.where(local_loc == -1, flat.shape[0], local_loc).astype(jnp.int32)
        flat = flat.at[safe].set(local_kv[:, 0], mode="drop")
        return flat.reshape(local_cache.shape)

    return scatter(fused_kv, loc, kv_cache)


if jax.default_backend() != "tpu":
    memory_pool.update_fused_kv_cache_vectorized = _cpu_fused_kv_scatter
