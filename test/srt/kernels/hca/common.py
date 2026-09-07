"""Shared production-shape setup for HCA TPU tests."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.mem_cache.hca_allocator import HCAKVPoolAllocator
from sgl_jax.srt.mem_cache.hca_pool import HCAKVPool, HCARecurrentStatePool
from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool


class HCATestFactory:
    """Build isolated meshes, pools, requests, and production-shape inputs."""

    @staticmethod
    def mesh():
        return jax.sharding.Mesh(
            np.asarray(jax.devices()[:1], object).reshape(1, 1),
            ("data", "tensor"),
            axis_types=(
                jax.sharding.AxisType.Explicit,
                jax.sharding.AxisType.Explicit,
            ),
        )

    @staticmethod
    def request():
        return SimpleNamespace(
            req_pool_idx=None,
            recurrent_pool_idx=None,
            is_chunked=0,
            kv_committed_len=0,
            dp_rank=0,
        )

    def runtime(self, *, requests=2, max_context_len=512, layer_ids=(0,)):
        mesh = self.mesh()
        with jax.set_mesh(mesh):
            state_pool = HCARecurrentStatePool(layer_ids, requests, mesh)
            kv_pool = HCAKVPool(
                max(requests * max_context_len, 512),
                128,
                jnp.bfloat16,
                len(layer_ids),
                mesh,
                max_num_requests=requests,
                max_context_len=max_context_len,
                layer_ids=layer_ids,
            )
            request_pool = HybridReqToTokenPool(
                requests,
                max_context_len,
                np.int32,
                state_pool,
                dp_size=1,
            )
            allocator = HCAKVPoolAllocator(kv_pool, request_pool)
        return mesh, kv_pool, state_pool, request_pool, allocator

    @staticmethod
    def inputs(mesh, tokens: int, *, seed: int = 20260814):
        key = jax.random.key(seed)

        def put(value, spec):
            return jax.device_put(value, NamedSharding(mesh, spec))

        hidden = put(jax.random.normal(key, (tokens, 4096), jnp.bfloat16), P("data", None))
        q = put(
            jax.random.normal(jax.random.fold_in(key, 1), (tokens, 64, 512), jnp.bfloat16),
            P("data", "tensor", None),
        )
        new_kv = put(
            jax.random.normal(jax.random.fold_in(key, 2), (tokens, 512), jnp.bfloat16),
            P("data", None),
        )
        wkv = put(
            jax.random.normal(jax.random.fold_in(key, 3), (512, 4096), jnp.bfloat16),
            P(None, None),
        )
        wgate = put(
            jax.random.normal(jax.random.fold_in(key, 4), (512, 4096), jnp.bfloat16),
            P(None, None),
        )
        ape = put(
            jax.random.normal(jax.random.fold_in(key, 5), (128, 512), jnp.float32),
            P(None, None),
        )
        norm = put(jnp.ones((512,), jnp.bfloat16), P(None))
        # Cover every absolute position the batch can reach; rows 0..511 stay
        # identical to the previous fixed-size table.
        rope_rows = max(512, tokens)
        angle = jnp.arange(rope_rows * 32, dtype=jnp.float32).reshape(rope_rows, 32) * 1e-3
        cos = put(jnp.cos(angle), P(None, None))
        sin = put(jnp.sin(angle), P(None, None))
        sink = put(jnp.zeros((64,), jnp.float32), P("tensor"))
        return hidden, q, new_kv, wkv, wgate, ape, norm, cos, sin, sink


__all__ = ["HCATestFactory"]
