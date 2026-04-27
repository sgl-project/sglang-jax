"""JIT cache stability for RecurrentStatePool / MemoryPools pytree aux.

Pytree aux determines the JIT cache key. If aux is unhashable, hash-unstable,
or not equal-by-value across same-parameter instances, every call re-traces
and re-compiles -- a perf disaster on the hot path.

These tests use a trace counter (a Python list mutated inside the JIT'd
function body) to count how many times the function is RE-TRACED (compiled).
Trace executes the Python body once on the first call with new abstract
inputs; subsequent calls with cache-equivalent inputs SKIP the trace.

Coverage:
- Same pool instance reused -> cache hit (baseline).
- Different pool instances with identical constructor args -> cache hit
  (proves aux is equal-by-value, not identity-based).
- Pool with diverging max_num_reqs / num_heads / num_k_heads / dtypes
  -> cache miss (proves cache key actually discriminates).
- MemoryPools wrapping equivalent pools -> cache hit.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MHATokenToKVPool
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool


def _mesh():
    return Mesh(np.array(jax.devices()), ("tensor",))


def _make_recurrent_pool(**overrides):
    defaults = dict(
        linear_recurrent_layer_ids=[0, 1],
        max_num_reqs=4,
        num_heads=2,
        head_dim=4,
        conv_kernel_size=4,
        mesh=_mesh(),
    )
    defaults.update(overrides)
    return RecurrentStatePool(**defaults)


class _BaseJitCache(unittest.TestCase):
    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")


class TestRecurrentStatePoolJitCacheKey(_BaseJitCache):
    """Pytree aux must produce a stable, equal-by-value JIT cache key."""

    def _make_traced_jit(self):
        """Return (jitted, trace_counter) where trace_counter[0] increments
        once per re-trace (cache miss)."""
        trace_counter = [0]

        @jax.jit
        def f(pool):
            trace_counter[0] += 1
            return pool.recurrent_buffers[0].sum()

        return f, trace_counter

    def test_same_instance_reused_caches(self):
        """Sanity: identical pool reference -> single trace."""
        f, counter = self._make_traced_jit()
        pool = _make_recurrent_pool()
        f(pool).block_until_ready()
        f(pool).block_until_ready()
        self.assertEqual(counter[0], 1, "Same pool reuse must cache-hit")

    def test_distinct_instances_same_params_share_cache(self):
        """Two distinct pool instances built with identical constructor args
        must hit the same JIT cache entry. Proves aux is equal-by-value
        (not identity-based) across constructions."""
        f, counter = self._make_traced_jit()
        pool_a = _make_recurrent_pool()
        pool_b = _make_recurrent_pool()
        self.assertIsNot(pool_a, pool_b, "Sanity: distinct instances")
        f(pool_a).block_until_ready()
        f(pool_b).block_until_ready()
        self.assertEqual(
            counter[0],
            1,
            "Distinct pools with identical params must share a cache entry "
            "-- aux must be equal-by-value across constructions.",
        )

    def test_different_max_num_reqs_misses_cache(self):
        """max_num_reqs change -> different leaf abstract shape -> cache miss."""
        f, counter = self._make_traced_jit()
        f(_make_recurrent_pool(max_num_reqs=4)).block_until_ready()
        f(_make_recurrent_pool(max_num_reqs=8)).block_until_ready()
        self.assertEqual(
            counter[0],
            2,
            "Different max_num_reqs must produce different cache entries "
            "(buffer shape changes).",
        )

    def test_different_num_heads_misses_cache(self):
        """num_heads change -> different recurrent_buffers shape -> cache miss."""
        f, counter = self._make_traced_jit()
        f(_make_recurrent_pool(num_heads=2)).block_until_ready()
        f(_make_recurrent_pool(num_heads=4)).block_until_ready()
        self.assertEqual(counter[0], 2, "Different num_heads must miss cache")

    def test_different_num_k_heads_misses_cache(self):
        """num_k_heads change -> proj_size change -> conv_buffers shape
        change -> cache miss. Locks the new GQA-API parameter into the
        cache key."""
        f, counter = self._make_traced_jit()
        f(_make_recurrent_pool(num_heads=4, head_dim=4, num_k_heads=4)).block_until_ready()
        f(_make_recurrent_pool(num_heads=4, head_dim=4, num_k_heads=2)).block_until_ready()
        self.assertEqual(
            counter[0],
            2,
            "Different num_k_heads must miss cache (proj_size diverges)",
        )

    def test_different_temporal_dtype_misses_cache(self):
        """dtype change -> different leaf abstract dtype -> cache miss."""
        f, counter = self._make_traced_jit()
        f(_make_recurrent_pool(temporal_dtype=jnp.float32)).block_until_ready()
        f(_make_recurrent_pool(temporal_dtype=jnp.bfloat16)).block_until_ready()
        self.assertEqual(counter[0], 2, "Different dtype must miss cache")


class TestMemoryPoolsJitCacheKey(_BaseJitCache):
    """MemoryPools wraps both pools as a pytree node; aux must be stable."""

    def _make_kv_pool(self, head_num=2):
        return MHATokenToKVPool(
            size=8,
            page_size=1,
            dtype=jnp.bfloat16,
            head_num=head_num,
            head_dim=4,
            layer_num=2,
            mesh=_mesh(),
        )

    def _make_memory_pools(self, **rsp_overrides):
        kv_overrides = {}
        if "num_heads" in rsp_overrides:
            kv_overrides["head_num"] = rsp_overrides["num_heads"]
        return MemoryPools(
            token_to_kv_pool=self._make_kv_pool(**kv_overrides),
            recurrent_state_pool=_make_recurrent_pool(**rsp_overrides),
        )

    def test_distinct_memory_pools_same_params_share_cache(self):
        """Two MemoryPools built from independent (but identically-configured)
        sub-pools must hit the same cache entry."""
        trace_counter = [0]

        @jax.jit
        def f(mp):
            trace_counter[0] += 1
            return mp.recurrent_state_pool.recurrent_buffers[0].sum()

        mp_a = self._make_memory_pools()
        mp_b = self._make_memory_pools()
        f(mp_a).block_until_ready()
        f(mp_b).block_until_ready()
        self.assertEqual(
            trace_counter[0],
            1,
            "Distinct MemoryPools with identical sub-pool configs must share " "a cache entry.",
        )


if __name__ == "__main__":
    unittest.main()
