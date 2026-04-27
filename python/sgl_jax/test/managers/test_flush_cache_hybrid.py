""": flush_cache must reset hybrid recurrent state via existing path.

scheduler.flush_cache calls self.req_to_token_pool.clear(); for hybrid
pools, HybridReqToTokenPool.clear  chains super().clear() +
recurrent_state_pool.clear(). This test guards that chain end-to-end
without exercising the full Scheduler.flush_cache path.

D5: no src change in this Task -- only a guard against future refactors
that might break the inheritance chain.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


def _mesh():
    """Single-device mesh with the canonical "tensor" axis name; matches the
    sharding axis RecurrentStatePool partitions H / proj_size on."""
    return Mesh(np.array(jax.devices()), ("tensor",))


class TestHybridReqToTokenPoolClearResetsRecurrentState(unittest.TestCase):
    """Direct invocation of HybridReqToTokenPool.clear -- the same call
    scheduler.flush_cache makes via self.req_to_token_pool.clear()."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

    def _make_pool(self):
        from sgl_jax.srt.mem_cache.memory_pool import HybridReqToTokenPool
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        rsp = RecurrentStatePool(
            linear_recurrent_layer_ids=[0, 1],
            max_num_reqs=4,
            num_heads=2,
            head_dim=4,
            conv_kernel_size=4,
            mesh=_mesh(),
        )
        return HybridReqToTokenPool(
            size=5, max_context_len=16, dtype=np.int32, recurrent_state_pool=rsp
        )

    def test_clear_resets_kv_recurrent_and_mapping(self):
        from types import SimpleNamespace

        pool = self._make_pool()
        # Allocate one slot; write non-zero data into recurrent buffer.
        req = SimpleNamespace(req_pool_idx=None, recurrent_pool_idx=None, is_chunked=0)
        pool.alloc([req])
        rec_idx = req.recurrent_pool_idx
        # Mutate buffer at the allocated slot to verify clear actually zeroes.
        layer0_buf, _conv = pool.recurrent_state_pool.get_linear_recurrent_layer_cache(0)
        pool.recurrent_state_pool.recurrent_buffers[0] = layer0_buf.at[rec_idx].set(7.0)

        # flush_cache -> req_to_token_pool.clear() -> HybridReqToTokenPool.clear
        pool.clear()

        # Recurrent buffer at the previously-allocated slot is zero again.
        layer0_buf_after, _ = pool.recurrent_state_pool.get_linear_recurrent_layer_cache(0)
        self.assertTrue(
            bool(jnp.all(layer0_buf_after[rec_idx] == 0)),
            "HybridReqToTokenPool.clear must zero the recurrent buffer slot",
        )
        # Mapping is reset to dummy slot 0.
        self.assertEqual(
            int(pool.req_index_to_recurrent_index_mapping[req.req_pool_idx]),
            0,
            "Mapping must be reset to dummy slot 0 after clear",
        )
        # KV-side free_slots also reset (parent class chain).
        # ReqToTokenPool.clear resets to all slots (range(size)); does not
        # reserve slot 0 (only RecurrentStatePool reserves slot 0 as dummy).
        self.assertEqual(
            len(pool.free_slots),
            pool.size,
            "Parent ReqToTokenPool free_slots must be fully replenished by clear",
        )


class TestSchedulerFlushCacheCallsClear(unittest.TestCase):
    """Static check: scheduler.flush_cache still calls req_to_token_pool.clear()
    after  refactors (defensive guard against future code drift)."""

    def test_flush_cache_source_calls_req_to_token_pool_clear(self):
        import inspect

        from sgl_jax.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.flush_cache)
        self.assertIn(
            "self.req_to_token_pool.clear()",
            src,
            "scheduler.flush_cache must dispatch to req_to_token_pool.clear() "
            "(HybridReqToTokenPool.clear chains recurrent reset; do not bypass)",
        )


if __name__ == "__main__":
    unittest.main()
