"""Phase 5: release_req must call free_recurrent_cache for hybrid pools.

Static source check + behavioral check via SimpleNamespace mock pool.
release_req has two free paths (radix_cache_disabled True/False); both
must invoke hybrid release adjacent to the KV slot free.
"""

import inspect
import unittest


class TestReleaseReqSourceHasHybridBranches(unittest.TestCase):
    def test_release_req_calls_free_recurrent_cache_in_both_paths(self):
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.release_req)
        # Both KV free paths must follow with a hybrid hasattr branch + call.
        self.assertGreaterEqual(
            src.count("_maybe_free_recurrent_cache(self.req_to_token_pool"),
            2,
            "release_req must call _maybe_free_recurrent_cache(self.req_to_token_pool, req) "
            "in both free paths (radix-disabled and radix-enabled)",
        )


class TestReleaseReqHybridBehavior(unittest.TestCase):
    """Functional check via direct invocation of the hybrid branch logic
    extracted as a module-level helper (mirrors Phase 4's
    _maybe_free_chunked_req_slot pattern)."""

    def test_helper_exists_and_calls_free_recurrent_cache_when_hybrid(self):
        from types import SimpleNamespace

        from sgl_jax.srt.managers.schedule_batch import _maybe_free_recurrent_cache

        free_calls = []
        hybrid_pool = SimpleNamespace(
            free_recurrent_cache=lambda r: free_calls.append(r),
        )
        req = SimpleNamespace(req_pool_idx=7, recurrent_pool_idx=3)
        _maybe_free_recurrent_cache(hybrid_pool, req)
        self.assertEqual(len(free_calls), 1)
        self.assertIs(free_calls[0], req)

    def test_helper_is_noop_when_non_hybrid(self):
        from types import SimpleNamespace

        from sgl_jax.srt.managers.schedule_batch import _maybe_free_recurrent_cache

        non_hybrid_pool = SimpleNamespace()  # no free_recurrent_cache attr
        req = SimpleNamespace(req_pool_idx=7, recurrent_pool_idx=None)
        # Must not raise; must not mutate.
        _maybe_free_recurrent_cache(non_hybrid_pool, req)


if __name__ == "__main__":
    unittest.main()
