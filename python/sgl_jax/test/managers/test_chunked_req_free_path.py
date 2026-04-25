"""Phase 4: scheduler.get_next_batch_to_run chunked_req free path branches by pool type.

For hybrid models (HybridReqToTokenPool, identified via duck typing on
free_recurrent_cache): skip the free, let HybridReqToTokenPool.alloc reuse
both req_pool_idx and recurrent_pool_idx next round.

For non-hybrid models: keep the existing free, but ALSO clear
chunked_req.req_pool_idx = None so the new alloc(reqs) list path treats
this req as fresh (D2).
"""

import inspect
import unittest


class TestChunkedReqFreePathBranches(unittest.TestCase):
    def test_helper_source_contains_hybrid_hasattr_branch(self):
        """_maybe_free_chunked_req_slot must check
        hasattr(req_to_token_pool, 'free_recurrent_cache') (D1)."""
        from sgl_jax.srt.managers.scheduler import _maybe_free_chunked_req_slot

        src = inspect.getsource(_maybe_free_chunked_req_slot)
        self.assertIn(
            'hasattr(req_to_token_pool, "free_recurrent_cache")',
            src,
            "_maybe_free_chunked_req_slot must duck-type-detect HybridReqToTokenPool "
            "via hasattr(...,'free_recurrent_cache') (D1)",
        )

    def test_helper_source_clears_stale_req_pool_idx_in_non_hybrid(self):
        """_maybe_free_chunked_req_slot non-hybrid path must clear
        chunked_req.req_pool_idx = None (D2: alloc(reqs) list path would
        otherwise treat stale ref as reuse and silently return a freed slot)."""
        from sgl_jax.srt.managers.scheduler import _maybe_free_chunked_req_slot

        src = inspect.getsource(_maybe_free_chunked_req_slot)
        self.assertIn(
            "chunked_req.req_pool_idx = None",
            src,
            "Non-hybrid free path must clear chunked_req.req_pool_idx after free",
        )

    def test_scheduler_calls_helper(self):
        """get_next_batch_to_run must dispatch chunked_req release through
        the helper (not call pool.free directly)."""
        from sgl_jax.srt.managers.scheduler import Scheduler

        src = inspect.getsource(Scheduler.get_next_batch_to_run)
        self.assertIn(
            "_maybe_free_chunked_req_slot(self.req_to_token_pool, self.chunked_req)",
            src,
            "get_next_batch_to_run must release the chunked_req slot via "
            "_maybe_free_chunked_req_slot helper",
        )
        self.assertNotIn(
            "self.req_to_token_pool.free(self.chunked_req.req_pool_idx)",
            src,
            "Old unconditional free of chunked_req.req_pool_idx must be removed; "
            "the helper now decides whether to free.",
        )


class TestChunkedReqFreePathBehavior(unittest.TestCase):
    """Functional behavior with a stand-in helper extracted from get_next_batch_to_run."""

    def test_hybrid_path_skips_free(self):
        """When pool exposes free_recurrent_cache, the chunked free is skipped."""
        from types import SimpleNamespace

        from sgl_jax.srt.managers.scheduler import _maybe_free_chunked_req_slot

        free_calls = []
        # Hybrid stub: has free_recurrent_cache (and free, but free should NOT be called).
        hybrid_pool = SimpleNamespace(
            free=lambda idx: free_calls.append(("free", idx)),
            free_recurrent_cache=lambda req: None,
        )
        chunked_req = SimpleNamespace(req_pool_idx=42)
        _maybe_free_chunked_req_slot(hybrid_pool, chunked_req)
        self.assertEqual(free_calls, [], "Hybrid path must skip pool.free for chunked_req")
        self.assertEqual(
            chunked_req.req_pool_idx,
            42,
            "Hybrid path must keep req_pool_idx so HybridReqToTokenPool.alloc reuses",
        )

    def test_non_hybrid_path_frees_and_clears(self):
        """When pool lacks free_recurrent_cache, free is called AND
        chunked_req.req_pool_idx is cleared (D2)."""
        from types import SimpleNamespace

        from sgl_jax.srt.managers.scheduler import _maybe_free_chunked_req_slot

        free_calls = []
        # Non-hybrid: only `free` (no free_recurrent_cache attr).
        non_hybrid_pool = SimpleNamespace(
            free=lambda idx: free_calls.append(("free", idx)),
        )
        chunked_req = SimpleNamespace(req_pool_idx=42)
        _maybe_free_chunked_req_slot(non_hybrid_pool, chunked_req)
        self.assertEqual(
            free_calls,
            [("free", 42)],
            "Non-hybrid path must call pool.free(req_pool_idx)",
        )
        self.assertIsNone(
            chunked_req.req_pool_idx,
            "Non-hybrid path must clear req_pool_idx after free (D2)",
        )


if __name__ == "__main__":
    unittest.main()
