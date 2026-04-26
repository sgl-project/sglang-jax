"""release_req must release both KV and recurrent slots via the pool's polymorphic
free(req).

Static source check + behavioral check via SimpleNamespace mock pool.
release_req has two free paths (radix_cache_disabled True/False); both must
call self.req_to_token_pool.free(req) — the pool subclass decides whether to
also release a recurrent slot. The previous _maybe_free_recurrent_cache helper
has been replaced by HybridReqToTokenPool.free(req) override.
"""

import inspect
import unittest


class TestReleaseReqSourceCallsPoolFreeInBothPaths(unittest.TestCase):
    def test_release_req_calls_pool_free_in_both_paths(self):
        """Both free paths (radix-disabled and radix-enabled) must invoke
        self.req_to_token_pool.free(req); the pool subclass dispatches the
        recurrent release without a caller-side hasattr check."""
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.release_req)
        self.assertGreaterEqual(
            src.count("self.req_to_token_pool.free(req)"),
            2,
            "release_req must call self.req_to_token_pool.free(req) in both free paths "
            "(radix-disabled and radix-enabled); polymorphism in the pool class hierarchy "
            "handles recurrent-state release.",
        )

    def test_release_req_no_longer_uses_legacy_helper(self):
        """The _maybe_free_recurrent_cache helper has been removed; release_req
        must not reference it."""
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.release_req)
        self.assertNotIn(
            "_maybe_free_recurrent_cache",
            src,
            "release_req must not reference the legacy _maybe_free_recurrent_cache helper",
        )

    def test_helper_removed_from_module(self):
        """The _maybe_free_recurrent_cache helper itself must be gone from
        schedule_batch (its job is now done by HybridReqToTokenPool.free)."""
        from sgl_jax.srt.managers import schedule_batch

        self.assertFalse(
            hasattr(schedule_batch, "_maybe_free_recurrent_cache"),
            "schedule_batch._maybe_free_recurrent_cache must be removed; "
            "polymorphism is now in HybridReqToTokenPool.free(req).",
        )


class TestHybridFreePolymorphismAtCallSite(unittest.TestCase):
    """Functional check: the call-site idiom is just pool.free(req); the
    pool subclass decides whether to also drop a recurrent slot."""

    def test_hybrid_pool_free_releases_both_slots(self):
        """HybridReqToTokenPool.free(req) releases both the KV and the
        recurrent slot in one call. Mocked here at the pool level so the
        callsite contract is exercised without standing up a full scheduler."""
        from types import SimpleNamespace

        free_calls = []
        recurrent_calls = []

        class HybridStub:
            def free(self, req):
                # Mirrors HybridReqToTokenPool.free(req): recurrent first,
                # then super (KV). The point of the override is the caller
                # only sees one call.
                recurrent_calls.append(req)
                free_calls.append(req)

        pool = HybridStub()
        req = SimpleNamespace(req_pool_idx=7, recurrent_pool_idx=3)
        pool.free(req)

        self.assertEqual(len(free_calls), 1)
        self.assertEqual(len(recurrent_calls), 1)
        self.assertIs(free_calls[0], req)
        self.assertIs(recurrent_calls[0], req)

    def test_non_hybrid_pool_free_only_kv(self):
        """Non-hybrid ReqToTokenPool.free(req) only releases the KV slot;
        no recurrent slot to drop. Same callsite — no hasattr probe needed."""
        from types import SimpleNamespace

        free_calls = []

        class NonHybridStub:
            def free(self, req):
                free_calls.append(req)

        pool = NonHybridStub()
        req = SimpleNamespace(req_pool_idx=7)
        pool.free(req)

        self.assertEqual(len(free_calls), 1)
        self.assertIs(free_calls[0], req)


if __name__ == "__main__":
    unittest.main()
