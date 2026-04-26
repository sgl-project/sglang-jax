"""cache_finished_req call sites must NOT invoke pool.free(req) directly —
release happens inside cache_finished_req via the pool's polymorphic free(req).

Two callsites in scheduler_output_processor_mixin.py both call
self.tree_cache.cache_finished_req(req); cache_finished_req's
implementations (chunk_cache.py, radix_cache.py, swa_radix_cache.py) all
already invoke self.req_to_token_pool.free(req) internally. Adding a
caller-side self.req_to_token_pool.free(req) on top would double-free the
KV slot (and trip the req_to_token_pool memory leak detector).

HybridReqToTokenPool.free(req) override releases the recurrent state slot
in the same call — no caller-side hasattr probe needed (the previous
_maybe_free_recurrent_cache helper has been removed).
"""

import inspect
import unittest


class TestCacheFinishedReqMixinDoesNotDoubleFree(unittest.TestCase):
    def test_no_caller_side_pool_free(self):
        """Mixin must NOT call self.req_to_token_pool.free(req) — release
        happens inside cache_finished_req via the pool's polymorphic
        free(req). Adding it here would double-free the KV slot."""
        from sgl_jax.srt.managers import scheduler_output_processor_mixin

        src = inspect.getsource(scheduler_output_processor_mixin)
        free_call_count = src.count("self.req_to_token_pool.free(req)")
        self.assertEqual(
            free_call_count,
            0,
            f"Expected 0 self.req_to_token_pool.free(req) calls in "
            f"scheduler_output_processor_mixin.py, got {free_call_count}. "
            "cache_finished_req(req) already releases the slot internally; "
            "an extra caller-side pool.free(req) double-frees the KV slot.",
        )

    def test_legacy_helper_no_longer_referenced(self):
        """The _maybe_free_recurrent_cache helper has been removed; the
        mixin must not import or call it."""
        from sgl_jax.srt.managers import scheduler_output_processor_mixin

        src = inspect.getsource(scheduler_output_processor_mixin)
        self.assertNotIn(
            "_maybe_free_recurrent_cache",
            src,
            "scheduler_output_processor_mixin must not reference the legacy "
            "_maybe_free_recurrent_cache helper; polymorphism lives in "
            "HybridReqToTokenPool.free(req).",
        )


if __name__ == "__main__":
    unittest.main()
