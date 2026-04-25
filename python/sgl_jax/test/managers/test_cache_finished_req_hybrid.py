"""Phase 5: cache_finished_req call sites must release hybrid recurrent state.

Two callsites in scheduler_output_processor_mixin.py (line 125 and 338);
both must invoke hybrid free_recurrent_cache(req) adjacent to the
self.tree_cache.cache_finished_req(req) call (D1: hasattr pattern).
"""

import inspect
import unittest


class TestCacheFinishedReqMixinHasHybridBranches(unittest.TestCase):
    def test_two_callsites_have_hybrid_hasattr_branches(self):
        from sgl_jax.srt.managers import scheduler_output_processor_mixin

        src = inspect.getsource(scheduler_output_processor_mixin)
        # Both callsites must follow cache_finished_req(req) with a hybrid
        # release. We expect >=2 invocations of the helper across the file.
        helper_call_count = src.count("_maybe_free_recurrent_cache(")
        self.assertGreaterEqual(
            helper_call_count,
            2,
            f"Expected >=2 _maybe_free_recurrent_cache(...) calls in "
            f"scheduler_output_processor_mixin.py, got {helper_call_count}. "
            "Both cache_finished_req(req) callsites must release recurrent state.",
        )


if __name__ == "__main__":
    unittest.main()
