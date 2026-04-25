"""Phase 4: ScheduleBatch.alloc_req_slots takes reqs (list) instead of num_reqs (int).

Backed by Phase 1's ReqToTokenPool.alloc(reqs) which already accepts
list[Req] and skips reqs whose req_pool_idx is set (chunked reuse path).
"""

import inspect
import unittest


class TestAllocReqSlotsSignature(unittest.TestCase):
    def test_signature_takes_reqs(self):
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        sig = inspect.signature(ScheduleBatch.alloc_req_slots)
        params = list(sig.parameters.keys())
        self.assertIn(
            "reqs",
            params,
            "alloc_req_slots must accept 'reqs' (list[Req])",
        )
        self.assertNotIn(
            "num_reqs",
            params,
            "alloc_req_slots must no longer accept 'num_reqs' (int) per D5",
        )

    def test_body_calls_alloc_with_reqs(self):
        """The body must pass reqs (the parameter) to req_to_token_pool.alloc."""
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch.alloc_req_slots)
        self.assertIn(
            "self.req_to_token_pool.alloc(reqs)",
            src,
            "alloc_req_slots must forward 'reqs' to req_to_token_pool.alloc",
        )

    def test_caller_passes_self_reqs(self):
        """prepare_for_extend (or any caller) must pass self.reqs to alloc_req_slots,
        not a count. Source-grep on ScheduleBatch source."""
        from sgl_jax.srt.managers.schedule_batch import ScheduleBatch

        src = inspect.getsource(ScheduleBatch)
        self.assertIn(
            "self.alloc_req_slots(self.reqs)",
            src,
            "Caller must pass self.reqs (list) to alloc_req_slots",
        )
        self.assertNotIn(
            "self.alloc_req_slots(bs)",
            src,
            "Old int-count caller pattern 'alloc_req_slots(bs)' must be replaced",
        )


if __name__ == "__main__":
    unittest.main()
