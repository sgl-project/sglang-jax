"""Regression: pop_committed/overallocated KV cache must be idempotent.

The PD prefill abort path can run the KV release a second time after a
handoff failure already freed the cache. Before the fix this tripped an
assertion in ``pop_committed_kv_cache`` and crashed the scheduler. The
methods now return a zero sentinel on repeat so the caller frees nothing
instead of double-freeing.
"""

import unittest

from sgl_jax.srt.managers.schedule_batch import Req
from sgl_jax.srt.sampling.sampling_params import SamplingParams


def _make_req() -> Req:
    req = Req("rid", "text", [1, 2, 3], SamplingParams(max_new_tokens=1))
    req.kv_committed_len = 10
    req.kv_allocated_len = 10
    return req


class TestKVCachePopIdempotent(unittest.TestCase):
    def test_pop_committed_idempotent(self):
        req = _make_req()
        self.assertEqual(req.pop_committed_kv_cache(), 10)
        # Second call must not raise; returns 0 so the caller frees nothing.
        self.assertEqual(req.pop_committed_kv_cache(), 0)
        self.assertEqual(req.pop_committed_kv_cache(), 0)

    def test_pop_overallocated_idempotent(self):
        req = _make_req()
        self.assertEqual(req.pop_overallocated_kv_cache(), (10, 10))
        self.assertEqual(req.pop_overallocated_kv_cache(), (0, 0))
        self.assertEqual(req.pop_overallocated_kv_cache(), (0, 0))


if __name__ == "__main__":
    unittest.main()
