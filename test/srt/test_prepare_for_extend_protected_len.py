"""prepare_for_extend must set cache_protected_len to the tree-owned prefix
(last_matched_prefix_len), not len(prefix_indices): after an off-boundary chunk
skip the latter includes request-owned KV that would leak on finish."""

import unittest
from unittest.mock import MagicMock

import numpy as np

from sgl_jax.srt.managers.schedule_batch import (
    ForwardMode,
    ScheduleBatch,
    ScheduleReqsInfo,
)


def _make_req(*, prefix_len, last_matched, total_len):
    """A req mid chunked continuation: prefix_indices spans the committed range."""
    req = MagicMock()
    req.prefix_indices = np.arange(prefix_len, dtype=np.int32)
    req.last_matched_prefix_len = last_matched
    req.fill_ids = list(range(total_len))
    req.extend_input_len = total_len - prefix_len
    req.logprob_start_len = 0
    req.return_logprob = False
    req.already_computed = 0
    req.cached_tokens = 0
    req.origin_input_ids = list(range(total_len))
    return req


class TestPrepareForExtendProtectedLen(unittest.TestCase):
    def _protected_len_after_prepare(self, req):
        pool = MagicMock()
        pool.alloc.return_value = [0]
        batch = ScheduleBatch(
            reqs_info=[ScheduleReqsInfo(reqs=[req])],
            dp_size=1,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=MagicMock(),
            tree_cache=None,
            model_config=MagicMock(vocab_size=32000),
            forward_mode=ForwardMode.EXTEND,
        )
        batch.is_hybrid_recurrent = False
        req.cache_protected_len = -1
        # Only the later token-slot allocation may fail (None tree_cache); narrow
        # to AttributeError so unrelated failures surface.
        try:
            batch.prepare_for_extend()
        except AttributeError:
            pass
        self.assertNotEqual(
            req.cache_protected_len,
            -1,
            "prepare_for_extend raised before the cache_protected_len assignment",
        )
        return req.cache_protected_len

    def test_protected_len_is_last_matched_not_prefix_len(self):
        # Off-boundary skip: committed 512, published 0.
        req = _make_req(prefix_len=512, last_matched=0, total_len=640)
        self.assertEqual(self._protected_len_after_prepare(req), 0)

    def test_protected_len_after_partial_publish(self):
        # Published a 128 boundary, then off-boundary skips to committed 512.
        req = _make_req(prefix_len=512, last_matched=128, total_len=640)
        self.assertEqual(self._protected_len_after_prepare(req), 128)

    def test_protected_len_equals_prefix_len_when_aligned(self):
        # Normal request: prefix_indices == tree-matched prefix (no skip tail).
        req = _make_req(prefix_len=256, last_matched=256, total_len=512)
        self.assertEqual(self._protected_len_after_prepare(req), 256)


if __name__ == "__main__":
    unittest.main()
