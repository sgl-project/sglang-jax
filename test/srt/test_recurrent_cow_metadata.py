"""`_build_recurrent_cow_src_indices` must return None (not an all-zero array)
when no clone is pending, so `_maybe_apply_recurrent_cow` skips the donated-buffer
scatter on cold/no-hit extends (which would corrupt state under multi-host SPMD).

One-shot *consumption* is covered end-to-end by the recurrent serving probe.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from sgl_jax.srt.managers.schedule_batch import _build_recurrent_cow_src_indices


def _req(rid, cow_src=None):
    req = MagicMock()
    req.rid = rid
    req.recurrent_cow_src_index = cow_src
    return req


class TestBuildRecurrentCowSrcIndices(unittest.TestCase):
    def test_all_zero_returns_none(self):
        self.assertIsNone(_build_recurrent_cow_src_indices([_req(0), _req(1)]))

    def test_empty_returns_none(self):
        self.assertIsNone(_build_recurrent_cow_src_indices([]))

    def test_mixed_returns_array(self):
        out = _build_recurrent_cow_src_indices([_req(0, 0), _req(1, 7)])
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out, np.array([0, 7], dtype=np.int32))

    def test_all_nonzero_returns_array(self):
        out = _build_recurrent_cow_src_indices([_req(0, 3), _req(1, 7)])
        np.testing.assert_array_equal(out, np.array([3, 7], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
