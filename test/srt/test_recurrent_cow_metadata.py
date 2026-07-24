"""CoW metadata keeps a stable array structure while zero remains the no-op sentinel."""

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
    def test_all_zero_returns_array(self):
        out = _build_recurrent_cow_src_indices([_req(0), _req(1)])
        np.testing.assert_array_equal(out, np.zeros(2, dtype=np.int32))

    def test_empty_returns_array(self):
        out = _build_recurrent_cow_src_indices([])
        np.testing.assert_array_equal(out, np.empty(0, dtype=np.int32))

    def test_mixed_returns_array(self):
        out = _build_recurrent_cow_src_indices([_req(0, 0), _req(1, 7)])
        self.assertIsNotNone(out)
        np.testing.assert_array_equal(out, np.array([0, 7], dtype=np.int32))

    def test_all_nonzero_returns_array(self):
        out = _build_recurrent_cow_src_indices([_req(0, 3), _req(1, 7)])
        np.testing.assert_array_equal(out, np.array([3, 7], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
