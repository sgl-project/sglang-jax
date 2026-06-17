"""Recurrent CoW metadata must be one-shot.

The multi-host donated-buffer scatter in `_maybe_apply_recurrent_cow` runs
whenever `ModelWorkerBatch.recurrent_cow_src_indices` is not None. An all-zero
array therefore triggers an unnecessary clone on every cold/no-hit extend, which
corrupts recurrent state under multi-host SPMD. `_build_recurrent_cow_src_indices`
returns None unless a real clone is pending, keeping the prepass one-shot.

One-shot *consumption* (a hit prefill clones exactly once; the following decode
tokens never replay it) is exercised end-to-end by the recurrent serving probe
with multi-token generation -- it depends on the full get_model_worker_batch /
prepare_for_decode machinery that is not worth mocking here.
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
    """The producer-side helper: None unless a real clone is pending."""

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
