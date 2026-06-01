"""Unit tests for Req.init_incremental_detokenize (incremental-cache version).

Verifies the ported implementation that avoids re-concatenating the full prompt
on every call. Asserts equivalence to a full-recompute ground truth across the
first call, incremental steps, a retract-then-resume, and finish with an extra
delayed (overlap-schedule) token that must be truncated at the stop position.

The method is pure list logic, so we build a Req via object.__new__ and set only
the attributes it touches (avoids the heavy real __init__).
"""

import unittest

from sgl_jax.srt.managers.schedule_batch import (
    INIT_INCREMENTAL_DETOKENIZATION_OFFSET as OFF,
)
from sgl_jax.srt.managers.schedule_batch import Req


def _make_req(origin, output=None):
    req = object.__new__(Req)
    req.origin_input_ids_unpadded = list(origin)
    req.output_ids = list(output or [])
    req.finished_len = None
    req.surr_offset = None
    req.read_offset = None
    req.surr_and_decode_ids = None
    req.cur_decode_ids_len = 0
    return req


def _ground_truth(req):
    surr = (
        max(len(req.origin_input_ids_unpadded) - OFF, 0)
        if req.surr_offset is None
        else req.surr_offset
    )
    out = req.output_ids[: req.finished_len] if req.finished_len is not None else req.output_ids
    return req.origin_input_ids_unpadded[surr:] + out


class TestInitIncrementalDetokenize(unittest.TestCase):
    def test_incremental_matches_full_recompute(self):
        origin = list(range(20))
        req = _make_req(origin, output=[100])

        ids, off = req.init_incremental_detokenize()
        self.assertEqual(off, OFF)
        self.assertEqual(ids, _ground_truth(req))

        # incremental growth
        for extra in ([101, 102], [103], [104, 105, 106]):
            req.output_ids += extra
            ids, off = req.init_incremental_detokenize()
            self.assertEqual(ids, _ground_truth(req))
            self.assertEqual(off, OFF)

    def test_no_full_prompt_reconcat(self):
        # surr_and_decode_ids must NOT contain prompt ids before surr_offset.
        origin = list(range(1000))
        req = _make_req(origin, output=[5, 6, 7])
        ids, _ = req.init_incremental_detokenize()
        # only the last OFF prompt ids are included
        self.assertEqual(ids[:OFF], origin[1000 - OFF :])
        self.assertEqual(len(ids), OFF + 3)

    def test_retract_then_resume(self):
        origin = list(range(20))
        req = _make_req(origin, output=[100, 101])
        req.init_incremental_detokenize()
        # reset_for_retract preserves output_ids / offsets / cache; decode resumes.
        req.output_ids += [102, 103]
        ids, _ = req.init_incremental_detokenize()
        self.assertEqual(ids, _ground_truth(req))

    def test_finish_truncates_delayed_token(self):
        origin = list(range(20))
        req = _make_req(origin, output=[100, 101, 102])
        req.init_incremental_detokenize()
        req.output_ids += [200]  # EOS
        req.finished_len = len(req.output_ids)
        req.output_ids += [201]  # delayed overlap token past stop
        ids, _ = req.init_incremental_detokenize()
        self.assertNotIn(201, ids)
        self.assertEqual(ids, origin[20 - OFF :] + [100, 101, 102, 200])
        # idempotent re-call (no growth)
        before = list(req.surr_and_decode_ids)
        ids2, _ = req.init_incremental_detokenize()
        self.assertEqual(ids2, before)


if __name__ == "__main__":
    unittest.main()
