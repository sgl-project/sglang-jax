"""Req.recurrent_pool_idx field for hybrid recurrent state slot tracking.

HybridReqToTokenPool.alloc(reqs) (Phase 1) reads/writes this field to
coordinate recurrent slot reuse during chunked prefill. Phase 4 wires up
the Req-side field so the round-trip works end-to-end.
"""

import inspect
import unittest


class TestReqRecurrentPoolIdxField(unittest.TestCase):
    """Req must expose recurrent_pool_idx (None by default) and reset it
    on retract, mirroring how req_pool_idx is handled."""

    def test_default_is_none(self):
        """Construct a Req instance via inspect (avoid full __init__ fixture)."""
        from sgl_jax.srt.managers.schedule_batch import Req

        # Req.__init__ requires complex args; verify the field is initialized
        # via source inspection (same trick Phase 3 used for model signature
        # checks).
        src = inspect.getsource(Req.__init__)
        self.assertIn(
            "self.recurrent_pool_idx: int | None = None",
            src,
            "Req.__init__ must initialize recurrent_pool_idx to None",
        )

    def test_retract_resets_recurrent_pool_idx(self):
        """reset_for_retract must clear recurrent_pool_idx alongside req_pool_idx."""
        from sgl_jax.srt.managers.schedule_batch import Req

        # The retract path lives inside Req; find the method that resets
        # req_pool_idx and verify recurrent_pool_idx is reset there too.
        retract_methods = [
            name
            for name, fn in inspect.getmembers(Req, predicate=inspect.isfunction)
            if "self.req_pool_idx = None" in inspect.getsource(fn)
        ]
        self.assertTrue(
            retract_methods,
            "Could not locate the Req method that resets req_pool_idx; "
            "if the codebase moved this logic, update the test fixture.",
        )
        for name in retract_methods:
            with self.subTest(method=name):
                src = inspect.getsource(getattr(Req, name))
                self.assertIn(
                    "self.recurrent_pool_idx = None",
                    src,
                    f"Req.{name} resets req_pool_idx but not recurrent_pool_idx",
                )


if __name__ == "__main__":
    unittest.main()
