"""Step 3.5 tp/dp invariance e2e (TPU multi-chip, real server, dummy weights).

I3 self-consistency: sharding must not change the output. Same microscale model,
same greedy prompt, run under different parallelism → identical output_ids. Needs
only RANDOM microscale weights (NOT the 398GB checkpoint) — it is a determinism
check, not an accuracy check — but it DOES need >= 2 chips (e.g. v6e-4).

Reuses the microscale config + /generate helpers from test_step3p5_serving_e2e.
The tp=1 single-chip output is the reference; tp=2 and dp=2 must match it.
"""

import unittest

from test_step3p5_serving_e2e import _BASE_ARGS, _INPUT_IDS, _config_dir, _generate

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def _launch(model_dir, parallel_args):
    # Disable the radix cache so the only variable across runs is parallelism.
    return popen_launch_server(
        model_dir,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        device="tpu",
        other_args=_BASE_ARGS + ["--disable-radix-cache"] + parallel_args,
    )


def _greedy_under(parallel_args):
    model_dir = _config_dir()
    proc = _launch(model_dir, parallel_args)
    try:
        return _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
    finally:
        kill_process_tree(proc.pid)


class TestStep3p5TPDPInvariance(CustomTestCase):
    """Greedy output must be identical under tp=1, tp=2, and dp=2."""

    @classmethod
    def setUpClass(cls):
        # tp=1 / dp=1 single-chip reference.
        cls.ref = _greedy_under(["--tp-size", "1"])

    def test_tp2_equals_tp1(self):
        out = _greedy_under(["--tp-size", "2"])
        self.assertEqual(out, self.ref, "tp=2 greedy output differs from tp=1")

    def test_dp2_equals_tp1(self):
        out = _greedy_under(["--dp-size", "2"])
        self.assertEqual(out, self.ref, "dp=2 greedy output differs from tp=1/dp=1")


if __name__ == "__main__":
    unittest.main()
