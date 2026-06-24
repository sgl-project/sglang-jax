"""Step 3.5 tp/dp invariance e2e (TPU 4-chip host, real server, dummy weights).

I3 self-consistency: sharding must not change the output. Same microscale model,
same greedy prompt, run under every tp×dp split of a 4-chip host → identical
output_ids. Needs only RANDOM microscale weights (NOT the 398GB checkpoint) — it
is a determinism check, not an accuracy check.

IMPORTANT (device-count constraint): the server requires
``tp_size * dp_size == device_count``. This test is written for a 4-chip host
(v6e-4 / one host of a 4x4), so every launch uses a tp×dp product of 4:
tp=4·dp=1 (reference), tp=2·dp=2, tp=1·dp=4. Expose exactly 4 chips (one host);
do NOT use a single-chip override and do NOT expose all 16 chips of a 4x4.

Reuses the microscale config + /generate helpers from test_step3p5_serving_e2e.
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


def _launch(model_dir, tp, dp):
    # tp*dp must equal the exposed device count (4). Disable the radix cache so the
    # only variable across runs is the tp/dp split.
    return popen_launch_server(
        model_dir,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        device="tpu",
        other_args=_BASE_ARGS
        + [
            "--disable-radix-cache",
            "--nnodes",
            "1",
            "--dist-init-addr",
            "0.0.0.0:10011",
            "--tp-size",
            str(tp),
            "--dp-size",
            str(dp),
        ],
    )


def _greedy_under(tp, dp):
    model_dir = _config_dir()
    proc = _launch(model_dir, tp, dp)
    try:
        return _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
    finally:
        kill_process_tree(proc.pid)


class TestStep3p5TPDPInvariance(CustomTestCase):
    """Greedy output must be identical across all tp×dp splits of a 4-chip host."""

    @classmethod
    def setUpClass(cls):
        # Reference: fully tensor-parallel (tp=4, dp=1) over the 4 chips.
        cls.ref = _greedy_under(4, 1)

    def test_tp2_dp2_equals_tp4(self):
        out = _greedy_under(2, 2)
        self.assertEqual(out, self.ref, "tp=2·dp=2 greedy output differs from tp=4·dp=1")

    def test_dp4_equals_tp4(self):
        out = _greedy_under(1, 4)
        self.assertEqual(out, self.ref, "tp=1·dp=4 greedy output differs from tp=4·dp=1")


if __name__ == "__main__":
    unittest.main()
