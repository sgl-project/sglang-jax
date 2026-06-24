"""Step 3.5 tp/dp invariance e2e (TPU 4-chip host, real server, dummy weights).

I3 self-consistency: sharding must not change the output. Same microscale model,
same greedy prompt, run under every parallelism layout of a 4-chip host →
identical output_ids. Needs only RANDOM microscale weights (NOT the 398GB
checkpoint) — it is a determinism check, not an accuracy check — but it DOES need
exactly 4 chips (one host, e.g. v6e-4).

Parallelism arg convention (scheduler.py: ici = [dp_size, tp_size // dp_size]):
``--tp-size`` is the TOTAL device count and ``--dp-size`` partitions it; the
per-group tensor width is tp_size // dp_size. So on a 4-chip host tp-size is
always 4 and only dp-size changes:
    dp=1 -> mesh [1, 4]  (full tensor-parallel)         <- reference
    dp=2 -> mesh [2, 2]  (2 data x 2 tensor)
    dp=4 -> mesh [4, 1]  (full data-parallel)

Expose exactly 4 chips (one host); do NOT use a single-chip override and do NOT
expose all 16 chips of a 4x4.

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

# Total devices on the host. tp_size is ALWAYS this; dp_size partitions it.
_DEVICES = 4


def _launch(model_dir, dp):
    # tp_size == device count (4); dp_size carves data-parallel groups out of it.
    # Disable the radix cache so the only variable across runs is the dp/tp layout.
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
            str(_DEVICES),
            "--dp-size",
            str(dp),
        ],
    )


def _greedy_under(dp):
    model_dir = _config_dir()
    proc = _launch(model_dir, dp)
    try:
        return _generate(DEFAULT_URL_FOR_TEST, _INPUT_IDS)
    finally:
        kill_process_tree(proc.pid)


class TestStep3p5TPDPInvariance(CustomTestCase):
    """Greedy output must be identical across all dp layouts of a 4-chip host."""

    @classmethod
    def setUpClass(cls):
        # Reference: full tensor-parallel (dp=1 -> mesh [1, 4]).
        cls.ref = _greedy_under(1)

    def test_dp2_equals_dp1(self):
        out = _greedy_under(2)  # mesh [2, 2]
        self.assertEqual(out, self.ref, "dp=2 (mesh [2,2]) greedy output differs from dp=1")

    def test_dp4_equals_dp1(self):
        out = _greedy_under(4)  # mesh [4, 1]
        self.assertEqual(out, self.ref, "dp=4 (mesh [4,1]) greedy output differs from dp=1")


if __name__ == "__main__":
    unittest.main()
