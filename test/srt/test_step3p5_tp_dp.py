"""Step 3.5 tp/dp invariance e2e (TPU 4-chip host, real server, dummy weights).

I3 self-consistency: sharding must not change the output. Same microscale model,
same greedy prompt, run under every parallelism layout of a 4-chip host. Needs
only RANDOM microscale weights (NOT the 398GB checkpoint) but DOES need exactly
4 chips (one host, e.g. v6e-4).

Parallelism arg convention (scheduler.py: ici = [dp_size, tp_size // dp_size]):
``--tp-size`` is the TOTAL device count, ``--dp-size`` partitions it, per-group
tensor width = tp_size // dp_size. So on a 4-chip host tp-size is always 4 and
only dp-size changes:
    dp=1 -> mesh [1, 4]  (tensor width 4)   <- reference
    dp=2 -> mesh [2, 2]  (tensor width 2)
    dp=4 -> mesh [4, 1]  (tensor width 1)

EVIDENCE-FIRST: each layout has a DIFFERENT tensor width, hence a different
bf16 reduction order in the sharded matmuls. With random dummy weights the
logits may have no margin, so a flipped argmax could be legitimate reduction
noise OR a real DP correctness bug — these are NOT distinguishable from the
emitted token alone. So this test dumps the FULL output logprob distribution per
layout and prints, for each variant vs the dp=1 reference:
  - max abs diff over the aligned per-token logprob vectors (bf16-level => noise);
  - the dp=1 top1/top2 margin (margin < diff => the argmax flip is within noise).
The hard gate is still argmax agreement; when it fails the printed table is the
evidence to decide noise-vs-bug (do NOT pre-judge from the token alone).

Expose exactly 4 chips (one host); not a single-chip override, not 16.
Reuses the microscale config + helpers from test_step3p5_serving_e2e.
"""

import unittest

import requests
from test_step3p5_serving_e2e import _BASE_ARGS, _INPUT_IDS, _VOCAB, _config_dir

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_DEVICES = 4  # tp_size is ALWAYS this; dp_size partitions it.


def _launch(model_dir, dp):
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


def _logprobs_under(dp):
    """Launch under the given dp layout; return (first_output_id, {token_id: logprob})
    for the first decode step (full-vocab top-logprobs)."""
    model_dir = _config_dir()
    proc = _launch(model_dir, dp)
    try:
        resp = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "input_ids": _INPUT_IDS,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "return_logprob": True,
                "top_logprobs_num": _VOCAB,  # full vocab
            },
            timeout=120,
        )
        resp.raise_for_status()
        body = resp.json()
        out_id = body["output_ids"][0]
        top = body["meta_info"]["output_top_logprobs"][0]  # [[logprob, token_id, ...], ...]
        dist = {int(tid): float(lp) for lp, tid, *_ in top}
        return out_id, dist
    finally:
        kill_process_tree(proc.pid)


def _max_abs_diff(ref, other):
    keys = set(ref) & set(other)
    return max((abs(ref[k] - other[k]) for k in keys), default=float("nan"))


def _top2_margin(dist):
    vals = sorted(dist.values(), reverse=True)
    return (vals[0] - vals[1]) if len(vals) >= 2 else float("nan")


class TestStep3p5TPDPInvariance(CustomTestCase):
    """Greedy output must be identical across all dp layouts of a 4-chip host."""

    @classmethod
    def setUpClass(cls):
        # Reference: full tensor-parallel (dp=1 -> mesh [1, 4], tensor width 4).
        cls.ref_id, cls.ref_dist = _logprobs_under(1)
        cls.ref_margin = _top2_margin(cls.ref_dist)

    def _check(self, dp, mesh):
        out_id, dist = _logprobs_under(dp)
        diff = _max_abs_diff(self.ref_dist, dist)
        print(
            f"\n=== dp={dp} {mesh} vs dp=1 [1,4] ===\n"
            f" argmax: dp1={self.ref_id}  dp{dp}={out_id}  agree={self.ref_id == out_id}\n"
            f" logprob max_abs_diff over aligned vocab = {diff:.4e}\n"
            f" dp=1 top1-top2 margin = {self.ref_margin:.4e}\n"
            f" => margin < diff means the argmax flip is WITHIN reduction noise;"
            f" margin >> diff (or large diff) means a real DP bug."
        )
        # Hard gate: decision must agree. On failure the table above is the evidence.
        self.assertEqual(
            self.ref_id,
            out_id,
            f"dp={dp} {mesh} argmax differs from dp=1 (see printed logprob evidence)",
        )

    def test_dp2_equals_dp1(self):
        self._check(2, "[2,2]")

    def test_dp4_equals_dp1(self):
        self._check(4, "[4,1]")


if __name__ == "__main__":
    unittest.main()
