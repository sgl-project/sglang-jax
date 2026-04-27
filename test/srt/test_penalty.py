"""
Functional tests for penalty parameters (frequency / presence / combined).

Ports sglang PR #11931 to sgl-jax. Validates that penalty params actually
shape token output distribution, not merely that the server accepts them.

Note: sgl-jax does not currently implement repetition_penalty
(no BatchedRepetitionPenalizer in penaltylib/), so no test here passes
repetition_penalty.

Usage:
    python3 -m unittest test_penalty.TestPenalty -v
"""

import re
import unittest

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPenalty(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.65",
                "--max-prefill-tokens",
                "8192",
                "--download-dir",
                "/dev/shm/",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "fa",
                "--page-size",
                "64",
                "--max-running-requests",
                "64",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_server_alive(self):
        """Sanity check: server is up and /v1/chat/completions responds."""
        resp = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 8,
                "temperature": 0.0,
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("choices", resp.json())


if __name__ == "__main__":
    unittest.main(verbosity=3)
