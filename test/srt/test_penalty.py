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

    def run_generate_with_prompt(self, prompt, sampling_params, max_tokens=100):
        """POST to /v1/chat/completions with the given prompt and params."""
        sampling_params.setdefault("temperature", 0.05)
        sampling_params.setdefault("top_p", 1.0)

        response = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                **sampling_params,
            },
        )
        self.assertEqual(response.status_code, 200)
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        return content

    def count_word_repetitions(self, text, word):
        """Count occurrences of `word` in `text` (case-insensitive, word-boundary)."""
        return len(re.findall(r"\b" + re.escape(word) + r"\b", text.lower()))

    def _test_penalty_effect(
        self,
        prompt,
        baseline_params,
        penalty_params,
        target_word,
        expected_reduction=True,
        max_tokens=50,
    ):
        """Run baseline vs penalty 5 times each; compare mean target-word counts."""
        baseline_counts = []
        penalty_counts = []

        for _ in range(5):
            baseline_output = self.run_generate_with_prompt(prompt, baseline_params, max_tokens)
            penalty_output = self.run_generate_with_prompt(prompt, penalty_params, max_tokens)

            baseline_counts.append(self.count_word_repetitions(baseline_output, target_word))
            penalty_counts.append(self.count_word_repetitions(penalty_output, target_word))

        avg_baseline = sum(baseline_counts) / len(baseline_counts)
        avg_penalty = sum(penalty_counts) / len(penalty_counts)

        if expected_reduction:
            self.assertLess(
                avg_penalty,
                avg_baseline,
                f"Penalty should reduce '{target_word}' repetition: "
                f"{avg_baseline:.1f} -> {avg_penalty:.1f}",
            )
        else:
            self.assertGreater(
                avg_penalty,
                avg_baseline,
                f"Negative penalty should increase '{target_word}' repetition: "
                f"{avg_baseline:.1f} -> {avg_penalty:.1f}",
            )

    def test_frequency_penalty_reduces_word_repetition(self):
        """frequency_penalty should reduce repetition of a target word."""
        prompt = (
            "Write exactly 10 very small sentences, each containing the word "
            "'data'. Use the word 'data' as much as possible."
        )
        baseline_params = {"frequency_penalty": 0.0}
        penalty_params = {"frequency_penalty": 1.99}
        self._test_penalty_effect(prompt, baseline_params, penalty_params, "data")

    def test_presence_penalty_reduces_topic_repetition(self):
        """presence_penalty should reduce topic repetition."""
        prompt = (
            "Write the word 'machine learning' exactly 20 times in a row, " "separated by spaces."
        )
        baseline_params = {"presence_penalty": 0.0}
        penalty_params = {"presence_penalty": 1.99}
        self._test_penalty_effect(prompt, baseline_params, penalty_params, "machine learning")

    def test_combined_penalties_reduce_repetition(self):
        """Combined frequency + presence penalties should reduce repetition."""
        prompt = (
            "Write exactly 10 short sentences, each containing the word 'data'. "
            "Use the word 'data' as much as possible."
        )
        baseline_params = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        penalty_params = {
            "frequency_penalty": 1.99,
            "presence_penalty": 1.99,
        }
        self._test_penalty_effect(prompt, baseline_params, penalty_params, "data", max_tokens=100)

    def test_penalty_edge_cases_negative_penalty_values(self):
        """Negative penalty values should increase repetition (expected_reduction=False)."""
        prompt = "Write the word 'test' exactly 15 times in a row, separated by spaces."
        baseline_params = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        negative_penalty_params = {
            "frequency_penalty": -0.5,
            "presence_penalty": -0.25,
        }
        self._test_penalty_effect(
            prompt,
            baseline_params,
            negative_penalty_params,
            "test",
            expected_reduction=False,
            max_tokens=60,
        )

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
