"""
Functional tests for penalty parameters (frequency / presence / combined).

Ports sglang PR #11931 to sgl-jax (aligned with the evolved upstream form
that uses vocabulary diversity + fixed seeds for stability). Validates that
penalty params actually shape token output distribution, not merely that
the server accepts them.

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

    def run_generate_with_prompt(self, prompt, sampling_params, max_tokens=100, seed=None):
        """POST to /v1/chat/completions with the given prompt and params."""
        sampling_params = sampling_params.copy()
        sampling_params.setdefault("temperature", 0.05)
        sampling_params.setdefault("top_p", 1.0)
        if seed is not None:
            sampling_params["seed"] = seed

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

    def _get_vocab_diversity(self, text):
        """Calculate vocabulary diversity as unique_words / total_words.

        Higher values mean more diverse (less repetitive) text.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 1.0
        return len(set(words)) / len(words)

    def _test_penalty_effect(
        self,
        prompt,
        baseline_params,
        penalty_params,
        expected_reduction=True,
        max_tokens=150,
    ):
        """Generic test for penalty effects using vocabulary diversity.

        Measures unique_words/total_words ratio instead of counting a specific
        word, because penalties affect ALL token probabilities -- the model may
        avoid some repeated tokens while using others more.
        """
        # Use higher temperature so penalties can actually affect token selection.
        # The default temperature (0.05) is near-greedy, making penalty adjustments
        # to logits ineffective since the top token still dominates.
        baseline_params = baseline_params.copy()
        penalty_params = penalty_params.copy()
        baseline_params.setdefault("temperature", 0.8)
        penalty_params.setdefault("temperature", 0.8)

        base_seed = 42
        baseline_diversities = []
        penalty_diversities = []

        for i in range(5):
            seed = base_seed + i
            baseline_output = self.run_generate_with_prompt(
                prompt, baseline_params, max_tokens, seed=seed
            )
            penalty_output = self.run_generate_with_prompt(
                prompt, penalty_params, max_tokens, seed=seed
            )

            baseline_diversities.append(self._get_vocab_diversity(baseline_output))
            penalty_diversities.append(self._get_vocab_diversity(penalty_output))

        avg_baseline = sum(baseline_diversities) / len(baseline_diversities)
        avg_penalty = sum(penalty_diversities) / len(penalty_diversities)

        if expected_reduction:
            self.assertGreater(
                avg_penalty,
                avg_baseline,
                f"Penalty should increase vocab diversity: "
                f"{avg_baseline:.3f} -> {avg_penalty:.3f}",
            )
        else:
            self.assertLess(
                avg_penalty,
                avg_baseline,
                f"Negative penalty should decrease vocab diversity: "
                f"{avg_baseline:.3f} -> {avg_penalty:.3f}",
            )

    def test_frequency_penalty_reduces_word_repetition(self):
        """frequency_penalty should increase vocabulary diversity."""
        prompt = (
            "Write exactly 10 very small sentences, each containing the word "
            "'data'. Use the word 'data' as much as possible."
        )
        baseline_params = {"frequency_penalty": 0.0}
        penalty_params = {"frequency_penalty": 1.99}
        self._test_penalty_effect(prompt, baseline_params, penalty_params)

    def test_presence_penalty_reduces_topic_repetition(self):
        """presence_penalty should increase vocabulary diversity."""
        prompt = (
            "Write the word 'machine learning' exactly 20 times in a row, " "separated by spaces."
        )
        baseline_params = {"presence_penalty": 0.0}
        penalty_params = {"presence_penalty": 1.99}
        self._test_penalty_effect(prompt, baseline_params, penalty_params)

    def test_combined_penalties_reduce_repetition(self):
        """Combined frequency + presence penalties should increase vocab diversity."""
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
        self._test_penalty_effect(prompt, baseline_params, penalty_params)

    def test_penalty_edge_cases_negative_penalty_values(self):
        """Negative penalty values should decrease vocabulary diversity."""
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
            expected_reduction=False,
        )

    def test_penalty_edge_cases_extreme_penalty_values(self):
        """Extreme penalty values should strongly increase vocabulary diversity."""
        prompt = "Write the word 'extreme' exactly 20 times in a row, separated by spaces."
        baseline_params = {
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }
        extreme_penalty_params = {
            "frequency_penalty": 2.0,
            "presence_penalty": 2.0,
        }
        self._test_penalty_effect(prompt, baseline_params, extreme_penalty_params)


if __name__ == "__main__":
    unittest.main(verbosity=3)
