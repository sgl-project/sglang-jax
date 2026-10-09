"""GLM-5.3-Flash end-to-end: accuracy (gsm8k) and long-context retrieval (needle).

Needs a TPU v7x-8 (tp8/ep8) and the zai-org/GLM-5.3-Flash weights, so it is gated:

    SGLANG_GLM53_FLASH_E2E=1 python3 test/srt/test_glm53_flash_models.py
"""

import os
import random
import unittest
from types import SimpleNamespace

import requests
from run_eval import run_eval

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase, popen_launch_server

MODEL = "zai-org/GLM-5.3-Flash"


class GLM53FlashMixin:
    """One server per class, launched with ``BASE_ARGS + EXTRA_ARGS``."""

    BASE_ARGS = [
        "--trust-remote-code",
        "--skip-server-warmup",
        "--dtype",
        "bfloat16",
        "--tp-size",
        "8",
        "--ep-size",
        "8",
        "--attention-backend",
        "dsa_sparse",
        "--moe-backend",
        "fused_v2",
        "--page-size",
        "128",
        "--context-length",
        "16384",
        "--chunked-prefill-size",
        "4096",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "16",
        "--enable-unified-radix-tree",
        "--enable-recurrent-extra-buffer",
        "--reasoning-parser",
        "glm53",
        "--precompile-token-paddings",
        "512",
        "1024",
        "2048",
        "4096",
    ]
    EXTRA_ARGS: list[str] = []
    # Measured 0.985-0.99 (200 questions).
    GSM8K_THRESHOLD = 0.97

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=3600,
            device="tpu",
            check_cache_miss=False,
            other_args=cls.BASE_ARGS + cls.EXTRA_ARGS,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _post(self, path: str, body: dict) -> dict:
        r = requests.post(f"{self.base_url}{path}", json=body, timeout=3600)
        r.raise_for_status()
        return r.json()

    def test_a_gsm8k(self):
        metrics = run_eval(
            SimpleNamespace(
                base_url=self.base_url,
                model=self.model,
                eval_name="gsm8k",
                num_examples=200,
                num_threads=16,
                max_tokens=4096,
            )
        )
        self.assertGreaterEqual(metrics["score"], self.GSM8K_THRESHOLD)

    def test_c_needle(self):
        """A passcode hidden in ~4K and ~14K tokens of filler is retrieved at every depth.
        Exercises multi-page sequences, the indexer's long-context buckets and sparse MLA
        over pages far from the query."""
        rng = random.Random(0)
        words = ["river", "granite", "lantern", "orchard", "copper", "harbor", "meadow", "cedar"]
        failures = []
        for n_notes in (180, 650):  # ~4K and ~14.5K prompt tokens
            for depth in (0.1, 0.5, 0.9):
                notes = [
                    f"Note {i}: the {' '.join(rng.sample(words, 2))} was moved next to the "
                    f"{' '.join(rng.sample(words, 2))} on day {rng.randint(1, 28)}."
                    for i in range(n_notes)
                ]
                code = str(rng.randint(10000, 99999))
                notes.insert(int(n_notes * depth), f"Important: the secret passcode is {code}.")
                prompt = (
                    "Here is a long list of notes.\n\n"
                    + "\n".join(notes)
                    + "\n\nWhat is the secret passcode? Answer with the number only."
                )
                r = self._post(
                    "/v1/chat/completions",
                    {
                        "model": self.model,
                        "temperature": 0,
                        "max_tokens": 1500,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                if code not in (r["choices"][0]["message"].get("content") or ""):
                    failures.append((r["usage"]["prompt_tokens"], depth))
        self.assertEqual(failures, [], f"needle missed at (prompt_tokens, depth): {failures}")


@unittest.skipUnless(
    os.getenv("SGLANG_GLM53_FLASH_E2E") == "1",
    "needs a TPU v7x-8 with the GLM-5.3-Flash weights; set SGLANG_GLM53_FLASH_E2E=1",
)
class TestGLM53Flash(GLM53FlashMixin, CustomTestCase):
    pass


if __name__ == "__main__":
    unittest.main()
