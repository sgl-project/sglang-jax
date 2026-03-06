import os
import re
import sys
import time
import unittest

import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import DEFAULT_URL_FOR_TEST, CustomTestCase, popen_launch_server


class TestW8Int8DynamicBlockQuant(CustomTestCase):
    model = "Qwen/Qwen3-8B"
    quantization_config_path = "int8_block_128_dynamic.yaml"
    other_args = [
        "--tp-size=4",
        "--download-dir=/dev/shm",
        "--max-running-requests=128",
        "--page-size=64",
        "--disable-precompile",
    ]

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--quantization-config-path",
            cls.quantization_config_path,
            *cls.other_args,
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=1200,
            other_args=other_args,
            check_cache_miss=False,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        try:
            cls.process.wait(timeout=30)
        except Exception:
            pass
        time.sleep(5)

    def _generate(self, prompt, max_new_tokens=16):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        return response.json()

    def test_basic_generation(self):
        prompt = "Answer with one word only. What is the capital of France?"
        data = self._generate(prompt, max_new_tokens=8)
        text = data.get("text", "")
        self.assertTrue(text.strip(), f"Empty generation response: {data}")
        self.assertRegex(
            text.lower(),
            re.compile(r"\bparis\b"),
            msg=f"Unexpected generation text: {text!r}",
        )


if __name__ == "__main__":
    unittest.main()
