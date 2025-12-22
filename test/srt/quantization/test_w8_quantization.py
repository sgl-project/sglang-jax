import time
import unittest
from types import SimpleNamespace

import requests

from sgl_jax.srt.utils import kill_process_tree
from run_eval import run_eval
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class BaseW8Test(CustomTestCase):
    model: str = None
    quantization_config_path: str = None
    gsm8k_accuracy_threshold: float = None
    throughput_threshold: float = None

    @classmethod
    def setUpClass(cls):
        if cls is BaseW8Test:
            raise unittest.SkipTest("Skip base test class")

        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = []
        if cls.quantization_config_path:
            other_args.extend(["--quantization-config-path", cls.quantization_config_path])
        if cls.other_args:
            other_args.extend(cls.other_args)

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        if cls is BaseW8Test:
            return
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        if self.gsm8k_accuracy_threshold is None:
            self.skipTest("gsm8k_accuracy_threshold not set for this test")

        args = SimpleNamespace(
            eval_name="gsm8k",
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            base_url=self.base_url,
        )
        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["accuracy"], self.gsm8k_accuracy_threshold)

    def run_decode(self, max_new_tokens):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "ignore_eos": True,
            },
        )
        return response.json()

    def test_throughput(self):

        max_tokens = 256
        tic = time.perf_counter()
        res = self.run_decode(max_tokens)
        tok = time.perf_counter()
        print(res["text"])
        throughput = max_tokens / (tok - tic)
        print(f"Throughput: {throughput} tokens/s")
        self.assertGreaterEqual(throughput, self.throughput_threshold)

class TestW8Int8(BaseW8Test):
    model = "Qwen/Qwen3-32B"
    quantization_config_path = "int8_all_modules_w_only.yaml"
    gsm8k_accuracy_threshold = 0.96
    throughput_threshold = 100
    other_args = [
        "--tp-size=4"
    ]


if __name__ == "__main__":
    unittest.main()
