import json
import multiprocessing
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestAbortAll(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--max-running-requests",
                8,
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.1",
                "--max-prefill-tokens",
                "4096",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_decode(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 8000,
                    "ignore_eos": True,
                },
            },
        )
        return response.json()

    def test_abort_all(self):
        num_requests = 32
        with ThreadPoolExecutor(num_requests) as executor:
            futures = [executor.submit(self._run_decode) for _ in range(num_requests)]

            # ensure the decode has been started
            time.sleep(2)

            requests.post(
                self.base_url + "/abort_request",
                json={
                    "abort_all": True,
                },
            )

            for future in as_completed(futures):
                self.assertEqual(
                    future.result()["meta_info"]["finish_reason"]["type"], "abort"
                )


if __name__ == "__main__":
    unittest.main()
