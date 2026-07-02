import os
import unittest

import requests


@unittest.skipUnless(
    os.getenv("SGLANG_NEXTN_E2E_URL"),
    "Set SGLANG_NEXTN_E2E_URL to a running multi-host NEXTN server.",
)
class TestNextNMultihostHostTransfer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.environ["SGLANG_NEXTN_E2E_URL"].rstrip("/")
        requests.get(f"{cls.base_url}/health", timeout=30).raise_for_status()

    def test_plain_decode_reaches_eagle_host_materialization(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "hello",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
            timeout=600,
        )
        response.raise_for_status()
        payload = response.json()
        self.assertGreater(payload["meta_info"]["completion_tokens"], 1)
        self.assertGreater(len(payload["text"]), 0)


if __name__ == "__main__":
    unittest.main()
