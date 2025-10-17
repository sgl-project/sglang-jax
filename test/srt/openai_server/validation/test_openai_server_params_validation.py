import unittest

import openai
import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestParamsValidation(CustomTestCase):
    """
    valid openai server parameters:
    temperature,
    top_p,
    request_length
    """

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"

        # Start server with auto truncate disabled
        cls.process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.2",
                "--chunked-prefill-size",
                "2048",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "16384",
                "--max-running-requests",
                "16",
                "--page-size",
                "64",
                "--max-total-tokens",
                "1000",
                "--context-length",
                "1000",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_invalid_temperature_parameter(self):
        """test invalid temperature parameter"""
        invalid_temperatures = [-1.0]
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        for temp in invalid_temperatures:
            with self.subTest(temperature=temp):
                with self.assertRaises(openai.BadRequestError) as cm:
                    client.completions.create(
                        model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                        prompt="Test prompt",
                        temperature=temp,
                        max_tokens=10,
                    )

                error_message = str(cm.exception).lower()
                self.assertIn("temperature must be non-negative", error_message)
                self.assertTrue(cm.exception.code, 400)

    def test_malformed_json_request(self):
        """test invalid json request"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # build a invalid json request
        malformed_json = '{"prompt": "test", "max_tokens": 10'

        try:
            response = requests.post(
                f"{self.base_url}/v1/completions",
                headers=headers,
                data=malformed_json,
                timeout=10,
            )
            # return 400 rather than failed
            self.assertEqual(response.status_code, 400, f"Expected 400, got {response.status_code}")
        except Exception as e:
            self.fail(f"Server should handle malformed JSON gracefully, but got: {e}")

    def test_invalid_top_p_parameter(self):
        """test invalid top_p parameter"""
        invalid_top_p_values = [-0.5, 1.5]
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        for top_p in invalid_top_p_values:
            with self.subTest(top_p=top_p):
                with self.assertRaises(openai.BadRequestError) as cm:
                    client.completions.create(
                        model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                        prompt="Test prompt",
                        top_p=top_p,
                        max_tokens=10,
                    )
            self.assertIn("top_p must be in (0, 1]", str(cm.exception))
            self.assertTrue(cm.exception.code, 400)

    def test_input_length_longer_than_context_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")
        # Will tokenize to more than context length
        long_text = "hello" * 1200

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )

        # print(str(cm.exception))
        # Error code: 400 - {'object': 'error', 'message': "The input (1205 tokens) is longer than the model's context length (1000 tokens).", 'type': 'BadRequestError', 'param': None, 'code': 400}
        self.assertIn("is longer than the model's context length", str(cm.exception))
        self.assertEqual(cm.exception.code, 400)

    def test_max_tokens_validation(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello"

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
                max_tokens=1200,
            )
        # print(str(cm.exception))
        # Error code: 400 - {'object': 'error', 'message': "Requested token count exceeds the model's maximum context length of 1000 tokens. You requested a total of 1206 tokens: 6 tokens from the input messages and 1200 tokens for the completion. Please reduce the number of tokens in the input messages or the completion to fit within the limit.", 'type': 'BadRequestError', 'param': None, 'code': 400}

        self.assertIn(
            "Requested token count exceeds the model's maximum context",
            str(cm.exception),
        )
        self.assertEqual(cm.exception.code, 400)

    def test_input_length_longer_than_maximum_allowed_length(self):
        client = openai.Client(api_key=self.api_key, base_url=f"{self.base_url}/v1")

        long_text = "hello" * 999

        with self.assertRaises(openai.BadRequestError) as cm:
            client.chat.completions.create(
                model=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
                messages=[
                    {"role": "user", "content": long_text},
                ],
                temperature=0,
            )
        # print(str(cm.exception))
        # Error code: 400 - {'object': 'error', 'message': "The input (1004 tokens) is longer than the model's context length (1000 tokens).", 'type': 'BadRequestError', 'param': None, 'code': 400}
        self.assertIn("is longer than the model's context length", str(cm.exception))
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
