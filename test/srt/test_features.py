import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import openai
import requests

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.run_curl import run_curl
from sgl_jax.test.run_eval import run_eval
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestFeatures(CustomTestCase):
    """
    Including:
    - BasicFeatures
      - ChunkPrefillSize
      - PageSizeGreaterThanOne
    - Abort
    - Logprobs
    - Penalties
      - FrequencyPenalty
      - PresencePenalty
      - MinNewTokens
      - CombinedPenalties

    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
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
                "--tp-size",
                "4",
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
                "--precompile-token-paddings",
                "16384",
                "--precompile-bs-paddings",
                "64",
                "--page-size",
                "64",
                "--max-running-requests",
                "64",
                "--chunked-prefill-size",
                "8192",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_features(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=64,
            max_tokens=1024,
        )

        metrics = run_eval(args)
        print(metrics)
        self.assertGreater(metrics["score"], 0.45)

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
                self.assertEqual(future.result()["meta_info"]["finish_reason"]["type"], "abort")

    def test_cache_miss_prefill(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=1,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)

    def test_cache_miss_decode(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=2,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)

    def test_logprobs(self):
        # Note: add test_logprobs until accuracy score is relatively high, we will update the following expected logits.
        # Now every accuracy improvement may result in tiny differences in value, so skip it now and support it in the future.
        return
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=6,
            return_logprob=True,
            top_logprobs_num=3,
            token_ids_logprob=[9370, 105180, 1],
            logprob_start_len=1,
        )

        resp = run_curl(args)

        # deal with result
        expected_input_token_logprobs = [
            [
                -0.58203125,
                315,
            ],
            [
                -3.53125,
                9625,
            ],
            [
                -2.84375,
                374,
            ],
        ]
        real_input_token_logprobs = resp["meta_info"]["input_token_logprobs"]
        for i, pair in enumerate(real_input_token_logprobs):
            if i == 0:
                continue
            real_prob, real_token = pair[0], pair[1]
            self.assertEqual(real_prob, expected_input_token_logprobs[i - 1][0])
            self.assertEqual(real_token, expected_input_token_logprobs[i - 1][1])

        expected_output_token_logprobs = [
            [
                -0.73046875,
                12095,
            ],
            [
                -0.9765625,
                13,
            ],
            [
                -1.3984375,
                1084,
            ],
            [
                -0.171875,
                374,
            ],
            [
                -0.42578125,
                7407,
            ],
            [
                -0.060546875,
                304,
            ],
        ]
        real_output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        for i, pair in enumerate(real_output_token_logprobs):
            real_prob, real_token = pair[0], pair[1]
            self.assertEqual(real_prob, expected_output_token_logprobs[i][0])
            self.assertEqual(real_token, expected_output_token_logprobs[i][1])

        expected_input_top_logprobs = [
            [
                [
                    -0.58203125,
                    315,
                ],
                [
                    -2.140625,
                    3283,
                ],
                [
                    -3.765625,
                    374,
                ],
            ],
            [
                [
                    -1.0390625,
                    279,
                ],
                [
                    -3.53125,
                    9625,
                ],
                [
                    -3.96875,
                    5616,
                ],
            ],
            [
                [
                    -2.09375,
                    13,
                ],
                [
                    -2.34375,
                    58883,
                ],
                [
                    -2.40625,
                    11,
                ],
            ],
        ]
        real_input_top_logprobs = resp["meta_info"]["input_top_logprobs"]
        for i, subitem in enumerate(real_input_top_logprobs):
            if i == 0:
                continue
            for j, pair in enumerate(subitem):
                real_prob, real_token = pair[0], pair[1]
                self.assertEqual(real_prob, expected_input_top_logprobs[i - 1][j][0])
                self.assertEqual(real_token, expected_input_top_logprobs[i - 1][j][1])

        expected_output_top_logprobs = [
            [
                [
                    -0.73046875,
                    12095,
                ],
                [
                    -2.734375,
                    2130,
                ],
                [
                    -3.046875,
                    32671,
                ],
            ],
            [
                [
                    -0.9765625,
                    13,
                ],
                [
                    -1.7265625,
                    58883,
                ],
                [
                    -2.53125,
                    11,
                ],
            ],
            [
                [
                    -1.3984375,
                    1084,
                ],
                [
                    -2.21875,
                    576,
                ],
                [
                    -2.40625,
                    12095,
                ],
            ],
            [
                [
                    -0.171875,
                    374,
                ],
                [
                    -2.296875,
                    594,
                ],
                [
                    -3.109375,
                    702,
                ],
            ],
            [
                [
                    -0.42578125,
                    7407,
                ],
                [
                    -2.296875,
                    264,
                ],
                [
                    -2.296875,
                    3881,
                ],
            ],
            [
                [
                    -0.060546875,
                    304,
                ],
                [
                    -3.0625,
                    389,
                ],
                [
                    -5.9375,
                    198,
                ],
            ],
        ]
        real_output_top_logprobs = resp["meta_info"]["output_top_logprobs"]
        for i, subitem in enumerate(real_output_top_logprobs):
            for j, pair in enumerate(subitem):
                real_prob, real_token = pair[0], pair[1]
                self.assertEqual(real_prob, expected_output_top_logprobs[i][j][0])
                self.assertEqual(real_token, expected_output_top_logprobs[i][j][1])

        expected_input_token_ids_logprobs = [
            [
                [
                    -8.1875,
                    9370,
                ],
                [
                    -15,
                    105180,
                ],
                [
                    -4.59375,
                    1,
                ],
            ],
            [
                [
                    -11,
                    9370,
                ],
                [
                    -18.125,
                    105180,
                ],
                [
                    -5.40625,
                    1,
                ],
            ],
            [
                [
                    -8.125,
                    9370,
                ],
                [
                    -13.8125,
                    105180,
                ],
                [
                    -3.28125,
                    1,
                ],
            ],
        ]
        real_input_token_ids_logprobs = resp["meta_info"]["input_token_ids_logprobs"]
        for i, subitem in enumerate(real_input_token_ids_logprobs):
            if i == 0:
                continue
            for j, pair in enumerate(subitem):
                real_prob, real_token = pair[0], pair[1]
                self.assertEqual(real_prob, expected_input_token_ids_logprobs[i - 1][j][0])
                self.assertEqual(real_token, expected_input_token_ids_logprobs[i - 1][j][1])

        expected_output_token_ids_logprobs = [
            [
                [
                    -9.8125,
                    9370,
                ],
                [
                    -16.125,
                    105180,
                ],
                [
                    -4.40625,
                    1,
                ],
            ],
            [
                [
                    -9.5,
                    9370,
                ],
                [
                    -15.5,
                    105180,
                ],
                [
                    -3.84375,
                    1,
                ],
            ],
            [
                [
                    -6.5625,
                    9370,
                ],
                [
                    -14.875,
                    105180,
                ],
                [
                    -16.125,
                    1,
                ],
            ],
            [
                [
                    -15.125,
                    9370,
                ],
                [
                    -18,
                    105180,
                ],
                [
                    -11.3125,
                    1,
                ],
            ],
            [
                [
                    -17.625,
                    9370,
                ],
                [
                    -22.375,
                    105180,
                ],
                [
                    -14.5,
                    1,
                ],
            ],
            [
                [
                    -17.875,
                    9370,
                ],
                [
                    -23.625,
                    105180,
                ],
                [
                    -12.5,
                    1,
                ],
            ],
        ]
        real_output_token_ids_logprobs = resp["meta_info"]["output_token_ids_logprobs"]
        for i, subitem in enumerate(real_output_token_ids_logprobs):
            for j, pair in enumerate(subitem):
                real_prob, real_token = pair[0], pair[1]
                self.assertEqual(real_prob, expected_output_token_ids_logprobs[i][j][0])
                self.assertEqual(real_token, expected_output_token_ids_logprobs[i][j][1])

    def test_frequency_penalty(self):
        """Test frequency penalty functionality."""
        # Test with frequency penalty enabled
        args = SimpleNamespace(
            base_url=self.base_url,
            text="Say hello hello hello",
            temperature=0.5,
            max_new_tokens=1,
            frequency_penalty=0.2,
        )

        # Test frequency penalty = 1.0
        resp_with_penalty = run_curl(args)

        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[frequency_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

        # decode
        args = SimpleNamespace(
            base_url=self.base_url,
            text="The weather is nice today. The weather",
            temperature=0.5,
            max_new_tokens=2,
            frequency_penalty=0.2,
        )
        resp_with_penalty = run_curl(args)
        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[frequency_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

    def test_presence_penalty(self):
        """Test presence penalty functionality."""
        args = SimpleNamespace(
            base_url=self.base_url,
            text="The weather is nice today. The weather",
            temperature=0.5,
            max_new_tokens=1,
            presence_penalty=0.2,
        )

        resp_with_penalty = run_curl(args)
        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[presence_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

        # decode
        args = SimpleNamespace(
            base_url=self.base_url,
            text="The weather is nice today. The weather",
            temperature=0.5,
            max_new_tokens=2,
            presence_penalty=0.2,
        )

        resp_with_penalty = run_curl(args)

        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[presence_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

    def test_min_new_tokens_penalty(self):
        """Test min_new_tokens penalty functionality."""
        args = SimpleNamespace(
            base_url=self.base_url,
            text="Hello",
            temperature=0.0,
            max_new_tokens=1,
            min_new_tokens=1,
        )

        resp_with_min_tokens = run_curl(args)

        if "cache_miss_count" not in resp_with_min_tokens["meta_info"]:
            raise "[min_new_tokens] cache_miss_count is missed in response"
        self.assertEqual(resp_with_min_tokens["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_min_tokens)

        # decode
        args = SimpleNamespace(
            base_url=self.base_url,
            text="Hello",
            temperature=0.0,
            max_new_tokens=2,
            min_new_tokens=2,
        )

        resp_with_min_tokens = run_curl(args)
        if "cache_miss_count" not in resp_with_min_tokens["meta_info"]:
            raise "[min_new_tokens] cache_miss_count is missed in response"
        self.assertEqual(resp_with_min_tokens["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_min_tokens)

    def test_combined_penalties(self):
        """Test multiple penalties applied together."""
        penalty_params = [
            {},
            {},
            {},
            {"frequency_penalty": 2},
            {"presence_penalty": 1},
            {"min_new_tokens": 16},
            {"frequency_penalty": 0.2},
            {"presence_penalty": 0.4},
            {"min_new_tokens": 8},
            {"frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"frequency_penalty": 0.4, "min_new_tokens": 12},
            {"presence_penalty": 0.8, "min_new_tokens": 12},
            {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
            {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
        ]

        for penalty_param in penalty_params:
            args = SimpleNamespace(
                base_url=self.base_url,
                text="Tell me about cats. Cats are",
                temperature=0.5,
                max_new_tokens=32,
                **penalty_param,
            )
            resp_combined = run_curl(args)
            if "cache_miss_count" not in resp_combined["meta_info"]:
                raise "[combined_penalties] cache_miss_count is missed in response"
            self.assertIn("text", resp_combined)


class TestNoOverlapSchedule(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
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
                "--tp-size",
                "4",
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
                "--precompile-token-paddings",
                "16384",
                "--precompile-bs-paddings",
                "64",
                "--page-size",
                "64",
                "--max-running-requests",
                "64",
                "--chunked-prefill-size",
                "8192",
                "--disable-overlap-schedule",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_frequency_penalty(self):
        """Test frequency penalty functionality."""
        # Test with frequency penalty enabled
        args = SimpleNamespace(
            base_url=self.base_url,
            text="Say hello hello hello",
            temperature=0.5,
            max_new_tokens=1,
            frequency_penalty=0.2,
        )

        # Test frequency penalty = 1.0
        resp_with_penalty = run_curl(args)

        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[frequency_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

        # decode
        args = SimpleNamespace(
            base_url=self.base_url,
            text="The weather is nice today. The weather",
            temperature=0.5,
            max_new_tokens=2,
            frequency_penalty=0.2,
        )
        resp_with_penalty = run_curl(args)
        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[frequency_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

    def test_presence_penalty(self):
        """Test presence penalty functionality."""
        args = SimpleNamespace(
            base_url=self.base_url,
            text="The weather is nice today. The weather",
            temperature=0.5,
            max_new_tokens=1,
            presence_penalty=0.2,
        )

        resp_with_penalty = run_curl(args)
        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[presence_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

        args = SimpleNamespace(
            base_url=self.base_url,
            text="The weather is nice today. The weather",
            temperature=0.5,
            max_new_tokens=2,
            presence_penalty=0.2,
        )

        resp_with_penalty = run_curl(args)

        if "cache_miss_count" not in resp_with_penalty["meta_info"]:
            raise "[presence_penalty] cache_miss_count is missed in response"
        self.assertEqual(resp_with_penalty["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_penalty)

    def test_min_new_tokens_penalty(self):
        """Test min_new_tokens penalty functionality."""
        args = SimpleNamespace(
            base_url=self.base_url,
            text="Hello",
            temperature=0.0,
            max_new_tokens=1,
            min_new_tokens=1,
        )

        resp_with_min_tokens = run_curl(args)
        if "cache_miss_count" not in resp_with_min_tokens["meta_info"]:
            raise "[min_new_tokens] cache_miss_count is missed in response"
        self.assertEqual(resp_with_min_tokens["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_min_tokens)

        args = SimpleNamespace(
            base_url=self.base_url,
            text="Hello",
            temperature=0.0,
            max_new_tokens=2,
            min_new_tokens=2,
        )

        # decode
        # Test with min_new_tokens
        resp_with_min_tokens = run_curl(args)
        if "cache_miss_count" not in resp_with_min_tokens["meta_info"]:
            raise "[min_new_tokens] cache_miss_count is missed in response"
        self.assertEqual(resp_with_min_tokens["meta_info"]["cache_miss_count"], 0)
        self.assertIn("text", resp_with_min_tokens)

    def test_combined_penalties(self):
        """Test multiple penalties applied together."""
        penalty_params = [
            {},
            {},
            {},
            {"frequency_penalty": 2},
            {"presence_penalty": 1},
            {"min_new_tokens": 16},
            {"frequency_penalty": 0.2},
            {"presence_penalty": 0.4},
            {"min_new_tokens": 8},
            {"frequency_penalty": 0.4, "presence_penalty": 0.8},
            {"frequency_penalty": 0.4, "min_new_tokens": 12},
            {"presence_penalty": 0.8, "min_new_tokens": 12},
            {"presence_penalty": -0.3, "frequency_penalty": 1.3, "min_new_tokens": 32},
            {"presence_penalty": 0.3, "frequency_penalty": -1.3, "min_new_tokens": 32},
        ]

        for penalty_param in penalty_params:
            args = SimpleNamespace(
                base_url=self.base_url,
                text="Tell me about cats. Cats are",
                temperature=0.5,
                max_new_tokens=32,
                **penalty_param,
            )
            resp_combined = run_curl(args)
            if "cache_miss_count" not in resp_combined["meta_info"]:
                raise "[combined_penalties] cache_miss_count is missed in response"
            self.assertIn("text", resp_combined)


cache_misses_common_args = [
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
    "--precompile-token-paddings",
    "16384",
    "--precompile-bs-paddings",
    "64",
    "--page-size",
    "64",
    "--max-running-requests",
    "64",
    "--chunked-prefill-size",
    "8192",
]


class TestCacheMissesTP1(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=cache_misses_common_args + ["--tp-size", "1"],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_cache_miss_prefill(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=1,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)

    def test_cache_miss_decode(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=2,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)


class TestCacheMissesTP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="tpu",
            other_args=cache_misses_common_args + ["--tp-size", "4"],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_cache_miss_prefill(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=1,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)

    def test_cache_miss_decode(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=2,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)

    def test_cache_miss_with_sampling_params(self):
        # prefill
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            top_k=5,
            top_p=0.5,
            min_p=0.2,
            max_new_tokens=1,
        )
        resp = run_curl(args)
        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)
        # decode
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            top_k=5,
            top_p=0.5,
            min_p=0.2,
            max_new_tokens=2,
        )
        resp = run_curl(args)
        if "cache_miss_count" not in resp["meta_info"]:
            raise "[prefill] cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)


if __name__ == "__main__":
    unittest.main()
