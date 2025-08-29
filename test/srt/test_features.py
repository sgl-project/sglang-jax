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
    - CacheMiss
    - OpenAIServer
    - Logprobs

    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
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
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--attention-backend",
                "fa",
                "--jax-precompile-prefill-token-paddings",
                "16384",
                "--jax-precompile-decode-bs-paddings",
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
                self.assertEqual(
                    future.result()["meta_info"]["finish_reason"]["type"], "abort"
                )

    def test_cache_miss(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            text="the capital of France is",
            temperature=0,
            max_new_tokens=6,
        )

        resp = run_curl(args)

        if "cache_miss_count" not in resp["meta_info"]:
            raise "cache_miss_count is missed in response"
        self.assertEqual(resp["meta_info"]["cache_miss_count"], 0)

    def run_completion(
        self, echo, logprobs, use_list_input, parallel_sample_num, token_input
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
            num_prompt_tokens *= 2
        else:
            prompt_arg = prompt_input
            num_choices = 1

        response = client.completions.create(
            model=self.model,
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
            n=parallel_sample_num,
        )

        assert len(response.choices) == num_choices * parallel_sample_num

        if echo:
            text = response.choices[0].text
            assert text.startswith(prompt)

        if logprobs:
            assert response.choices[0].logprobs
            assert isinstance(response.choices[0].logprobs.tokens[0], str)
            assert isinstance(response.choices[0].logprobs.top_logprobs[1], dict)
            ret_num_top_logprobs = len(response.choices[0].logprobs.top_logprobs[1])

            # FIXME: Sometimes, some top_logprobs are missing in the return value. The reason is that some output id maps to the same output token and duplicate in the map
            # assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"
            assert ret_num_top_logprobs > 0

            # when echo=True and request.logprobs>0, logprob_start_len is 0, so the first token's logprob would be None.
            if not echo:
                assert response.choices[0].logprobs.token_logprobs[0]

        assert response.id
        assert response.created
        assert (
            response.usage.prompt_tokens == num_prompt_tokens
        ), f"{response.usage.prompt_tokens} vs {num_prompt_tokens}"
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_completion_stream(
        self, echo, logprobs, use_list_input, parallel_sample_num, token_input
    ):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        prompt = "The capital of France is"
        if token_input:
            prompt_input = self.tokenizer.encode(prompt)
            num_prompt_tokens = len(prompt_input)
        else:
            prompt_input = prompt
            num_prompt_tokens = len(self.tokenizer.encode(prompt))

        if use_list_input:
            prompt_arg = [prompt_input, prompt_input]
            num_choices = len(prompt_arg)
            num_prompt_tokens *= 2
        else:
            prompt_arg = prompt_input
            num_choices = 1

        generator = client.completions.create(
            model=self.model,
            prompt=prompt_arg,
            temperature=0,
            max_tokens=32,
            echo=echo,
            logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, f"usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, f"usage.completion_tokens was zero"
                assert usage.total_tokens > 0, f"usage.total_tokens was zero"
                continue

            index = response.choices[0].index
            is_first = is_firsts.get(index, True)

            if logprobs:
                assert response.choices[0].logprobs, f"no logprobs in response"
                assert isinstance(
                    response.choices[0].logprobs.tokens[0], str
                ), f"{response.choices[0].logprobs.tokens[0]} is not a string"
                if not (is_first and echo):
                    assert isinstance(
                        response.choices[0].logprobs.top_logprobs[0], dict
                    ), f"top_logprobs was not a dictionary"
                    ret_num_top_logprobs = len(
                        response.choices[0].logprobs.top_logprobs[0]
                    )
                    # FIXME: Sometimes, some top_logprobs are missing in the return value. The reason is that some output id maps to the same output token and duplicate in the map
                    # assert ret_num_top_logprobs == logprobs, f"{ret_num_top_logprobs} vs {logprobs}"
                    assert ret_num_top_logprobs > 0, f"ret_num_top_logprobs was 0"

            if is_first:
                if echo:
                    assert response.choices[0].text.startswith(
                        prompt
                    ), f"{response.choices[0].text} and all args {echo} {logprobs} {token_input} {is_first}"
                is_firsts[index] = False
            assert response.id, f"no id in response"
            assert response.created, f"no created in response"

        for index in [i for i in range(parallel_sample_num * num_choices)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

    def run_chat_completion(self, logprobs, parallel_sample_num):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {
                    "role": "user",
                    "content": "What is the capital of France? Answer in a few words.",
                },
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            n=parallel_sample_num,
        )

        if logprobs:
            assert isinstance(
                response.choices[0].logprobs.content[0].top_logprobs[0].token, str
            )

            ret_num_top_logprobs = len(
                response.choices[0].logprobs.content[0].top_logprobs
            )
            assert (
                ret_num_top_logprobs == logprobs
            ), f"{ret_num_top_logprobs} vs {logprobs}"

        assert len(response.choices) == parallel_sample_num
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.id
        assert response.created
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def run_chat_completion_stream(self, logprobs, parallel_sample_num=1):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        generator = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "What is the capital of France?"},
            ],
            temperature=0,
            logprobs=logprobs is not None and logprobs > 0,
            top_logprobs=logprobs,
            stream=True,
            stream_options={"include_usage": True},
            n=parallel_sample_num,
        )

        is_firsts = {}
        is_finished = {}
        finish_reason_counts = {}
        for response in generator:
            usage = response.usage
            if usage is not None:
                assert usage.prompt_tokens > 0, f"usage.prompt_tokens was zero"
                assert usage.completion_tokens > 0, f"usage.completion_tokens was zero"
                assert usage.total_tokens > 0, f"usage.total_tokens was zero"
                continue

            index = response.choices[0].index
            finish_reason = response.choices[0].finish_reason
            if finish_reason is not None:
                is_finished[index] = True
                finish_reason_counts[index] = finish_reason_counts.get(index, 0) + 1

            data = response.choices[0].delta

            if is_firsts.get(index, True):
                assert (
                    data.role == "assistant"
                ), f"data.role was not 'assistant' for first chunk"
                is_firsts[index] = False
                continue

            if logprobs and not is_finished.get(index, False):
                assert response.choices[0].logprobs, f"logprobs was not returned"
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs[0].token, str
                ), f"top_logprobs token was not a string"
                assert isinstance(
                    response.choices[0].logprobs.content[0].top_logprobs, list
                ), f"top_logprobs was not a list"
                ret_num_top_logprobs = len(
                    response.choices[0].logprobs.content[0].top_logprobs
                )
                assert (
                    ret_num_top_logprobs == logprobs
                ), f"{ret_num_top_logprobs} vs {logprobs}"

            assert (
                isinstance(data.content, str)
                or isinstance(data.reasoning_content, str)
                or (isinstance(data.tool_calls, list) and len(data.tool_calls) > 0)
                or response.choices[0].finish_reason
            )
            assert response.id
            assert response.created

        for index in [i for i in range(parallel_sample_num)]:
            assert not is_firsts.get(
                index, True
            ), f"index {index} is not found in the response"

        # Verify that each choice gets exactly one finish_reason chunk
        for index in range(parallel_sample_num):
            assert (
                index in finish_reason_counts
            ), f"No finish_reason found for index {index}"
            assert (
                finish_reason_counts[index] == 1
            ), f"Expected 1 finish_reason chunk for index {index}, got {finish_reason_counts[index]}"

    def test_completion(self):
        for echo in [False, True]:
            for logprobs in [None]:
                for use_list_input in [True, False]:
                    for parallel_sample_num in [1]:
                        for token_input in [False, True]:
                            self.run_completion(
                                echo,
                                logprobs,
                                use_list_input,
                                parallel_sample_num,
                                token_input,
                            )

    def test_completion_stream(self):
        # parallel sampling and list input are not supported in streaming mode
        for echo in [False, True]:
            for logprobs in [None]:
                for use_list_input in [False]:
                    for parallel_sample_num in [1]:
                        for token_input in [False, True]:
                            self.run_completion_stream(
                                echo,
                                logprobs,
                                use_list_input,
                                parallel_sample_num,
                                token_input,
                            )

    def test_chat_completion(self):
        for logprobs in [None]:
            for parallel_sample_num in [1]:
                self.run_chat_completion(logprobs, parallel_sample_num)

    def test_chat_completion_stream(self):
        for logprobs in [None]:
            for parallel_sample_num in [1]:
                self.run_chat_completion_stream(logprobs, parallel_sample_num)

    def test_penalty(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant"},
                {"role": "user", "content": "Introduce the capital of France."},
            ],
            temperature=0,
            max_tokens=32,
            frequency_penalty=1.0,
        )
        text = response.choices[0].message.content
        assert isinstance(text, str)

    def test_model_list(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        models = list(client.models.list())
        assert len(models) == 1
        assert isinstance(getattr(models[0], "max_model_len", None), int)

    def test_retrieve_model(self):
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Test retrieving an existing model
        retrieved_model = client.models.retrieve(self.model)
        self.assertEqual(retrieved_model.id, self.model)
        self.assertEqual(retrieved_model.root, self.model)

        # Test retrieving a non-existent model
        with self.assertRaises(openai.NotFoundError):
            client.models.retrieve("non-existent-model")

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
                self.assertEqual(
                    real_prob, expected_input_token_ids_logprobs[i - 1][j][0]
                )
                self.assertEqual(
                    real_token, expected_input_token_ids_logprobs[i - 1][j][1]
                )

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
                self.assertEqual(
                    real_token, expected_output_token_ids_logprobs[i][j][1]
                )


if __name__ == "__main__":
    unittest.main()
