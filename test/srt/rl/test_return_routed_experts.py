import os
import time
import unittest

import numpy as np
import requests

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.layers.routed_experts_capturer import (
    extract_routed_experts_from_meta_info,
)
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_MOE_30B,
    CustomTestCase,
    popen_launch_server,
)

PROMPT1 = "AI is a field of computer science focused on"
PROMPT2 = "Computer science is the study of"
PROMPT3 = "Write a short story."


class TestReturnRoutedExperts(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = QWEN3_MOE_30B
        cls.model_num_hidden_layers = 2
        cls.model_num_experts_per_tok = 8
        cls.multi_prompts_prefill_baseline_file_name = (
            "test/srt/rl/dumped_data/return_routed_experts_prefill_baseline.txt"
        )
        cls.multi_prompts_decode_baseline_file_name = (
            "test/srt/rl/dumped_data/return_routed_experts_decode_baseline.txt"
        )
        cls.prompt1_tokenized_num = 9
        cls.prompt2_tokenized_num = 6

        # cls.generate_baseline_data()
        cls.prompt1_seq_layer_topk, cls.prompt2_seq_layer_topk, cls.prompt3_seq_layer_topk = (
            cls.get_baseline_data()
        )

    @classmethod
    def generate_baseline_data(cls):
        """
        Note: reserve this func to generate more results for future if it is necessary
        Steps:
        cd test/srt/rl/dumped_data
        python3 ../test_return_routed_experts.py
        There are some post-operations, please contact to aolemila.
        """
        print(f"Initialize an Engine to get baseline expert ids!")
        os.environ["DUMP_TOPK_IDS_FILEINFO"] = f"prompt_prefill.txt,prompt_decode.txt"
        engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=4,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.95,
            chunked_prefill_size=64,
            download_dir="/tmp",
            dtype="float32",
            precompile_bs_paddings=[64],
            max_running_requests=64,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[64],
            page_size=16,
            log_requests=False,
            enable_deterministic_sampling=True,
            watchdog_timeout=3000,
            enable_return_routed_experts=True,
            disable_overlap_schedule=True,
            model_layer_nums=2,
        )

        time.sleep(5)

        sampling_params = {
            "max_new_tokens": 2,
            "temperature": 0.0,
        }

        prompts = [PROMPT1, PROMPT2, PROMPT3]
        engine.generate(
            prompt=prompts,
            sampling_params=[sampling_params] * len(prompts),
            return_routed_experts=[True, True, False],
        )

        engine.shutdown()
        print(f"Shutdown the Engine to get baseline expert ids!")

    @classmethod
    def get_baseline_data(cls):
        prefill_data = np.array(
            [
                61,
                95,
                64,
                91,
                47,
                75,
                4,
                37,
                68,
                114,
                55,
                90,
                0,
                9,
                126,
                28,
                120,
                104,
                16,
                114,
                53,
                107,
                0,
                111,
                126,
                86,
                107,
                109,
                13,
                3,
                22,
                14,
                106,
                27,
                50,
                110,
                114,
                0,
                48,
                32,
                86,
                99,
                116,
                3,
                109,
                126,
                107,
                13,
                2,
                23,
                112,
                22,
                28,
                91,
                36,
                62,
                29,
                114,
                90,
                67,
                48,
                13,
                109,
                68,
                68,
                107,
                31,
                69,
                53,
                101,
                81,
                93,
                75,
                83,
                29,
                85,
                103,
                66,
                25,
                2,
                91,
                87,
                86,
                17,
                95,
                105,
                112,
                28,
                90,
                60,
                75,
                68,
                114,
                13,
                103,
                8,
                86,
                112,
                87,
                91,
                105,
                122,
                70,
                66,
                38,
                13,
                104,
                14,
                25,
                66,
                67,
                52,
                120,
                116,
                126,
                29,
                67,
                77,
                12,
                51,
                11,
                107,
                13,
                14,
                45,
                114,
                99,
                86,
                81,
                69,
                93,
                61,
                31,
                53,
                92,
                120,
                83,
                39,
                91,
                76,
                45,
                46,
                86,
                41,
                86,
                17,
                87,
                75,
                105,
                95,
                91,
                6,
                68,
                114,
                90,
                0,
                55,
                9,
                104,
                96,
                86,
                87,
                112,
                91,
                105,
                122,
                124,
                30,
                13,
                104,
                14,
                38,
                25,
                20,
                66,
                16,
                120,
                16,
                104,
                114,
                53,
                107,
                71,
                0,
                86,
                13,
                107,
                2,
                104,
                14,
                96,
                34,
                106,
                57,
                114,
                127,
                71,
                110,
                103,
                50,
                86,
                48,
                9,
                116,
                3,
                106,
                96,
                32,
                87,
                77,
                23,
                112,
                28,
                66,
                91,
                29,
                48,
                45,
                67,
                16,
                29,
                114,
                86,
                88,
                68,
                107,
                81,
                31,
                69,
                53,
                93,
                57,
                75,
                83,
                67,
                45,
                91,
                62,
                48,
                52,
                126,
                3,
                97,
                21,
                75,
                2,
                124,
                36,
                68,
                114,
                55,
                0,
                9,
                90,
                88,
                6,
                106,
                27,
                61,
                114,
                50,
                110,
                78,
                49,
                88,
                99,
                45,
                62,
                3,
                105,
                91,
                4,
                34,
                17,
                87,
                62,
                61,
                28,
                124,
                125,
                88,
                99,
                45,
                48,
                75,
                7,
                90,
                4,
                87,
                125,
                23,
                124,
                41,
                28,
                105,
                30,
                45,
                67,
                38,
                98,
                88,
                7,
                29,
                46,
                98,
                65,
                114,
                43,
                45,
                73,
                72,
                106,
                18,
                94,
                38,
                119,
                127,
                70,
                80,
                50,
            ],
            dtype=np.int32,
        )
        decode_data = np.array(
            [
                82,
                64,
                33,
                91,
                90,
                28,
                72,
                51,
                55,
                106,
                19,
                60,
                64,
                9,
                70,
                28,
                126,
                77,
                87,
                3,
                67,
                62,
                91,
                29,
                75,
                78,
                45,
                123,
                114,
                67,
                52,
                83,
                82,
                33,
                9,
                90,
                64,
                22,
                39,
                20,
                37,
                55,
                44,
                6,
                19,
                88,
                106,
                91,
            ],
            dtype=np.int32,
        )
        reshaped_prefill_data = prefill_data.reshape(
            -1, cls.model_num_hidden_layers, cls.model_num_experts_per_tok
        )
        reshaped_decode_data = decode_data.reshape(
            -1, cls.model_num_hidden_layers, cls.model_num_experts_per_tok
        )

        prompt1_prefill_data = reshaped_prefill_data[: cls.prompt1_tokenized_num]
        prompt2_prefill_data = reshaped_prefill_data[
            cls.prompt1_tokenized_num : cls.prompt1_tokenized_num + cls.prompt2_tokenized_num
        ]
        prompt1_decode_data = reshaped_decode_data[:1]
        prompt2_decode_data = reshaped_decode_data[1:2]

        return (
            np.concatenate([prompt1_prefill_data, prompt1_decode_data], axis=0),
            np.concatenate([prompt2_prefill_data, prompt2_decode_data], axis=0),
            None,
        )

    def case_01_multi_mixed_prompts_with_engine(self):
        """
        Mixed means requests with return_routed_experts and without return_routed_experts are sent together
        1. Use the same test_01 engine
        2. Use engine.async_generate() to get response concurrently for PROMPT1, PROMPT2, PROMPT3
        3. Compare the every item with parsed_response and cls.prompt1|2|3_seq_layer_topk_ids, ensure every item is the same
        4. Shutdown engine
        """
        print(f"Begin to run case_01_multi_mixed_prompts_with_engine!", flush=True)
        os.unsetenv("DUMP_TOPK_IDS_FILEINFO")
        # Launch engine with same arguments as baseline, plus enable_return_routed_experts
        engine = Engine(
            model_path=self.model_path,
            trust_remote_code=True,
            tp_size=4,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.95,
            chunked_prefill_size=64,
            download_dir="/tmp",
            dtype="float32",
            precompile_bs_paddings=[64],
            max_running_requests=64,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[64],
            page_size=16,
            log_requests=False,
            enable_deterministic_sampling=True,
            watchdog_timeout=3000,
            enable_return_routed_experts=True,  # Enable routed experts feature
            model_layer_nums=2,
        )

        # Sampling params with return_routed_experts enabled
        sampling_params = {
            "max_new_tokens": 64,
            "temperature": 0.0,
        }

        # Send async requests for all three prompts
        prompts = [PROMPT1, PROMPT2, PROMPT3]
        outputs = engine.generate(
            prompt=prompts,
            sampling_params=[sampling_params] * len(prompts),
            return_routed_experts=[True, True, False],
        )

        # Expected baselines for each prompt
        baselines = [
            self.prompt1_seq_layer_topk,
            self.prompt2_seq_layer_topk,
            self.prompt3_seq_layer_topk,
        ]

        # Collect results and compare each response
        for idx, (output, baseline) in enumerate(zip(outputs, baselines)):
            # Extract expert IDs from response
            expert_ids = extract_routed_experts_from_meta_info(output)

            if idx == 2:
                assert expert_ids == None, f"expert_ids of {idx+1=} are expected to be None"
                continue

            # Reshape to [seq_len, num_layers, num_experts_per_tok]
            seq_len = len(expert_ids) // (
                self.model_num_hidden_layers * self.model_num_experts_per_tok
            )
            expert_ids = expert_ids.reshape(
                seq_len, self.model_num_hidden_layers, self.model_num_experts_per_tok
            )
            # adjust only compare the first two tokens
            expert_ids = expert_ids[:-62]

            np.testing.assert_array_equal(
                expert_ids,
                baseline,
                f"Prompt {idx+1}: Expert IDs mismatch! Shape: {expert_ids.shape} vs {baseline.shape}, Items: {expert_ids} vs {baseline}",
            )

        # Shutdown engine
        engine.shutdown()
        time.sleep(10)
        print(
            f"✅Successfully complete running case_01_multi_mixed_prompts_with_engine!", flush=True
        )

    def case_02_single_prompt_with_enable_single_process_http_server(self):
        """
        1. Use the same arguments with test_01 to launch the http_server
        2. Call /generate endpoint to get response
        3. Compare the every item with parsed_response and cls.prompt1_seq_layer_topk_ids, ensure every item is the same
        4. Close http_server
        """
        print(f"Begin to run case_02_single_prompt_with_enable_single_process_http_server!")
        os.unsetenv("DUMP_TOPK_IDS_FILEINFO")
        # Launch HTTP server with enable_return_routed_experts
        base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            self.model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "4",
                "--device",
                "tpu",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.95",
                "--chunked-prefill-size",
                "64",
                "--download-dir",
                "/tmp",
                "--dtype",
                "float32",
                "--precompile-bs-paddings",
                "64",
                "--max-running-requests",
                "64",
                "--skip-server-warmup",
                "--attention-backend",
                "fa",
                "--precompile-token-paddings",
                "64",
                "--page-size",
                "16",
                "--enable-deterministic-sampling",
                "--watchdog-timeout",
                "3000",
                "--enable-return-routed-experts",  # Enable routed experts feature
                "--model-layer-nums",
                "2",
                "--enable-single-process",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jit_cache",
            },
        )

        try:
            # Call /generate endpoint for PROMPT1
            response = requests.post(
                base_url + "/generate",
                json={
                    "text": PROMPT1,
                    "sampling_params": {
                        "temperature": 0.0,
                        "max_new_tokens": 2,
                    },
                    "return_routed_experts": True,
                },
            )

            # Get the response JSON
            result = response.json()

            # Extract expert IDs from meta_info
            expert_ids = extract_routed_experts_from_meta_info(result)

            # Reshape to [seq_len, num_layers, num_experts_per_tok]
            seq_len = len(expert_ids) // (
                self.model_num_hidden_layers * self.model_num_experts_per_tok
            )
            expert_ids = expert_ids.reshape(
                seq_len, self.model_num_hidden_layers, self.model_num_experts_per_tok
            )

            # Compare with baseline
            baseline = self.prompt1_seq_layer_topk
            np.testing.assert_array_equal(
                expert_ids,
                baseline,
                f"Expert IDs mismatch! Shape: {expert_ids.shape} vs {baseline.shape}, Items: {expert_ids} vs {baseline}",
            )

        finally:
            # Shutdown the HTTP server
            kill_process_tree(process.pid)

        print(
            f"✅Successfully complete running case_02_single_prompt_with_enable_single_process_http_server!"
        )

    def test_run_test_one_by_one(self):
        self.case_01_multi_mixed_prompts_with_engine()
        self.case_02_single_prompt_with_enable_single_process_http_server()

    @classmethod
    def tearDownClass(cls):
        pass


if __name__ == "__main__":
    unittest.main()
