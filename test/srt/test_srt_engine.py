"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_1_engine_prompt_ids_output_ids
"""

from typing import List

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.test.test_utils import QWEN3_8B, CustomTestCase


class TestSRTEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = QWEN3_8B
        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            chunked_prefill_size=1024,
            download_dir="/tmp",
            dtype="bfloat16",
            precompile_bs_paddings=[8],
            max_running_requests=8,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_requests=False,
            enable_deterministic_sampling=True,
        )
        cls.tokenizer = get_tokenizer(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def tokenize(self, input_string: str) -> List[int]:
        """Tokenizes the input string."""
        tokenizer = TestSRTEngine.tokenizer
        input_ids = tokenizer.encode(input_string)
        bos_tok = (
            [tokenizer.bos_token_id]
            if tokenizer.bos_token_id is not None
            and tokenizer.bos_token_id
            and input_ids[0] != tokenizer.bos_token_id
            else []
        )
        eos_tok = (
            [tokenizer.eos_token_id]
            if tokenizer.eos_token_id is not None
            and input_ids[-1] != tokenizer.eos_token_id
            else []
        )
        return bos_tok + input_ids + eos_tok

    def test_1_engine_prompt_ids_output_ids(self):
        input_strings = ["the capital of China is", "the capital of France is"]

        sampling_params = TestSRTEngine.engine.get_default_sampling_params()
        sampling_params.max_new_tokens = 10
        sampling_params.n = 1
        sampling_params.temperature = 0
        sampling_params.stop_token_ids = [TestSRTEngine.tokenizer.eos_token_id]
        sampling_params.skip_special_tokens = True

        sampling_params_dict = sampling_params.convert_to_dict()

        prompt_ids_list = [self.tokenize(x) for x in input_strings]
        outputs = TestSRTEngine.engine.generate(
            input_ids=prompt_ids_list,
            sampling_params=[sampling_params_dict] * 2,
        )

        self.assertEqual(len(outputs), 2)
        for item in outputs:
            decoded_output = TestSRTEngine.tokenizer.decode(
                item["output_ids"],
                True,
            )
            self.assertEqual(decoded_output, item["text"])

    def test_2_engine_prompt_ids_with_sample_n_output_ids(self):
        input_strings = ["the capital of China is", "the capital of France is"]

        sampling_params = TestSRTEngine.engine.get_default_sampling_params()
        sampling_params.max_new_tokens = 10
        sampling_params.n = 2
        sampling_params.temperature = 0
        sampling_params.stop_token_ids = [TestSRTEngine.tokenizer.eos_token_id]
        sampling_params.skip_special_tokens = True

        sampling_params_dict = sampling_params.convert_to_dict()

        prompt_ids_list = [self.tokenize(x) for x in input_strings]
        outputs = TestSRTEngine.engine.generate(
            input_ids=prompt_ids_list,
            sampling_params=[sampling_params_dict] * 2,
        )

        self.assertEqual(len(outputs), 4)
        for item in outputs:
            decoded_output = TestSRTEngine.tokenizer.decode(
                item["output_ids"],
                True,
            )
            self.assertEqual(decoded_output, item["text"])

    def test_3_engine_sampling_temperature_top_p_top_k_min_p(self):
        input_strings = ["the capital of France is"]

        def get_sampling_params(max_new_tokens: int = 1):
            sampling_params = TestSRTEngine.engine.get_default_sampling_params()
            sampling_params.max_new_tokens = max_new_tokens
            sampling_params.n = max_new_tokens
            sampling_params.temperature = 0
            sampling_params.stop_token_ids = [TestSRTEngine.tokenizer.eos_token_id]
            sampling_params.skip_special_tokens = True
            return sampling_params

        def update_sampling_params(
            params: SamplingParams,
            temperature: float = None,
            top_p: float = None,
            top_k: int = None,
            min_p: float = None,
            sampling_seed: int = None,
        ):
            if temperature is not None:
                params.temperature = temperature
            if top_p is not None:
                params.top_p = top_p
            if top_k is not None:
                params.top_k = top_k
            if min_p is not None:
                params.min_p = min_p
            if sampling_seed is not None:
                params.sampling_seed = sampling_seed

        cases = {
            "[greedy] temperature[0.0]_top_p[1.0]_top_k[-1]_min_p[0.0]": (
                0.0,
                1.0,
                -1,
                0.0,
            ),
            "[greedy] temperature[0.5]_top_p[1.0]_top_k[1]_min_p[0.0]": (
                0.5,
                1.0,
                1,
                0.0,
            ),
            "[not_greedy_top_p_0.9] temperature[0.6]_top_p[0.9]_top_k[-1]_min_p[0.0]": (
                0.6,
                0.9,
                -1,
                0.0,
            ),
            "[not_greedy_top_k_10] temperature[0.6]_top_p[1.0]_top_k[10]_min_p[0.0]": (
                0.6,
                1.0,
                10,
                0.0,
            ),
            "[not_greedy_min_p_0.5] temperature[0.6]_top_p[1.0]_top_k[-1]_min_p[0.5]": (
                0.6,
                1.0,
                -1,
                0.5,
            ),  # need_min_p_sampling
            "[not_greedy_sampling_seed_36] temperature[0.6]_top_p[1.0]_top_k[-1]_min_p[0.0]_sampling_seed[36]": (
                0.5,
                1.0,
                -1,
                0.0,
                36,
            ),
            "[not_greedy_tempeture_top_p_top_min_p_sampling_seed] temperature[0.5]_top_p[0.9]_top_k[10]_min_p[0.5]_sampling_seed[40]": (
                0.5,
                0.9,
                10,
                0.5,
                40,
            ),
        }

        prompt_ids_list = [self.tokenize(x) for x in input_strings]

        # prefill
        sampling_params_prefill = get_sampling_params(1)
        for case_name, args in cases.items():
            print(f"[prefill, {case_name}] begins to run")
            update_sampling_params(sampling_params_prefill, *args)
            sampling_params_dict = sampling_params_prefill.convert_to_dict()
            outputs = TestSRTEngine.engine.generate(
                input_ids=prompt_ids_list,
                sampling_params=sampling_params_dict,
            )
            self.assertEqual(int(outputs[0]["meta_info"]["cache_miss_count"]), 0)

        # decode
        sampling_params_decode = get_sampling_params(2)
        for case_name, args in cases.items():
            print(f"[decode, {case_name}] begins to run")
            update_sampling_params(sampling_params_decode, *args)
            sampling_params_dict = sampling_params_decode.convert_to_dict()
            outputs = TestSRTEngine.engine.generate(
                input_ids=prompt_ids_list,
                sampling_params=sampling_params_dict,
            )
            self.assertEqual(int(outputs[0]["meta_info"]["cache_miss_count"]), 0)

    def test_4_engine_prompt_ids_with_sample_n_output_ids(self):
        input_strings = ["the capital of China is"]

        sampling_params = TestSRTEngine.engine.get_default_sampling_params()
        sampling_params.max_new_tokens = 10
        sampling_params.n = 20
        sampling_params.temperature = 0
        sampling_params.stop_token_ids = [TestSRTEngine.tokenizer.eos_token_id]
        sampling_params.skip_special_tokens = True
        # sampling_params.sampling_seed= 30

        sampling_params_dict = sampling_params.convert_to_dict()

        prompt_ids_list = [self.tokenize(x) for x in input_strings]
        outputs = TestSRTEngine.engine.generate(
            input_ids=prompt_ids_list,
            sampling_params=[sampling_params_dict] * 2,
        )

        # self.assertEqual(len(outputs), 4)
        # for item in outputs:
        #     decoded_output = TestSRTEngine.tokenizer.decode(
        #         item["output_ids"],
        #         True,
        #     )
        #     self.assertEqual(decoded_output, item["text"])
