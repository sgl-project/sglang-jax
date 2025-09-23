"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_1_engine_prompt_ids_output_ids
"""

from typing import List


from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import (
    CustomTestCase,
    QWEN3_8B,
)
from sgl_jax.srt.entrypoints.engine import Engine


class TestSRTEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_path = QWEN3_8B
        cls.engine = Engine(
            model_path= cls.model_path, 
            trust_remote_code=True, 
            tp_size=1, 
            device='tpu',
            random_seed=3, 
            node_rank=0, 
            mem_fraction_static=0.6, 
            chunked_prefill_size=1024, 
            download_dir='/tmp', 
            dtype='bfloat16', 
            precompile_bs_paddings = [8], 
            max_running_requests = 8, 
            skip_server_warmup=True, 
            attention_backend='fa',
            precompile_token_paddings=[1024], 
            page_size=64,
            log_requests=False,
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
            if tokenizer.bos_token_id is not None and tokenizer.bos_token_id and input_ids[0] != tokenizer.bos_token_id
            else []
        )
        eos_tok = (
            [tokenizer.eos_token_id]
            if tokenizer.eos_token_id is not None and input_ids[-1] != tokenizer.eos_token_id
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
            sampling_params=[sampling_params_dict]*2,
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
            sampling_params=[sampling_params_dict]*2,
        )

        self.assertEqual(len(outputs), 4)
        for item in outputs:
            decoded_output = TestSRTEngine.tokenizer.decode(
                item["output_ids"],
                True,
            )
            self.assertEqual(decoded_output, item["text"])  