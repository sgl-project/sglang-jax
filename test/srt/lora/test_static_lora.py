"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import unittest
from typing import List

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import CustomTestCase

DTYPE = "bfloat16"

PROMPT = """
### Instruction:
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
"""


class TestStaticLoRA(CustomTestCase):
    def test_diff_after_apply_dummy(self):
        print("=================== test_diff_after_apply_dummy =======================")
        model_path = "Qwen/Qwen3-4B"
        lora_target_modules = ["gate_proj"]
        import os

        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"
        engine = Engine(
            model_path=model_path,
            trust_remote_code=True,
            tp_size=1,
            device="tpu",
            random_seed=3,
            node_rank=0,
            mem_fraction_static=0.6,
            chunked_prefill_size=1024,
            download_dir="/tmp",
            dtype=DTYPE,
            precompile_bs_paddings=[8],
            max_running_requests=8,
            skip_server_warmup=True,
            attention_backend="fa",
            precompile_token_paddings=[1024],
            page_size=64,
            log_requests=False,
            enable_deterministic_sampling=True,
            enable_static_lora=True,
            max_loras_per_batch=1,
            lora_scaling=0.5,
            max_lora_rank=8,
            lora_target_modules=lora_target_modules,
            enable_single_process=True,
        )

        tokenizer = get_tokenizer(model_path)

        def tokenize(input_string: str) -> List[int]:
            """Tokenizes the input string."""
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
                if tokenizer.eos_token_id is not None and input_ids[-1] != tokenizer.eos_token_id
                else []
            )
            return bos_tok + input_ids + eos_tok

        sampling_params = engine.get_default_sampling_params()
        sampling_params.max_new_tokens = 20
        sampling_params.temperature = 0.0

        sampling_params_dict = sampling_params.convert_to_dict()

        try:
            prompt_ids = tokenize(PROMPT)
            outputs_base = engine.generate(
                input_ids=prompt_ids,
                sampling_params=sampling_params_dict,
            )

            print(f"{outputs_base['output_ids']=}")

            engine.apply_dummy_lora_ab_buffer(target_modules=lora_target_modules)

            outputs_updated_lora_ab = engine.generate(
                input_ids=prompt_ids,
                sampling_params=sampling_params_dict,
            )

            print(f"{outputs_updated_lora_ab['output_ids']=}")

            self.assertNotEqual(outputs_base["output_ids"], outputs_updated_lora_ab["output_ids"])

        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main(warnings="ignore")
