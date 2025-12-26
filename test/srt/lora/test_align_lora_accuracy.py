from ast import dump
import unittest
from typing import List
import os
import json

import numpy as np


from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import CustomTestCase

LORA_SETS = [
    {
        "base": "Qwen/Qwen3-4B",
        "lora": [
            "y9760210/Qwen3-4B-lora_model",
        ],
    },
]

LORA_PATH = ["y9760210/Qwen3-4B-lora_model"]
BASE_MODEL = "Qwen/Qwen3-4B" 
# Note: According to https://github.com/sgl-project/sglang-jax/issues/587, adjust dtype from bfloat16 to float32.
DTYPE = "float32" 

PROMPT = [
    "AI is a field of computer science focused on",
]
THRESHOLD = 2e-3


class TestStaticLoRA(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        local_dataset_path = os.path.join(
            MOUNT_ROOT, "dataset/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json"
        )

        if os.path.exists(local_dataset_path):
            print(f"Using path: {local_dataset_path}")
            cls.sharegpt_dataset_path = local_dataset_path
        else:
            print(f"Local dataset not found at '{local_dataset_path}'.")
            cls.sharegpt_dataset_path = download_and_cache_file(SHAREGPT_URL)

        print(f"Dataset is ready at location: {cls.sharegpt_dataset_path}")


    def setupClass(cls):
        print("=================== test_diff_after_apply_dummy =======================")
        model_path = "Qwen/Qwen3-4B"
        lora_target_modules = ["all"]
        import os

        os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jit_cache"
        os.environ["JAX_PLATFORMS"] = "cpu"
        cls.prefill_dump_filename = f"./{'_'.join(LORA_PATH[0].split('/'))}/prefill.json"
        cls.decode_dump_filename = f"./{'_'.join(LORA_PATH[0].split('/'))}/decode.json"
        os.environ["DUMP_LAST_LAYER_LOGITS_FILENAMES"] = f"{cls.prefill_dump_filename},{cls.decode_dump_filename}"
        cls.engine = Engine(
            model_path=model_path,
            trust_remote_code=True,
            tp_size=1,
            device="cpu",
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
            lora_paths=LORA_SETS['loras'],
            max_loras_per_batch=3,
            lora_target_modules=lora_target_modules,
        )

        cls.tokenizer = get_tokenizer(model_path)

    def get_sglang_jax_last_layer_logits(self):
        sampling_params= {
            "max_new_tokens":2, # prefill and decode
            "temperature": 0.0,
        },

        outputs = self.engine.generate(
            prompt=PROMPT,
            sampling_params=sampling_params,
            lora_path=LORA_PATH,
        )

        self.assertEqual(len(outputs), 2)

        if not os.path.exists(self.prefill_dump_filename) or not os.path.exists(self.decode_dump_filename):
            raise ValueError(f"{self.prefill_dump_filename} or {self.decode_dump_filename} does not exist!")

        with open(self.prefill_dump_filename, 'r') as prefill_f, open(self.decode_dump_filename, 'r') as decode_f:
            prefill_data = json.load(prefill_f)
            prefill_logits = np.array(prefill_data,dtype=np.float32)
            decode_data = json.load(decode_f)
            decode_logits = np.array(decode_data,dtype=np.float32)
        return prefill_logits, decode_logits

        
    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_align_last_layer_logits_with_hf(self):
        # Run logits comparison test if reference file exists
        hf_lora_prefill_logits_file = os.path.join(os.path.dirname(__file__), "hf_lora_prefill_logits.json")
        hf_lora_decode_logits_file = os.path.join(os.path.dirname(__file__), "hf_lora_decode_logits.json")

        assert os.path.exists(hf_lora_prefill_logits_file) and os.path.exists(hf_lora_decode_logits_file), f"{hf_lora_prefill_logits_file} or {hf_lora_decode_logits_file} does not exist, please generate them firstly!"

        """Compare logprobs from sglang-jax with HuggingFace reference."""
        print("=================== testing logits comparison ======================")

        # Load HF reference logprobs
        try:
            with open(hf_lora_prefill_logits_file, "r") as prefill_f, open(hf_lora_prefill_logits_file, "r") as decode_f:
                hf_prefill_logits = json.load(prefill_f)
                hf_decode_logits = json.load(decode_f)
        except Exception as e:
            raise ValueError(f"Fail to load {hf_lora_prefill_logits_file} and {hf_lora_decode_logits_file} and meet err: {e}")

        sgl_prefill_logits, sgl_decode_logits=self.get_sglang_jax_last_layer_logits()

        # Calculate differences (similar to original sglang)
        print(f"{hf_prefill_logits.shape=}, {hf_prefill_logits[...,:20]}")
        print(f"{hf_decode_logits.shape=}, {hf_decode_logits[...,:20]}")
        print(f"{sgl_prefill_logits.shape=}, {sgl_prefill_logits[...,:20]}")
        print(f"{sgl_decode_logits.shape=}, {sgl_decode_logits[...,:20]}")

        prefill_diff = np.abs(hf_prefill_logits - sgl_prefill_logits)
        prefill_max_diff = np.max(prefill_diff)
        prefill_mean_diff = np.mean(prefill_diff)

        decode_diff = np.abs(hf_decode_logits - sgl_decode_logits)
        decode_max_diff = np.max(decode_diff)
        decode_mean_diff = np.mean(decode_diff)

        print(f"\n  Prefill  Max diff:   {prefill_max_diff:.6e}")
        print(f"    Prefill Mean diff:  {prefill_mean_diff:.6e}")

        print(f"\n  Decode  Max diff:   {decode_max_diff:.6e}")
        print(f"    Decode Mean diff:  {decode_mean_diff:.6e}")

        # Check threshold
        if prefill_max_diff > THRESHOLD:
            print(f"    WARNING: Prefill Max diff {prefill_max_diff:.6e} exceeds threshold {THRESHOLD:.0e}")
        else:
            print(f"    ✓ PASS: Prefill Max diff within threshold {THRESHOLD:.0e}")

        if decode_max_diff > THRESHOLD:
            print(f"    WARNING: Decode Max diff {prefill_max_diff:.6e} exceeds threshold {THRESHOLD:.0e}")
        else:
            print(f"    ✓ PASS: Decode Max diff within threshold {THRESHOLD:.0e}")

        print("\n✓ Logprobs comparison completed")



if __name__ == "__main__":
    unittest.main()