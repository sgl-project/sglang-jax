"""
TP hidden states alignment test.

Verifies hidden states remain numerically correct when model weights
and computation are sharded across multiple TPU chips via Tensor Parallelism.

Environment: TPU v6e-4, tp_size=4
Model: Qwen/Qwen3-8B (36 layers, hidden_dim=4096)
"""

import unittest

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    assert_hidden_states_aligned,
)


class TestHiddenStatesAlignmentTP(CustomTestCase):
    """
    TP hidden states alignment test (v6e-4, tp_size=4).

    Compares prefill-phase per-layer hidden states from SGLang-JAX engine
    (with 4-way TP) against HuggingFace Transformers reference (CPU, float32).
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = DEFAULT_MODEL_NAME_FOR_TEST  # Qwen/Qwen3-8B

        # Load HuggingFace model on CPU (float32)
        print(f"Loading HuggingFace model: {cls.model_path} on CPU...")
        cls.hf_tokenizer = AutoTokenizer.from_pretrained(cls.model_path, trust_remote_code=True)
        cls.hf_model = AutoModelForCausalLM.from_pretrained(
            cls.model_path,
            trust_remote_code=True,
            dtype=torch.float32,
        ).eval()
        print("HuggingFace model loaded.")

        # Load SGLang-JAX engine on TPU with TP=4
        print(f"Loading SGLang-JAX engine: {cls.model_path} on TPU (tp_size=4)...")
        cls.engine = Engine(
            model_path=cls.model_path,
            trust_remote_code=True,
            tp_size=4,
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
            enable_return_hidden_states=True,
        )
        print("SGLang-JAX engine loaded.")

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()
        del cls.hf_model

    def _get_hf_hidden_states(self, prompt):
        inputs = self.hf_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.hf_model(**inputs, output_hidden_states=True)
        return outputs.hidden_states

    def _get_sgl_prefill_hidden_states(self, prompt):
        outputs = self.engine.generate(
            prompt=[prompt],
            sampling_params={"temperature": 0, "max_new_tokens": 1},
            return_hidden_states=True,
        )
        prefill_hs = outputs[0]["meta_info"]["hidden_states"][0]
        if not isinstance(prefill_hs, np.ndarray):
            prefill_hs = np.array(prefill_hs)
        return prefill_hs

    def _compare_per_layer(self, prompt):
        hf_hs = self._get_hf_hidden_states(prompt)
        sgl_hs = self._get_sgl_prefill_hidden_states(prompt)
        assert_hidden_states_aligned(self, hf_hs, sgl_hs, prompt)

    def test_tp_single_prompt_alignment(self):
        """Per-layer hidden states match HF after TP all-reduce."""
        self._compare_per_layer("The capital of France is")

    def test_tp_multi_prompt_alignment(self):
        """Batch + TP correctness."""
        prompts = ["Hello world", "The future of AI is"]
        for prompt in prompts:
            self._compare_per_layer(prompt)


if __name__ == "__main__":
    unittest.main()
