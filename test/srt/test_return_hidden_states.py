"""
Usage:
python3 -m unittest test_return_hidden_states.TestReturnHiddenStates.test_return_hidden_states
"""

import unittest

import numpy as np

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import QWEN3_8B, CustomTestCase

# Qwen3-8B has 36 hidden layers and hidden_dim=4096
EXPECTED_NUM_LAYERS = 36
EXPECTED_HIDDEN_DIM = 4096


class TestReturnHiddenStates(CustomTestCase):
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
            enable_return_hidden_states=True,
        )
        cls.tokenizer = get_tokenizer(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_return_hidden_states(self):
        prompts = [
            "The capital of France is",
            "The future of AI is",
        ]

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 10,
        }

        outputs = self.engine.generate(
            prompt=prompts,
            sampling_params=sampling_params,
            return_hidden_states=True,
        )

        self.assertEqual(len(outputs), 2)
        for prompt, output in zip(prompts, outputs):
            # Convert each hidden state to numpy array
            for i in range(len(output["meta_info"]["hidden_states"])):
                hs = output["meta_info"]["hidden_states"][i]
                if not isinstance(hs, np.ndarray):
                    output["meta_info"]["hidden_states"][i] = np.array(hs)

            prompt_tokens = output["meta_info"]["prompt_tokens"]
            completion_tokens = output["meta_info"]["completion_tokens"]
            total_tokens = prompt_tokens + completion_tokens

            print("===============================")
            print(
                f"Prompt: {prompt}\n"
                f"Generated text: {output['text']}\n"
                f"Prompt_Tokens: {prompt_tokens}\t"
                f"Completion_tokens: {completion_tokens}"
            )

            # Prefill hidden states: [prompt_tokens, num_layers, hidden_dim]
            # Decode hidden states: [num_layers, hidden_dim] per token
            raw_hs = output["meta_info"]["hidden_states"]
            print(f"Number of hidden state chunks: {len(raw_hs)}")
            for idx, h in enumerate(raw_hs):
                print(f"  Chunk {idx} shape: {h.shape}")

            # Concatenate: expand decode [num_layers, hidden_dim] to [1, num_layers, hidden_dim]
            hidden_states = np.concatenate(
                [np.expand_dims(h, 0) if h.ndim == 2 else h for h in raw_hs]
            )
            print(f"Combined hidden states shape: {hidden_states.shape}")
            # In autoregressive generation, the last completion token is never fed
            # into the model as input, so its hidden state is not available.
            # Expected: [prompt_tokens + completion_tokens - 1, num_layers, hidden_dim]
            self.assertEqual(hidden_states.shape[0], total_tokens - 1)
            self.assertEqual(hidden_states.shape[1], EXPECTED_NUM_LAYERS)
            self.assertEqual(hidden_states.shape[2], EXPECTED_HIDDEN_DIM)

            # Verify per-layer hidden states are distinct (not all zeros, not all same)
            for layer_idx in [0, EXPECTED_NUM_LAYERS // 2, EXPECTED_NUM_LAYERS - 1]:
                layer_hs = hidden_states[0, layer_idx, :]
                self.assertGreater(
                    np.abs(layer_hs).max(),
                    0,
                    f"Layer {layer_idx} hidden states are all zeros",
                )

            print()

            self.assertIsNotNone(hidden_states)
            self.assertGreater(hidden_states.size, 0)

    def test_no_hidden_states_by_default(self):
        prompts = ["The capital of France is"]

        sampling_params = {
            "temperature": 0,
            "max_new_tokens": 5,
        }

        outputs = self.engine.generate(
            prompt=prompts,
            sampling_params=sampling_params,
        )

        self.assertEqual(len(outputs), 1)
        hidden_states = outputs[0]["meta_info"].get("hidden_states")
        # When return_hidden_states is not set, hidden_states should be absent or empty
        if hidden_states is not None:
            self.assertEqual(len(hidden_states), 0)


if __name__ == "__main__":
    unittest.main()
