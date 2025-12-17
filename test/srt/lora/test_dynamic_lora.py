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

import jax.numpy as jnp
import requests
from flax import nnx

from sgl_jax.srt.entrypoints.engine import Engine
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    kill_process_tree,
    popen_launch_server,
)

LORA_SETS = [
    {
        "base": "Qwen/Qwen3-4B",
        "loras": [
            "nissenj/Qwen3-4B-lora-v2",
            "y9760210/Qwen3-4B-lora_model",
        ],
    },
]

DTYPES = ["bfloat16"]

PROMPTS = [
    '\n### Instruction:\nWrite a poem about the transformers Python library.\nMention the word "large language models" in that poem.\n### Response:\nThe Transformers are large language models,\nThey\'re used to make predictions on text.\n',
    '\n### Instruction:\nTell me about llamas and alpacas\n### Response:\nLlamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.\n### Question 2:\nWhat do you know about llamas?\n### Answer:\n',
]


class TestLoRA(CustomTestCase):
    def run_base_inference(self, base_url, prompts, max_new_tokens):
        """Run inference on base model (no LoRA loaded) and return responses."""
        print("=================== collecting base model responses =======================")
        headers = {"Content-Type": "application/json"}
        responses = []

        for prompt in prompts:
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            }

            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                headers=headers,
                timeout=60,
            )
            self.assertEqual(response.status_code, 200)
            responses.append(response.json())

        return responses

    def run_inference_test(self, base_url, prompts, lora_set, max_new_tokens):
        """Test inference with mixed batch (some with LoRA, some without)."""
        print("=================== testing mixed inference =======================")
        all_lora_paths = lora_set["loras"]
        batch_lora_paths = [None]
        i = 0
        for _ in range(len(prompts) - 1):
            lora_path = all_lora_paths[i]
            lora_name = lora_path.split("/")[-1]
            batch_lora_paths.append(lora_name)
            i = (i + 1) % len(all_lora_paths)

        headers = {"Content-Type": "application/json"}
        responses = []

        for prompt, lora_path in zip(prompts, batch_lora_paths):
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            }
            if lora_path is not None:
                payload["lora_path"] = lora_path

            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                headers=headers,
                timeout=60,
            )
            self.assertEqual(response.status_code, 200)
            responses.append(response.json())

        # Verify we got valid responses
        for i, response in enumerate(responses):
            self.assertIn("text", response)
            print(f"Response {i} (lora={batch_lora_paths[i]}): {response['text'][:100]}")

    def run_base_comparison_test(self, base_url, prompts, max_new_tokens, expected_responses):
        """
        Verify that running the base model (without lora_path) on a LoRA-enabled server
        produces the same output as the pure base model.
        """
        print("=================== testing base model equivalence =======================")
        headers = {"Content-Type": "application/json"}

        for i, prompt in enumerate(prompts):
            payload = {
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            }

            response = requests.post(
                f"{base_url}/generate",
                json=payload,
                headers=headers,
                timeout=60,
            )
            self.assertEqual(response.status_code, 200)
            actual_response = response.json()

            print(f"No LoRA support: {expected_responses[i]['text'][:100]}")
            print(f"With LoRA support (unused): {actual_response['text'][:100]}")

            self.assertEqual(
                expected_responses[i]["text"],
                actual_response["text"],
                f"Base model output changed when LoRA support enabled (request {i})",
            )

    def run_lora_effect_test(self, base_url, prompts, lora_set, max_new_tokens, base_responses):
        """Verify that LoRA actually changes the output compared to base model."""
        print("=================== testing LoRA vs base difference =======================")
        lora_path = lora_set["loras"][0]
        lora_name = lora_path.split("/")[-1]
        test_prompt = prompts[0]
        base_text = base_responses[0]["text"]

        headers = {"Content-Type": "application/json"}
        payload_lora = {
            "text": test_prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            "lora_path": lora_name,
        }

        response_lora = requests.post(
            f"{base_url}/generate",
            json=payload_lora,
            headers=headers,
            timeout=60,
        )
        self.assertEqual(response_lora.status_code, 200)
        lora_output = response_lora.json()
        lora_text = lora_output["text"]

        print(f"\nPrompt: {test_prompt[:80]}...")
        print(f"\nBase model output:\n{base_text}")
        print(f"\nLoRA model output:\n{lora_text}")

        self.assertNotEqual(
            base_text, lora_text, "LoRA output should differ from base model output! "
        )
        print("\n✓ SUCCESS: LoRA produces different output than base model")

    def run_test_suite(self, lora_set, dtype, tp_size=1, max_new_tokens=32):
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        base_url = DEFAULT_URL_FOR_TEST

        # ----------------------------------------------------------------
        # Phase 1: Base Model Server (No LoRA)
        # ----------------------------------------------------------------
        print(f"\n[Phase 1] Launching base server (No LoRA) for {base_path}")
        server_args_no_lora = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
        ]

        process_no_lora = popen_launch_server(
            base_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args_no_lora,
        )

        base_responses = []
        try:
            base_responses = self.run_base_inference(base_url, PROMPTS, max_new_tokens)
        finally:
            kill_process_tree(process_no_lora.pid)

        # ----------------------------------------------------------------
        # Phase 2: LoRA Model Server (With LoRA)
        # ----------------------------------------------------------------
        print(f"\n[Phase 2] Launching server with LoRA support for {base_path}")
        server_args_lora = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
            "--lora-paths",
            *all_lora_paths,
            "--max-loras-per-batch",
            "3",
            "--disable-overlap-schedule",
        ]

        process_lora = popen_launch_server(
            base_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args_lora,
        )

        try:
            # Run all tests that require LoRA server
            self.run_inference_test(base_url, PROMPTS, lora_set, max_new_tokens)
            self.run_base_comparison_test(base_url, PROMPTS, max_new_tokens, base_responses)
            self.run_lora_effect_test(base_url, PROMPTS, lora_set, max_new_tokens, base_responses)
        finally:
            kill_process_tree(process_lora.pid)

    def test_all(self):
        for lora_set in LORA_SETS:
            for dtype in DTYPES:
                self.run_test_suite(lora_set, dtype)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
