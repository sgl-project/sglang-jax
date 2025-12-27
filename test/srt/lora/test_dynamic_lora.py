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

import asyncio
import json
import os
import unittest
from typing import List

import aiohttp
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

DTYPES = [
    "float32"
]  # Note: According to https://github.com/sgl-project/sglang-jax/issues/587, adjust dtype from bfloat16 to float32.

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]


async def send_request(session, base_url, headers, payload):
    async with session.post(
        f"{base_url}/generate",
        json=payload,
        headers=headers,
        timeout=aiohttp.ClientTimeout(total=60),
    ) as response:
        assert response.status == 200, f"Status code: {response.status}"
        actual_response = await response.json()
        return actual_response


async def run_async_generate(base_url, headers, payloads):
    async with aiohttp.ClientSession() as session:
        tasks = [
            send_request(
                session,
                base_url,
                headers,
                payload,
            )
            for payload in payloads
        ]
        actual_responses = await asyncio.gather(*tasks)

    return actual_responses


class TestLoRA(CustomTestCase):
    def run_base_inference(self, base_url, prompts, max_new_tokens):
        """Run inference on base model (no LoRA loaded) and return responses."""
        print("=================== collecting base model responses =======================")
        headers = {"Content-Type": "application/json"}

        payloads = []
        for prompt in prompts:
            payloads.append(
                {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                }
            )

        return asyncio.run(run_async_generate(base_url, headers, payloads))

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

        payloads = []
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
            payloads.append(payload)

        responses = asyncio.run(run_async_generate(base_url, headers, payloads))

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

        payloads = []
        for prompt in prompts:
            payloads.append(
                {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                }
            )

        actual_responses = asyncio.run(run_async_generate(base_url, headers, payloads))

        for i, (expected, actual) in enumerate(zip(expected_responses, actual_responses)):
            print(f"No LoRA support: {expected['text']}")
            print(f"With LoRA support (unused): {actual['text']}")
            assert (
                expected["text"] == actual["text"]
            ), f"Base model output changed when LoRA support enabled (request {i})"

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
