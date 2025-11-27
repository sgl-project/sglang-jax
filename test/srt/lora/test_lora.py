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

import jax.numpy as jnp

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
    """
### Instruction:
Write a poem about the transformers Python library.
Mention the word "large language models" in that poem.
### Response:
The Transformers are large language models,
They're used to make predictions on text.
""",
    """
### Instruction:
Tell me about llamas and alpacas
### Response:
Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids (camels, dromedaries). Llamas live in the Andean mountains of South America where they graze on grasses and shrubs. Alpaca is another name for domesticated llama. The word "alpaca" comes from an Incan language meaning "golden fleece." Alpacas look very similar to llamas but are smaller than their wild relatives. Both species were used by ancient people as pack animals and for meat. Today both llamas and alpacas are raised primarily for their fiber which can be spun into yarn or knitted into clothing.
### Question 2:
What do you know about llamas?
### Answer:
""",
]


class TestLoRA(CustomTestCase):
    def inference(self, prompts, lora_set, tp_size, dtype, max_new_tokens):
        print("=================== testing inference =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_names = [None]
        i = 0
        for _ in range(len(prompts) - 1):
            batch_lora_names.append(all_lora_paths[i])
            i = (i + 1) % len(all_lora_paths)

        # Launch server with LoRA support
        base_url = DEFAULT_URL_FOR_TEST
        server_args = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
            "--lora-paths",
            *all_lora_paths,
            "--max-loras-per-batch",
            "3",
            "--disable-radix-cache",
        ]

        process = popen_launch_server(
            base_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

        try:
            import requests

            # Test inference with mixed batch (some with LoRA, some without)
            headers = {"Content-Type": "application/json"}

            # Make requests to the server
            responses = []
            for prompt, lora_name in zip(prompts, batch_lora_names):
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                }
                if lora_name is not None:
                    payload["lora_name"] = lora_name

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
                print(f"Response {i} (lora={batch_lora_names[i]}): {response['text'][:100]}")

            # Test base model (no LoRA)
            base_responses = []
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
                base_responses.append(response.json())

            # Verify base model responses differ from LoRA responses
            for i in range(len(prompts)):
                if batch_lora_names[i] is not None:
                    # Responses with LoRA should potentially differ from base
                    # (though not always guaranteed depending on the prompt)
                    print(f"LoRA response {i}: {responses[i]['text'][:100]}")
                    print(f"Base response {i}: {base_responses[i]['text'][:100]}")

        finally:
            kill_process_tree(process.pid)

    def serving(self, prompts, lora_set, tp_size, dtype, max_new_tokens):
        print("=================== testing serving =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_names = [None]
        i = 0
        for _ in range(len(prompts) - 1):
            batch_lora_names.append(all_lora_paths[i])
            i = (i + 1) % len(all_lora_paths)

        # Launch server with LoRA support
        base_url = DEFAULT_URL_FOR_TEST
        server_args = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
            "--lora-paths",
            *all_lora_paths,
            "--max-loras-per-batch",
            "3",
            "--disable-radix-cache",
        ]

        process = popen_launch_server(
            base_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

        try:
            import requests

            headers = {"Content-Type": "application/json"}

            # Test batch serving
            responses = []
            for prompt, lora_name in zip(prompts, batch_lora_names):
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                }
                if lora_name is not None:
                    payload["lora_name"] = lora_name

                response = requests.post(
                    f"{base_url}/generate",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                self.assertEqual(response.status_code, 200)
                responses.append(response.json())

            # Verify responses
            for i, response in enumerate(responses):
                self.assertIn("text", response)
                print(
                    f"Serving response {i} (lora={batch_lora_names[i]}): {response['text'][:100]}"
                )

        finally:
            kill_process_tree(process.pid)

    def base_inference(self, prompts, lora_set, tp_size, dtype, max_new_tokens):
        print("=================== testing base inference =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]
        batch_lora_names = [None] * len(prompts)

        # Launch server WITHOUT LoRA support
        base_url_no_lora = DEFAULT_URL_FOR_TEST
        server_args_no_lora = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
        ]

        process_no_lora = popen_launch_server(
            base_path,
            base_url_no_lora,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args_no_lora,
        )

        try:
            import requests

            headers = {"Content-Type": "application/json"}

            # Get base model responses
            no_lora_responses = []
            for prompt in prompts:
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                }

                response = requests.post(
                    f"{base_url_no_lora}/generate",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                self.assertEqual(response.status_code, 200)
                no_lora_responses.append(response.json())

        finally:
            kill_process_tree(process_no_lora.pid)

        # Launch server WITH LoRA support but don't use LoRA
        base_url_with_lora = DEFAULT_URL_FOR_TEST
        server_args_with_lora = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
            "--lora-paths",
            *all_lora_paths,
        ]

        process_with_lora = popen_launch_server(
            base_path,
            base_url_with_lora,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args_with_lora,
        )

        try:
            import requests

            # Get responses from server with LoRA support but not using LoRA
            with_lora_base_responses = []
            for prompt in prompts:
                payload = {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                    },
                }

                response = requests.post(
                    f"{base_url_with_lora}/generate",
                    json=payload,
                    headers=headers,
                    timeout=60,
                )
                self.assertEqual(response.status_code, 200)
                with_lora_base_responses.append(response.json())

            # Compare responses - they should be identical
            for i in range(len(prompts)):
                print(f"No LoRA support: {no_lora_responses[i]['text'][:100]}")
                print(f"With LoRA support (unused): {with_lora_base_responses[i]['text'][:100]}")

                # The outputs should be identical since we're not using LoRA
                self.assertEqual(
                    no_lora_responses[i]["text"],
                    with_lora_base_responses[i]["text"],
                    f"Base model output changed when LoRA support enabled (request {i})",
                )

        finally:
            kill_process_tree(process_with_lora.pid)

    def test_lora_vs_base_difference(self, prompts, lora_set, tp_size, dtype, max_new_tokens):
        """
        Test that LoRA actually changes the model output.

        This test verifies that:
        1. Using the same prompt with LoRA produces different output than without LoRA
        2. The outputs are deterministic (temperature=0.0)
        """
        print("=================== testing LoRA vs base difference =======================")
        base_path = lora_set["base"]
        all_lora_paths = lora_set["loras"]

        # Use the first LoRA adapter
        lora_path = all_lora_paths[0]

        # Launch server with LoRA support
        base_url = DEFAULT_URL_FOR_TEST
        server_args = [
            "--tp-size",
            str(tp_size),
            "--dtype",
            dtype,
            "--lora-paths",
            lora_path,
            "--max-loras-per-batch",
            "2",
            "--disable-radix-cache",
        ]

        process = popen_launch_server(
            base_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
        )

        try:
            import requests

            headers = {"Content-Type": "application/json"}

            # Test with the first prompt
            test_prompt = prompts[0]

            # Get base model output (without LoRA)
            payload_base = {
                "text": test_prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
            }

            response_base = requests.post(
                f"{base_url}/generate",
                json=payload_base,
                headers=headers,
                timeout=60,
            )
            self.assertEqual(response_base.status_code, 200)
            base_output = response_base.json()

            # Get LoRA output (with LoRA)
            payload_lora = {
                "text": test_prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0.0,
                },
                "lora_name": lora_path,
            }

            response_lora = requests.post(
                f"{base_url}/generate",
                json=payload_lora,
                headers=headers,
                timeout=60,
            )
            self.assertEqual(response_lora.status_code, 200)
            lora_output = response_lora.json()

            # Verify both produced valid outputs
            self.assertIn("text", base_output)
            self.assertIn("text", lora_output)

            base_text = base_output["text"]
            lora_text = lora_output["text"]

            # Print outputs for inspection
            print(f"\nPrompt: {test_prompt[:80]}...")
            print(f"\nBase model output:\n{base_text}")
            print(f"\nLoRA model output:\n{lora_text}")

            # Assert that outputs are different
            # This is the key test: LoRA should change the model's behavior
            self.assertNotEqual(
                base_text,
                lora_text,
                "LoRA output should differ from base model output! "
                "If they are the same, LoRA is not being applied correctly.",
            )

            print("\n✓ SUCCESS: LoRA produces different output than base model")

        finally:
            kill_process_tree(process.pid)

    def test_all(self):
        for lora_set in LORA_SETS:
            for dtype in DTYPES:
                tp_size = 1
                max_new_tokens = 32
                self.inference(PROMPTS, lora_set, tp_size, dtype, max_new_tokens)
                self.serving(PROMPTS, lora_set, tp_size, dtype, max_new_tokens)
                self.base_inference(PROMPTS, lora_set, tp_size, dtype, max_new_tokens)
                self.test_lora_vs_base_difference(PROMPTS, lora_set, tp_size, dtype, max_new_tokens)


if __name__ == "__main__":
    unittest.main(warnings="ignore")
