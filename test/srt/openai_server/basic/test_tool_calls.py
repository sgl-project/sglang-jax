"""
E2E test for tool_calls functionality.

Run with:
    python -m unittest test.srt.openai_server.basic.test_tool_calls
"""

import json
import unittest

import openai

from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    QWEN3_CODER_30B_A3B_INSTRUCT,
    CustomTestCase,
    popen_launch_server,
)


class TestToolCalls(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_CODER_30B_A3B_INSTRUCT
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--trust-remote-code",
                "--skip-server-warmup",
                "--random-seed",
                "3",
                "--mem-fraction-static",
                "0.8",
                "--chunked-prefill-size",
                "2048",
                "--download-dir",
                "/tmp/",
                "--dtype",
                "bfloat16",
                "--precompile-bs-paddings",
                "16",
                "--precompile-token-paddings",
                "16384",
                "--max-running-requests",
                "16",
                "--page-size",
                "64",
                "--tool-call-parser",
                "qwen3_coder",
                "--tp-size",
                "4",
            ],
            env={
                "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
            },
        )
        cls.base_url += "/v1"
        cls.tokenizer = get_tokenizer(QWEN3_CODER_30B_A3B_INSTRUCT, trust_remote_code=True)

        # Define tool schema (same as e2e test)
        cls.tools = [
            {
                "type": "function",
                "function": {
                    "name": "square_the_number",
                    "description": "Calculate the square of a number.",
                    "parameters": {
                        "type": "object",
                        "required": ["input_num"],
                        "properties": {
                            "input_num": {
                                "type": "number",
                                "description": "The number to be squared.",
                            }
                        },
                    },
                },
            }
        ]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_tool_calls_e2e(self):
        """E2E test: user asks to square a number, model calls tool, returns result"""
        client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Step 1: User input
        user_input = "square the number 1024"
        messages = [{"role": "user", "content": user_input}]

        # Step 2: First LLM call - model decides to call tool
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            temperature=0.0,
        )

        # Verify response structure
        assert response.id
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

        # Get assistant message
        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        # Step 3: Verify tool call was made
        assert (
            assistant_message.tool_calls is not None
        ), f"Expected tool call but got none. Content: {assistant_message.content}"
        assert len(assistant_message.tool_calls) > 0, "Expected at least one tool call"

        tool_call = assistant_message.tool_calls[0]
        assert tool_call.function.name == "square_the_number"
        assert tool_call.function.arguments

        # Step 4: Parse and verify arguments
        try:
            func_args = json.loads(tool_call.function.arguments)
            assert "input_num" in func_args
            input_num = func_args["input_num"]
            assert input_num == 1024
        except json.JSONDecodeError:
            self.fail(f"Failed to parse tool call arguments: {tool_call.function.arguments}")

        # Step 5: Execute tool function (simulate)
        result = input_num**2
        assert result == 1048576

        # Step 6: Add tool result to messages
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )

        # Step 7: Second LLM call - generate final response
        final_response = client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # Verify final response
        assert final_response.choices[0].message.role == "assistant"
        assert final_response.choices[0].message.content is not None
        final_answer = final_response.choices[0].message.content

        # The answer should mention the result
        assert "1048576" in final_answer or "1,048,576" in final_answer


if __name__ == "__main__":
    unittest.main()
