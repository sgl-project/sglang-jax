"""
python3 -m unittest test.srt.openai_server.features.test_structural_tag
"""

import json
import re
import unittest
from typing import Any

import openai

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


def setup_class(cls, backend: str):
    cls.model = DEFAULT_MODEL_NAME_FOR_TEST
    cls.base_url = DEFAULT_URL_FOR_TEST

    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
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
            "/dev/shm",
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
            "--grammar-backend",
            backend,
        ],
        env={
            "JAX_COMPILATION_CACHE_DIR": "/tmp/jax_compilation_cache",
        },
    )


class TestStructuralTagXGrammarBackend(CustomTestCase):
    model: str
    base_url: str
    process: Any

    @classmethod
    def setUpClass(cls):
        setup_class(cls, backend="llguidance")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_stag_constant_str_openai(self):
        client = openai.Client(api_key="EMPTY", base_url=f"{self.base_url}/v1")

        tool_get_current_weather = {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The city to find the weather for, e.g. 'San Francisco'",
                        },
                        "state": {
                            "type": "string",
                            "description": "the two-letter abbreviation for the state that the city is"
                            " in, e.g. 'CA' which would mean 'California'",
                        },
                        "unit": {
                            "type": "string",
                            "description": "The unit to fetch the temperature in",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city", "state", "unit"],
                },
            },
        }

        tool_get_current_date = {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": "Get the current date and time for a given timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {
                            "type": "string",
                            "description": "The timezone to fetch the current date and time for, e.g. 'America/New_York'",
                        }
                    },
                    "required": ["timezone"],
                },
            },
        }

        schema_get_current_weather = tool_get_current_weather["function"]["parameters"]
        schema_get_current_date = tool_get_current_date["function"]["parameters"]

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"""
# Tool Instructions
- Always execute python code in messages that you share.
- When looking for real time information use relevant functions if available else fallback to brave_search
You have access to the following functions:
Use the function 'get_current_weather' to: Get the current weather in a given location
{tool_get_current_weather["function"]}
Use the function 'get_current_date' to: Get the current date and time for a given timezone
{tool_get_current_date["function"]}
If a you choose to call a function ONLY reply in the following format:
<{{start_tag}}={{function_name}}>{{parameters}}{{end_tag}}
where
start_tag => `<function`
parameters => a JSON dict with the function argument name as key and function argument value as value.
end_tag => `</function>`
Here is an example,
<function=example_function_name>{{"example_name": "example_value"}}</function>
Reminder:
- Function calls MUST follow the specified format
- Required parameters MUST be specified
- Only call one function at a time
- Put the entire function call reply on one line
- Always add your sources when using search results to answer the user query
You are a helpful assistant.""",
                },
                {
                    "role": "user",
                    "content": "You are in New York. Please get the current date and time, and the weather.",
                },
            ],
            response_format={
                "type": "structural_tag",
                "structures": [
                    {
                        "begin": "<function=get_current_weather>",
                        "schema": schema_get_current_weather,
                        "end": "</function>",
                    },
                    {
                        "begin": "<function=get_current_date>",
                        "schema": schema_get_current_date,
                        "end": "</function>",
                    },
                ],
                "triggers": ["<function="],
            },
        )

        text = response.choices[0].message.content
        print(f"{text=}")
        # Use non-greedy matching to capture only the first closing tag
        m = re.search(r"<function=get_current_date>(.*?)</function>", text, flags=re.S)
        self.assertIsNotNone(m, f"Tagged date call not found in: {text}")
        inner = m.group(1).strip()
        print(f"{inner=}")
        parsed = json.loads(inner)
        self.assertEqual(parsed["timezone"], "America/New_York")

        # Optionally also verify weather call if present
        m_w = re.search(r"<function=get_current_weather>(.*?)</function>", text, flags=re.S)
        if m_w is not None:
            inner_w = m_w.group(1).strip()
            try:
                parsed_w = json.loads(inner_w)
            except Exception:
                self.fail(f"Weather function body is not valid JSON: {inner_w}")
            # Basic keys should exist per schema
            self.assertIn("city", parsed_w)
            self.assertIn("state", parsed_w)
            self.assertIn("unit", parsed_w)


if __name__ == "__main__":
    unittest.main()
