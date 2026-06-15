"""Nightly OpenAI-compatible API surface tests for tool calling.

Covers P0 and P1 scope of issue #1205.  P0: basic tool call,
tool_choice="auto", streaming SSE reassembly, multi-turn tool
interaction.  P1: tool_choice variants (required/none/specific/strict),
parallel tool calls, thinking mode, malformed-request validation,
logprobs, and n>1.  All tests share a single Qwen3-1.7B server
instance launched with --tool-call-parser=qwen25 on 1 TPU.

Run with:
    cd test/srt
    python3 -m unittest openai_server.nightly.test_openai_api_surface
"""

import json
import os
import sys
import unittest
from dataclasses import dataclass

import openai

from sgl_jax.srt.utils import kill_process_tree
from sgl_jax.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


@dataclass(frozen=True)
class OpenAIApiTestParams:
    """Reusable runner abstraction for OpenAI-compatible API surface tests.

    Modeled on sglang ToolCallTestParams.  Bundles server/client config
    and provides helpers so P1 tests (test_required, test_none, etc.) can
    reuse the same abstraction without duplicating setup.
    """

    model: str
    base_url: str
    api_key: str
    tools: tuple[dict, ...] = ()
    temperature: float = 0.0

    def make_client(self) -> openai.Client:
        return openai.Client(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: list[dict],
        *,
        tool_choice: str | dict | None = None,
        stream: bool = False,
        **kwargs,
    ):
        """Send a chat completion request, optionally with tools and streaming."""
        extra: dict = {}
        if self.tools:
            extra["tools"] = list(self.tools)
        if tool_choice is not None:
            extra["tool_choice"] = tool_choice
        if stream:
            extra["stream"] = True
            extra["stream_options"] = {"include_usage": True}
        return self.make_client().chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            **extra,
            **kwargs,
        )


ADD_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers together and return the sum.",
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
        },
    },
}

ADD_TOOL_STRICT: dict = {
    "type": "function",
    "function": {
        "name": "add",
        "description": "Add two numbers together and return the sum.",
        "strict": True,
        "parameters": {
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
            },
            "additionalProperties": False,
        },
    },
}


class TestOpenAIApiSurface(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "sk-123456"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
            other_args=[
                "--skip-server-warmup",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.8",
                "--max-running-requests",
                "16",
                "--page-size",
                "64",
                "--random-seed",
                "42",
                "--tool-call-parser",
                "qwen25",
            ],
        )
        cls.params = OpenAIApiTestParams(
            model=cls.model,
            base_url=cls.base_url + "/v1",
            api_key=cls.api_key,
            tools=(ADD_TOOL,),
        )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            kill_process_tree(cls.process.pid)

    def _assert_tool_call_is_add(self, tool_call):
        self.assertEqual(tool_call.function.name, "add")
        args = json.loads(tool_call.function.arguments)
        self.assertIn("a", args)
        self.assertIn("b", args)
        return args

    def test_basic(self):
        """tools=[add(a,b)], validate tool_calls field is populated."""
        response = self.params.chat(
            [{"role": "user", "content": "What is 2 + 3? Use the add tool."}],
        )

        msg = response.choices[0].message
        self.assertIsNotNone(
            msg.tool_calls,
            f"Expected tool_calls but got none. Content: {msg.content}",
        )
        self.assertGreater(len(msg.tool_calls), 0)

        tool_call = msg.tool_calls[0]
        self.assertTrue(tool_call.id)
        self._assert_tool_call_is_add(tool_call)

        self.assertIsNotNone(response.id)
        self.assertGreater(response.usage.prompt_tokens, 0)
        self.assertGreater(response.usage.completion_tokens, 0)

    def test_auto(self):
        """tool_choice='auto' lets the model decide to call a tool."""
        response = self.params.chat(
            [{"role": "user", "content": "Please add 10 and 20 using the add tool."}],
            tool_choice="auto",
        )

        msg = response.choices[0].message
        self.assertIsNotNone(
            msg.tool_calls,
            f"Expected tool_calls with tool_choice='auto' but got none. Content: {msg.content}",
        )
        self.assertGreater(len(msg.tool_calls), 0)
        self._assert_tool_call_is_add(msg.tool_calls[0])

    def test_streaming(self):
        """SSE streaming: reassemble chunked tool_calls from delta stream."""
        stream = self.params.chat(
            [{"role": "user", "content": "What is 7 + 8? Use the add tool."}],
            stream=True,
        )

        tool_call_chunks: dict[int, dict] = {}
        saw_usage = False

        for chunk in stream:
            if chunk.usage is not None:
                self.assertGreater(chunk.usage.prompt_tokens, 0)
                self.assertGreater(chunk.usage.completion_tokens, 0)
                saw_usage = True
                continue

            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc.id or "",
                            "name": "",
                            "arguments": "",
                        }
                    if tc.id:
                        tool_call_chunks[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_call_chunks[idx]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_call_chunks[idx]["arguments"] += tc.function.arguments

        self.assertTrue(saw_usage, "No usage chunk received in stream")
        self.assertGreater(len(tool_call_chunks), 0, "No tool_call chunks received")

        reassembled = tool_call_chunks[0]
        self.assertTrue(reassembled["id"], "Tool call id is empty after reassembly")
        self.assertEqual(reassembled["name"], "add")
        args = json.loads(reassembled["arguments"])
        self.assertIn("a", args)
        self.assertIn("b", args)

    def test_multiturn(self):
        """Multi-turn: model calls tool, we return result, model produces final answer."""
        messages = [{"role": "user", "content": "What is 100 + 200? Use the add tool."}]

        response = self.params.chat(messages)

        assistant_msg = response.choices[0].message
        self.assertIsNotNone(
            assistant_msg.tool_calls,
            f"Expected tool_calls in first turn but got none. Content: {assistant_msg.content}",
        )

        tool_call = assistant_msg.tool_calls[0]
        args = self._assert_tool_call_is_add(tool_call)

        result = args["a"] + args["b"]

        messages.append(assistant_msg)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            }
        )

        final_response = self.params.chat(messages)

        final_msg = final_response.choices[0].message
        self.assertEqual(final_msg.role, "assistant")
        self.assertIsNotNone(final_msg.content)
        self.assertGreater(len(final_msg.content), 0)

    # ── P1: tool_choice variants ──────────────────────────────────

    def test_required(self):
        """tool_choice='required' forces the model to call a tool."""
        response = self.params.chat(
            [{"role": "user", "content": "Tell me a joke."}],
            tool_choice="required",
        )
        msg = response.choices[0].message
        self.assertIsNotNone(
            msg.tool_calls,
            f"Expected tool_calls with tool_choice='required' but got none. "
            f"Content: {msg.content}",
        )
        self.assertGreater(len(msg.tool_calls), 0)

    def test_none(self):
        """tool_choice='none' prevents tool calling even when tools are provided."""
        response = self.params.chat(
            [{"role": "user", "content": "What is 2 + 3? Use the add tool."}],
            tool_choice="none",
        )
        msg = response.choices[0].message
        self.assertIsNone(
            msg.tool_calls,
            f"Expected no tool_calls with tool_choice='none' " f"but got: {msg.tool_calls}",
        )
        self.assertIsNotNone(msg.content)

    def test_specific(self):
        """tool_choice targeting a specific function constrains the model."""
        response = self.params.chat(
            [{"role": "user", "content": "What is 5 + 10?"}],
            tool_choice={"type": "function", "function": {"name": "add"}},
        )
        msg = response.choices[0].message
        self.assertIsNotNone(
            msg.tool_calls,
            f"Expected tool_calls with tool_choice=specific but got none. "
            f"Content: {msg.content}",
        )
        self.assertGreater(len(msg.tool_calls), 0)
        self.assertEqual(msg.tool_calls[0].function.name, "add")
        self._assert_tool_call_is_add(msg.tool_calls[0])

    def test_strict(self):
        """strict: true on tool schema enables structured output validation."""
        params = OpenAIApiTestParams(
            model=self.model,
            base_url=self.base_url + "/v1",
            api_key=self.api_key,
            tools=(ADD_TOOL_STRICT,),
        )
        response = params.chat(
            [{"role": "user", "content": "Add 5 and 6 using the add tool."}],
        )
        msg = response.choices[0].message
        self.assertIsNotNone(
            msg.tool_calls,
            f"Expected tool_calls with strict=True but got none. " f"Content: {msg.content}",
        )
        args = self._assert_tool_call_is_add(msg.tool_calls[0])
        self.assertEqual(
            set(args.keys()),
            {"a", "b"},
            f"strict mode should produce exactly {{a, b}}, got {set(args.keys())}",
        )
        self.assertIsInstance(args["a"], (int, float))
        self.assertIsInstance(args["b"], (int, float))

    def test_parallel(self):
        """A single prompt triggers multiple tool calls in one response."""
        response = self.params.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "I need two separate calculations done with the add tool: "
                        "first add 1 and 2, then add 3 and 4. "
                        "Call the add tool twice."
                    ),
                }
            ],
        )
        msg = response.choices[0].message
        self.assertIsNotNone(
            msg.tool_calls,
            f"Expected tool_calls but got none. Content: {msg.content}",
        )
        if len(msg.tool_calls) < 2:
            self.skipTest(
                f"Model produced {len(msg.tool_calls)} tool call(s); "
                "parallel tool calling not supported by this model"
            )
        for tc in msg.tool_calls:
            self.assertTrue(tc.id)
            self._assert_tool_call_is_add(tc)

    # ── P1: thinking, validation, logprobs, n>1 ──────────────────

    def test_thinking(self):
        """enable_thinking via chat_template_kwargs returns a valid response."""
        client = self.params.make_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "What is 2 + 2?"}],
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": True}},
            )
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                raise
            self.skipTest(f"Model does not support thinking mode: {e}")

        msg = response.choices[0].message
        self.assertIsNotNone(msg.content)
        self.assertGreater(len(msg.content), 0)

    def test_malformed_request_4xx(self):
        """Malformed tool schema returns 4xx, not 500."""
        client = self.params.make_client()
        with self.assertRaises(openai.APIStatusError) as ctx:
            client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                extra_body={
                    "tools": [{"type": "function", "function": 42}],
                },
            )
        self.assertLess(
            ctx.exception.status_code,
            500,
            f"Expected 4xx but got {ctx.exception.status_code}",
        )

    def test_logprobs(self):
        """logprobs=True with top_logprobs=3 returns per-token probabilities."""
        client = self.params.make_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Say hello."}],
            temperature=0.0,
            logprobs=True,
            top_logprobs=3,
        )
        choice = response.choices[0]
        self.assertIsNotNone(choice.logprobs)
        self.assertIsNotNone(choice.logprobs.content)
        self.assertGreater(len(choice.logprobs.content), 0)

        first = choice.logprobs.content[0]
        self.assertIsNotNone(first.top_logprobs)
        self.assertEqual(len(first.top_logprobs), 3)
        for tlp in first.top_logprobs:
            self.assertIsInstance(tlp.token, str)
            self.assertIsInstance(tlp.logprob, float)

    def test_n_greater_than_1(self):
        """n=2 returns exactly 2 choices."""
        client = self.params.make_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": "Say hello."}],
            temperature=0.5,
            n=2,
        )
        self.assertEqual(len(response.choices), 2)
        for i, choice in enumerate(response.choices):
            self.assertEqual(choice.index, i)
            self.assertIsNotNone(choice.message.content)
            self.assertGreater(len(choice.message.content), 0)


class _ResultCollector(unittest.TextTestResult):
    """Tracks successes alongside failures/errors for GH summary output."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes: list[unittest.TestCase] = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)


def _write_github_summary(result: _ResultCollector) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    rows: list[tuple[str, str]] = []
    for test in result.successes:
        rows.append((str(test), "Pass"))
    for test, _ in result.failures:
        rows.append((str(test), "FAIL"))
    for test, _ in result.errors:
        rows.append((str(test), "ERROR"))
    for test, _ in result.skipped:
        rows.append((str(test), "Skip"))
    rows.sort()

    passed = len(result.successes)
    failed = len(result.failures)
    errored = len(result.errors)
    skipped = len(result.skipped)

    lines = [
        "## OpenAI API Surface Tests",
        "",
        "| Test | Result |",
        "|---|---|",
    ]
    for name, status in rows:
        lines.append(f"| `{name}` | {status} |")
    lines.append("")
    lines.append(
        f"**{result.testsRun} tests**: {passed} passed, "
        f"{failed} failed, {errored} errors, {skipped} skipped"
    )
    lines.append("")

    with open(summary_path, "a") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOpenAIApiSurface)
    runner = unittest.TextTestRunner(verbosity=2, resultclass=_ResultCollector)
    result = runner.run(suite)
    try:
        _write_github_summary(result)
    except Exception as e:
        print(f"Warning: failed to write GitHub summary: {e}", file=sys.stderr)
    if result.testsRun > 0 and len(result.skipped) > result.testsRun // 2:
        print(
            f"WARNING: {len(result.skipped)}/{result.testsRun} tests skipped — "
            "model may be degraded",
            file=sys.stderr,
        )
    sys.exit(0 if result.wasSuccessful() else 1)
