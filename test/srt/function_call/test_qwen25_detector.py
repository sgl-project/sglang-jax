"""Unit tests for Qwen25Detector (function-call parser).

Covers the Qwen 2.5 / Qwen 3 / Ling-2.6 tool-call format:

    <tool_call>
    {"name": "...", "arguments": {...}}
    </tool_call>

Run with:
    python -m unittest test.srt.function_call.test_qwen25_detector
"""

import json

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool
from sgl_jax.srt.function_call.qwen25_detector import Qwen25Detector
from sgl_jax.test.test_utils import CustomTestCase


def _tool(name: str, properties: dict | None = None) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": properties or {}},
        ),
    )


def _weather_tool() -> Tool:
    return _tool("get_weather", {"location": {"type": "string"}})


def _square_tool() -> Tool:
    return _tool("square", {"x": {"type": "number"}})


class TestQwen25Detector(CustomTestCase):
    def test_has_tool_call(self):
        d = Qwen25Detector()
        self.assertTrue(d.has_tool_call("foo<tool_call>\n{...}\n</tool_call>"))
        self.assertFalse(d.has_tool_call("plain text only"))
        # Missing trailing newline after <tool_call> should not match: the
        # format strictly requires "<tool_call>\n" as the bot_token.
        self.assertFalse(d.has_tool_call("<tool_call>{...}</tool_call>"))

    def test_detect_and_parse_single_call(self):
        text = (
            "thinking done\n"
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"location": "Beijing"}}\n'
            "</tool_call>"
        )
        result = Qwen25Detector().detect_and_parse(text, [_weather_tool()])
        self.assertEqual(result.normal_text, "thinking done\n")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"location": "Beijing"})

    def test_detect_and_parse_two_calls(self):
        text = (
            "<tool_call>\n"
            '{"name": "get_weather", "arguments": {"location": "Beijing"}}\n'
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "square", "arguments": {"x": 7}}\n'
            "</tool_call>"
        )
        result = Qwen25Detector().detect_and_parse(text, [_weather_tool(), _square_tool()])
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {"location": "Beijing"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"x": 7})

    def test_detect_and_parse_no_tool_call(self):
        text = "just a normal answer with no tool"
        result = Qwen25Detector().detect_and_parse(text, [_weather_tool()])
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_unknown_function(self):
        text = (
            "<tool_call>\n"
            '{"name": "mystery", "arguments": {"x": 1}}\n'
            "</tool_call>"
        )
        result = Qwen25Detector().detect_and_parse(text, [_weather_tool()])
        # Unknown function: parse_base_json logs a warning and drops it.
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_malformed_json_skipped(self):
        text = (
            "<tool_call>\n"
            "this is not json\n"
            "</tool_call>\n"
            "<tool_call>\n"
            '{"name": "square", "arguments": {"x": 9}}\n'
            "</tool_call>"
        )
        result = Qwen25Detector().detect_and_parse(text, [_square_tool()])
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "square")

    def test_streaming_split_chunks(self):
        det = Qwen25Detector()
        tools = [_square_tool()]
        chunks = [
            "<tool_call>\n",
            '{"name": "squ',
            'are", "arguments": {"x": ',
            "42}}\n",
            "</tool_call>",
        ]
        out_calls = []
        out_normal = ""
        for ch in chunks:
            r = det.parse_streaming_increment(ch, tools)
            out_normal += r.normal_text
            out_calls.extend(r.calls)
        names = [c.name for c in out_calls if c.name]
        self.assertEqual(names, ["square"])
        joined = "".join(c.parameters for c in out_calls)
        self.assertEqual(json.loads(joined), {"x": 42})
        self.assertNotIn("</tool_call>", out_normal)

    def test_streaming_normal_text_then_call(self):
        det = Qwen25Detector()
        tools = [_square_tool()]
        r1 = det.parse_streaming_increment("plain text. ", tools)
        self.assertEqual(r1.normal_text, "plain text. ")
        self.assertEqual(r1.calls, [])
        r2 = det.parse_streaming_increment(
            "<tool_call>\n"
            '{"name": "square", "arguments": {"x": 3}}\n'
            "</tool_call>",
            tools,
        )
        names = [c.name for c in r2.calls if c.name]
        self.assertIn("square", names)
        self.assertNotIn("<tool_call>", r2.normal_text)
        self.assertNotIn("</tool_call>", r2.normal_text)

    def test_structure_info(self):
        info = Qwen25Detector().structure_info()("get_weather")
        self.assertEqual(info.trigger, "<tool_call>")
        self.assertEqual(info.end, "}\n</tool_call>")
        self.assertTrue(info.begin.startswith('<tool_call>\n{"name":"get_weather"'))

    def test_build_ebnf_contains_tool_name(self):
        grammar = Qwen25Detector().build_ebnf([_weather_tool(), _square_tool()])
        self.assertIn("get_weather", grammar)
        self.assertIn("square", grammar)


if __name__ == "__main__":
    import unittest

    unittest.main()
