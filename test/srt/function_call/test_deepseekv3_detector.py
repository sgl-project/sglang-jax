"""Fixture-only tests for the DeepSeek V3 native tool-call parser."""

import json
import unittest

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool
from sgl_jax.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser


def make_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name, description=name, parameters={"type": "object", "properties": {}}
        ),
    )


class TestDeepSeekV3Detector(unittest.TestCase):
    def setUp(self):
        self.tools = [make_tool("get_weather"), make_tool("search")]
        self.detector = DeepSeekV3Detector()

    def call(self, name: str, arguments: dict) -> str:
        return (
            f"{self.detector.call_begin}function{self.detector.separator}{name}\n```json\n"
            + json.dumps(arguments)
            + f"\n```{self.detector.call_end}"
        )

    def test_parser_is_registered(self):
        self.assertIsInstance(
            FunctionCallParser(self.tools, "deepseekv3").detector, DeepSeekV3Detector
        )

    def test_parses_tagged_calls_and_leading_text(self):
        text = (
            "Checking. "
            + self.detector.calls_begin
            + self.call("get_weather", {"city": "Beijing"})
            + self.detector.calls_end
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Beijing"})

    def test_parses_multiple_calls_and_ignores_unknown_tools(self):
        text = (
            self.detector.calls_begin
            + self.call("get_weather", {"city": "Tokyo"})
            + self.call("search", {"query": "food"})
            + self.detector.calls_end
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([call.name for call in result.calls], ["get_weather", "search"])
        unknown = self.detector.detect_and_parse(
            self.detector.calls_begin + self.call("unknown", {}) + self.detector.calls_end,
            self.tools,
        )
        self.assertEqual(unknown.calls, [])

    def test_keeps_plain_or_malformed_text_as_content(self):
        self.assertEqual(self.detector.detect_and_parse("plain", self.tools).normal_text, "plain")
        malformed = self.detector.calls_begin + self.call("get_weather", {"city": "x"}).replace(
            '"x"', "bad"
        )
        self.assertEqual(self.detector.detect_and_parse(malformed, self.tools).calls, [])

    def test_streams_split_marker_arguments_and_multiple_calls(self):
        text = (
            self.detector.calls_begin
            + self.call("get_weather", {"city": "Shanghai"})
            + self.call("search", {"query": "food"})
            + self.detector.calls_end
        )
        calls = []
        for chunk in (text[:17], text[17:51], text[51:91], text[91:]):
            calls.extend(self.detector.parse_streaming_increment(chunk, self.tools).calls)
        names = [call.name for call in calls if call.name]
        parameters_by_index = {}
        for call in calls:
            if call.parameters:
                parameters_by_index.setdefault(call.tool_index, "")
                parameters_by_index[call.tool_index] += call.parameters
        self.assertEqual(names, ["get_weather", "search"])
        self.assertEqual(json.loads(parameters_by_index[0]), {"city": "Shanghai"})
        self.assertEqual(json.loads(parameters_by_index[1]), {"query": "food"})

    def test_structure_and_ebnf(self):
        info = self.detector.structure_info()("get_weather")
        self.assertEqual(info.trigger, self.detector.calls_begin)
        self.assertIn("get_weather", info.begin)
        grammar = self.detector.build_ebnf(self.tools)
        self.assertIn(self.detector.calls_begin, grammar)
        self.assertIn("get_weather", grammar)


if __name__ == "__main__":
    unittest.main()
