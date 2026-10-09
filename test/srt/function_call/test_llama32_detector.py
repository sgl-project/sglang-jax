"""Unit tests for the Llama 3.2 tool-call parser; no model loading required."""

import json
import unittest

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool
from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser
from sgl_jax.srt.function_call.llama32_detector import Llama32Detector


def make_tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={"type": "object", "properties": {}},
        ),
    )


class TestLlama32Detector(unittest.TestCase):
    def setUp(self):
        self.tools = [make_tool("get_weather"), make_tool("search")]

    def test_parser_is_registered(self):
        self.assertIsInstance(FunctionCallParser(self.tools, "llama3").detector, Llama32Detector)

    def test_detects_and_parses_tagged_json(self):
        result = Llama32Detector().detect_and_parse(
            'I will check. <|python_tag|>{"name":"get_weather","arguments":{"city":"Beijing"}}',
            self.tools,
        )
        self.assertEqual(result.normal_text, "I will check. ")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Beijing"})

    def test_parses_markerless_json_and_multiple_calls(self):
        result = Llama32Detector().detect_and_parse(
            '{"name":"get_weather","arguments":{"city":"Beijing"}}; '
            '{"name":"search","arguments":{"query":"restaurants"}}',
            self.tools,
        )
        self.assertEqual([call.name for call in result.calls], ["get_weather", "search"])
        self.assertEqual(json.loads(result.calls[1].parameters), {"query": "restaurants"})

    def test_ignores_plain_text_and_unknown_tools(self):
        detector = Llama32Detector()
        plain = detector.detect_and_parse("The weather is sunny.", self.tools)
        unknown = detector.detect_and_parse(
            '<|python_tag|>{"name":"unknown","arguments":{}}', self.tools
        )
        self.assertEqual(plain.normal_text, "The weather is sunny.")
        self.assertEqual(plain.calls, [])
        self.assertEqual(unknown.calls, [])

    def test_python_dict_fallback(self):
        result = Llama32Detector().detect_and_parse(
            "<|python_tag|>{'name': 'get_weather', 'arguments': {'city': 'Tokyo'}}",
            self.tools,
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Tokyo"})

    def test_streams_a_split_marker_and_arguments(self):
        detector = Llama32Detector()
        calls = []
        normal_text = ""
        for chunk in (
            "I will check. <|python_",
            'tag|>{"name":"get_weather",',
            '"arguments":{"city":"Tokyo"',
            "}}",
        ):
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            calls.extend(result.calls)

        self.assertEqual(normal_text, "I will check. ")
        self.assertEqual([call.name for call in calls if call.name], ["get_weather"])
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls if call.parameters)),
            {"city": "Tokyo"},
        )

    def test_structure_info(self):
        info = Llama32Detector().structure_info()("get_weather")
        self.assertEqual(info.trigger, "<|python_tag|>")
        self.assertIn("get_weather", info.begin)
        self.assertEqual(info.end, "}")

    def test_build_ebnf_contains_the_marker_and_tool_names(self):
        grammar = Llama32Detector().build_ebnf(self.tools)
        self.assertIn("<|python_tag|>", grammar)
        self.assertIn("get_weather", grammar)
        self.assertIn("search", grammar)


if __name__ == "__main__":
    unittest.main()
