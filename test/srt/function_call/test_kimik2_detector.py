"""Unit tests for KimiK2Detector (function-call parser).

Covers the Kimi-K2 / K2.5 tool-call format:

    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>
    <|tool_calls_section_end|>

Subset of sglang's test/registered/function_call/test_kimik2_detector.py.

Run with:
    python test/srt/function_call/test_kimik2_detector.py
"""

import json
import unittest

from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser
from sgl_jax.srt.function_call.kimik2_detector import KimiK2Detector
from sgl_jax.srt.function_call.utils import get_schema_properties
from sgl_jax.test.test_utils import CustomTestCase
from sgl_jax.test.tool_parser_test_config import ToolParserTestConfig as C

BEGIN = "<|tool_calls_section_begin|>"
END = "<|tool_calls_section_end|>"


def _call(tool_call_id: str, args: str) -> str:
    return f"<|tool_call_begin|>{tool_call_id}<|tool_call_argument_begin|>{args}<|tool_call_end|>"


def _tools():
    return [
        C.make_tool("read_file", {"path": {"type": "string"}}),
        C.make_tool("get_weather", {"city": {"type": "string"}, "unit": {"type": "string"}}),
    ]


def _stream(detector, chunks, tools):
    """Feed chunks and reassemble tool calls the way an OpenAI client would."""
    calls, normal = [], ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        normal += result.normal_text
        for item in result.calls:
            while len(calls) <= item.tool_index:
                calls.append({"name": None, "arguments": ""})
            if item.name:
                calls[item.tool_index]["name"] = item.name
            calls[item.tool_index]["arguments"] += item.parameters
    return calls, normal


class TestKimiK2DetectorNonStreaming(CustomTestCase):
    def test_has_tool_call(self):
        d = KimiK2Detector()
        self.assertTrue(d.has_tool_call("text" + BEGIN))
        self.assertFalse(d.has_tool_call("just normal text"))

    def test_no_tool_call(self):
        text = "just a normal answer"
        result = KimiK2Detector().detect_and_parse(text, _tools())
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_single_call_with_text_before(self):
        text = "Let me check." + BEGIN + _call("functions.read_file:0", '{"path": "/a.py"}') + END
        result = KimiK2Detector().detect_and_parse(text, _tools())
        self.assertEqual(result.normal_text, "Let me check.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "read_file")
        self.assertEqual(json.loads(result.calls[0].parameters), {"path": "/a.py"})

    def test_multiple_calls_use_local_tool_index(self):
        """The model's `:N` suffix is a conversation-wide counter; tool_index is 0-based."""
        text = (
            BEGIN
            + _call("functions.read_file:5", '{"path": "/a.py"}')
            + _call("functions.get_weather:6", '{"city": "Tokyo"}')
            + END
        )
        calls = KimiK2Detector().detect_and_parse(text, _tools()).calls
        self.assertEqual(
            [(c.tool_index, c.name) for c in calls], [(0, "read_file"), (1, "get_weather")]
        )
        self.assertEqual(json.loads(calls[1].parameters), {"city": "Tokyo"})

    def test_id_without_functions_prefix_and_hyphenated_name(self):
        tools = [C.make_tool("mcp__portal__search-docs", {"q": {"type": "string"}})]
        text = BEGIN + _call("mcp__portal__search-docs:2", '{"q": "x"}') + END
        calls = KimiK2Detector().detect_and_parse(text, tools).calls
        self.assertEqual([c.name for c in calls], ["mcp__portal__search-docs"])

    def test_bare_counter_id_single_tool(self):
        tools = [C.make_tool("search", {"q": {"type": "string"}})]
        text = BEGIN + _call("3", '{"q": "x"}') + END
        calls = KimiK2Detector().detect_and_parse(text, tools).calls
        self.assertEqual([c.name for c in calls], ["search"])

    def test_bare_counter_id_infers_name_from_arguments(self):
        text = BEGIN + _call("0", '{"city": "Tokyo"}') + END
        calls = KimiK2Detector().detect_and_parse(text, _tools()).calls
        self.assertEqual([c.name for c in calls], ["get_weather"])

    def test_bare_counter_id_with_unknown_arguments_is_skipped(self):
        text = BEGIN + _call("0", '{"unknown": 1}') + END
        tools = [C.make_tool("a"), C.make_tool("b")]
        self.assertEqual(KimiK2Detector().detect_and_parse(text, tools).calls, [])

    def test_unparsable_id_is_skipped(self):
        text = (
            BEGIN
            + _call("weird@id", '{"city": "London"}')
            + _call("functions.get_weather:1", '{"city": "Delhi"}')
            + END
        )
        calls = KimiK2Detector().detect_and_parse(text, _tools()).calls
        self.assertEqual([(c.tool_index, c.name) for c in calls], [(0, "get_weather")])


class TestKimiK2DetectorStreaming(CustomTestCase):
    def test_single_call_across_chunks(self):
        chunks = [
            "Checking. ",
            BEGIN,
            "<|tool_call_begin|>",
            "functions.get_weather:0",
            "<|tool_call_argument_begin|>",
            '{"city": ',
            '"Paris"}',
            "<|tool_call_end|>",
            END,
        ]
        calls, normal = _stream(KimiK2Detector(), chunks, _tools())
        self.assertEqual(normal, "Checking. ")
        self.assertEqual(calls, [{"name": "get_weather", "arguments": '{"city": "Paris"}'}])

    def test_two_calls_keep_raw_arguments(self):
        """Arguments are streamed as raw text and never parsed: prev_tool_call_arr keeps
        `arguments == {}` (serving_chat's finish-chunk top-up relies on detecting that)."""
        d = KimiK2Detector()
        chunks = [
            BEGIN + _call("functions.read_file:0", '{"path": "/a.py"}'),
            _call("functions.get_weather:1", '{"city": "Paris"}') + END,
        ]
        calls, normal = _stream(d, chunks, _tools())
        self.assertEqual(normal, "")
        self.assertEqual(
            calls,
            [
                {"name": "read_file", "arguments": '{"path": "/a.py"}'},
                {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            ],
        )
        self.assertEqual(d.prev_tool_call_arr[1], {"name": "get_weather", "arguments": {}})
        self.assertEqual(d.streamed_args_for_tool, ['{"path": "/a.py"}', '{"city": "Paris"}'])
        self.assertEqual(d.current_tool_id, 2)
        self.assertEqual(d._buffer, "")

    def test_trailing_left_angle_is_not_dropped(self):
        result = KimiK2Detector().parse_streaming_increment("a < b <", _tools())
        self.assertEqual(result.normal_text, "a < b <")

    def test_unparsable_id_does_not_wedge_stream(self):
        chunks = [
            BEGIN + _call("weird@id", '{"city": "London"}'),
            _call("functions.get_weather:1", '{"city": "Delhi"}') + END,
        ]
        calls, _ = _stream(KimiK2Detector(), chunks, _tools())
        self.assertEqual(calls, [{"name": "get_weather", "arguments": '{"city": "Delhi"}'}])

    def test_orphan_tool_call_begin_is_discarded(self):
        chunks = [
            BEGIN + "<|tool_call_begin|>functions.read_file:0",
            _call("functions.get_weather:1", '{"city": "Oslo"}') + END,
        ]
        calls, _ = _stream(KimiK2Detector(), chunks, _tools())
        self.assertEqual(calls, [{"name": "get_weather", "arguments": '{"city": "Oslo"}'}])

    def test_whitespace_inside_section_is_not_content(self):
        """Newlines between/after calls are dropped, as in detect_and_parse."""
        chunks = ["Checking.", "\n", BEGIN, "\n", _call("functions.read_file:0", '{"path": "/a"}')]
        chunks += ["\n", _call("functions.get_weather:1", '{"city": "A"}'), "\n", END, "\n"]
        calls, normal = _stream(KimiK2Detector(), chunks, _tools())
        self.assertEqual(normal, "Checking.\n")
        self.assertEqual(
            normal, KimiK2Detector().detect_and_parse("".join(chunks), _tools()).normal_text
        )
        self.assertEqual([c["name"] for c in calls], ["read_file", "get_weather"])

    def test_argument_whitespace_is_stripped(self):
        head = "<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>"
        chunks = [BEGIN, head, " ", '{"city": ', '"A"} ', "\n", "<|tool_call_end|>", END]
        calls, _ = _stream(KimiK2Detector(), chunks, _tools())
        self.assertEqual(calls, [{"name": "get_weather", "arguments": '{"city": "A"}'}])
        parsed = KimiK2Detector().detect_and_parse("".join(chunks), _tools()).calls
        self.assertEqual(calls[0]["arguments"], parsed[0].parameters)


class TestKimiK2DetectorIntegration(CustomTestCase):
    def test_registered_and_parse_non_stream(self):
        parser = FunctionCallParser(_tools(), "kimi_k2")
        text = "ok" + BEGIN + _call("functions.read_file:0", '{"path": "/a"}') + END
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "ok")
        self.assertEqual([c.name for c in calls], ["read_file"])

    def test_no_structural_tag_required_uses_json_schema(self):
        tools = _tools()
        tools[0].function.strict = True
        parser = FunctionCallParser(tools, "kimi_k2")
        self.assertFalse(parser.detector.supports_structural_tag())
        self.assertIsNone(parser.get_structure_constraint("auto"))
        self.assertEqual(parser.get_structure_constraint("required")[0], "json_schema")
        with self.assertRaises(NotImplementedError):
            parser.detector.structure_info()
        with self.assertRaises(NotImplementedError):
            parser.detector.build_ebnf(tools)

    def test_get_schema_properties_descends_into_combinators(self):
        self.assertEqual(get_schema_properties({"properties": {"a": {}}}), {"a": {}})
        schema = {"anyOf": [{"properties": {"a": {"type": "string"}}}, {"properties": {"b": {}}}]}
        self.assertEqual(set(get_schema_properties(schema)), {"a", "b"})
        self.assertEqual(get_schema_properties(None), {})


if __name__ == "__main__":
    unittest.main()
