"""Unit tests for Qwen3CoderDetector (function-call parser).

Covers the Qwen 3 Coder tool-call format:

    <tool_call>
    <function=execute_bash>
    <parameter=command>pwd && ls</parameter>
    </function>
    </tool_call>

Run with:
    python -m unittest test.srt.function_call.test_qwen3_coder_detector
"""

import json

from sgl_jax.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sgl_jax.test.test_utils import CustomTestCase
from sgl_jax.test.tool_parser_test_config import ToolParserTestConfig as C


class TestQwen3CoderDetector(CustomTestCase):
    def test_has_tool_call(self):
        d = Qwen3CoderDetector()
        self.assertTrue(d.has_tool_call("foo<tool_call>bar"))
        self.assertFalse(d.has_tool_call("just normal text"))

    def test_detect_and_parse_no_tool_call(self):
        text = "just a normal answer with no tool"
        result = Qwen3CoderDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_call(self):
        text = (
            "thinking done\n"
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>pwd && ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = Qwen3CoderDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(result.normal_text, "thinking done\n")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"command": "pwd && ls"})

    def test_detect_and_parse_two_calls(self):
        text = (
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>ls</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>pwd</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = Qwen3CoderDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "ls"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"command": "pwd"})

    def test_streaming_split_chunks(self):
        det = Qwen3CoderDetector()
        tools = [C.bash_tool()]
        chunks = [
            "plain text. ",
            "<tool_call>\n",
            "<function=execute_bash>\n",
            "<parameter=command>ls</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]
        out_calls = []
        out_normal = ""
        for ch in chunks:
            r = det.parse_streaming_increment(ch, tools)
            out_normal += r.normal_text
            out_calls.extend(r.calls)
        names = [c.name for c in out_calls if c.name]
        self.assertEqual(names, ["execute_bash"])
        joined = "".join(c.parameters for c in out_calls)
        self.assertEqual(json.loads(joined), {"command": "ls"})
        self.assertNotIn("<tool_call>", out_normal)
        self.assertNotIn("</tool_call>", out_normal)

    def test_detect_and_parse_malformed_skipped(self):
        text = (
            "<tool_call>\n"
            "this is not valid xml\n"
            "</tool_call>\n"
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = Qwen3CoderDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")

    def test_detect_and_parse_unknown_function(self):
        text = (
            "<tool_call>\n"
            "<function=mystery>\n"
            "<parameter=x>1</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = Qwen3CoderDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(result.calls, [])

    def test_safe_val_html_unescape(self):
        text = (
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>echo &quot;hi&quot; &amp;&amp; ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = Qwen3CoderDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["command"], 'echo "hi" && ls')

    def test_build_ebnf_contains_tool_name(self):
        grammar = Qwen3CoderDetector().build_ebnf([C.bash_tool(), C.weather_tool()])
        self.assertIn("execute_bash", grammar)
        self.assertIn("get_weather", grammar)

    def test_build_ebnf_compiles(self):
        from llguidance import grammar_from

        grammar = Qwen3CoderDetector().build_ebnf([C.bash_tool(), C.weather_tool()])
        grammar_from("lark", grammar)


if __name__ == "__main__":
    import unittest

    unittest.main()
