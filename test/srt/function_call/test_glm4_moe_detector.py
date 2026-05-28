"""Unit tests for Glm4MoeDetector (function-call parser).

Covers the GLM-4.5 / GLM-4.6 tool-call format:

    <tool_call>get_weather
    <arg_key>city</arg_key>
    <arg_value>Beijing</arg_value>
    </tool_call>

Run with:
    python -m unittest test.srt.function_call.test_glm4_moe_detector
"""

import json

from sgl_jax.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sgl_jax.test.test_utils import CustomTestCase
from sgl_jax.test.tool_parser_test_config import ToolParserTestConfig as C


class TestGlm4MoeDetector(CustomTestCase):
    def test_has_tool_call(self):
        d = Glm4MoeDetector()
        self.assertTrue(d.has_tool_call("foo<tool_call>bar"))
        self.assertFalse(d.has_tool_call("just normal text"))

    def test_detect_and_parse_no_tool_call(self):
        text = "just a normal answer with no tool"
        result = Glm4MoeDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_call(self):
        text = (
            "thinking done\n"
            "<tool_call>execute_bash\n"
            "<arg_key>command</arg_key>\n"
            "<arg_value>ls</arg_value>\n"
            "</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(result.normal_text, "thinking done")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"command": "ls"})

    def test_detect_and_parse_two_calls(self):
        text = (
            "<tool_call>execute_bash\n"
            "<arg_key>command</arg_key>\n"
            "<arg_value>ls</arg_value>\n"
            "</tool_call>"
            "<tool_call>execute_bash\n"
            "<arg_key>command</arg_key>\n"
            "<arg_value>pwd</arg_value>\n"
            "</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "ls"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"command": "pwd"})

    def test_streaming_split_chunks(self):
        det = Glm4MoeDetector()
        tools = [C.bash_tool()]

        r1 = det.parse_streaming_increment("plain text. ", tools)
        self.assertEqual(r1.normal_text, "plain text. ")
        self.assertEqual(r1.calls, [])

        chunks = [
            "<tool_call>execute_bash\n",
            "<arg_key>command</arg_key>\n",
            "<arg_value>ls</arg_value>\n",
            "</tool_call>",
        ]
        out_calls = []
        for ch in chunks:
            r = det.parse_streaming_increment(ch, tools)
            out_calls.extend(r.calls)
        names = [c.name for c in out_calls if c.name]
        self.assertEqual(names, ["execute_bash"])
        joined = "".join(c.parameters for c in out_calls)
        self.assertEqual(json.loads(joined), {"command": "ls"})

    def test_detect_and_parse_malformed_skipped(self):
        text = (
            "<tool_call>no_newline_separator</tool_call>"
            "<tool_call>execute_bash\n"
            "<arg_key>command</arg_key>\n"
            "<arg_value>ls</arg_value>\n"
            "</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")

    def test_detect_and_parse_unknown_function(self):
        text = (
            "<tool_call>mystery\n"
            "<arg_key>x</arg_key>\n"
            "<arg_value>1</arg_value>\n"
            "</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_literal_newline(self):
        """GLM4 MOE also supports literal \\n characters as separators."""
        text = (
            "<tool_call>execute_bash\\n"
            "<arg_key>command</arg_key>\\n"
            "<arg_value>ls</arg_value>\\n"
            "</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, [C.bash_tool()])
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"command": "ls"})

    def test_detect_and_parse_param_types(self):
        text = (
            "<tool_call>do_thing\n"
            "<arg_key>n</arg_key>\n"
            "<arg_value>42</arg_value>\n"
            "<arg_key>ratio</arg_key>\n"
            "<arg_value>3.14</arg_value>\n"
            "<arg_key>flag</arg_key>\n"
            "<arg_value>true</arg_value>\n"
            "<arg_key>obj</arg_key>\n"
            '<arg_value>{"k": 1}</arg_value>\n'
            "</tool_call>"
        )
        result = Glm4MoeDetector().detect_and_parse(text, [C.typed_tool()])
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["n"], 42)
        self.assertEqual(args["ratio"], 3.14)
        self.assertIs(args["flag"], True)
        self.assertEqual(args["obj"], {"k": 1})


if __name__ == "__main__":
    import unittest

    unittest.main()
