"""Unit tests for MiMoDetector (function-call parser).

Run with:
    python -m unittest test.srt.function_call.test_mimo_detector
"""

import json

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool
from sgl_jax.srt.function_call.mimo_detector import MiMoDetector
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


def _bash_tool() -> Tool:
    return _tool(
        "execute_bash",
        {"command": {"type": "string"}},
    )


def _typed_tool() -> Tool:
    return _tool(
        "do_thing",
        {
            "n": {"type": "integer"},
            "ratio": {"type": "number"},
            "flag": {"type": "boolean"},
            "obj": {"type": "object"},
        },
    )


class TestMiMoDetector(CustomTestCase):
    def test_has_tool_call(self):
        d = MiMoDetector()
        self.assertTrue(d.has_tool_call("foo<tool_call>bar"))
        self.assertFalse(d.has_tool_call("just normal text"))

    def test_detect_and_parse_single_call(self):
        text = (
            "thinking done\n"
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>pwd && ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = MiMoDetector().detect_and_parse(text, [_bash_tool()])
        self.assertEqual(result.normal_text, "thinking done\n")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"command": "pwd && ls"})

    def test_detect_and_parse_param_types(self):
        text = (
            "<tool_call>\n"
            "<function=do_thing>\n"
            "<parameter=n>42</parameter>\n"
            "<parameter=ratio>3.14</parameter>\n"
            "<parameter=flag>true</parameter>\n"
            '<parameter=obj>{"k": 1}</parameter>\n'
            "</function>\n"
            "</tool_call>"
        )
        result = MiMoDetector().detect_and_parse(text, [_typed_tool()])
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["n"], 42)
        self.assertEqual(args["ratio"], 3.14)
        self.assertIs(args["flag"], True)
        self.assertEqual(args["obj"], {"k": 1})

    def test_detect_and_parse_unknown_function(self):
        text = (
            "before\n"
            "<tool_call>\n"
            "<function=mystery>\n"
            "<parameter=x>1</parameter>\n"
            "</function>\n"
            "</tool_call>\n"
            "after"
        )
        result = MiMoDetector().detect_and_parse(text, [_bash_tool()])
        self.assertEqual(result.calls, [])
        self.assertIn("<function=mystery>", result.normal_text)
        self.assertTrue(result.normal_text.startswith("before\n"))

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
        result = MiMoDetector().detect_and_parse(text, [_bash_tool()])
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "ls"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"command": "pwd"})

    def test_streaming_normal_text_then_call(self):
        det = MiMoDetector()
        tools = [_bash_tool()]
        r1 = det.parse_streaming_increment("plain text. ", tools)
        self.assertEqual(r1.normal_text, "plain text. ")
        self.assertEqual(r1.calls, [])

        r2 = det.parse_streaming_increment(
            "<tool_call>\n<function=execute_bash>\n"
            "<parameter=command>ls</parameter>\n</function>\n</tool_call>",
            tools,
        )
        self.assertEqual(len(r2.calls), 1)
        self.assertEqual(r2.calls[0].name, "execute_bash")
        self.assertEqual(json.loads(r2.calls[0].parameters), {"command": "ls"})
        self.assertNotIn("<tool_call>", r2.normal_text)
        self.assertNotIn("</tool_call>", r2.normal_text)

    def test_param_value_python_literal(self):
        text = (
            "<tool_call>\n"
            "<function=do_thing>\n"
            "<parameter=obj>{'k': 1, 'v': [1, 2]}</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = MiMoDetector().detect_and_parse(text, [_typed_tool()])
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["obj"], {"k": 1, "v": [1, 2]})

    def test_param_value_html_unescape(self):
        text = (
            "<tool_call>\n"
            "<function=execute_bash>\n"
            "<parameter=command>echo &quot;hi&quot; &amp;&amp; ls</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = MiMoDetector().detect_and_parse(text, [_bash_tool()])
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["command"], 'echo "hi" && ls')


if __name__ == "__main__":
    import unittest

    unittest.main()
