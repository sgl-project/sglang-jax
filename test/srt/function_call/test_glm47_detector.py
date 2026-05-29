import json
import unittest

from sgl_jax.srt.entrypoints.openai.protocol import Function, Tool
from sgl_jax.srt.function_call.glm47_moe_detector import Glm47MoeDetector
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


class TestGlm47Detector(CustomTestCase):
    def test_has_tool_call(self):
        d = Glm47MoeDetector()
        self.assertTrue(d.has_tool_call("foo<tool_call>bar"))
        self.assertFalse(d.has_tool_call("just normal text"))

    def test_detect_and_parse_single_call(self):
        text = (
            "thinking done\n"
            "<tool_call>execute_bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
        )
        result = Glm47MoeDetector().detect_and_parse(text, [_bash_tool()])
        self.assertEqual(result.normal_text, "thinking done")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "execute_bash")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"command": "ls"})

    def test_detect_and_parse_two_calls(self):
        text = (
            "<tool_call>execute_bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
            "<tool_call>execute_bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>"
        )
        result = Glm47MoeDetector().detect_and_parse(text, [_bash_tool()])
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "execute_bash")
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "ls"})
        self.assertEqual(result.calls[1].name, "execute_bash")
        self.assertEqual(json.loads(result.calls[1].parameters), {"command": "pwd"})


if __name__ == "__main__":
    unittest.main()
