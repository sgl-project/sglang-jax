import json
import unittest

from sgl_jax.srt.function_call.function_call_parser import FunctionCallParser
from sgl_jax.srt.function_call.ling3_detector import Ling3Detector
from sgl_jax.test.test_utils import CustomTestCase
from sgl_jax.test.tool_parser_test_config import ToolParserTestConfig as C


class TestLing3Detector(CustomTestCase):
    def test_parser_registered(self):
        parser = FunctionCallParser([C.bash_tool()], "ling3")
        self.assertIsInstance(parser.detector, Ling3Detector)

    def test_non_stream_with_newline(self):
        parser = FunctionCallParser([C.bash_tool()], "ling3")
        normal, calls = parser.parse_non_stream(
            "thinking done\n<tool_call>execute_bash  \n"
            "<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
        )
        self.assertEqual(normal, "thinking done")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_bash")
        self.assertEqual(json.loads(calls[0].parameters), {"command": "ls"})

    def test_non_stream_without_newline(self):
        parser = FunctionCallParser([C.bash_tool()], "ling3")
        normal, calls = parser.parse_non_stream(
            "<tool_call>execute_bash<arg_key>command</arg_key>"
            "<arg_value>pwd</arg_value></tool_call>"
        )
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_bash")
        self.assertEqual(json.loads(calls[0].parameters), {"command": "pwd"})

    def test_non_stream_without_arguments(self):
        parser = FunctionCallParser([C.bash_tool()], "ling3")
        normal, calls = parser.parse_non_stream("<tool_call>execute_bash</tool_call>")
        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "execute_bash")
        self.assertEqual(json.loads(calls[0].parameters), {})

    def test_streaming_without_newline(self):
        detector = Ling3Detector()
        tools = [C.bash_tool()]
        chunks = [
            "<tool_call>execute_bash",
            "<arg_key>command</arg_key>",
            "<arg_value>ls</arg_value>",
            "</tool_call>",
        ]
        calls = []
        for chunk in chunks:
            calls.extend(detector.parse_streaming_increment(chunk, tools).calls)

        self.assertEqual([call.name for call in calls if call.name], ["execute_bash"])
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls)), {"command": "ls"}
        )


if __name__ == "__main__":
    unittest.main()
