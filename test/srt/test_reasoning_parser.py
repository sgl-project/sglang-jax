import unittest

from sgl_jax.srt.reasoning_parser import ReasoningParser
from sgl_jax.test.test_utils import CustomTestCase


class TestReasoningParserGlm(CustomTestCase):
    def test_glm47_reasoning(self):
        parser = ReasoningParser(model_type="glm47")
        reasoning, normal = parser.parse_non_stream("<think>this is reasoning</think>this is normal")
        self.assertEqual(reasoning, "this is reasoning")
        self.assertEqual(normal, "this is normal")

    def test_glm47_interruption(self):
        parser = ReasoningParser(model_type="glm47")
        # Glm47Detector uses tool_start_token="<tool_call>"
        reasoning, normal = parser.parse_non_stream("<think>thinking...<tool_call>func")
        self.assertEqual(reasoning, "thinking...")
        self.assertEqual(normal, "<tool_call>func")


if __name__ == "__main__":
    unittest.main()
