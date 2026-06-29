import unittest

from sgl_jax.srt.reasoning_parser import ReasoningParser
from sgl_jax.test.test_utils import CustomTestCase


class TestReasoningParserQwen3(CustomTestCase):
    def test_qwen3_no_open_tag(self):
        """Qwen3 chat_template emits the `<think>\\n` opener as part of the
        prompt, so the completion text starts mid-reasoning without it.
        Detector must still split on `</think>`.
        """
        parser = ReasoningParser(model_type="qwen3")
        reasoning, normal = parser.parse_non_stream("step1 step2\n</think>\n\nfinal answer")
        self.assertEqual(reasoning, "step1 step2\n")
        self.assertEqual(normal, "final answer")

    def test_qwen3_with_open_tag(self):
        parser = ReasoningParser(model_type="qwen3")
        reasoning, normal = parser.parse_non_stream("<think>\nstep1\n</think>\n\nans")
        self.assertEqual(reasoning, "step1\n")
        self.assertEqual(normal, "ans")

    def test_qwen3_truncated_reasoning(self):
        """No `</think>` (length-truncated): whole text is reasoning."""
        parser = ReasoningParser(model_type="qwen3")
        reasoning, normal = parser.parse_non_stream("step1 step2 step3")
        self.assertEqual(reasoning, "step1 step2 step3")
        self.assertEqual(normal, "")

    def test_qwen3_streaming_no_open_tag(self):
        parser = ReasoningParser(model_type="qwen3")
        r1, n1 = parser.parse_stream_chunk("step1 ")
        r2, n2 = parser.parse_stream_chunk("step2</think>")
        r3, n3 = parser.parse_stream_chunk("ans")
        self.assertEqual((r1 + r2 + r3).strip(), "step1 step2")
        self.assertEqual((n1 + n2 + n3).strip(), "ans")


class TestReasoningParserGlm(CustomTestCase):
    def test_glm47_reasoning(self):
        parser = ReasoningParser(model_type="glm45")
        reasoning, normal = parser.parse_non_stream(
            "<think>this is reasoning</think>this is normal"
        )
        self.assertEqual(reasoning, "this is reasoning")
        self.assertEqual(normal, "this is normal")

    def test_glm47_interruption(self):
        parser = ReasoningParser(model_type="glm45")
        # Glm45Detector uses tool_start_token="<tool_call>"
        reasoning, normal = parser.parse_non_stream("<think>thinking...<tool_call>func")
        self.assertEqual(reasoning, "thinking...")
        self.assertEqual(normal, "<tool_call>func")


if __name__ == "__main__":
    unittest.main()
