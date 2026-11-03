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

    def test_mimo_shares_qwen3_detector(self):
        """mimo maps to Qwen3Detector so also gets force_reasoning=True.
        serving_chat gates mimo on `enable_thinking is True`, so the detector
        is only invoked when reasoning is active — same safety as qwen3."""
        parser = ReasoningParser(model_type="mimo")
        reasoning, normal = parser.parse_non_stream("thinking\n</think>\n\nans")
        self.assertEqual(reasoning, "thinking\n")
        self.assertEqual(normal, "ans")

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


class TestReasoningParserLing3(CustomTestCase):
    def test_ling3_registered_and_defaults_to_reasoning(self):
        parser = ReasoningParser(model_type="ling3")
        reasoning, normal = parser.parse_non_stream("reasoning</think>answer")
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(normal, "answer")

    def test_ling3_open_reasoning_is_preserved_as_content(self):
        parser = ReasoningParser(model_type="ling3")
        reasoning, normal = parser.parse_non_stream("only reasoning")
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "only reasoning")

    def test_ling3_tool_call_interrupts_reasoning(self):
        parser = ReasoningParser(model_type="ling3")
        reasoning, normal = parser.parse_non_stream(
            "thinking...<tool_call>execute_bash<arg_key>command</arg_key>"
            "<arg_value>ls</arg_value></tool_call>"
        )
        self.assertEqual(reasoning, "thinking...")
        self.assertTrue(normal.startswith("<tool_call>execute_bash"))


_KIMI_SECTION = (
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
    '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>'
)
# The same completion split at token boundaries (every marker is a single token).
_KIMI_SECTION_TOKENS = [
    "<|tool_calls_section_begin|>",
    "<|tool_call_begin|>",
    "functions.get_weather:0",
    "<|tool_call_argument_begin|>",
    '{"city": ',
    '"Paris"}',
    "<|tool_call_end|>",
    "<|tool_calls_section_end|>",
]


class TestReasoningParserKimiK2(CustomTestCase):
    """Kimi-K2.5: the prompt ends with `<think>`; reasoning ends at the first of `</think>` or
    `<|tool_calls_section_begin|>`, and a `</think>` after that point is dropped (vLLM parity)."""

    def _stream(self, chunks, stream_reasoning=True):
        parser = ReasoningParser(model_type="kimi_k2", stream_reasoning=stream_reasoning)
        reasoning, normal = "", ""
        for chunk in chunks:
            r, n = parser.parse_stream_chunk(chunk)
            reasoning += r
            normal += n
        return reasoning, normal

    def test_kimi_k2_registered_and_kimi_unchanged(self):
        self.assertEqual(
            type(ReasoningParser(model_type="kimi_k2").detector).__name__, "KimiK2Detector"
        )
        self.assertEqual(ReasoningParser(model_type="kimi").detector.think_start_token, "◁think▷")

    def test_think_then_content_then_tools(self):
        parser = ReasoningParser(model_type="kimi_k2")
        reasoning, normal = parser.parse_non_stream("plan</think>Sure." + _KIMI_SECTION)
        self.assertEqual(reasoning, "plan")
        self.assertEqual(normal, "Sure." + _KIMI_SECTION)

    def test_tool_section_ends_reasoning_without_think_end(self):
        parser = ReasoningParser(model_type="kimi_k2")
        reasoning, normal = parser.parse_non_stream("plan" + _KIMI_SECTION)
        self.assertEqual(reasoning, "plan")
        self.assertEqual(normal, _KIMI_SECTION)

    def test_think_end_after_tool_section_is_dropped(self):
        """Kimi-K2.5 RFC example: `</think>` is emitted after `<|tool_calls_section_end|>`."""
        parser = ReasoningParser(model_type="kimi_k2")
        reasoning, normal = parser.parse_non_stream("plan" + _KIMI_SECTION + "</think>")
        self.assertEqual(reasoning, "plan")
        self.assertEqual(normal, _KIMI_SECTION)

    def test_text_after_late_think_end_stays_content(self):
        parser = ReasoningParser(model_type="kimi_k2")
        reasoning, normal = parser.parse_non_stream("plan" + _KIMI_SECTION + "</think>tail")
        self.assertEqual(reasoning, "plan")
        self.assertEqual(normal, _KIMI_SECTION + "tail")

    def test_truncated_reasoning(self):
        parser = ReasoningParser(model_type="kimi_k2")
        reasoning, normal = parser.parse_non_stream("step1 step2")
        self.assertEqual(reasoning, "step1 step2")
        self.assertEqual(normal, "")

    def test_stray_think_end_in_content_is_dropped(self):
        parser = ReasoningParser(model_type="kimi_k2")
        reasoning, normal = parser.parse_non_stream("plan</think>answer</think>")
        self.assertEqual(reasoning, "plan")
        self.assertEqual(normal, "answer")

    def test_streaming_per_token_think_end_after_tool_section(self):
        chunks = ["pl", "an"] + _KIMI_SECTION_TOKENS + ["</think>"]
        self.assertEqual(self._stream(chunks), ("plan", _KIMI_SECTION))

    def test_streaming_coalesced_chunks(self):
        tokens = ["pl", "an"] + _KIMI_SECTION_TOKENS + ["</think>"]
        for split in range(1, len(tokens)):
            chunks = ["".join(tokens[:split]), "".join(tokens[split:])]
            self.assertEqual(self._stream(chunks), ("plan", _KIMI_SECTION), chunks)
        self.assertEqual(self._stream(["".join(tokens)]), ("plan", _KIMI_SECTION))

    def test_streaming_think_then_content_then_tools(self):
        chunks = ["plan", "</think>", "Sure", "."] + _KIMI_SECTION_TOKENS
        self.assertEqual(self._stream(chunks), ("plan", "Sure." + _KIMI_SECTION))

    def test_streaming_content_with_angle_bracket_after_reasoning(self):
        self.assertEqual(self._stream(["plan", "</think>", "a ", "<", " b"]), ("plan", "a < b"))

    def test_streaming_without_stream_reasoning(self):
        parser = ReasoningParser(model_type="kimi_k2", stream_reasoning=False)
        self.assertEqual(parser.parse_stream_chunk("pl"), ("", ""))
        self.assertEqual(parser.parse_stream_chunk("an"), ("", ""))
        self.assertEqual(
            parser.parse_stream_chunk(_KIMI_SECTION_TOKENS[0]), ("plan", _KIMI_SECTION_TOKENS[0])
        )


if __name__ == "__main__":
    unittest.main()
