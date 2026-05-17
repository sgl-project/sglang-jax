import json
import logging
import re

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    _GetInfoFunc,
)
from sgl_jax.srt.function_call.ebnf_composer import EBNFComposer

logger = logging.getLogger(__name__)


class Qwen25Detector(BaseFormatDetector):
    """
    Detector for Qwen 2.5 / Qwen 3 / Ling-2.6 tool-call format.

    Format:
        <tool_call>
        {"name": "...", "arguments": {...}}
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>\n"
        self.eot_token = "\n</tool_call>"
        self.tool_call_separator = "\n"
        # Buffer for handling end-tokens that arrive split across streaming chunks.
        self._normal_text_buffer = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        normal_text = text[:idx] if idx != -1 else text
        if idx == -1:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        calls = []
        for match in re.findall(pattern, text, re.DOTALL):
            try:
                parsed = json.loads(match.strip())
                calls.extend(self.parse_base_json(parsed, tools))
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse Qwen25 tool_call JSON: %s (%s)", match, e)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        result = super().parse_streaming_increment(new_text, tools)
        if not result.normal_text:
            return result

        self._normal_text_buffer += result.normal_text
        # eot_token is "\n</tool_call>"; the leading "\n" is the separator that
        # the base parser already returns as part of normal_text after a complete
        # call. Detect/strip the trailing "</tool_call>" tag here.
        end_no_nl = self.eot_token[1:]

        if end_no_nl in self._normal_text_buffer:
            cleaned = self._normal_text_buffer.replace(end_no_nl, "")
            self._normal_text_buffer = ""
            result.normal_text = cleaned
        else:
            partial = self._ends_with_partial_token(self._normal_text_buffer, end_no_nl)
            if partial:
                result.normal_text = self._normal_text_buffer[:-partial]
                self._normal_text_buffer = self._normal_text_buffer[-partial:]
            else:
                result.normal_text = self._normal_text_buffer
                self._normal_text_buffer = ""
        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )

    def build_ebnf(self, tools: list[Tool]) -> str:
        return EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            individual_call_start_token=self.bot_token.replace("\n", "\\n"),
            individual_call_end_token=self.eot_token.replace("\n", "\\n"),
            tool_call_separator="\\n",
        )
