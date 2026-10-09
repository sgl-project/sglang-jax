import json
import re

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from sgl_jax.srt.function_call.ebnf_composer import EBNFComposer
from sgl_jax.srt.function_call.utils import _is_complete_json


class DeepSeekV3Detector(BaseFormatDetector):
    """Parse DeepSeek V3's native fenced-JSON tool-call format."""

    # Keep these ASCII-escaped because Windows console encodings can otherwise
    # corrupt DeepSeek's full-width-bar and sentence-piece characters.
    calls_begin = "<\uff5ctool\u2581calls\u2581begin\uff5c>"
    calls_end = "<\uff5ctool\u2581calls\u2581end\uff5c>"
    call_begin = "<\uff5ctool\u2581call\u2581begin\uff5c>"
    call_end = "<\uff5ctool\u2581call\u2581end\uff5c>"
    separator = "<\uff5ctool\u2581sep\uff5c>"

    def __init__(self):
        super().__init__()
        self.bot_token = self.calls_begin
        self.eot_token = self.calls_end
        self._last_arguments = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        start = text.find(self.bot_token)
        if start < 0:
            return StreamingParseResult(normal_text=text, calls=[])

        pattern = re.compile(
            re.escape(self.call_begin)
            + r"function"
            + re.escape(self.separator)
            + r"(?P<name>[^\n]+)\n```json\n(?P<arguments>.*?)\n```"
            + re.escape(self.call_end),
            re.DOTALL,
        )
        actions = []
        try:
            for match in pattern.finditer(text):
                actions.append(
                    {
                        "name": match.group("name").strip(),
                        "arguments": json.loads(match.group("arguments")),
                    }
                )
        except json.JSONDecodeError:
            return StreamingParseResult(normal_text=text, calls=[])

        return StreamingParseResult(
            normal_text=text[:start].strip(),
            calls=self.parse_base_json(actions, tools),
        )

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if self.bot_token not in self._buffer and self.call_begin not in self._buffer:
            partial = self._ends_with_partial_token(self._buffer, self.bot_token)
            if partial:
                return StreamingParseResult()
            normal_text, self._buffer = self._buffer, ""
            return StreamingParseResult(normal_text=normal_text.replace(self.eot_token, ""))

        calls: list[ToolCallItem] = []
        pattern = re.compile(
            re.escape(self.call_begin)
            + r"function"
            + re.escape(self.separator)
            + r"(?P<name>[^\n]+)\n```json\n(?P<arguments>.*)",
            re.DOTALL,
        )

        # A generation chunk can contain more than one complete call (for example,
        # with multi-token prediction), so consume complete calls until only a
        # partial call remains.
        while True:
            match = pattern.search(self._buffer)
            if not match:
                break

            name = match.group("name").strip()
            if name not in self._get_tool_indices(tools):
                return StreamingParseResult(calls=calls)
            arguments = match.group("arguments").split("\n```", 1)[0]

            if self.current_tool_id < 0:
                self.current_tool_id = 0
            if not self.current_tool_name_sent:
                calls.append(
                    ToolCallItem(tool_index=self.current_tool_id, name=name, parameters="")
                )
                self.current_tool_name_sent = True
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")
                self.prev_tool_call_arr[self.current_tool_id] = {"name": name, "arguments": {}}

            argument_diff = (
                arguments[len(self._last_arguments) :]
                if arguments.startswith(self._last_arguments)
                else arguments
            )
            if argument_diff:
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id, name=None, parameters=argument_diff
                    )
                )
                self._last_arguments += argument_diff
                self.streamed_args_for_tool[self.current_tool_id] += argument_diff

            call_end_index = self._buffer.find(self.call_end, match.start())
            if call_end_index < 0 or not _is_complete_json(arguments):
                break

            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = json.loads(arguments)
            self._buffer = self._buffer[call_end_index + len(self.call_end) :]
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            self._last_arguments = ""
        return StreamingParseResult(calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=self.calls_begin
            + self.call_begin
            + "function"
            + self.separator
            + name
            + "\n```json\n",
            end="\n```" + self.call_end + self.calls_end,
            trigger=self.calls_begin,
        )

    def build_ebnf(self, tools: list[Tool]) -> str:
        return EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            sequence_start_token=self.calls_begin,
            sequence_end_token=self.calls_end,
            individual_call_start_token=self.call_begin,
            individual_call_end_token=self.call_end,
            tool_call_separator="\\n",
            call_rule_fmt=(
                f'"function{self.separator}{{name}}\\n```json\\n" ' '{arguments_rule} "\\n```"'
            ),
        )
