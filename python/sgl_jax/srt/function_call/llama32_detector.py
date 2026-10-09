import ast
import json
from json import JSONDecodeError, JSONDecoder

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import StreamingParseResult, StructureInfo, _GetInfoFunc
from sgl_jax.srt.function_call.ebnf_composer import EBNFComposer


class Llama32Detector(BaseFormatDetector):
    """Parse Llama 3.2's ``<|python_tag|>`` JSON tool-call format."""

    def __init__(self):
        super().__init__()
        self.bot_token = "<|python_tag|>"
        # Llama's template uses semicolons when it emits multiple calls.
        self.tool_call_separator = ";"

    @staticmethod
    def _convert_python_dict_to_json(text: str) -> str:
        """Convert a complete Python dict literal to JSON when possible."""
        try:
            parsed = ast.literal_eval(text.strip())
        except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
            return text
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=False)
        return text

    def has_tool_call(self, text: str) -> bool:
        # Some Llama chat templates omit the marker before a tool call.
        return self.bot_token in text or text.lstrip().startswith("{")

    @staticmethod
    def _find_dict_end(text: str, start: int) -> int | None:
        """Return the exclusive end offset of the dict beginning at ``start``."""
        depth = 0
        quote = None
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
                continue
            if char in ("'", '"'):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return index + 1
        return None

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        if self.bot_token in text:
            normal_text, action_text = text.split(self.bot_token, maxsplit=1)
        elif text.lstrip().startswith("{"):
            normal_text, action_text = "", text.lstrip()
        else:
            return StreamingParseResult(normal_text=text, calls=[])

        decoder = JSONDecoder()
        actions = []
        index = 0
        consumed = 0
        while index < len(action_text):
            while index < len(action_text) and action_text[index].isspace():
                index += 1
            if action_text.startswith(self.tool_call_separator, index):
                index += len(self.tool_call_separator)
                continue

            try:
                action, end = decoder.raw_decode(action_text, index)
            except JSONDecodeError:
                end = self._find_dict_end(action_text, index)
                if end is None:
                    break
                converted = self._convert_python_dict_to_json(action_text[index:end])
                if converted == action_text[index:end]:
                    break
                try:
                    action, _ = decoder.raw_decode(converted)
                except JSONDecodeError:
                    break
            actions.append(action)
            index = end
            consumed = end

        calls = self.parse_base_json(actions, tools) if actions else []
        trailing = action_text[consumed:].strip() if consumed < len(action_text) else ""
        return StreamingParseResult(normal_text=normal_text + trailing, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        self._buffer += new_text

        marker_index = self._buffer.find(self.bot_token)
        if marker_index > 0:
            normal_text = self._buffer[:marker_index]
            self._buffer = self._buffer[marker_index:]
            return StreamingParseResult(normal_text=normal_text)

        # The base implementation already handles incremental JSON. Convert the
        # Python-literal variant emitted by some templates before delegating.
        converted = self._buffer
        try:
            start = converted.find(self.bot_token)
            literal = converted[start + len(self.bot_token) :] if start >= 0 else converted
            json_literal = self._convert_python_dict_to_json(literal)
            if json_literal != literal:
                converted = (
                    converted[: start + len(self.bot_token)] + json_literal
                    if start >= 0
                    else json_literal
                )
        except (MemoryError, RecursionError, SyntaxError, TypeError, ValueError):
            converted = self._buffer

        original = self._buffer
        self._buffer = converted
        result = super().parse_streaming_increment("", tools)
        if self._buffer == converted:
            self._buffer = original
        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<|python_tag|>{"name":"' + name + '", "arguments":',
            end="}",
            trigger=self.bot_token,
        )

    def build_ebnf(self, tools: list[Tool]) -> str:
        grammar = EBNFComposer.build_ebnf(
            tools,
            function_format="json",
            tool_call_separator=self.tool_call_separator,
        )
        root = 'root ::= function_call ( ";" function_call )*'
        return grammar.replace(
            root, f'root ::= "{self.bot_token}" function_call ( ";" function_call )*', 1
        )
