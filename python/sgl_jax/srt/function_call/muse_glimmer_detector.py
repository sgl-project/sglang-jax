"""ATEM tool-call detector for Muse Glimmer."""

from __future__ import annotations

import json
import re

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
)

_INVOKE_RE = re.compile(r"<atem:invoke\b.*?</atem:invoke>", re.DOTALL)
_NAME_RE = re.compile(r'<atem:invoke\b[^>]*?\bname="([^"]+)"')
_PARAM_RE = re.compile(
    r'<atem:parameter\b[^>]*?\bname="(?P<key>[^"]+)"[^>]*?>(?P<value>.*?)</atem:parameter>',
    re.DOTALL,
)
_OPEN = "<atem:function_calls>"
_CLOSE = "</atem:function_calls>"


def _decode_value(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


class MuseGlimmerDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.bot_token = _OPEN
        self.eot_token = _CLOSE
        self._emitted_calls = 0

    @staticmethod
    def _normalize_name(name: str, tools: list[Tool]) -> str:
        registered = {tool.function.name for tool in tools if tool.function.name}
        if name in registered:
            return name
        head, separator, tail = name.partition(".")
        if separator and head == tail and head in registered:
            return head
        return name

    @classmethod
    def _parse_calls(cls, text: str, tools: list[Tool]) -> list[ToolCallItem]:
        registered = {tool.function.name for tool in tools if tool.function.name}
        calls = []
        for index, invoke in enumerate(_INVOKE_RE.findall(text)):
            name_match = _NAME_RE.search(invoke)
            if name_match is None:
                continue
            name = cls._normalize_name(name_match.group(1), tools)
            if registered and name not in registered:
                continue
            arguments = {
                match.group("key"): _decode_value(match.group("value"))
                for match in _PARAM_RE.finditer(invoke)
            }
            calls.append(
                ToolCallItem(
                    tool_index=index,
                    name=name,
                    parameters=json.dumps(arguments, ensure_ascii=False),
                )
            )
        return calls

    def has_tool_call(self, text: str) -> bool:
        return _OPEN in text or "<atem:invoke" in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        calls = self._parse_calls(text, tools)
        if calls:
            return StreamingParseResult(calls=calls)
        if self.has_tool_call(text):
            return StreamingParseResult()
        return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        if not self.has_tool_call(self._buffer):
            partial = self._ends_with_partial_token(self._buffer, self.bot_token)
            if partial:
                normal = self._buffer[:-partial]
                self._buffer = self._buffer[-partial:]
            else:
                normal = self._buffer
                self._buffer = ""
            return StreamingParseResult(normal_text=normal)

        calls = self._parse_calls(self._buffer, tools)
        if len(calls) <= self._emitted_calls:
            return StreamingParseResult()
        emitted = calls[self._emitted_calls :]
        self._emitted_calls = len(calls)
        return StreamingParseResult(calls=emitted)

    def supports_structural_tag(self) -> bool:
        return False

    def parses_required_natively(self) -> bool:
        return True

    def structure_info(self):
        return lambda name: StructureInfo(begin="", end="", trigger=_OPEN)

    def build_ebnf(self, tools: list[Tool]) -> str:
        return ""
