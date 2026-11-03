import json
import logging
import re

from sgl_jax.srt.entrypoints.openai.protocol import Tool
from sgl_jax.srt.function_call.base_format_detector import BaseFormatDetector
from sgl_jax.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sgl_jax.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)

_KIMI_K2_SPECIAL_TOKENS = [
    "<|tool_calls_section_begin|>",
    "<|tool_calls_section_end|>",
    "<|tool_call_begin|>",
    "<|tool_call_end|>",
    "<|tool_call_argument_begin|>",
]


def _strip_special_tokens(text: str) -> str:
    """Remove all Kimi-K2 tool-call special tokens from text."""
    for token in _KIMI_K2_SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text


class KimiK2Detector(BaseFormatDetector):
    """
    Detector for Kimi K2 / K2.5 model function call format.

    Ported from sglang (python/sglang/srt/function_call/kimik2_detector.py @ 6aacca2)
    without the xgrammar structural-tag support, which sgl-jax does not have.

    Format Structure (standard):
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Format Structure (bare counter — model omits function name):
    ```
    <|tool_call_begin|>{counter}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    ```

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    """

    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"
        self.tool_call_argument_begin_token: str = "<|tool_call_argument_begin|>"

        # Capture tool_call_id broadly: the model may emit standard IDs
        # like "functions.ReadFile:0" or bare call counters like "3".
        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>",
            re.DOTALL,
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[^\s<|]+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)",
            re.DOTALL,
        )

        self._last_arguments = ""
        self._current_stream_function_name: str | None = None
        # Set once <|tool_calls_section_begin|> has streamed; afterwards
        # whitespace-only text (newlines between calls) is not content.
        self._in_tool_section = False

        # Standard ID: "functions.search:0", "search:0"
        self.tool_call_id_regex = re.compile(r"^(?:functions\.)?(?P<name>[\w.\-]+):(?P<index>\d+)$")
        # Bare call counter: "0", "3" (model uses auto-incrementing counter)
        self.tool_call_id_counter_regex = re.compile(r"^\d+$")

    def _parse_tool_call_id(
        self, function_id: str, tools: list[Tool], function_args: str | None = None
    ):
        """Parse a tool call ID into (function_name, call_index).

        Standard format: "functions.ReadFile:0" → ("ReadFile", 0)
        Bare counter:    "3" → call_index=3, infer name from arguments.

        The bare counter is a conversation-level auto-increment, NOT an index
        into the tools list. The function name is inferred by matching argument
        keys against tool parameter schemas.
        """
        m = self.tool_call_id_regex.match(function_id)
        if m:
            return m.group("name"), int(m.group("index"))

        if self.tool_call_id_counter_regex.match(function_id):
            call_index = int(function_id)
            name = self._infer_tool_name(tools, function_args)
            if name:
                return name, call_index
            return None, call_index

        logger.warning("Unexpected tool_call_id format: %s", function_id)
        return None, 0

    def _infer_tool_name(self, tools: list[Tool], function_args: str | None = None):
        """Infer function name when the model omits it (bare counter ID).

        Matches argument keys against tool parameter schemas, preferring the
        tool whose declared properties best match the actual arguments.
        """
        if not tools:
            return None
        if len(tools) == 1:
            return tools[0].function.name

        if not function_args:
            logger.debug("No function_args for tool name inference with %d tools", len(tools))
            return None

        try:
            arg_keys = set(json.loads(function_args).keys())
        except (json.JSONDecodeError, TypeError):
            logger.debug(
                "Could not parse function_args for tool name inference "
                "(may be partial JSON in streaming)"
            )
            return None

        # Pick the tool whose properties best match the argument keys.
        best_name = None
        best_score = -1
        for tool in tools:
            params = tool.function.parameters or {}
            props = set(get_schema_properties(params).keys())
            if not props:
                continue
            overlap = len(arg_keys & props)
            extra = len(arg_keys - props)
            score = overlap - extra
            if score > best_score:
                best_score = score
                best_name = tool.function.name

        return best_name

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a KimiK2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: StreamingParseResult with normal_text (content before tool calls) and calls (parsed items).
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            function_call_tuples = self.tool_call_regex.findall(text)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            # ``tool_index`` is the per-response 0-based position of the call
            # (OpenAI spec); enumerate parsed calls locally and ignore the
            # model's ``:N`` suffix, which is a conversation-level counter.
            # ``serving_chat._process_tool_call_id()`` later offsets these by
            # ``history_tool_calls_cnt`` for multi-turn responses.
            local_tool_index = 0
            for match in function_call_tuples:
                function_id, function_args = match
                function_name, _ = self._parse_tool_call_id(function_id, tools, function_args)
                if function_name is None:
                    continue

                logger.debug("function_name %s", function_name)

                tool_calls.append(
                    ToolCallItem(
                        tool_index=local_tool_index,
                        name=function_name,
                        parameters=function_args,
                    )
                )
                local_tool_index += 1

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.exception("Error in detect_and_parse: %s", e)
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing tool calls for KimiK2 format."""
        self._buffer += new_text

        # Fast path: no tool call in flight and no markers yet -- emit as
        # normal text, holding back any trailing partial start token.
        if (
            self._current_stream_function_name is None
            and self.bot_token not in self._buffer
            and self.tool_call_start_token not in self._buffer
        ):
            emit, hold = self._split_pending_start(self._buffer)
            self._buffer = hold
            return StreamingParseResult(normal_text=self._normal_text(emit))

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        normal_text_parts: list[str] = []
        calls: list[ToolCallItem] = []

        try:
            while True:
                buffer = self._buffer

                # Locate next <|tool_call_begin|>, draining any prefix as text.
                begin_idx = self._locate_tool_call_start(buffer, normal_text_parts)
                if begin_idx is None:
                    break
                buffer = self._buffer

                # If another <|tool_call_begin|> appears before the header
                # closes with <|tool_call_argument_begin|>, the section is
                # malformed -- discard and restart at the orphan.
                arg_begin_idx = buffer.find(self.tool_call_argument_begin_token)
                next_begin = buffer.find(
                    self.tool_call_start_token, len(self.tool_call_start_token)
                )
                if next_begin != -1 and (arg_begin_idx == -1 or next_begin < arg_begin_idx):
                    logger.warning(
                        "Kimi-K2 tool_call_begin without preceding tool_call_end; "
                        "discarding incomplete section."
                    )
                    self._buffer = buffer[next_begin:]
                    self._reset_inflight_call_state()
                    continue

                if arg_begin_idx == -1:
                    # Header not fully arrived yet.
                    break

                id_start = len(self.tool_call_start_token)
                function_id = buffer[id_start:arg_begin_idx].strip()
                args_start = arg_begin_idx + len(self.tool_call_argument_begin_token)
                end_idx = buffer.find(self.tool_call_end_token)

                # Resolve function name (cached across chunks within a section).
                name_just_resolved = False
                if self._current_stream_function_name is None:
                    args_for_inference = (
                        buffer[args_start:end_idx] if end_idx != -1 else buffer[args_start:]
                    )
                    resolved = self._resolve_function_name(function_id, tools, args_for_inference)
                    if resolved is None:
                        if end_idx == -1:
                            # Wait for the end marker before deciding.
                            break
                        logger.warning(
                            "Kimi-K2 unrecognized tool_call_id %r; skipping section.",
                            function_id,
                        )
                        self._buffer = buffer[end_idx + len(self.tool_call_end_token) :]
                        self._reset_inflight_call_state()
                        continue
                    name = resolved
                    self._current_stream_function_name = name
                    name_just_resolved = True

                    # ``tool_index`` is the per-response 0-based position
                    # (OpenAI streaming spec); ignore the model's ``:N`` suffix
                    # which is a conversation-level counter.
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.prev_tool_call_arr = []
                        self.streamed_args_for_tool = [""]
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")
                    # Arguments are streamed as raw model text and never parsed,
                    # so "arguments" stays {}; serving_chat's finish-chunk top-up
                    # must not trust it (see _process_tool_call_stream).
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": name,
                        "arguments": {},
                    }
                    self.current_tool_name_sent = True

                # Stream newly-arrived args, combining the first event with
                # the freshly-resolved name. Strip surrounding whitespace like
                # the non-streaming regex; trailing whitespace is held back
                # until more text (or <|tool_call_end|>) arrives.
                args_full = buffer[args_start:end_idx] if end_idx != -1 else buffer[args_start:]
                args_full = args_full.strip()
                argument_diff = args_full[len(self._last_arguments) :]
                if argument_diff or name_just_resolved:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=(
                                self._current_stream_function_name if name_just_resolved else None
                            ),
                            parameters=argument_diff,
                        )
                    )
                    if argument_diff:
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                if end_idx == -1:
                    # Args still streaming.
                    break

                # Section finalized -- advance buffer and prepare next call.
                self._buffer = buffer[end_idx + len(self.tool_call_end_token) :]
                self.current_tool_id += 1
                self._reset_inflight_call_state()

            return StreamingParseResult(normal_text="".join(normal_text_parts), calls=calls)

        except Exception as e:
            logger.exception("Error in parse_streaming_increment: %s", e)
            # Drop the buffer to avoid leaking raw special tokens.
            self._buffer = ""
            self._reset_inflight_call_state()
            return StreamingParseResult(normal_text="".join(normal_text_parts), calls=calls)

    def _reset_inflight_call_state(self) -> None:
        """Reset per-section streaming state after finalize/discard."""
        self._last_arguments = ""
        self.current_tool_name_sent = False
        self._current_stream_function_name = None

    def _locate_tool_call_start(self, buffer: str, normal_text_parts: list) -> int | None:
        """Find the next <|tool_call_begin|>; drain any prefix as normal text.

        Returns 0 on success, or ``None`` when no start token is present yet.
        """
        begin_idx = buffer.find(self.tool_call_start_token)
        if begin_idx == -1:
            emit, hold = self._split_pending_start(buffer)
            if emit:
                normal_text_parts.append(self._normal_text(emit))
            self._buffer = hold
            return None

        if begin_idx > 0:
            normal_text_parts.append(self._normal_text(buffer[:begin_idx]))
            self._buffer = buffer[begin_idx:]
        return 0

    def _normal_text(self, text: str) -> str:
        """Strip special tokens from text drained outside a tool call.

        Text after <|tool_calls_section_begin|> that is only whitespace
        (e.g. newlines between calls) is dropped, matching
        ``detect_and_parse``, which returns only the text before the section.
        """
        bot_idx = text.find(self.bot_token)
        if bot_idx != -1:
            before, after = text[:bot_idx], text[bot_idx:]
            self._in_tool_section = True
        elif self._in_tool_section:
            before, after = "", text
        else:
            before, after = text, ""
        after = _strip_special_tokens(after)
        if not after.strip():
            after = ""
        return _strip_special_tokens(before) + after

    def _split_pending_start(self, text: str) -> tuple[str, str]:
        """Hold back a trailing fragment that could be the start of
        <|tool_calls_section_begin|> or <|tool_call_begin|>. Everything
        before it is safe to emit as normal text.
        """
        candidates = (self.bot_token, self.tool_call_start_token)
        max_tail = max(len(t) for t in candidates) - 1
        for n in range(min(len(text), max_tail), 1, -1):
            tail = text[-n:]
            if any(t.startswith(tail) for t in candidates):
                return text[:-n], tail
        return text, ""

    def _resolve_function_name(
        self, function_id: str, tools: list[Tool], function_args: str
    ) -> str | None:
        """Map a Kimi-K2 tool_call_id to a tool name, or ``None`` if unknown."""
        if not function_id:
            return self._infer_tool_name(tools, function_args)

        m = self.tool_call_id_regex.match(function_id)
        if m:
            return m.group("name")

        if self.tool_call_id_counter_regex.match(function_id):
            return self._infer_tool_name(tools, function_args)

        return None

    # sgl-jax has no Kimi structural tag / EBNF; tool_choice="required" and named
    # tool_choice use the generic json_schema constraint + serving_chat JSON fallback.
    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def build_ebnf(self, tools: list[Tool]) -> str:
        raise NotImplementedError
