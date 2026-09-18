import re


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(self, normal_text: str = "", reasoning_text: str = ""):
        self.normal_text = normal_text
        self.reasoning_text = reasoning_text


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: str | None = None,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.tool_start_token = tool_start_token

        self._buffer = ""
        self.stripped_think_start = False

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        in_reasoning = self._in_reasoning or text.startswith(self.think_start_token)

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        processed_text = text.replace(self.think_start_token, "").strip()

        if self.think_end_token not in processed_text:
            # Check for tool_start_token interruption
            if self.tool_start_token is not None and self.tool_start_token in processed_text:
                tool_idx = processed_text.find(self.tool_start_token)
                reasoning_text = processed_text[:tool_idx].strip()
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
            # Assume reasoning was truncated before `</think>` token
            return StreamingParseResult(reasoning_text=processed_text)

        # Extract reasoning content
        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0]
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        self._buffer += new_text
        current_text = self._buffer

        # If the current text is a prefix of the think token, keep buffering
        tokens_to_check = [self.think_start_token, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text for token in tokens_to_check
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and self.think_start_token in current_text:
            current_text = current_text.replace(self.think_start_token, "")
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        # Continue with reasoning content
        if self._in_reasoning:
            # Check for tool_start_token interruption
            if self.tool_start_token and self.tool_start_token in current_text:
                tool_idx = current_text.find(self.tool_start_token)
                reasoning_text = current_text[:tool_idx]
                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
            if self.stream_reasoning:
                # Stream the content immediately
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=new_text)

        return StreamingParseResult()


class DeepSeekR1Detector(BaseReasoningFormatDetector):
    """
    Detector for DeepSeek-R1 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True):
        # DeepSeek-R1 is assumed to be reasoning until `</think>` token
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )
        # https://github.com/sgl-project/sglang/pull/3202#discussion_r1950153599


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 model.
    Assumes reasoning format:
      (<think>)*(.*)</think>
    Returns all the text before the </think> tag as `reasoning_text`
    and the rest of the text as `normal_text`.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True):
        # Qwen3 chat_template emits the `<think>\n` opener as part of the prompt,
        # so the completion starts mid-reasoning without the start tag. The
        # `enable_thinking=False` case is gated upstream in serving_chat
        # (`_get_reasoning_from_request`), so this detector is only invoked when
        # reasoning is active.
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=True,
            stream_reasoning=stream_reasoning,
        )


class KimiDetector(BaseReasoningFormatDetector):
    """
    Detector for Kimi Thinking model.
    Assumes reasoning format:
      ◁think▷*(.*)◁/think▷
    Returns all the text before the ◁/think▷ tag as `reasoning_text`
    and the rest of the text as `normal_text`.
    """

    def __init__(self, stream_reasoning: bool = True):
        super().__init__(
            "◁think▷",
            "◁/think▷",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )


class Glm45Detector(BaseReasoningFormatDetector):
    """
    Detector for GLM-4.5 / 4.6 / 4.7 models.
    Assumes reasoning format:
      (<think>)*(.*)</think>

    GLM models use `<tool_call>` as the tool start token to switch from reasoning mode to normal mode.

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(self, stream_reasoning: bool = True, force_reasoning: bool = False):
        super().__init__(
            "<think>",
            "</think>",
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token="<tool_call>",
        )


class Gemma4Detector(BaseReasoningFormatDetector):
    """
    Detector for Gemma 4 model.
    Assumes reasoning format:
      <|channel>thought\n*(.*)<channel|>
    """

    def __init__(self, stream_reasoning: bool = True):
        super().__init__(
            "<|channel>thought\n",
            "<channel|>",
            force_reasoning=False,
            stream_reasoning=stream_reasoning,
        )

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        if text.startswith("<|channel>thought\n"):
            text = text[len("<|channel>thought\n") :]
            self._in_reasoning = True
        elif text.startswith("thought\n"):
            text = text[len("thought\n") :]
            self._in_reasoning = True

        if not self._in_reasoning and "<channel|>" in text:
            self._in_reasoning = True

        if not self._in_reasoning:
            return StreamingParseResult(normal_text=text)

        processed_text = text.strip()

        if self.think_end_token not in processed_text:
            return StreamingParseResult(reasoning_text=processed_text)

        splits = processed_text.split(self.think_end_token, maxsplit=1)
        reasoning_text = splits[0].strip()
        normal_text = splits[1].strip()

        return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        tokens_to_check = [self.think_start_token, "thought\n", self.think_end_token]
        if any(
            token.startswith(current_text) and token != current_text for token in tokens_to_check
        ):
            return StreamingParseResult()

        if not self.stripped_think_start:
            if current_text.startswith("<|channel>thought\n"):
                current_text = current_text[len("<|channel>thought\n") :]
                self.stripped_think_start = True
                self._in_reasoning = True
            elif current_text.startswith("thought\n"):
                current_text = current_text[len("thought\n") :]
                self.stripped_think_start = True
                self._in_reasoning = True

        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)
            reasoning_text = current_text[:end_idx]
            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]
            return StreamingParseResult(
                normal_text=normal_text, reasoning_text=reasoning_text.rstrip()
            )

        if self._in_reasoning:
            if self.stream_reasoning:
                self._buffer = ""
                return StreamingParseResult(reasoning_text=current_text)
            else:
                return StreamingParseResult()

        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=new_text)

        return StreamingParseResult()


class Ling3Detector(Glm45Detector):
    """Preserve unfinished reasoning so a truncated answer is not returned empty."""

    def __init__(self, stream_reasoning: bool = True):
        super().__init__(stream_reasoning=stream_reasoning, force_reasoning=True)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        result = super().detect_and_parse(text)
        if result.reasoning_text and not result.normal_text and "<tool_call>" not in text:
            return StreamingParseResult(normal_text=result.reasoning_text)
        return result


class MuseGlimmerDetector(BaseReasoningFormatDetector):
    """Split Muse Glimmer Harmony channels into reasoning and visible content."""

    _header_re = re.compile(
        r"(?:<\|start\|>assistant\s*)?\s*to=(?P<recipient>[^\s<]+)<\|message\|>"
    )
    _end_re = re.compile(r"<\|eom\|>|<\|eot\|>")
    _markers = ("<|eom|>", "<|eot|>", "<|start|>", "<|message|>")

    def __init__(self, stream_reasoning: bool = True):
        super().__init__("to=self<|message|>", "<|eom|>", stream_reasoning=stream_reasoning)
        self._full_text = ""
        self._emitted_reasoning = 0
        self._emitted_normal = 0

    @classmethod
    def _safe_open_body(cls, body: str) -> str:
        for overlap in range(min(len(body), max(map(len, cls._markers)) - 1), 0, -1):
            suffix = body[-overlap:]
            if any(marker.startswith(suffix) for marker in cls._markers):
                return body[:-overlap]
        return body

    @classmethod
    def _classify(cls, text: str) -> tuple[str, str]:
        reasoning = []
        normal = []
        matches = list(cls._header_re.finditer(text))
        if not matches:
            stripped = text.lstrip()
            if (
                stripped in {"t", "to", "to="}
                or stripped.startswith("to=")
                or stripped.startswith("<|start|")
            ):
                return "", ""
            return "", text
        for index, match in enumerate(matches):
            body_start = match.end()
            next_header = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            end_match = cls._end_re.search(text, body_start, next_header)
            closed = end_match is not None
            body_end = end_match.start() if end_match is not None else next_header
            body = text[body_start:body_end]
            if not closed and index == len(matches) - 1:
                body = cls._safe_open_body(body)
            recipient = match.group("recipient")
            if recipient == "self":
                reasoning.append(body)
            elif recipient == "user":
                normal.append(body)
            else:
                normal.append(body)
        return "".join(reasoning), "".join(normal)

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        reasoning, normal = self._classify(text)
        return StreamingParseResult(normal_text=normal, reasoning_text=reasoning)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._full_text += new_text
        reasoning, normal = self._classify(self._full_text)
        reasoning_delta = reasoning[self._emitted_reasoning :]
        normal_delta = normal[self._emitted_normal :]
        self._emitted_reasoning = len(reasoning)
        self._emitted_normal = len(normal)
        return StreamingParseResult(
            normal_text=normal_delta,
            reasoning_text=reasoning_delta,
        )


class ReasoningParser:
    """
    Parser that handles both streaming and non-streaming scenarios for extracting
    reasoning content from model outputs.

    Args:
        model_type (str): Type of model to parse reasoning from
        stream_reasoning (bool): If False, accumulates reasoning content until complete.
            If True, streams reasoning content as it arrives.
    """

    DetectorMap: dict[str, type[BaseReasoningFormatDetector]] = {
        "deepseek-r1": DeepSeekR1Detector,
        "qwen3": Qwen3Detector,
        "mimo": Qwen3Detector,
        "kimi": KimiDetector,
        "glm45": Glm45Detector,
        "gemma4": Gemma4Detector,
        "ling3": Ling3Detector,
        "muse_glimmer": MuseGlimmerDetector,
    }

    def __init__(self, model_type: str | None = None, stream_reasoning: bool = True):
        if not model_type:
            raise ValueError("Model type must be specified")

        detector_class = self.DetectorMap.get(model_type.lower())
        if not detector_class:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.detector = detector_class(stream_reasoning=stream_reasoning)

    def parse_non_stream(self, full_text: str) -> tuple[str, str]:
        """Non-streaming call: one-time parsing"""
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> tuple[str, str]:
        """Streaming call: incremental parsing"""
        ret = self.detector.parse_streaming_increment(chunk_text)
        return ret.reasoning_text, ret.normal_text
