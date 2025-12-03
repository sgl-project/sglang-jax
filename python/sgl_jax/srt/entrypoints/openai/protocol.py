"""Pydantic models for OpenAI API protocol"""

import time
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, root_validator, validator


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "sglang"
    root: str | None = None
    max_model_len: int | None = None


class ModelList(BaseModel):
    """Model list consists of model cards."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    type: str
    param: str | None = None
    code: int


class LogProbs(BaseModel):
    text_offset: list[int] = Field(default_factory=list)
    token_logprobs: list[float | None] = Field(default_factory=list)
    tokens: list[str] = Field(default_factory=list)
    top_logprobs: list[dict[str, float] | None] = Field(default_factory=list)


class TopLogprob(BaseModel):
    token: str
    bytes: list[int]
    logprob: float


class ChatCompletionTokenLogprob(BaseModel):
    token: str
    bytes: list[int]
    logprob: float
    top_logprobs: list[TopLogprob]


class ChoiceLogprobs(BaseModel):
    # build for v1/chat/completions response
    content: list[ChatCompletionTokenLogprob]


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0
    # only used to return cached tokens when --enable-cache-report is set
    prompt_tokens_details: dict[str, int] | None = None


class StreamOptions(BaseModel):
    include_usage: bool | None = False


class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: str | None = None
    # use alias to workaround pydantic conflict
    schema_: dict[str, object] | None = Field(alias="schema", default=None)
    strict: bool | None = False


class FileRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/files/create
    file: bytes  # The File object (not file name) to be uploaded
    purpose: str = "batch"  # The intended purpose of the uploaded file, default is "batch"


class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str


class FileDeleteResponse(BaseModel):
    id: str
    object: str = "file"
    deleted: bool


class BatchRequest(BaseModel):
    input_file_id: str  # The ID of an uploaded file that contains requests for the new batch
    endpoint: str  # The endpoint to be used for all requests in the batch
    completion_window: str  # The time frame within which the batch should be processed
    metadata: dict | None = None  # Optional custom metadata for the batch


class BatchResponse(BaseModel):
    id: str
    object: str = "batch"
    endpoint: str
    errors: dict | None = None
    input_file_id: str
    completion_window: str
    status: str = "validating"
    output_file_id: str | None = None
    error_file_id: str | None = None
    created_at: int
    in_progress_at: int | None = None
    expires_at: int | None = None
    finalizing_at: int | None = None
    completed_at: int | None = None
    failed_at: int | None = None
    expired_at: int | None = None
    cancelling_at: int | None = None
    cancelled_at: int | None = None
    request_counts: dict | None = None
    metadata: dict | None = None


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: list[int] | list[list[int]] | str | list[str]
    best_of: int | None = None
    echo: bool = False
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: int | None = None
    max_tokens: int = 16
    n: int = 1
    presence_penalty: float = 0.0
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    suffix: str | None = None
    temperature: float = 1.0
    top_p: float = 1.0
    user: str | None = None
    return_hidden_states: bool = False

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    json_schema: str | None = None
    regex: str | None = None
    ebnf: str | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    skip_special_tokens: bool = True
    lora_path: list[str | None] | str | None | None = None
    extra_key: str | None = None
    session_params: dict | None = None

    # For PD disaggregation
    bootstrap_host: str | None = None
    bootstrap_port: int | None = None
    bootstrap_room: int | None = None

    # For request id
    rid: list[str] | str | None = None

    @validator("max_tokens")
    @classmethod
    def validate_max_tokens_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_tokens must be positive")
        return v


class CompletionResponseChoice(BaseModel):
    index: int
    text: str
    logprobs: LogProbs | None = None
    finish_reason: Literal["stop", "length", "content_filter", "abort"] | None = None
    matched_stop: None | int | str = None
    hidden_states: object | None = None

    # @model_serializer(mode="wrap")  # Not available in pydantic v1
    # def _serialize(self, handler):
    #     data = handler(self)
    #     if self.hidden_states is None:
    #         data.pop("hidden_states", None)
    #     return data


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseChoice]
    usage: UsageInfo


class CompletionResponseStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: LogProbs | None = None
    finish_reason: Literal["stop", "length", "content_filter", "abort"] | None = None
    matched_stop: None | int | str = None
    hidden_states: object | None = None

    # @model_serializer(mode="wrap")  # Not available in pydantic v1
    # def _serialize(self, handler):
    #     data = handler(self)
    #     if self.hidden_states is None:
    #         data.pop("hidden_states", None)
    #     return data


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionResponseStreamChoice]
    usage: UsageInfo | None = None


class ChatCompletionMessageContentTextPart(BaseModel):
    type: Literal["text"]
    text: str


class ChatCompletionMessageContentImageURL(BaseModel):
    url: str
    detail: Literal["auto", "low", "high"] | None = "auto"


class ChatCompletionMessageContentVideoURL(BaseModel):
    url: str


class ChatCompletionMessageContentAudioURL(BaseModel):
    url: str


class ChatCompletionMessageContentImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ChatCompletionMessageContentImageURL
    modalities: Literal["image", "multi-images", "video"] | None = "image"


class ChatCompletionMessageContentVideoPart(BaseModel):
    type: Literal["video_url"]
    video_url: ChatCompletionMessageContentVideoURL


class ChatCompletionMessageContentAudioPart(BaseModel):
    type: Literal["audio_url"]
    audio_url: ChatCompletionMessageContentAudioURL


ChatCompletionMessageContentPart = (
    ChatCompletionMessageContentTextPart
    | ChatCompletionMessageContentImagePart
    | ChatCompletionMessageContentVideoPart
    | ChatCompletionMessageContentAudioPart
)


class FunctionResponse(BaseModel):
    """Function response."""

    name: str | None = None
    arguments: str | None = None


class ToolCall(BaseModel):
    """Tool call response."""

    id: str | None = None
    index: int | None = None
    type: Literal["function"] = "function"
    function: FunctionResponse


class ChatCompletionMessageGenericParam(BaseModel):
    role: Literal["system", "assistant", "tool"]
    content: str | list[ChatCompletionMessageContentTextPart] | None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])

    @validator("role", pre=True)
    @classmethod
    def _normalize_role(cls, v):
        if isinstance(v, str):
            v_lower = v.lower()
            if v_lower not in {"system", "assistant", "tool"}:
                raise ValueError(
                    "'role' must be one of 'system', 'assistant', or 'tool' (case-insensitive)."
                )
            return v_lower
        raise ValueError("'role' must be a string")


class ChatCompletionMessageUserParam(BaseModel):
    role: Literal["user"]
    content: str | list[ChatCompletionMessageContentPart]


ChatCompletionMessageParam = ChatCompletionMessageGenericParam | ChatCompletionMessageUserParam


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JsonSchemaResponseFormat | None = None


class StructuresResponseFormat(BaseModel):
    begin: str
    schema_: dict[str, object] | None = Field(alias="schema", default=None)
    end: str


class StructuralTagResponseFormat(BaseModel):
    type: Literal["structural_tag"]
    structures: list[StructuresResponseFormat]
    triggers: list[str]


class Function(BaseModel):
    """Function descriptions."""

    description: str | None = Field(default=None, examples=[None])
    name: str | None = None
    parameters: object | None = None
    strict: bool = False


class Tool(BaseModel):
    """Function wrapper."""

    type: str = Field(default="function", examples=["function"])
    function: Function


class ToolChoiceFuncName(BaseModel):
    """The name of tool choice function."""

    name: str | None = None


class ToolChoice(BaseModel):
    """The tool choice definition."""

    function: ToolChoiceFuncName
    type: Literal["function"] = Field(default="function", examples=["function"])


class ChatCompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: list[ChatCompletionMessageParam]
    model: str
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    logprobs: bool = False
    top_logprobs: int | None = None
    max_tokens: int | None = Field(
        default=None,
        deprecated="max_tokens is deprecated in favor of the max_completion_tokens field",
        description="The maximum number of tokens that can be generated in the chat completion. ",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        description="The maximum number of completion tokens for a chat completion request, "
        "including visible output tokens and reasoning tokens. Input tokens are not included. ",
    )
    n: int = 1
    presence_penalty: float = 0.0
    response_format: ResponseFormat | StructuralTagResponseFormat | None = None
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float = 0.7
    top_p: float = 1.0
    user: str | None = None
    tools: list[Tool] | None = Field(default=None, examples=[None])
    tool_choice: ToolChoice | Literal["auto", "required", "none"] = Field(
        default="auto", examples=["none"]
    )  # noqa
    return_hidden_states: bool = False

    @root_validator(pre=True)
    @classmethod
    def set_tool_choice_default(cls, values):
        if values.get("tool_choice") is None:
            if values.get("tools") is None:
                values["tool_choice"] = "none"
            else:
                values["tool_choice"] = "auto"
        return values

    # Extra parameters for SRT backend only and will be ignored by OpenAI models.
    top_k: int = -1
    min_p: float = 0.0
    min_tokens: int = 0
    regex: str | None = None
    ebnf: str | None = None
    repetition_penalty: float = 1.0
    stop_token_ids: list[int] | None = None
    no_stop_trim: bool = False
    ignore_eos: bool = False
    continue_final_message: bool = False
    skip_special_tokens: bool = True
    lora_path: list[str | None] | str | None | None = None
    extra_key: str | None = None
    session_params: dict | None = None
    separate_reasoning: bool = True
    stream_reasoning: bool = True
    chat_template_kwargs: dict | None = None

    # For request id
    rid: list[str] | str | None = None

    # For PD disaggregation
    bootstrap_host: str | None = None
    bootstrap_port: int | None = None
    bootstrap_room: int | None = None


class ChatMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: LogProbs | ChoiceLogprobs | None = None
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call", "abort"] | None
    ) = None
    matched_stop: None | int | str = None
    hidden_states: object | None = None

    # @model_serializer(mode="wrap")  # Not available in pydantic v1
    # def _serialize(self, handler):
    #     data = handler(self)
    #     if self.hidden_states is None:
    #         data.pop("hidden_states", None)
    #     return data


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, examples=[None])
    hidden_states: object | None = None

    # @model_serializer(mode="wrap")  # Not available in pydantic v1
    # def _serialize(self, handler):
    #     data = handler(self)
    #     if self.hidden_states is None:
    #         data.pop("hidden_states", None)
    #     return data


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    logprobs: LogProbs | ChoiceLogprobs | None = None
    finish_reason: (
        Literal["stop", "length", "tool_calls", "content_filter", "function_call", "abort"] | None
    ) = None
    matched_stop: None | int | str = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None


class MultimodalEmbeddingInput(BaseModel):
    text: str | None = None
    image: str | None = None


EmbeddingInput = list[int] | list[list[int]] | str | list[str] | list[MultimodalEmbeddingInput]


class EmbeddingRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/embeddings/create
    input: EmbeddingInput
    model: str
    encoding_format: str = "float"
    dimensions: int | None = None
    user: str | None = None

    # The request id.
    rid: list[str] | str | None = None


class EmbeddingObject(BaseModel):
    embedding: list[float]
    index: int
    object: str = "embedding"


class EmbeddingResponse(BaseModel):
    data: list[EmbeddingObject]
    model: str
    object: str = "list"
    usage: UsageInfo | None = None


class ScoringRequest(BaseModel):
    query: str | list[int] | None = None  # Query text or pre-tokenized token IDs
    items: str | list[str] | list[list[int]] | None = (
        None  # Item text(s) or pre-tokenized token IDs
    )
    label_token_ids: list[int] | None = None  # Token IDs to compute probabilities for
    apply_softmax: bool = False
    item_first: bool = False
    model: str


class ScoringResponse(BaseModel):
    scores: list[
        list[float]
    ]  # List of lists of probabilities, each in the order of label_token_ids
    model: str
    usage: UsageInfo | None = None
    object: str = "scoring"


class V1RerankReqInput(BaseModel):
    query: str
    documents: list[str]


class RerankResponse(BaseModel):
    score: float
    document: str
    index: int
    meta_info: dict | None = None


OpenAIServingRequest = (
    ChatCompletionRequest | CompletionRequest | EmbeddingRequest | ScoringRequest | V1RerankReqInput
)


@dataclass
class MessageProcessingResult:
    """Result of processing chat messages and applying templates.

    This dataclass encapsulates all the outputs from message processing including
    prompt generation, multimodal data extraction, and constraint preparation.
    Used internally by OpenAIServingChat to pass processed data between methods.

    Args:
        prompt: The final text prompt after applying chat template
        prompt_ids: Either the text prompt (str) or tokenized IDs (List[int])
        image_data: Extracted image data from messages, if any
        audio_data: Extracted audio data from messages, if any
        modalities: List of modality types present in the messages
        stop: Combined stop strings from template and request
        tool_call_constraint: Optional constraint for structured tool calls
    """

    prompt: str
    prompt_ids: str | list[int]
    image_data: Any | None
    audio_data: Any | None
    video_data: Any | None
    modalities: list[str]
    stop: list[str]
    tool_call_constraint: Any | None = None
