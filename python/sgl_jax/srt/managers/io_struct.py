import copy
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from sgl_jax.srt.managers.schedule_batch import BaseFinishReason


@dataclass
class BatchStrOut:
    # The request id
    rids: list[str]
    # The finish reason
    finished_reasons: list[dict]
    # The output decoded strings
    output_strs: list[str]
    # The token ids
    output_ids: list[int] | None

    # Token counts
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]

    # Logprobs
    input_token_logprobs_val: list[float]
    input_token_logprobs_idx: list[int]
    output_token_logprobs_val: list[float]
    output_token_logprobs_idx: list[int]
    input_top_logprobs_val: list[list]
    input_top_logprobs_idx: list[list]
    output_top_logprobs_val: list[list]
    output_top_logprobs_idx: list[list]
    input_token_ids_logprobs_val: list[list]
    input_token_ids_logprobs_idx: list[list]
    output_token_ids_logprobs_val: list[list]
    output_token_ids_logprobs_idx: list[list]

    # Hidden states
    output_hidden_states: list[list[float]]

    # Cache miss count
    cache_miss_count: int = None


@dataclass
class BatchTokenIDOut:
    # The request id
    rids: list[str]
    # The finish reason
    finished_reasons: list[BaseFinishReason]
    # For incremental decoding
    decoded_texts: list[str]
    decode_ids: list[list[int]]
    read_offsets: list[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: list[int] | None
    # Detokenization configs
    skip_special_tokens: list[bool]
    spaces_between_special_tokens: list[bool]
    no_stop_trim: list[bool]

    # Token counts
    prompt_tokens: list[int]
    completion_tokens: list[int]
    cached_tokens: list[int]

    # Logprobs
    input_token_logprobs_val: list[float]
    input_token_logprobs_idx: list[int]
    output_token_logprobs_val: list[float]
    output_token_logprobs_idx: list[int]
    input_top_logprobs_val: list[list]
    input_top_logprobs_idx: list[list]
    output_top_logprobs_val: list[list]
    output_top_logprobs_idx: list[list]
    input_token_ids_logprobs_val: list[list]
    input_token_ids_logprobs_idx: list[list]
    output_token_ids_logprobs_val: list[list]
    output_token_ids_logprobs_idx: list[list]

    # Hidden states
    output_hidden_states: list[list[float]]

    # Cache miss count
    cache_miss_count: int = None


@dataclass
class TokenizedGenerateReqInput:
    # The request id.
    rid: list[str] | str | None = None
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: list[str] | str | None = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: list[list[int]] | list[int] | None = None
    # The sampling_params. See descriptions below.
    sampling_params: list[dict] | dict | None = None
    # Whether to return logprobs.
    return_logprob: list[bool] | bool | None = None
    # Whether to return onutput logprobs only.
    return_output_logprob_only: bool | None = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: list[int] | int | None = -1
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: list[int] | int | None = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: list[list[int]] | list[int] | None = None
    # Whether to stream output
    stream: bool = False
    # LoRA related
    lora_id: str | None = None  # None means just use the base model


@dataclass
class AbortReq:
    # The request id
    rid: str = ""
    # Whether to abort all requests
    abort_all: bool = False


# Additional classes needed for engine.py imports
@dataclass
class EmbeddingReqInput:
    """Request input for embedding generation."""

    rid: str = None
    text: str = ""
    input_ids: list[int] = None
    normalize: bool = True


@dataclass
class GenerateReqInput:
    """Request input for text generation."""

    batch_size: int = 1
    rid: list[str] | str | None = None
    text: list[str] | str | None = None
    input_ids: list[int] = None
    # The embeddings for input_ids; one can specify either text or input_ids or input_embeds.
    input_embeds: list[list[list[float]]] | list[list[float]] | None = None
    sampling_params: Any | None = (
        None  # Using Any for now to avoid SamplingParams serialization issues
    )
    stream: bool = False
    is_single: bool = True
    return_logprob: list[bool] | bool | None = None
    return_output_logprob_only: bool | None = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: list[int] | int | None = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: list[int] | int | None = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: list[list[int]] | list[int] | None = None
    # Whether to detokenize tokens in text in the returned logprobs.
    return_text_in_logprobs: bool = True
    # The path to the LoRA adaptors
    lora_path: list[str] | str | None = None
    # The uid of LoRA adaptors, should be initialized by tokenizer manager
    lora_id: list[str] | str | None = None

    def _normalize_rid(self, num):
        """Normalize request IDs for batch processing."""
        if self.rid is None:
            self.rid = [uuid.uuid4().hex for _ in range(num)]
        elif isinstance(self.rid, str):
            new_rids = [f"{self.rid}_{i}" for i in range(num)]
            self.rid = new_rids
        elif isinstance(self.rid, list):
            # Note: the length of rid shall be the same as the batch_size,
            # as the rid would be expanded for parallel sampling in tokenizer_manager
            if len(self.rid) != self.batch_size:
                raise ValueError(
                    "The specified rids length mismatch with the batch_size for batch processing."
                )
        else:
            raise ValueError("The rid should be a string or a list of strings.")

    def normalize_batch_and_arguments(self):
        self._validate_inputs()
        self._determine_batch_size()
        self._handle_parallel_sampling()

        if self.is_single:
            self._normalize_single_inputs()
        else:
            self._normalize_batch_inputs()

    def _normalize_single_inputs(self):
        """Normalize inputs for a single example."""
        if self.sampling_params is None:
            self.sampling_params = {}
        if self.rid is None:
            self.rid = uuid.uuid4().hex
        if self.return_logprob is None:
            self.return_logprob = False
        if self.logprob_start_len is None:
            self.logprob_start_len = -1
        if self.top_logprobs_num is None:
            self.top_logprobs_num = 0
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = None

    def _handle_parallel_sampling(self):
        """Handle parallel sampling parameters and adjust batch size if needed."""
        # Determine parallel sample count
        if self.sampling_params is None:
            self.parallel_sample_num = 1
            return
        elif isinstance(self.sampling_params, dict):
            self.parallel_sample_num = self.sampling_params.get("n", 1)
        else:  # isinstance(self.sampling_params, list):
            self.parallel_sample_num = self.sampling_params[0].get("n", 1)
            for sampling_params in self.sampling_params:
                if self.parallel_sample_num != sampling_params.get("n", 1):
                    raise ValueError(
                        "The parallel_sample_num should be the same for all samples in sample params."
                    )

        # If using parallel sampling with a single example, convert to batch
        if self.parallel_sample_num > 1 and self.is_single:
            self.is_single = False
            if self.text is not None:
                self.text = [self.text]
            if self.input_ids is not None:
                self.input_ids = [self.input_ids]
            if self.input_embeds is not None:
                self.input_embeds = [self.input_embeds]

    def _normalize_batch_inputs(self):
        """Normalize inputs for a batch of examples, including parallel sampling expansion."""
        # Calculate expanded batch size
        if self.parallel_sample_num == 1:
            num = self.batch_size
        else:
            # Expand parallel_sample_num
            num = self.batch_size * self.parallel_sample_num

        # Expand input based on type
        self._expand_inputs(num)
        self._normalize_rid(num)
        self._normalize_sampling_params(num)
        self._normalize_logprob_params(num)
        self._normalize_lora_paths(num)

    def _expand_inputs(self, num):
        """Expand the main inputs (text, input_ids, input_embeds) for parallel sampling."""
        if self.text is not None:
            if not isinstance(self.text, list):
                raise ValueError("Text should be a list for batch processing.")
            self.text = self.text * self.parallel_sample_num
        elif self.input_ids is not None:
            if not isinstance(self.input_ids, list) or not isinstance(self.input_ids[0], list):
                raise ValueError("input_ids should be a list of lists for batch processing.")
            self.input_ids = self.input_ids * self.parallel_sample_num
        elif self.input_embeds is not None:
            if not isinstance(self.input_embeds, list):
                raise ValueError("input_embeds should be a list for batch processing.")
            self.input_embeds = self.input_embeds * self.parallel_sample_num

    def _validate_inputs(self):
        """Validate that the input configuration is valid."""
        if (self.text is None and self.input_ids is None) or (
            self.text is not None and self.input_ids is not None
        ):
            raise ValueError("Either text or input_ids should be provided.")

    def _normalize_sampling_params(self, num):
        """Normalize sampling parameters for batch processing."""
        if self.sampling_params is None:
            self.sampling_params = [{}] * num
        elif isinstance(self.sampling_params, dict):
            self.sampling_params = [self.sampling_params] * num
        else:  # Already a list
            self.sampling_params = self.sampling_params * self.parallel_sample_num

    def _determine_batch_size(self):
        """Determine if this is a single example or a batch and the batch size."""
        if self.text is not None:
            if isinstance(self.text, str):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.text)
            self.input_embeds = None
        elif self.input_ids is not None:
            if len(self.input_ids) == 0:
                raise ValueError("input_ids cannot be empty.")
            if isinstance(self.input_ids[0], int):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_ids)
            self.input_embeds = None
        else:
            if isinstance(self.input_embeds[0][0], float):
                self.is_single = True
                self.batch_size = 1
            else:
                self.is_single = False
                self.batch_size = len(self.input_embeds)

    def _normalize_logprob_params(self, num):
        """Normalize logprob-related parameters for batch processing."""

        # Helper function to normalize a parameter
        def normalize_param(param, default_value, param_name):
            if param is None:
                return [default_value] * num
            elif not isinstance(param, list):
                return [param] * num
            else:
                if self.parallel_sample_num > 1:
                    raise ValueError(f"Cannot use list {param_name} with parallel_sample_num > 1")
                return param

        # Normalize each logprob parameter
        self.return_logprob = normalize_param(self.return_logprob, False, "return_logprob")
        self.logprob_start_len = normalize_param(self.logprob_start_len, -1, "logprob_start_len")
        self.top_logprobs_num = normalize_param(self.top_logprobs_num, 0, "top_logprobs_num")

        # Handle token_ids_logprob specially due to its nested structure
        if not self.token_ids_logprob:  # covers both None and []
            self.token_ids_logprob = [None] * num
        elif not isinstance(self.token_ids_logprob, list):
            self.token_ids_logprob = [[self.token_ids_logprob] for _ in range(num)]
        elif not isinstance(self.token_ids_logprob[0], list):
            self.token_ids_logprob = [copy.deepcopy(self.token_ids_logprob) for _ in range(num)]
        elif self.parallel_sample_num > 1:
            raise ValueError("Cannot use list token_ids_logprob with parallel_sample_num > 1")

    def _normalize_lora_paths(self, num):
        """Normalize LoRA paths for batch processing."""
        if self.lora_path is not None:
            if isinstance(self.lora_path, str):
                self.lora_path = [self.lora_path] * num
            elif isinstance(self.lora_path, list):
                self.lora_path = self.lora_path * self.parallel_sample_num
            else:
                raise ValueError("lora_path should be a list or a string.")

    def regenerate_rid(self):
        """Generate a new request ID and return it."""
        self.rid = uuid.uuid4().hex
        return self.rid

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            lora_id=self.lora_id[i] if self.lora_id is not None else None,
        )


@dataclass
class RpcReqInput:
    """Base class for RPC request input."""

    request_id: str


@dataclass
class RpcReqOutput:
    """Base class for RPC request output."""

    request_id: str
    success: bool = True
    error_msg: str = ""


@dataclass
class ReleaseMemoryOccupationReqInput(RpcReqInput):
    """Request to release memory occupation."""

    memory_size: int


@dataclass
class ResumeMemoryOccupationReqInput(RpcReqInput):
    """Request to resume memory occupation."""

    memory_size: int


# Additional output classes
@dataclass
class BatchEmbeddingOut:
    """Batch embedding output."""

    rids: list[str]
    embeddings: list[list[float]]
    prompt_tokens: list[int]


# Request input classes for sessions
@dataclass
class OpenSessionReqInput(RpcReqInput):
    """Request to open a session."""

    session_id: str


@dataclass
class CloseSessionReqInput(RpcReqInput):
    """Request to close a session."""

    session_id: str


@dataclass
class TokenizedEmbeddingReqInput:
    """Tokenized embedding request input."""

    rid: str
    text: str
    input_ids: list[int]
    normalize: bool = True


# Configuration and logging classes
@dataclass
class ConfigureLoggingReq(RpcReqInput):
    """Request to configure logging."""

    log_level: str
    log_file: str | None = None


@dataclass
class FlushCacheReqInput(RpcReqInput):
    """Request to flush cache."""

    request_id: str = ""
    cache_type: str = "all"

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"flush_cache_{int(time.time())}"


# Internal state classes
@dataclass
class GetInternalStateReq:
    pass


@dataclass
class GetInternalStateReqOutput:
    internal_state: dict[Any, Any]


@dataclass
class SetInternalStateReq(RpcReqInput):
    """Request to set internal state."""

    state_data: dict[str, Any]


# Profile classes


@dataclass
class ProfileReqInput:
    output_dir: str | None = None
    start_step: int | None = None
    num_steps: int | None = None
    # Sets the trace level for host-side activities.
    # 0: Disables host (CPU) tracing entirely.
    # 1: Enables tracing of only user-instrumented TraceMe events (this is the default).
    # 2: Includes level 1 traces plus high-level program execution details like expensive XLA operations.
    # 3: Includes level 2 traces plus more verbose, low-level program execution details such as cheap XLA operations.
    host_tracer_level: int | None = None
    # Controls whether Python tracing is enabled.
    # 0: Disables Python function call tracing.
    # 1: Enables Python tracing (this is the default).
    python_tracer_level: int | None = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq:
    type: ProfileReqType
    output_dir: str | None = None
    start_step: int | None = None
    num_steps: int | None = None
    host_tracer_level: int | None = None
    python_tracer_level: int | None = None
    profile_id: str | None = None


@dataclass
class ProfileReqOutput:
    success: bool
    message: str


# Health check
@dataclass
class HealthCheckOutput:
    """Health check output."""

    status: str = "healthy"
    timestamp: float = 0.0


@dataclass
class ReleaseMemoryOccupationReqOutput(RpcReqOutput):
    """Output for ReleaseMemoryOccupationReqInput."""

    pass


@dataclass
class ResumeMemoryOccupationReqOutput(RpcReqOutput):
    """Output for ResumeMemoryOccupationReqInput."""

    pass


@dataclass
class OpenSessionReqOutput(RpcReqOutput):
    """Output for OpenSessionReqInput."""

    session_id: str = ""


@dataclass
class CloseSessionReqOutput(RpcReqOutput):
    """Output for CloseSessionReqInput."""

    pass


@dataclass
class ConfigureLoggingReqOutput(RpcReqOutput):
    """Output for ConfigureLoggingReq."""

    pass


@dataclass
class FlushCacheReqOutput(RpcReqOutput):
    """Output for FlushCacheReqInput."""

    flushed_items: int = 0


@dataclass
class SetInternalStateReqOutput(RpcReqOutput):
    """Output for SetInternalStateReq."""

    pass


# Additional missing request classes
@dataclass
class ParseFunctionCallReq(RpcReqInput):
    """Request to parse function calls."""

    text: str
    parser_type: str = "default"


@dataclass
class SeparateReasoningReqInput(RpcReqInput):
    """Request to separate reasoning."""

    text: str
    reasoning_type: str = "default"


@dataclass
class VertexGenerateReqInput(GenerateReqInput):
    """Vertex AI compatible generate request input."""

    pass


@dataclass
class StartTraceReqInput(RpcReqInput):
    """Request to start precision tracing."""

    req_num: int | None = None  # Maximum number of requests to trace
    output_file: str | None = None  # Output file path
    request_id: str = ""  # Override base class field with default
    save_tensor: bool = False  # Save the specific tensor content

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"start_trace_{int(time.time())}"


@dataclass
class StopTraceReqInput(RpcReqInput):
    """Request to stop precision tracing."""

    request_id: str = ""  # Override base class field with default

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"stop_trace_{int(time.time())}"


@dataclass
class TraceStatusReqInput(RpcReqInput):
    """Request to get trace status."""

    request_id: str = ""  # Override base class field with default

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"trace_status_{int(time.time())}"
