import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sgl_jax.srt.managers.schedule_batch import BaseFinishReason


@dataclass
class BatchStrOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[dict]
    # The output decoded strings
    output_strs: List[str]
    # The token ids
    output_ids: Optional[List[int]]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]

    # Hidden states
    output_hidden_states: List[List[float]]


@dataclass
class BatchTokenIDOut:
    # The request id
    rids: List[str]
    # The finish reason
    finished_reasons: List[BaseFinishReason]
    # For incremental decoding
    decoded_texts: List[str]
    decode_ids: List[List[int]]
    read_offsets: List[int]
    # Only used when `--skip-tokenizer-init` is on
    output_ids: Optional[List[int]]
    # Detokenization configs
    skip_special_tokens: List[bool]
    spaces_between_special_tokens: List[bool]
    no_stop_trim: List[bool]

    # Token counts
    prompt_tokens: List[int]
    completion_tokens: List[int]
    cached_tokens: List[int]

    # Logprobs
    input_token_logprobs_val: List[float]
    input_token_logprobs_idx: List[int]
    output_token_logprobs_val: List[float]
    output_token_logprobs_idx: List[int]
    input_top_logprobs_val: List[List]
    input_top_logprobs_idx: List[List]
    output_top_logprobs_val: List[List]
    output_top_logprobs_idx: List[List]
    input_token_ids_logprobs_val: List[List]
    input_token_ids_logprobs_idx: List[List]
    output_token_ids_logprobs_val: List[List]
    output_token_ids_logprobs_idx: List[List]

    # Hidden states
    output_hidden_states: List[List[float]]


@dataclass
class TokenizedGenerateReqInput:
    # The request id.
    rid: Optional[Union[List[str], str]] = None
    # The input prompt. It can be a single prompt or a batch of prompts.
    text: Optional[Union[List[str], str]] = None
    # The token ids for text; one can specify either text or input_ids
    input_ids: Optional[Union[List[List[int]], List[int]]] = None
    # The sampling_params. See descriptions below.
    sampling_params: Optional[Union[List[Dict], Dict]] = None
    # Whether to return logprobs.
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    # By default, this value is "-1", which means it will only return logprobs for output tokens.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None
    # Whether to stream output
    stream: bool = False


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
    input_ids: List[int] = None
    normalize: bool = True


@dataclass
class GenerateReqInput:
    """Request input for text generation."""

    batch_size: int = 1
    rid: Optional[Union[List[str], str]] = None
    text: Optional[Union[List[str], str]] = None
    input_ids: List[int] = None
    sampling_params: Optional[Any] = (
        None  # Using Any for now to avoid SamplingParams serialization issues
    )
    stream: bool = False
    is_single: bool = True
    return_logprob: Optional[Union[List[bool], bool]] = None
    # If return logprobs, the start location in the prompt for returning logprobs.
    logprob_start_len: Optional[Union[List[int], int]] = None
    # If return logprobs, the number of top logprobs to return at each position.
    top_logprobs_num: Optional[Union[List[int], int]] = None
    # If return logprobs, the token ids to return logprob for.
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None

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
        self._normalize_rid(num=self.batch_size)

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
            stream=self.stream,
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

    rids: List[str]
    embeddings: List[List[float]]
    prompt_tokens: List[int]


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
    input_ids: List[int]
    normalize: bool = True


# Configuration and logging classes
@dataclass
class ConfigureLoggingReq(RpcReqInput):
    """Request to configure logging."""

    log_level: str
    log_file: Optional[str] = None


@dataclass
class FlushCacheReqInput(RpcReqInput):
    """Request to flush cache."""

    cache_type: str = "all"


# Internal state classes
@dataclass
class GetInternalStateReq:
    pass


@dataclass
class GetInternalStateReqOutput:
    internal_state: Dict[Any, Any]


@dataclass
class SetInternalStateReq(RpcReqInput):
    """Request to set internal state."""

    state_data: Dict[str, Any]


# Profile classes


@dataclass
class ProfileReqInput:
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None


class ProfileReqType(Enum):
    START_PROFILE = 1
    STOP_PROFILE = 2


@dataclass
class ProfileReq:
    type: ProfileReqType
    output_dir: Optional[str] = None
    start_step: Optional[int] = None
    num_steps: Optional[int] = None
    profile_id: Optional[str] = None


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
