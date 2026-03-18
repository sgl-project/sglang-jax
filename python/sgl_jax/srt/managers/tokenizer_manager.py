"""TokenizerManager is a process that tokenizes the text."""

import asyncio
import copy
import dataclasses
import json
import logging
import math
import os
import pickle
import signal
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from http import HTTPStatus
from typing import Any

import fastapi
import uvloop
import zmq
import zmq.asyncio
from fastapi import BackgroundTasks
from scipy.special import softmax

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.hf_transformers_utils import get_tokenizer
from sgl_jax.srt.lora.lora_registry import LoRARegistry
from sgl_jax.srt.managers.io_struct import (
    AbortReq,
    BatchEmbeddingOut,
    BatchStrOut,
    BatchTokenIDOut,
    CloseSessionReqInput,
    ConfigureLoggingReq,
    ContinueGenerationReqInput,
    EmbeddingReqInput,
    FlushCacheReqInput,
    FlushCacheReqOutput,
    GenerateReqInput,
    GetInternalStateReq,
    GetInternalStateReqOutput,
    HealthCheckOutput,
    OpenSessionReqInput,
    OpenSessionReqOutput,
    PauseGenerationReqInput,
    ProfileReq,
    ProfileReqOutput,
    ProfileReqType,
    ReleaseMemoryOccupationReqInput,
    ReleaseMemoryOccupationReqOutput,
    ReleaseScoringCacheReqInput,
    ReleaseScoringCacheReqOutput,
    ResumeMemoryOccupationReqInput,
    ResumeMemoryOccupationReqOutput,
    ScoreFromCacheReqInput,
    ScoreFromCacheReqOutput,
    SetInternalStateReq,
    SetInternalStateReqOutput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sgl_jax.srt.multimodal.tokenizer_utils import resolve_tokenizer_subdir
from sgl_jax.srt.sampling.sampling_params import SamplingParams
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    dataclass_to_string_truncated,
    get_bool_env_var,
    get_zmq_socket,
    kill_process_tree,
)
from sgl_jax.srt.validation import (
    ValidationError,
    validate_score_request,
)
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ReqState:
    """Store the state a request."""

    out_list: list[dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateReqInput | EmbeddingReqInput

    # For metrics
    created_time: float
    finished_time: float = 0.0
    first_token_time: float = 0.0
    last_time: float = 0.0
    last_completion_tokens: int = 1

    # For streaming output
    last_output_offset: int = 0

    text: str = ""
    output_ids: list[int] = dataclasses.field(default_factory=list)
    input_token_logprobs_val: list[float] = dataclasses.field(default_factory=list)
    input_token_logprobs_idx: list[int] = dataclasses.field(default_factory=list)
    output_token_logprobs_val: list[float] = dataclasses.field(default_factory=list)
    output_token_logprobs_idx: list[int] = dataclasses.field(default_factory=list)
    input_top_logprobs_val: list[list[float]] = dataclasses.field(default_factory=list)
    input_top_logprobs_idx: list[list[int]] = dataclasses.field(default_factory=list)
    output_top_logprobs_val: list[list[float]] = dataclasses.field(default_factory=list)
    output_top_logprobs_idx: list[list[int]] = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_val: list = dataclasses.field(default_factory=list)
    input_token_ids_logprobs_idx: list = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_val: list = dataclasses.field(default_factory=list)
    output_token_ids_logprobs_idx: list = dataclasses.field(default_factory=list)


class TokenizerManager:
    """TokenizerManager is a process that tokenizes the text."""

    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
    ):
        # Parse args
        self.server_args = server_args
        self.log_requests = server_args.log_requests
        self.log_requests_level = server_args.log_requests_level
        self.preferred_sampling_params = (
            json.loads(server_args.preferred_sampling_params)
            if server_args.preferred_sampling_params
            else None
        )
        self.crash_dump_folder = server_args.crash_dump_folder
        self.crash_dump_performed = False  # Flag to ensure dump is only called once
        self.event_loop = None  # Store the event loop to use

        # Init inter-process communication
        context = zmq.asyncio.Context(2)
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )

        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )

        # Read model args
        self.model_path = server_args.model_path
        self.served_model_name = server_args.served_model_name
        if not server_args.multimodal:
            self.model_config = ModelConfig.from_server_args(server_args)
            self.is_generation = self.model_config.is_generation
            self.context_len = self.model_config.context_len
            self.image_token_id = self.model_config.image_token_id
        else:
            self.model_config = None
        self.is_pause = False
        self.is_pause_cond = asyncio.Condition()
        self._updating = False
        self._cond = asyncio.Condition()

        self.mm_processor = None

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            tokenizer_subdir = ""
            if server_args.multimodal:
                tokenizer_subdir = resolve_tokenizer_subdir(
                    server_args.model_path, server_args.tokenizer_path
                )
            self.tokenizer = get_tokenizer(
                server_args.tokenizer_path,
                tokenizer_mode=server_args.tokenizer_mode,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
                sub_dir=tokenizer_subdir,
            )

        # Store states
        self.no_create_loop = False
        self.rid_to_state: dict[str, ReqState] = {}
        self.health_check_failed = False
        self.gracefully_exit = False
        self.last_receive_tstamp = 0
        self.dump_requests_folder = ""  # By default do not dump
        self.dump_requests_threshold = 1000
        self.dump_request_list: list[tuple] = []
        self.crash_dump_request_list: deque[tuple] = deque()
        self.log_request_metadata = self.get_log_request_metadata()
        self.session_futures = {}  # session_id -> asyncio event
        self.max_req_input_len = None
        self.asyncio_tasks = set()

        # For load balancing
        self.current_load = 0
        self.current_load_lock = asyncio.Lock()

        # Communicators
        self.release_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.resume_memory_occupation_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.flush_cache_communicator = _Communicator(self.send_to_scheduler, server_args.dp_size)
        self.release_scoring_cache_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.score_from_cache_v2_communicator = _CorrelatedCommunicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.profile_communicator = _Communicator(self.send_to_scheduler, server_args.dp_size)
        self.get_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.set_internal_state_communicator = _Communicator(
            self.send_to_scheduler, server_args.dp_size
        )
        self.score_fastpath_attempted = 0
        self.score_fastpath_succeeded = 0
        self.score_fastpath_fallback = 0
        self.score_fastpath_fallback_reasons: dict[str, int] = {}

        # LoRA
        self.lora_registry = LoRARegistry(self.server_args.lora_paths)

        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (
                        BatchStrOut,
                        BatchEmbeddingOut,
                        BatchTokenIDOut,
                    ),
                    self._handle_batch_output,
                ),
                (AbortReq, self._handle_abort_req),
                (OpenSessionReqOutput, self._handle_open_session_req_output),
                (
                    ReleaseMemoryOccupationReqOutput,
                    self.release_memory_occupation_communicator.handle_recv,
                ),
                (
                    ResumeMemoryOccupationReqOutput,
                    self.resume_memory_occupation_communicator.handle_recv,
                ),
                (
                    FlushCacheReqOutput,
                    self.flush_cache_communicator.handle_recv,
                ),
                (
                    ReleaseScoringCacheReqOutput,
                    self.release_scoring_cache_communicator.handle_recv,
                ),
                (
                    ScoreFromCacheReqOutput,
                    self.score_from_cache_v2_communicator.handle_recv,
                ),
                (
                    ProfileReqOutput,
                    self.profile_communicator.handle_recv,
                ),
                (
                    GetInternalStateReqOutput,
                    self.get_internal_state_communicator.handle_recv,
                ),
                (
                    SetInternalStateReqOutput,
                    self.set_internal_state_communicator.handle_recv,
                ),
                (HealthCheckOutput, lambda x: None),
            ]
        )
        self.wait_timeout = int(os.environ.get("SGLANG_WAIT_TIMEOUT", "4"))
        self.scheduler_pids: list[int] = []
        self.scheduler_unavailable_error: str | None = None

    async def generate_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        request: fastapi.Request | None = None,
    ):

        created_time = time.time()
        async with self.is_pause_cond:
            await self.is_pause_cond.wait_for(lambda: not self.is_pause)

        self.auto_create_handle_loop()
        obj.normalize_batch_and_arguments()

        # Acquire LoRA ID if lora_path is provided
        if isinstance(obj, GenerateReqInput) and self.server_args.enable_lora and obj.lora_path:
            obj.lora_id = await self.lora_registry.acquire(obj.lora_path)

        if isinstance(obj, EmbeddingReqInput) and self.is_generation:
            raise ValueError(
                "This model does not appear to be an embedding model by default. "
                "Please add `--is-embedding` when launching the server or try another model."
            )

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )

        if obj.is_single:
            tokenized_obj = await self._tokenize_one_request(obj)
            state = self._send_one_request(obj, tokenized_obj, created_time)
            async for response in self._wait_one_response(obj, state, request):
                yield response
        else:
            async for response in self._handle_batch_request(obj, request, created_time):
                yield response

    async def _tokenize_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
    ):
        """Tokenize one request."""

        # Tokenize
        input_text = obj.text
        input_ids = obj.input_ids
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        self._validate_one_request(obj, input_ids)
        return self._create_tokenized_object(obj, input_text, input_ids)

    def _validate_one_request(
        self, obj: GenerateReqInput | EmbeddingReqInput, input_ids: list[int]
    ) -> None:
        """Validates that the input token count and the requested token count doesn't exceed the model's context length."""

        input_token_num = len(input_ids) if input_ids is not None else 0
        # Check if input alone exceeds context length
        if input_token_num >= self.context_len:
            raise ValueError(
                f"The input ({input_token_num} tokens) is longer than the "
                f"model's context length ({self.context_len} tokens)."
            )

        # Check total tokens (input + max_new_tokens)
        max_new_tokens = obj.sampling_params.get("max_new_tokens")
        if max_new_tokens is not None and (max_new_tokens + input_token_num) >= self.context_len:
            total_tokens = max_new_tokens + input_token_num
            error_msg = (
                f"Requested token count exceeds the model's maximum context length "
                f"of {self.context_len} tokens. You requested a total of {total_tokens} "
                f"tokens: {input_token_num} tokens from the input messages and "
                f"{max_new_tokens} tokens for the completion. Please reduce the number "
                f"of tokens in the input messages or the completion to fit within the limit."
            )
            raise ValueError(error_msg)

    def _validate_input_ids_in_vocab(self, input_ids: list[int], vocab_size: int) -> None:
        if any(id >= vocab_size for id in input_ids):
            raise ValueError(
                f"The input_ids {input_ids} contains values greater than the vocab size ({vocab_size})."
            )

    def _create_tokenized_object(
        self,
        obj: GenerateReqInput,
        input_text: str,
        input_ids: list[int],
    ) -> TokenizedGenerateReqInput:
        """Create a tokenized request object from common parameters."""
        # Parse sampling parameters
        # Note: if there are preferred sampling params, we use them if they are not
        # explicitly passed in sampling_params
        if self.preferred_sampling_params:
            sampling_kwargs = {**self.preferred_sampling_params, **obj.sampling_params}
        else:
            sampling_kwargs = obj.sampling_params
        sampling_params = SamplingParams(**sampling_kwargs)
        sampling_params.normalize(self.tokenizer)
        sampling_params.verify(self.model_config.vocab_size)

        # Build return object

        tokenized_obj = TokenizedGenerateReqInput(
            rid=obj.rid,
            text=input_text,
            input_ids=input_ids,
            sampling_params=sampling_params,
            return_logprob=obj.return_logprob,
            return_output_logprob_only=obj.return_output_logprob_only,
            logprob_start_len=obj.logprob_start_len,
            top_logprobs_num=obj.top_logprobs_num,
            token_ids_logprob=obj.token_ids_logprob,
            stream=obj.stream,
            lora_id=obj.lora_id,
            extra_key=obj.extra_key,
            return_routed_experts=obj.return_routed_experts,
            cache_for_scoring=bool(obj.cache_for_scoring),
            extend_from_cache=obj.extend_from_cache,
        )
        # note: When only `return_logprob` is specified, we assume that only the output probability is required.
        if (
            tokenized_obj.return_logprob
            and (obj.logprob_start_len is None or obj.logprob_start_len == -1)
            and (obj.top_logprobs_num == 0 or obj.top_logprobs_num is None)
            and obj.token_ids_logprob is None
        ):
            tokenized_obj.return_logprob = False
            obj.return_output_logprob_only = True
            tokenized_obj.return_output_logprob_only = True

        return tokenized_obj

    async def _batch_tokenize_and_process(
        self, batch_size: int, obj: GenerateReqInput
    ) -> list[TokenizedGenerateReqInput | TokenizedEmbeddingReqInput]:
        """Handle batch tokenization for text inputs only."""
        logger.debug("Starting batch tokenization for %s text requests", batch_size)

        # Collect requests and texts
        requests = [obj[i] for i in range(batch_size)]
        texts = [req.text for req in requests]

        # Batch tokenize all texts
        encoded = self.tokenizer(texts)
        input_ids_list = encoded["input_ids"]

        # Process all requests
        tokenized_objs = []
        for i, req in enumerate(requests):
            # self._validate_token_len(obj[i], input_ids_list[i])
            tokenized_objs.append(self._create_tokenized_object(req, req.text, input_ids_list[i]))
        logger.debug("Completed batch processing for %s requests", batch_size)
        return tokenized_objs

    def _validate_batch_tokenization_constraints(
        self, batch_size: int, obj: GenerateReqInput | EmbeddingReqInput
    ) -> None:
        """Validate constraints for batch tokenization processing."""
        for i in range(batch_size):
            if self.is_generation and obj[i].contains_mm_input():
                raise ValueError(
                    "For multimodal input processing do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_ids is not None:
                raise ValueError(
                    "Batch tokenization is not needed for pre-tokenized input_ids. Do not set `enable_tokenizer_batch_encode`."
                )
            if obj[i].input_embeds is not None:
                raise ValueError(
                    "Batch tokenization is not needed for input_embeds. Do not set `enable_tokenizer_batch_encode`."
                )

    def _send_one_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        tokenized_obj: TokenizedGenerateReqInput | TokenizedEmbeddingReqInput,
        created_time: float | None = None,
    ):
        self._raise_if_scheduler_unavailable()
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = ReqState([], False, asyncio.Event(), obj, created_time=created_time)
        # Handle rid being a list (single element) or string
        rid_key = obj.rid[0] if isinstance(obj.rid, list) else obj.rid
        self.rid_to_state[rid_key] = state
        return state

    def _send_batch_requests(
        self,
        objs: list[GenerateReqInput | EmbeddingReqInput],
        tokenized_objs: list[TokenizedGenerateReqInput | TokenizedEmbeddingReqInput],
        created_time: float | None = None,
    ) -> list[ReqState]:
        if len(objs) != len(tokenized_objs):
            raise ValueError("objs and tokenized_objs must have the same length")
        if not objs:
            return []

        self._raise_if_scheduler_unavailable()
        payload = tokenized_objs[0] if len(tokenized_objs) == 1 else tokenized_objs
        self.send_to_scheduler.send_pyobj(payload)

        states: list[ReqState] = []
        for obj in objs:
            state = ReqState([], False, asyncio.Event(), obj, created_time=created_time)
            rid_key = obj.rid[0] if isinstance(obj.rid, list) else obj.rid
            self.rid_to_state[rid_key] = state
            states.append(state)
        return states

    @staticmethod
    def _is_process_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _build_scheduler_unavailable_message(self) -> str | None:
        if not self.scheduler_pids:
            return None
        dead_pids = [pid for pid in self.scheduler_pids if not self._is_process_alive(pid)]
        if not dead_pids:
            return None
        return (
            "Scheduler subprocess is unavailable "
            f"(dead pid(s): {', '.join(str(pid) for pid in dead_pids)}). "
            "Please restart the server."
        )

    def _fail_pending_requests(self, message: str) -> None:
        for rid, state in list(self.rid_to_state.items()):
            if state.finished:
                continue
            state.finished = True
            state.out_list.append(
                {
                    "text": "",
                    "meta_info": {
                        "id": rid,
                        "finish_reason": {
                            "type": "abort",
                            "message": message,
                            "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                        },
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                    },
                }
            )
            state.event.set()
            self.rid_to_state.pop(rid, None)

    def _mark_scheduler_unavailable(self, message: str) -> None:
        if self.scheduler_unavailable_error is None:
            logger.error(message)
        self.scheduler_unavailable_error = message
        self.health_check_failed = True
        self._fail_pending_requests(message)

    def _check_scheduler_health(self) -> bool:
        if self.scheduler_unavailable_error is not None:
            return False
        message = self._build_scheduler_unavailable_message()
        if message is None:
            return True
        self._mark_scheduler_unavailable(message)
        return False

    def _raise_if_scheduler_unavailable(self) -> None:
        if self._check_scheduler_health():
            return
        raise ValueError(
            self.scheduler_unavailable_error
            or "Scheduler subprocess is unavailable. Please restart the server."
        )

    async def _wait_one_response(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        state: ReqState,
        request: fastapi.Request | None = None,
    ):
        """Wait for the response of one request."""
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    # Abort the request for disconnected requests (non-streaming, waiting queue)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    try:
                        raise ValueError(
                            f"Request is disconnected from the client side (type 1). Abort request rid={obj.rid}"
                        )
                    except ValueError as e:
                        raise ValueError(
                            f"Request is disconnected from the client side (type 1). Abort request rid={obj.rid}"
                        ) from e
                if not self._check_scheduler_health():
                    raise ValueError(
                        self.scheduler_unavailable_error
                        or "Scheduler subprocess is unavailable. Please restart the server."
                    ) from None
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                # Check if this was an abort/error created by scheduler
                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if finish_reason.get("type") == "abort":
                        raise ValueError(
                            finish_reason.get("message") or "Request aborted by scheduler."
                        )

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    # Abort the request for disconnected requests (non-streaming, running)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
                    )

    async def _handle_batch_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        request: fastapi.Request | None = None,
        created_time: float | None = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self.server_args.enable_tokenizer_batch_encode:
                # Validate batch tokenization constraints
                self._validate_batch_tokenization_constraints(batch_size, obj)

                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)
                batched_objs = [obj[i] for i in range(batch_size)]
                if self.server_args.enable_tokenizer_batch_send:
                    states = self._send_batch_requests(
                        batched_objs,
                        tokenized_objs,
                        created_time,
                    )
                    for tmp_obj, state in zip(batched_objs, states, strict=True):
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
                else:
                    for tmp_obj, tokenized_obj in zip(batched_objs, tokenized_objs, strict=True):
                        state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                batched_objs = [obj[i] for i in range(batch_size)]
                tokenized_objs = []
                for tmp_obj in batched_objs:
                    tokenized_objs.append(await self._tokenize_one_request(tmp_obj))

                if self.server_args.enable_tokenizer_batch_send:
                    states = self._send_batch_requests(
                        batched_objs,
                        tokenized_objs,
                        created_time,
                    )
                    for tmp_obj, state in zip(batched_objs, states, strict=True):
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
                else:
                    for tmp_obj, tokenized_obj in zip(batched_objs, tokenized_objs, strict=True):
                        state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(
                *(self._tokenize_one_request(obj) for obj in objs)
            )

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, state, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        if not is_stream:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    async def flush_cache(self) -> FlushCacheReqOutput:
        self.auto_create_handle_loop()
        return (await self.flush_cache_communicator(FlushCacheReqInput()))[0]

    def abort_request(self, rid: str = "", abort_all: bool = False):
        if not abort_all and rid not in self.rid_to_state:
            return
        req = AbortReq(rid=rid, abort_all=abort_all)
        self.send_to_scheduler.send_pyobj(req)

    async def start_profile(
        self,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        host_tracer_level: int | None = None,
        python_tracer_level: int | None = None,
    ):
        self.auto_create_handle_loop()
        req = ProfileReq(
            type=ProfileReqType.START_PROFILE,
            output_dir=output_dir,
            start_step=start_step,
            num_steps=num_steps,
            host_tracer_level=host_tracer_level,
            python_tracer_level=python_tracer_level,
            profile_id=str(time.time()),
        )
        return await self._execute_profile(req)

    async def stop_profile(self):
        self.auto_create_handle_loop()
        req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
        return await self._execute_profile(req)

    async def _execute_profile(self, req: ProfileReq):
        result = (await self.profile_communicator(req))[0]
        if not result.success:
            raise RuntimeError(result.message)
        return result

    async def pause_generation(self, obj: PauseGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = True
            if obj.mode != "abort":
                await self.send_to_scheduler.send_pyobj(obj)
            else:
                # use len(self.rid_to_state) == 0 to ensure all requests are aborted
                while True:
                    self.abort_request(abort_all=True)
                    if len(self.rid_to_state) == 0:
                        break
                    await asyncio.sleep(0.1)

    async def continue_generation(self, obj: ContinueGenerationReqInput):
        async with self.is_pause_cond:
            self.is_pause = False
            await self.send_to_scheduler.send_pyobj(obj)
            self.is_pause_cond.notify_all()

    async def release_memory_occupation(
        self,
        obj: ReleaseMemoryOccupationReqInput,
        request: fastapi.Request | None = None,
    ):
        self.auto_create_handle_loop()
        await self.release_memory_occupation_communicator(obj)

    async def resume_memory_occupation(
        self,
        obj: ResumeMemoryOccupationReqInput,
        request: fastapi.Request | None = None,
    ):
        self.auto_create_handle_loop()
        await self.resume_memory_occupation_communicator(obj)

    async def open_session(self, obj: OpenSessionReqInput, request: fastapi.Request | None = None):
        self.auto_create_handle_loop()

        if obj.session_id is None:
            obj.session_id = uuid.uuid4().hex
        elif obj.session_id in self.session_futures:
            return None

        self.send_to_scheduler.send_pyobj(obj)

        self.session_futures[obj.session_id] = asyncio.Future()
        session_id = await self.session_futures[obj.session_id]
        del self.session_futures[obj.session_id]
        return session_id

    async def close_session(
        self, obj: CloseSessionReqInput, request: fastapi.Request | None = None
    ):
        await self.send_to_scheduler.send_pyobj(obj)

    async def get_internal_state(self) -> list[dict[Any, Any]]:
        self.auto_create_handle_loop()
        req = GetInternalStateReq()
        responses: list[GetInternalStateReqOutput] = await self.get_internal_state_communicator(req)
        # Many DP ranks
        return [res.internal_state for res in responses]

    async def get_load(self) -> dict:
        if not self.current_load_lock.locked():
            async with self.current_load_lock:
                internal_state = await self.get_internal_state()
                self.current_load = internal_state[0]["load"]
        return {"load": self.current_load}

    async def set_internal_state(self, obj: SetInternalStateReq) -> SetInternalStateReqOutput:
        self.auto_create_handle_loop()
        responses: list[SetInternalStateReqOutput] = await self.set_internal_state_communicator(obj)
        return (
            responses[0]
            if responses
            else SetInternalStateReqOutput(
                request_id=obj.request_id,
                success=False,
                error_msg="No response from scheduler",
            )
        )

    def get_log_request_metadata(self):
        max_length = None
        skip_names = None
        out_skip_names = None
        if self.log_requests:
            if self.log_requests_level == 0:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                        "lora_path",
                        "sampling_params",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                        "embedding",
                    ]
                )
            elif self.log_requests_level == 1:
                max_length = 1 << 30
                skip_names = set(
                    [
                        "text",
                        "input_ids",
                        "input_embeds",
                        "image_data",
                        "audio_data",
                        "lora_path",
                    ]
                )
                out_skip_names = set(
                    [
                        "text",
                        "output_ids",
                        "embedding",
                    ]
                )
            elif self.log_requests_level == 2:
                max_length = 2048
            elif self.log_requests_level == 3:
                max_length = 1 << 30
            else:
                raise ValueError(f"Invalid --log-requests-level: {self.log_requests_level=}")
        return max_length, skip_names, out_skip_names

    def configure_logging(self, obj: ConfigureLoggingReq):
        if obj.log_requests is not None:
            self.log_requests = obj.log_requests
        if obj.log_requests_level is not None:
            self.log_requests_level = obj.log_requests_level
        if obj.dump_requests_folder is not None:
            self.dump_requests_folder = obj.dump_requests_folder
        if obj.dump_requests_threshold is not None:
            self.dump_requests_threshold = obj.dump_requests_threshold
        if obj.crash_dump_folder is not None:
            self.crash_dump_folder = obj.crash_dump_folder
        self.log_request_metadata = self.get_log_request_metadata()

    def create_abort_task(self, obj: GenerateReqInput):
        # Abort the request if the client is disconnected.
        async def abort_request():
            await asyncio.sleep(2)
            if obj.is_single:
                self.abort_request(obj.rid)
            else:
                for rid in obj.rid:
                    self.abort_request(rid)

        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return background_tasks

    def auto_create_handle_loop(self):
        if self.no_create_loop:
            return

        self.no_create_loop = True
        # Use the provided event loop if available, otherwise get the current one
        loop = self.event_loop if self.event_loop is not None else asyncio.get_event_loop()

        try:
            current_running_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_running_loop = None

        if current_running_loop == loop:
            task = loop.create_task(print_exception_wrapper(self.handle_loop))
            self.asyncio_tasks.add(task)
        else:
            asyncio.run_coroutine_threadsafe(print_exception_wrapper(self.handle_loop), loop)

        # We cannot add signal handler when the tokenizer manager is not in
        # the main thread due to the CPython limitation.
        if threading.current_thread() is threading.main_thread():
            signal_handler = SignalHandler(self)
            loop.add_signal_handler(signal.SIGTERM, signal_handler.sigterm_handler)
            # Update the signal handler for the process. It overrides the sigquit handler in the launch phase.
            loop.add_signal_handler(signal.SIGQUIT, signal_handler.running_phase_sigquit_handler)
        else:
            logger.warning(
                "Signal handler is not added because the tokenizer manager is "
                "not in the main thread. This disables graceful shutdown of the "
                "tokenizer manager when SIGTERM is received."
            )
        self.asyncio_tasks.add(loop.create_task(print_exception_wrapper(self.sigterm_watchdog)))

    def dump_requests_before_crash(self):
        if self.crash_dump_performed:
            logger.info(
                "SIGTERM/SIGQUIT/Exception triggered, but crash dump already performed, skipping."
            )
            return
        logger.error(
            "Dumping requests before crash. crash_dump_folder=%s",
            self.crash_dump_folder,
        )
        self.crash_dump_performed = True
        if not self.crash_dump_folder:
            return

        data_to_dump = []
        if self.crash_dump_request_list:
            data_to_dump.extend(self.crash_dump_request_list)

        # Add unfinished requests from rid_to_state
        unfinished_requests = []
        for rid, state in self.rid_to_state.items():
            if not state.finished:
                unfinished_requests.append((state.obj, {}, state.created_time, time.time()))
        if unfinished_requests:
            data_to_dump.extend(unfinished_requests)

        if not data_to_dump:
            return

        filename = os.path.join(
            self.crash_dump_folder,
            os.getenv("HOSTNAME", None),
            f"crash_dump_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl",
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # Include server_args in the dump
        data_to_dump_with_server_args = {
            "server_args": self.server_args,
            "requests": data_to_dump,
        }
        with open(filename, "wb") as f:
            pickle.dump(data_to_dump_with_server_args, f)
        logger.error(
            "Dumped %d finished and %d unfinished requests before crash to %s",
            len(self.crash_dump_request_list),
            len(unfinished_requests),
            filename,
        )

    async def sigterm_watchdog(self):
        while not self.gracefully_exit:
            await asyncio.sleep(5)

        # Drain requests
        while True:
            remain_num_req = len(self.rid_to_state)

            if self.health_check_failed:
                # if health check failed, we should exit immediately
                logger.error(
                    "Signal SIGTERM received while health check failed. Exiting... remaining number of requests: %d",
                    remain_num_req,
                )
                self.dump_requests_before_crash()
                break

            elif get_bool_env_var("SGL_FORCE_SHUTDOWN"):
                # if force shutdown flag set, exit immediately
                logger.error(
                    "Signal SIGTERM received while force shutdown flag set. Force exiting... remaining number of requests: %d",
                    remain_num_req,
                )
                break

            logger.info(
                "Gracefully exiting... remaining number of requests %d",
                remain_num_req,
            )
            if remain_num_req > 0:
                await asyncio.sleep(5)
            else:
                self.dump_requests_before_crash()
                break

        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(0)

    async def handle_loop(self):
        """The event loop that handles requests"""
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()
            self._result_dispatcher(recv_obj)
            self.last_receive_tstamp = time.perf_counter()

    def _handle_batch_output(
        self,
        recv_obj: BatchStrOut | BatchEmbeddingOut | BatchTokenIDOut,
    ):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    "Received output for rid=%s but the state was deleted in TokenizerManager.",
                    rid,
                )
                continue

            # Build meta_info and return value
            meta_info = {
                "id": rid,
                "finish_reason": recv_obj.finished_reasons[i],
                "prompt_tokens": recv_obj.prompt_tokens[i],
            }

            if getattr(state.obj, "return_logprob", False) or getattr(
                state.obj, "return_output_logprob_only", False
            ):
                self.convert_logprob_style(
                    meta_info,
                    state,
                    state.obj.top_logprobs_num,
                    state.obj.token_ids_logprob,
                    state.obj.return_text_in_logprobs and not self.server_args.skip_tokenizer_init,
                    recv_obj,
                    i,
                )

            if not isinstance(recv_obj, BatchEmbeddingOut):
                meta_info.update(
                    {
                        "completion_tokens": recv_obj.completion_tokens[i],
                        "cached_tokens": recv_obj.cached_tokens[i],
                    }
                )

            if getattr(recv_obj, "output_hidden_states", None):
                meta_info["hidden_states"] = recv_obj.output_hidden_states[i]

            if getattr(recv_obj, "output_routed_experts", None):
                meta_info["routed_experts"] = recv_obj.output_routed_experts[i]

            if getattr(recv_obj, "cache_miss_count", None) is not None:
                if (
                    get_bool_env_var("SGLANG_JAX_ENABLE_CACHE_MISS_CHECK")
                    and recv_obj.cache_miss_count > 0
                ):
                    raise RuntimeError(
                        f"Cache miss occurred {recv_obj.cache_miss_count} times, please check if the precompile logic covers the current scenario"
                    )
                meta_info["cache_miss_count"] = recv_obj.cache_miss_count
            if getattr(recv_obj, "scheduler_queue_wait_s", None) is not None:
                meta_info["scheduler_queue_wait_s"] = recv_obj.scheduler_queue_wait_s[i]
            if getattr(recv_obj, "scheduler_device_compute_s", None) is not None:
                meta_info["scheduler_device_compute_s"] = recv_obj.scheduler_device_compute_s[i]
            if getattr(recv_obj, "scheduler_host_overhead_s", None) is not None:
                meta_info["scheduler_host_overhead_s"] = recv_obj.scheduler_host_overhead_s[i]
            if getattr(recv_obj, "scheduler_dispatch_count", None) is not None:
                meta_info["scheduler_dispatch_count"] = recv_obj.scheduler_dispatch_count[i]

            if isinstance(recv_obj, BatchStrOut):
                state.text += recv_obj.output_strs[i]
                state.output_ids += recv_obj.output_ids[i]
                out_dict = {
                    "text": state.text,
                    "output_ids": state.output_ids,
                    "meta_info": meta_info,
                }
            elif isinstance(recv_obj, BatchTokenIDOut):
                if self.server_args.stream_output and state.obj.stream:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids[state.last_output_offset :]
                    state.last_output_offset = len(state.output_ids)
                else:
                    state.output_ids.extend(recv_obj.output_ids[i])
                    output_token_ids = state.output_ids.copy()

                out_dict = {
                    "output_ids": output_token_ids,
                    "meta_info": meta_info,
                }
            else:
                assert isinstance(recv_obj, BatchEmbeddingOut)
                out_dict = {
                    "embedding": recv_obj.embeddings[i],
                    "meta_info": meta_info,
                }

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                state.finished_time = time.time()
                meta_info["e2e_latency"] = state.finished_time - state.created_time
                # Release LoRA ID if it was acquired
                # Note: Only GenerateReqInput supports LoRA, not EmbeddingReqInput
                if (
                    isinstance(state.obj, GenerateReqInput)
                    and self.server_args.enable_lora
                    and state.obj.lora_id
                ):
                    asyncio.create_task(self.lora_registry.release(state.obj.lora_id))
                del self.rid_to_state[rid]

            state.out_list.append(out_dict)
            state.event.set()

            # Log metrics and dump
            if self.dump_requests_folder and state.finished and state.obj.log_metrics:
                self.dump_requests(state, out_dict)
            if self.crash_dump_folder and state.finished and state.obj.log_metrics:
                self.record_request_for_crash_dump(state, out_dict)

    def convert_logprob_style(
        self,
        meta_info: dict,
        state: ReqState,
        top_logprobs_num: int,
        token_ids_logprob: list[int],
        return_text_in_logprobs: bool,
        recv_obj: BatchStrOut,
        recv_obj_index: int,
    ):
        if state.obj.return_output_logprob_only:
            state.output_token_logprobs_val.extend(
                recv_obj.output_token_logprobs_val[recv_obj_index]
            )
            state.output_token_logprobs_idx.extend(
                recv_obj.output_token_logprobs_idx[recv_obj_index]
            )
            meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
                state.output_token_logprobs_val,
                state.output_token_logprobs_idx,
                return_text_in_logprobs,
            )
            if (
                token_ids_logprob is not None
                and recv_obj.output_token_ids_logprobs_val is not None
                and len(recv_obj.output_token_ids_logprobs_val) > 0
            ):
                state.output_token_ids_logprobs_val.extend(
                    recv_obj.output_token_ids_logprobs_val[recv_obj_index]
                )
                state.output_token_ids_logprobs_idx.extend(
                    recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
                )
                meta_info["output_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                    state.output_token_ids_logprobs_val,
                    state.output_token_ids_logprobs_idx,
                    return_text_in_logprobs,
                )
            return
        if recv_obj.input_token_logprobs_val is None:
            return
        if len(recv_obj.input_token_logprobs_val) > 0:
            state.input_token_logprobs_val.extend(recv_obj.input_token_logprobs_val[recv_obj_index])
            state.input_token_logprobs_idx.extend(recv_obj.input_token_logprobs_idx[recv_obj_index])
        state.output_token_logprobs_val.extend(recv_obj.output_token_logprobs_val[recv_obj_index])
        state.output_token_logprobs_idx.extend(recv_obj.output_token_logprobs_idx[recv_obj_index])
        meta_info["input_token_logprobs"] = self.detokenize_logprob_tokens(
            state.input_token_logprobs_val,
            state.input_token_logprobs_idx,
            return_text_in_logprobs,
        )
        meta_info["output_token_logprobs"] = self.detokenize_logprob_tokens(
            state.output_token_logprobs_val,
            state.output_token_logprobs_idx,
            return_text_in_logprobs,
        )

        if top_logprobs_num > 0:
            if len(recv_obj.input_top_logprobs_val) > 0:
                state.input_top_logprobs_val.extend(recv_obj.input_top_logprobs_val[recv_obj_index])
                state.input_top_logprobs_idx.extend(recv_obj.input_top_logprobs_idx[recv_obj_index])
            state.output_top_logprobs_val.extend(recv_obj.output_top_logprobs_val[recv_obj_index])
            state.output_top_logprobs_idx.extend(recv_obj.output_top_logprobs_idx[recv_obj_index])
            meta_info["input_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_top_logprobs_val,
                state.input_top_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_top_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.output_top_logprobs_val,
                state.output_top_logprobs_idx,
                return_text_in_logprobs,
            )

        if token_ids_logprob is not None:
            if len(recv_obj.input_token_ids_logprobs_val) > 0:
                state.input_token_ids_logprobs_val.extend(
                    recv_obj.input_token_ids_logprobs_val[recv_obj_index]
                )
                state.input_token_ids_logprobs_idx.extend(
                    recv_obj.input_token_ids_logprobs_idx[recv_obj_index]
                )
            state.output_token_ids_logprobs_val.extend(
                recv_obj.output_token_ids_logprobs_val[recv_obj_index]
            )
            state.output_token_ids_logprobs_idx.extend(
                recv_obj.output_token_ids_logprobs_idx[recv_obj_index]
            )
            meta_info["input_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.input_token_ids_logprobs_val,
                state.input_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )
            meta_info["output_token_ids_logprobs"] = self.detokenize_top_logprobs_tokens(
                state.output_token_ids_logprobs_val,
                state.output_token_ids_logprobs_idx,
                return_text_in_logprobs,
            )

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        if not decode_to_text:
            return [
                (logprob, token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(zip(token_logprobs_val, token_logprobs_idx, token_texts))

    def detokenize_top_logprobs_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        # We should batch all top-k tokens in all positions.
        ret = []
        for i in range(len(token_logprobs_val)):
            if token_logprobs_val[i]:
                ret.append(
                    self.detokenize_logprob_tokens(
                        token_logprobs_val[i], token_logprobs_idx[i], decode_to_text
                    )
                )
            else:
                ret.append(None)
        return ret

    def dump_requests(self, state: ReqState, out_dict: dict):
        self.dump_request_list.append((state.obj, out_dict, state.created_time, time.time()))

        if len(self.dump_request_list) >= self.dump_requests_threshold:
            filename = os.path.join(
                self.dump_requests_folder,
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".pkl",
            )
            logger.info("Dump %s requests to %s", len(self.dump_request_list), filename)

            to_dump = self.dump_request_list
            self.dump_request_list = []

            to_dump_with_server_args = {
                "server_args": self.server_args,
                "requests": to_dump,
            }

            def background_task():
                os.makedirs(self.dump_requests_folder, exist_ok=True)
                with open(filename, "wb") as f:
                    pickle.dump(to_dump_with_server_args, f)

            # Schedule the task to run in the background without awaiting it
            asyncio.create_task(asyncio.to_thread(background_task))

    def record_request_for_crash_dump(self, state: ReqState, out_dict: dict):
        current_time = time.time()
        self.crash_dump_request_list.append((state.obj, out_dict, state.created_time, current_time))
        # Remove requests older than 5 minutes based on finish time
        while (
            self.crash_dump_request_list
            and current_time - self.crash_dump_request_list[0][3] >= 300
        ):
            self.crash_dump_request_list.popleft()

    def _handle_abort_req(self, recv_obj):
        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.out_list.append(
            {
                "text": "",
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": "Abort before prefill",
                    },
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                },
            }
        )
        state.event.set()

    def _handle_open_session_req_output(self, recv_obj):
        self.session_futures[recv_obj.session_id].set_result(
            recv_obj.session_id if recv_obj.success else None
        )

    async def score_request(
        self,
        query: str | list[int] | None = None,
        items: str | list[str] | list[list[int]] | None = None,
        label_token_ids: list[int] | None = None,
        apply_softmax: bool = False,
        item_first: bool = False,
        request: Any | None = None,
    ) -> list[list[float]]:
        """
        See Engine.score() for more details.
        """
        logger.debug(
            "Score request: query_type=%s, items_len=%s, label_token_ids=%s, "
            "apply_softmax=%s, item_first=%s",
            type(query),
            len(items) if items is not None else 0,
            label_token_ids,
            apply_softmax,
            item_first,
        )
        # Comprehensive validation per RFC-006
        vocab_size = len(self.tokenizer) if self.tokenizer is not None else None
        try:
            validate_score_request(
                query=query,
                items=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                item_first=item_first,
                vocab_size=vocab_size,
            )
        except ValidationError as e:
            raise ValueError(e.message) from e

        # Tokenize inputs if necessary
        query_tokens = query
        if isinstance(query, str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text scoring.")
            query_tokens = self.tokenizer.encode(query, add_special_tokens=False)

        item_tokens_list = items
        if isinstance(items, str):
            item_tokens_list = [items]

        if item_tokens_list and isinstance(item_tokens_list[0], str):
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for text scoring.")
            item_tokens_list = [
                self.tokenizer.encode(item, add_special_tokens=False)
                for item in item_tokens_list
            ]

        if len(item_tokens_list) > 1:
            if item_first:
                logger.warning("Ignoring item_first=True for prefill+extend strategy.")

            return await self.score_prefill_extend(
                query_tokens=query_tokens,
                item_tokens_list=item_tokens_list,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
            )

        def _convert_logprobs(logprobs_data: list) -> list[float]:
            logprobs = {}
            for logprob, token_id, _ in logprobs_data:
                if token_id in label_token_ids:
                    logprobs[token_id] = logprob
            score_list = [logprobs.get(token_id, float("-inf")) for token_id in label_token_ids]
            if apply_softmax:
                return softmax(score_list).tolist()
            return [math.exp(x) if x != float("-inf") else 0.0 for x in score_list]

        # Handle string or tokenized query/items in single-item mode
        if isinstance(query, str) and (
            isinstance(items, str)
            or (isinstance(items, list) and (not items or isinstance(items[0], str)))
        ):
            # Both query and items are text
            items_list = [items] if isinstance(items, str) else items
            if item_first:
                prompts = [f"{item}{query}" for item in items_list]
            else:
                prompts = [f"{query}{item}" for item in items_list]
            batch_request = GenerateReqInput(
                text=prompts,
                return_logprob=True,
                token_ids_logprob=label_token_ids,
                stream=False,
                sampling_params={"max_new_tokens": 0},  # Prefill-only: no generation needed
                is_multi_item_scoring=False,
            )
            logger.debug(
                "Scoring text prompts: num_items=%d, first_prompt_len=%d",
                len(prompts),
                len(prompts[0]),
            )
        elif (
            isinstance(query, list)
            and isinstance(items, list)
            and items
            and isinstance(items[0], list)
        ):
            # Both query and items are token IDs
            if item_first:
                input_ids_list = [item + query for item in items]
            else:
                input_ids_list = [query + item for item in items]
            batch_request = GenerateReqInput(
                input_ids=input_ids_list,
                return_logprob=True,
                token_ids_logprob=label_token_ids,
                stream=False,
                sampling_params={"max_new_tokens": 0},  # Prefill-only: no generation needed
                is_multi_item_scoring=False,
            )
            logger.debug(
                "Scoring token IDs: num_items=%d, first_ids_len=%d",
                len(input_ids_list),
                len(input_ids_list[0]),
            )
        else:
            raise ValueError("Invalid combination of query/items types for score_request.")

        results = await self.generate_request(batch_request, request).__anext__()
        scores = []

        for result in results:
            output_logprobs = result["meta_info"].get("output_token_ids_logprobs", [])
            if not output_logprobs or len(output_logprobs) == 0:
                raise RuntimeError(
                    f"output_token_ids_logprobs is empty for request "
                    f"{result['meta_info'].get('id', '<unknown>')}. "
                    "This indicates token_ids_logprobs were not computed properly."
                )
            scores.append(_convert_logprobs(output_logprobs[0]))

        return scores

    async def _prefill_and_cache(self, query_tokens: list[int]) -> str:
        """Prefill query and return handle to cached KV."""
        cache_handle = uuid.uuid4().hex
        logger.debug(
            "Prefill+extend: starting prefill cache request rid=%s query_tokens=%d",
            cache_handle,
            len(query_tokens),
        )
        req = GenerateReqInput(
            # Use a single request (flat token list), not a batch-of-1. This keeps
            # the cache handle stable and avoids rid suffix rewrites during normalize.
            input_ids=query_tokens,
            sampling_params={"max_new_tokens": 0},  # Prefill only
            return_logprob=False,
            cache_for_scoring=True,  # New flag
            is_single=True,
            rid=cache_handle,
        )

        # Execute request
        async for _ in self.generate_request(req):
            pass

        logger.debug("Prefill+extend: prefill cache ready rid=%s", cache_handle)
        return cache_handle

    async def _batched_extend_score(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        scores, _ = await self._batched_extend_score_with_metrics(
            cache_handle=cache_handle,
            items=items,
            label_token_ids=label_token_ids,
            apply_softmax=apply_softmax,
        )
        return scores

    async def _batched_extend_score_with_metrics(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> tuple[list[list[float]], dict[str, float | int]]:
        """Score items by extending from cached prefix."""
        if not items:
            return (
                [],
                {
                    "dispatch_count": 0,
                    "queue_wait_s": 0.0,
                    "device_compute_s": 0.0,
                    "host_orchestration_s": 0.0,
                    "lifecycle_requests_sent": 0,
                    "lifecycle_results_received": 0,
                },
            )
        logger.debug(
            "Prefill+extend: scoring extend batch handle=%s batch_items=%d",
            cache_handle,
            len(items),
        )

        requests = GenerateReqInput(
            input_ids=items,
            sampling_params={"max_new_tokens": 0},
            return_logprob=True,
            return_output_logprob_only=False,
            token_ids_logprob=label_token_ids,
            extend_from_cache=cache_handle,
            stream=False,
            # We don't mark is_multi_item_scoring here because these are treated as individual requests
            # that happen to share a prefix.
        )

        results = []
        async for res in self.generate_request(requests):
            # res is a list of results for the batch
            if isinstance(res, list):
                results.extend(res)
            else:
                results.append(res)

        # Sort results by index when present so scores align with request order.
        if all("index" in result for result in results):
            results.sort(key=lambda x: x["index"])

        if len(results) != len(items):
            raise RuntimeError(
                f"Expected {len(items)} extend results for cache handle {cache_handle}, "
                f"but got {len(results)}."
            )

        scores = []
        scheduler_dispatch_counts = []
        scheduler_queue_wait = []
        scheduler_device_compute = []
        scheduler_host_overhead = []
        for result in results:
            meta_info = result.get("meta_info", {})
            finish_reason = meta_info.get("finish_reason")
            if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                raise RuntimeError(
                    "Prefill+extend extend request aborted for "
                    f"{meta_info.get('id', '<unknown>')}: {finish_reason}"
                )

            output_logprobs = meta_info.get("output_token_ids_logprobs", [])
            if not output_logprobs or not output_logprobs[0]:
                raise RuntimeError(
                    "output_token_ids_logprobs is empty for prefill+extend request "
                    f"{meta_info.get('id', '<unknown>')}."
                )

            logprobs_map = {}
            for logprob, token_id, _ in output_logprobs[0]:
                if token_id in label_token_ids:
                    logprobs_map[token_id] = logprob

            item_scores = [
                logprobs_map.get(token_id, float("-inf")) for token_id in label_token_ids
            ]
            if all(score == float("-inf") for score in item_scores):
                raise RuntimeError(
                    "No requested label token IDs were found in output_token_ids_logprobs for "
                    f"{meta_info.get('id', '<unknown>')}."
                )

            if apply_softmax:
                scores.append(softmax(item_scores).tolist())
            else:
                scores.append([math.exp(x) if x != float("-inf") else 0.0 for x in item_scores])
            if meta_info.get("scheduler_dispatch_count") is not None:
                scheduler_dispatch_counts.append(int(meta_info["scheduler_dispatch_count"]))
            if meta_info.get("scheduler_queue_wait_s") is not None:
                scheduler_queue_wait.append(float(meta_info["scheduler_queue_wait_s"]))
            if meta_info.get("scheduler_device_compute_s") is not None:
                scheduler_device_compute.append(float(meta_info["scheduler_device_compute_s"]))
            if meta_info.get("scheduler_host_overhead_s") is not None:
                scheduler_host_overhead.append(float(meta_info["scheduler_host_overhead_s"]))

        logger.debug(
            "Prefill+extend: completed extend batch handle=%s batch_items=%d",
            cache_handle,
            len(items),
        )
        return (
            scores,
            {
                "dispatch_count": (
                    max(scheduler_dispatch_counts) if scheduler_dispatch_counts else 1
                ),
                "queue_wait_s": (max(scheduler_queue_wait) if scheduler_queue_wait else 0.0),
                "device_compute_s": (
                    max(scheduler_device_compute) if scheduler_device_compute else 0.0
                ),
                "host_orchestration_s": (
                    max(scheduler_host_overhead) if scheduler_host_overhead else 0.0
                ),
                "lifecycle_requests_sent": len(items),
                "lifecycle_results_received": len(results),
            },
        )

    async def _release_cache(self, cache_handle: str):
        """Release the cached query."""
        self.auto_create_handle_loop()
        logger.debug("Prefill+extend: releasing cache handle=%s", cache_handle)
        timeout_s = float(
            getattr(self.server_args, "multi_item_prefill_extend_cache_timeout", 60.0)
        )
        try:
            outputs = await self.release_scoring_cache_communicator(
                ReleaseScoringCacheReqInput(rid=cache_handle),
                timeout=timeout_s if timeout_s > 0 else None,
            )
        except TimeoutError:
            logger.error(
                "Timed out releasing prefill+extend cache handle=%s (timeout=%.2fs).",
                cache_handle,
                timeout_s,
            )
            return False
        except Exception:
            logger.exception(
                "Unexpected failure while releasing prefill+extend cache handle=%s.",
                cache_handle,
            )
            return False

        if not outputs:
            logger.warning("Release scoring cache returned no output for handle=%s", cache_handle)
            return False

        for out in outputs:
            if not out.success:
                logger.error(
                    "Failed to release scoring cache handle=%s: %s",
                    cache_handle,
                    out.error_msg,
                )
                return False
            logger.debug(
                "Prefill+extend: released cache handle=%s released_items=%d",
                cache_handle,
                out.released_items,
            )
        return True

    def _record_score_fastpath_fallback(self, reason: str):
        self.score_fastpath_fallback += 1
        self.score_fastpath_fallback_reasons[reason] = (
            self.score_fastpath_fallback_reasons.get(reason, 0) + 1
        )

    async def _score_from_cache_fastpath_v2(
        self,
        cache_handle: str,
        items: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool,
    ) -> ScoreFromCacheReqOutput:
        self.auto_create_handle_loop()
        req_rid = f"scorev2-{uuid.uuid4().hex}"
        timeout_s = float(
            getattr(self.server_args, "multi_item_prefill_extend_cache_timeout", 60.0)
        )
        items_per_step = int(
            getattr(
                self.server_args,
                "multi_item_score_from_cache_v2_items_per_step",
                ServerArgs.multi_item_score_from_cache_v2_items_per_step,
            )
        )
        outputs = await self.score_from_cache_v2_communicator(
            ScoreFromCacheReqInput(
                rid=req_rid,
                cache_handle=cache_handle,
                items_2d=items,
                label_token_ids=label_token_ids,
                apply_softmax=apply_softmax,
                items_per_step=items_per_step,
            ),
            timeout=timeout_s if timeout_s > 0 else None,
        )
        if not outputs:
            return ScoreFromCacheReqOutput(
                success=False,
                scores=[],
                fallback_reason="no_scheduler_response",
                error_msg="No score-from-cache v2 response from scheduler.",
            )
        return outputs[0]

    def _maybe_log_score_path_metrics(self, metrics: dict):
        if not getattr(self.server_args, "multi_item_score_fastpath_log_metrics", False):
            return
        logger.info(
            "ScorePathMetrics path=%s items=%d dispatches=%d lifecycle_sent=%d lifecycle_recv=%d "
            "queue_wait_s=%.6f device_compute_s=%.6f host_orchestration_s=%.6f "
            "fastpath_attempted=%s fastpath_succeeded=%s fastpath_fallback_reason=%s",
            metrics.get("path", "unknown"),
            int(metrics.get("items", 0)),
            int(metrics.get("dispatch_count", 0)),
            int(metrics.get("lifecycle_requests_sent", 0)),
            int(metrics.get("lifecycle_results_received", 0)),
            float(metrics.get("queue_wait_s", 0.0)),
            float(metrics.get("device_compute_s", 0.0)),
            float(metrics.get("host_orchestration_s", 0.0)),
            bool(metrics.get("fastpath_attempted", False)),
            bool(metrics.get("fastpath_succeeded", False)),
            metrics.get("fastpath_fallback_reason"),
        )

    async def score_prefill_extend(
        self,
        query_tokens: list[int],
        item_tokens_list: list[list[int]],
        label_token_ids: list[int],
        apply_softmax: bool = False,
    ) -> list[list[float]]:
        """
        Score items using prefill+extend strategy.
        """
        if not item_tokens_list:
            return []

        logger.debug(
            "Prefill+extend: begin scoring query_tokens=%d items=%d",
            len(query_tokens),
            len(item_tokens_list),
        )
        metrics = {
            "path": "prefill_extend_baseline",
            "items": len(item_tokens_list),
            "dispatch_count": 0,
            "lifecycle_requests_sent": 0,
            "lifecycle_results_received": 0,
            "queue_wait_s": 0.0,
            "device_compute_s": 0.0,
            "host_orchestration_s": 0.0,
            "fastpath_attempted": False,
            "fastpath_succeeded": False,
            "fastpath_fallback_reason": None,
        }
        # Step 1: Prefill query and get cache handle
        cache_handle = await self._prefill_and_cache(query_tokens)
        metrics["lifecycle_requests_sent"] += 1
        metrics["lifecycle_results_received"] += 1

        try:
            if getattr(self.server_args, "multi_item_enable_score_from_cache_v2", False):
                metrics["fastpath_attempted"] = True
                self.score_fastpath_attempted += 1
                try:
                    fastpath_out = await self._score_from_cache_fastpath_v2(
                        cache_handle=cache_handle,
                        items=item_tokens_list,
                        label_token_ids=label_token_ids,
                        apply_softmax=apply_softmax,
                    )
                    metrics["lifecycle_requests_sent"] += 1
                    metrics["lifecycle_results_received"] += 1
                except TimeoutError:
                    fastpath_out = ScoreFromCacheReqOutput(
                        success=False,
                        scores=[],
                        fallback_reason="timeout",
                        error_msg="Timed out waiting for score-from-cache v2 response.",
                    )
                except Exception:
                    logger.exception("Fastpath v2 request failed before scheduler response.")
                    fastpath_out = ScoreFromCacheReqOutput(
                        success=False,
                        scores=[],
                        fallback_reason="runtime_exception",
                        error_msg="Fastpath v2 communicator exception.",
                    )

                fallback_reason = None
                fallback_error_msg = fastpath_out.error_msg
                if fastpath_out.success:
                    if len(fastpath_out.scores) != len(item_tokens_list):
                        fallback_reason = "invalid_response_count"
                        fallback_error_msg = (
                            "Fastpath v2 returned wrong score count: "
                            f"{len(fastpath_out.scores)} != {len(item_tokens_list)}."
                        )
                    else:
                        self.score_fastpath_succeeded += 1
                        metrics["path"] = "score_from_cache_v2"
                        metrics["fastpath_succeeded"] = True
                        metrics["dispatch_count"] += int(fastpath_out.dispatch_count)
                        metrics["queue_wait_s"] += float(fastpath_out.queue_wait_s)
                        metrics["device_compute_s"] += float(fastpath_out.device_compute_s)
                        metrics["host_orchestration_s"] += float(fastpath_out.host_orchestration_s)
                        metrics["lifecycle_requests_sent"] += int(
                            fastpath_out.lifecycle_requests_sent
                        )
                        metrics["lifecycle_results_received"] += int(
                            fastpath_out.lifecycle_results_received
                        )
                        self._maybe_log_score_path_metrics(metrics)
                        return fastpath_out.scores
                else:
                    fallback_reason = fastpath_out.fallback_reason or "runtime_exception"

                if fallback_reason is not None:
                    metrics["fastpath_fallback_reason"] = fallback_reason
                    self._record_score_fastpath_fallback(fallback_reason)
                    logger.warning(
                        "Fastpath v2 falling back to baseline: reason=%s error=%s",
                        fallback_reason,
                        fallback_error_msg,
                    )

            # Step 2: Process items in batches
            all_scores = []
            batch_size = int(getattr(self.server_args, "multi_item_extend_batch_size", 32))
            if batch_size <= 0:
                batch_size = len(item_tokens_list) or 1

            for i in range(0, len(item_tokens_list), batch_size):
                batch = item_tokens_list[i : i + batch_size]
                logger.debug(
                    "Prefill+extend: processing batch start=%d size=%d total=%d",
                    i,
                    len(batch),
                    len(item_tokens_list),
                )
                # Keep extend batch shape stable to avoid extra compile on trailing
                # partial batches (e.g., 10 items with batch size 4 -> 4,4,2).
                # We drop padded scores after the call.
                padded_batch = batch
                padded_count = 0
                if len(batch) < batch_size and len(batch) > 0:
                    padded_count = batch_size - len(batch)
                    padded_batch = batch + [batch[-1]] * padded_count
                batch_scores, batch_metrics = await self._batched_extend_score_with_metrics(
                    cache_handle=cache_handle,
                    items=padded_batch,
                    label_token_ids=label_token_ids,
                    apply_softmax=apply_softmax,
                )
                if padded_count > 0:
                    batch_scores = batch_scores[: len(batch)]
                all_scores.extend(batch_scores)
                metrics["dispatch_count"] += int(batch_metrics["dispatch_count"])
                metrics["queue_wait_s"] += float(batch_metrics["queue_wait_s"])
                metrics["device_compute_s"] += float(batch_metrics["device_compute_s"])
                metrics["host_orchestration_s"] += float(batch_metrics["host_orchestration_s"])
                # Only real items should contribute to lifecycle counters.
                real_items = len(batch)
                metrics["lifecycle_requests_sent"] += real_items
                metrics["lifecycle_results_received"] += real_items

            logger.debug("Prefill+extend: complete items=%d", len(item_tokens_list))
            self._maybe_log_score_path_metrics(metrics)
            return all_scores
        finally:
            # Step 3: Release cache
            released = await self._release_cache(cache_handle)
            if not released:
                logger.warning(
                    "Prefill+extend cache handle=%s was not cleanly released.", cache_handle
                )


async def print_exception_wrapper(func):
    """
    Sometimes an asyncio function does not print exception.
    We do another wrapper to handle the exception.
    """
    try:
        await func()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("TokenizerManager hit an exception: %s", traceback)
        if hasattr(func, "__self__") and isinstance(func.__self__, TokenizerManager):
            func.__self__.dump_requests_before_crash()
        kill_process_tree(os.getpid(), include_parent=True)
        sys.exit(1)


class SignalHandler:
    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def sigterm_handler(self, signum=None, frame=None):
        logger.warning(
            "SIGTERM received. signum=%s frame=%s. Draining requests and shutting down...",
            signum,
            frame,
        )
        self.tokenizer_manager.gracefully_exit = True

    def running_phase_sigquit_handler(self, signum=None, frame=None):
        logger.error("Received sigquit from a child process. It usually means the child failed.")
        self.tokenizer_manager.dump_requests_before_crash()
        kill_process_tree(os.getpid())


@dataclasses.dataclass
class _CorrelatedWaiter[T]:
    event: asyncio.Event
    values: list[T]


class _CorrelatedCommunicator[T]:
    """Allow multiple in-flight RPCs by correlating responses with `rid`."""

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._pending: dict[str, _CorrelatedWaiter[T]] = {}

    async def __call__(self, obj, timeout: float | None = None):
        rid = getattr(obj, "rid", None)
        if not rid:
            raise ValueError(
                "Correlated communicator requires request objects with non-empty `rid`."
            )
        if rid in self._pending:
            raise RuntimeError(f"Duplicate in-flight correlated request rid={rid!r}.")

        waiter = _CorrelatedWaiter(event=asyncio.Event(), values=[])
        self._pending[rid] = waiter
        try:
            if obj is not None:
                self._sender.send_pyobj(obj)

            wait_coro = waiter.event.wait()
            if timeout is not None and timeout > 0:
                await asyncio.wait_for(wait_coro, timeout=timeout)
            else:
                await wait_coro
            return list(waiter.values)
        finally:
            self._pending.pop(rid, None)

    def handle_recv(self, recv_obj: T):
        rid = getattr(recv_obj, "rid", None)
        if not rid:
            logger.warning(
                "Dropping correlated communicator response missing rid. type=%s",
                type(recv_obj).__name__,
            )
            return

        waiter = self._pending.get(rid)
        if waiter is None:
            logger.warning(
                "Dropping correlated communicator response with no active waiter. rid=%s type=%s",
                rid,
                type(recv_obj).__name__,
            )
            return

        waiter.values.append(recv_obj)
        if len(waiter.values) >= self._fan_out:
            waiter.event.set()


class _Communicator[T]:
    """Note: The communicator now only run up to 1 in-flight request at any time."""

    def __init__(self, sender, fan_out: int):
        self._sender = sender
        self._fan_out = fan_out
        self._lock = asyncio.Lock()
        self._result_event: asyncio.Event | None = None
        self._result_values: list[T] | None = None

    async def __call__(self, obj, timeout: float | None = None):
        async with self._lock:
            if self._result_event is not None or self._result_values is not None:
                raise RuntimeError(
                    "Communicator received a new call while a previous call is still active."
                )

            self._result_event = asyncio.Event()
            self._result_values = []
            try:
                if obj is not None:
                    self._sender.send_pyobj(obj)

                wait_coro = self._result_event.wait()
                if timeout is not None and timeout > 0:
                    await asyncio.wait_for(wait_coro, timeout=timeout)
                else:
                    await wait_coro

                return list(self._result_values)
            finally:
                self._result_event = None
                self._result_values = None

    def handle_recv(self, recv_obj: T):
        if self._result_values is None or self._result_event is None:
            logger.warning(
                "Dropping communicator response with no active waiter. type=%s",
                type(recv_obj).__name__,
            )
            return
        self._result_values.append(recv_obj)
        if len(self._result_values) >= self._fan_out:
            self._result_event.set()
