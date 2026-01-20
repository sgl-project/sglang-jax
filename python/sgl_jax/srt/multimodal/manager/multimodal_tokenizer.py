import asyncio
import dataclasses
import logging
import signal
import time
import uuid
from http import HTTPStatus
from typing import Any

import fastapi
import psutil
import setproctitle
from transformers import AutoImageProcessor

from sgl_jax.srt.managers.io_struct import AbortReq
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager
from sgl_jax.srt.multimodal.manager.io_struct import (
    GenerateMMReqInput,
    TokenizedGenerateMMReqInput,
)
from sgl_jax.srt.server_args import PortArgs, ServerArgs
from sgl_jax.srt.utils import (
    configure_logger,
    dataclass_to_string_truncated,
    kill_itself_when_parent_died,
)
from sgl_jax.utils import TypeBasedDispatcher, get_exception_traceback

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class MMReqState:
    """Store the state of a request."""

    rid: str
    out_list: list[dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateMMReqInput
    created_time: float


class MultimodalTokenizer(TokenizerManager):
    """Tokenization manager for multimodal requests.

    `MultimodalTokenizer` accepts high-level multimodal generation requests
    (`GenerateMMReqInput`), tokenizes text inputs (and prepares image
    references when supported), forwards tokenized requests to the
    scheduler pipeline, and waits for/streams back results. It tracks the
    state of outstanding requests via `MMReqState` and uses a
    `TypeBasedDispatcher` to handle results arriving from the pipeline.
    """

    def __init__(self, server_args, port_args):
        """Initialize tokenizer, processor and result dispatcher.

        Loads an image processor (best-effort), initializes an in-memory
        map `rid_to_state` to track request state objects, and prepares a
        result dispatcher that routes batches of outputs back to
        `_handle_batch_output`.
        """
        super().__init__(server_args, port_args)
        # Use slow image processor to avoid torchvision dependency warning
        try:
            self.mm_processor = AutoImageProcessor.from_pretrained(
                server_args.model_path, use_fast=False
            )
        except Exception:
            logger.warning("Failed to load processor from %s", server_args.model_path)
        self.rid_to_state: dict[str, MMReqState] = {}
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (list),
                    self._handle_batch_output,
                ),
                (
                    AbortReq,
                    self._handle_abort_req,
                ),
            ]
        )

    def _handle_batch_output(self, reqs: list):
        """Handle a batch of outputs returned from the pipeline.

        Marks the corresponding `MMReqState` as finished, sets its event to
        wake any waiters, and stores a simple success meta record. If a
        result arrives for an unknown `rid` it logs a warning.
        """
        if len(reqs) > 0 and self.server_args.log_requests:
            logger.info("handle_batch_output %s, self.rid_to_state %s", reqs, self.rid_to_state)
        for req in reqs:
            if req.rid in self.rid_to_state:
                self.rid_to_state[req.rid].finished = True
                self.rid_to_state[req.rid].event.set()
                self.rid_to_state[req.rid].out_list = [{"success": True, "meta_info": {}}]
            else:
                logger.warning(
                    "Received result for unknown request rid=%s. Known rids: %s",
                    req.rid,
                    list(self.rid_to_state.keys()),
                )

    def _handle_abort_req(self, recv_obj: AbortReq):
        """Handle an AbortReq returned from the scheduler.

        When a request is aborted (e.g., removed from the scheduler's queue
        before processing started), the scheduler sends an AbortReq back to
        notify the tokenizer. This method marks the request as finished with
        an abort status and wakes any waiting coroutines.
        """
        if recv_obj.rid not in self.rid_to_state:
            logger.warning(
                "Received abort for unknown request rid=%s. Known rids: %s",
                recv_obj.rid,
                list(self.rid_to_state.keys()),
            )
            return

        state = self.rid_to_state[recv_obj.rid]
        state.finished = True
        state.out_list.append(
            {
                "success": False,
                "meta_info": {
                    "id": recv_obj.rid,
                    "finish_reason": {
                        "type": "abort",
                        "message": recv_obj.aborted_message or "Request aborted",
                        "status_code": HTTPStatus.BAD_REQUEST,
                    },
                },
            }
        )
        state.event.set()
        logger.info("Abort completed for rid=%s", recv_obj.rid)

    async def generate_request(
        self,
        obj: GenerateMMReqInput,
        request: fastapi.Request | None = None,
    ):
        """High level API: accept a generation request and stream responses.

        This coroutine tokenizes the input (text and optional image refs),
        sends the tokenized request to the scheduler pipeline, and then
        asynchronously yields results as they arrive (supporting streaming
        if `obj.stream` is True). It respects client disconnects and a
        configured wait timeout.
        """
        created_time = time.time()
        async with self._cond:
            await self._cond.wait_for(lambda: not self._updating)

        self.auto_create_handle_loop()

        if self.log_requests:
            max_length, skip_names, _ = self.log_request_metadata
            logger.info(
                "Receive: obj=%s",
                dataclass_to_string_truncated(obj, max_length, skip_names=skip_names),
            )

        tokenized_obj = await self._tokenize_one_request(obj)
        state = self._send_one_request(obj, tokenized_obj, created_time)
        async for response in self._wait_one_response(obj, state, request):
            yield response

    async def _tokenize_one_request(self, obj: GenerateMMReqInput):
        """
        Converts text fields to token ids using the configured tokenizer.
        Image preprocessing / references are noted as TODO; when provided
        `input_ids` are passed through unchanged.
        """
        # Support both 'prompt' (multimodal) and 'text' (text-only) fields
        input_text = getattr(obj, "prompt", None) or getattr(obj, "text", None)
        neg_input_text = getattr(obj, "neg_prompt", None) or getattr(obj, "text", None)
        input_ids = getattr(obj, "input_ids", None)
        neg_input_ids = getattr(obj, "neg_input_ids", None)
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        if neg_input_ids is None and neg_input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but neg_input_text requires tokenization"
                )
            encoded = self.tokenizer(neg_input_text)
            neg_input_ids = encoded["input_ids"]
        if getattr(obj, "input_reference", None) is not None:
            # TODO: Handle image preprocessing for multimodal inputs
            pass

        return self._create_tokenized_object(
            obj, input_text, input_ids, neg_input_text, neg_input_ids
        )

    def _create_tokenized_object(
        self, obj: GenerateMMReqInput, input_text, input_ids, neg_input_text, neg_input_ids
    ):
        """Build `TokenizedGenerateMMReqInput` from the original request.

        Ensures a request id (`rid`) exists, and copies over relevant
        properties such as size, num_frames, data type and save_output flag.
        """
        rid = getattr(obj, "rid", None)
        if rid is None:
            rid = uuid.uuid4().hex

        tokenized_obj = TokenizedGenerateMMReqInput(
            rid=rid,
            prompt=input_text,
            negative_prompt=neg_input_text,
            input_ids=input_ids,
            negative_input_ids=neg_input_ids,
            size=getattr(obj, "size", None),
            num_frames=getattr(obj, "num_frames", None),
            num_inference_steps=getattr(obj, "num_inference_steps", None),
            data_type=getattr(obj, "data_type", None),
            save_output=getattr(obj, "save_output", True),
        )
        return tokenized_obj

    def _send_one_request(
        self,
        obj: GenerateMMReqInput,
        tokenized_obj: TokenizedGenerateMMReqInput,
        created_time: float | None = None,
    ):
        """Send a tokenized request into the scheduling pipeline and track it.

        Constructs an `MMReqState` to wait for results and stores it in
        `rid_to_state` keyed by the request id.
        """
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = MMReqState(
            rid=tokenized_obj.rid,
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=obj,
            created_time=created_time,
        )
        self.rid_to_state[tokenized_obj.rid] = state
        return state

    async def _wait_one_response(
        self,
        obj: GenerateMMReqInput,
        state: MMReqState,
        request: fastapi.Request | None = None,
    ):
        """Wait for results for a single request, yielding responses.

        This method waits on `state.event` with a timeout (`self.wait_timeout`),
        handles client disconnects (aborting the request), and yields
        intermediate/final outputs according to `obj.stream`.
        """
        while True:
            try:
                await asyncio.wait_for(state.event.wait(), timeout=self.wait_timeout)
            except TimeoutError:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    ) from None
                continue

            out = state.out_list[-1]

            state.out_list = []
            if state.finished:
                if self.log_requests:
                    max_length, skip_names, out_skip_names = self.log_request_metadata
                    msg = f"Finish: obj={dataclass_to_string_truncated(obj, max_length, skip_names=skip_names)}, out={dataclass_to_string_truncated(out, max_length, skip_names=out_skip_names)}"
                    logger.info(msg)

                if isinstance(out["meta_info"].get("finish_reason"), dict):
                    finish_reason = out["meta_info"]["finish_reason"]
                    if (
                        finish_reason.get("type") == "abort"
                        and finish_reason.get("status_code") == HTTPStatus.BAD_REQUEST
                    ):
                        raise ValueError(finish_reason["message"])

                yield out
                break

            state.event.clear()

            if obj.stream:
                yield out
            else:
                if request is not None and await request.is_disconnected():
                    self.abort_request(state.rid)
                    raise ValueError(
                        f"Request is disconnected from the client side. Abort request rid={state.rid}"
                    )


def run_multimodal_tokenizer_process(
    server_args: ServerArgs,
    port_args: PortArgs,
):
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang-jax::multimodal_tokenizer")
    configure_logger(server_args)
    parent_process = psutil.Process().parent()

    try:
        tokenizer = MultimodalTokenizer(server_args, port_args)
        tokenizer.event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("MultimodalTokenizerManager hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGQUIT)

    return tokenizer
