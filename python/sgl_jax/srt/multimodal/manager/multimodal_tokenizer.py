import asyncio
import dataclasses
import logging
import signal
import time
from http import HTTPStatus
from typing import Any

import fastapi
import psutil
import setproctitle
from transformers import AutoImageProcessor

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
    """Store the state a request."""

    out_list: list[dict[Any, Any]]
    finished: bool
    event: asyncio.Event
    obj: GenerateMMReqInput

    created_time: float


class MultimodalTokenizer(TokenizerManager):
    def __init__(self, server_args, port_args):
        super().__init__(server_args, port_args)
        self.mm_processor = AutoImageProcessor.from_pretrained(
            server_args.model_path, use_fast=True
        )
        self.rid_to_state: dict[str, MMReqState] = {}
        # todo: rewrite this
        self._result_dispatcher = TypeBasedDispatcher(
            [
                (
                    (list),
                    self._handle_batch_output,
                ),
            ]
        )

    def _handle_batch_output(self, reqs: list):
        print(f"handle_batch_output {reqs}, self.rid_to_state {self.rid_to_state}")
        for req in reqs:
            if req.rid in self.rid_to_state:
                self.rid_to_state[req.rid].finished = True
                self.rid_to_state[req.rid].event.set()
                self.rid_to_state[req.rid].out_list = [{"success": True, "meta_info": {}}]

    async def generate_request(
        self,
        obj: GenerateMMReqInput,
        request: fastapi.Request | None = None,
    ):
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
        """Tokenize one request."""

        # Tokenize
        input_text = obj.prompt
        input_ids = obj.input_ids
        if input_ids is None and input_text is not None:
            if self.tokenizer is None:
                raise ValueError(
                    "Tokenizer is not initialized but input_text requires tokenization"
                )
            encoded = self.tokenizer(input_text)
            input_ids = encoded["input_ids"]
        if obj.input_reference is not None:
            # todo: need handle image process
            pass
        return self._create_tokenized_object(obj, input_text, input_ids)

    def _create_tokenized_object(self, obj: GenerateMMReqInput, input_text, input_ids):
        tokenized_obj = TokenizedGenerateMMReqInput(
            obj.rid,
            obj.prompt,
            input_ids,
            size=obj.size,
            num_frames=obj.num_frames,
        )
        return tokenized_obj

    def _send_one_request(
        self,
        obj: GenerateMMReqInput,
        tokenized_obj: TokenizedGenerateMMReqInput,
        created_time: float | None = None,
    ):
        self.send_to_scheduler.send_pyobj(tokenized_obj)
        state = MMReqState([], False, asyncio.Event(), obj, created_time=created_time)
        # Handle rid being a list (single element) or string
        rid_key = obj.rid[0] if isinstance(obj.rid, list) else obj.rid
        self.rid_to_state[rid_key] = state
        return state

    async def _wait_one_response(
        self,
        obj: GenerateMMReqInput,
        state: MMReqState,
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
                    # Abort the request for disconnected requests (non-streaming, running)
                    self.abort_request(obj.rid)
                    # Use exception to kill the whole call stack and asyncio task
                    raise ValueError(
                        f"Request is disconnected from the client side (type 3). Abort request {obj.rid=}"
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
