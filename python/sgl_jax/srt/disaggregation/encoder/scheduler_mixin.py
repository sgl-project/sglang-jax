from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from sgl_jax.srt.disaggregation.encoder import create_encoder_client
from sgl_jax.srt.disaggregation.encoder.client import (
    EncoderClient,
    PendingEncoderRequest,
)
from sgl_jax.srt.managers.io_struct import AbortReq, TokenizedGenerateReqInput

if TYPE_CHECKING:
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerDisaggregationEncoderMixin:
    """Encoder-disaggregation request handling for the language scheduler."""

    encoder_client: EncoderClient | None
    encoder_waiting: dict[str, PendingEncoderRequest]

    def init_encoder_disaggregation(self: Scheduler) -> None:
        self.encoder_client = None
        self.encoder_waiting = {}
        if not self.server_args.language_only:
            return
        if self.nnodes > 1:
            raise RuntimeError("encoder disaggregation does not support multi-host schedulers yet")
        if self._mm_processor is None:
            raise ValueError("encoder disaggregation requires a multimodal processor")
        token_buckets = (
            []
            if self.server_args.disable_precompile
            else self.tp_worker.get_precompile_paddings()[0]
        )
        self.encoder_client = create_encoder_client(
            self.server_args,
            self.tp_worker.model_runner,
            token_buckets,
            self._apply_encoder_result,
        )

    def process_encoder_requests(
        self: Scheduler,
        recv_reqs: list,
    ) -> list:
        ready = []
        now = time.monotonic()

        for recv_req in recv_reqs:
            if (
                not isinstance(recv_req, TokenizedGenerateReqInput)
                or not recv_req.need_wait_for_mm_inputs
            ):
                ready.append(recv_req)
                continue
            if recv_req.rid in self.encoder_waiting:
                continue

            try:
                pending = self.encoder_client.receive(recv_req)
            except Exception as exc:
                self._abort_encoder_request(recv_req, str(exc))
                continue

            self.encoder_waiting[recv_req.rid] = pending

        for pending in self.encoder_client.drain_completed():
            recv_req = pending.recv_req
            rid = recv_req.rid
            if self.encoder_waiting.get(rid) is not pending:
                continue
            try:
                pending.poll()
            except Exception as exc:
                self._abort_encoder_request(recv_req, str(exc))
            else:
                ready.append(recv_req)
            self._remove_encoder_waiting(rid)

        timeout = self.server_args.encoder_request_timeout_seconds
        if timeout > 0:
            for rid, pending in list(self.encoder_waiting.items()):
                if now - pending.started_at >= timeout:
                    self._abort_encoder_request(
                        pending.recv_req, f"encoder timed out after {timeout}s"
                    )
                    self._remove_encoder_waiting(rid)

        return ready

    def _apply_encoder_result(self: Scheduler, recv_req, result: dict) -> None:
        embeddings = result.get("embeddings")
        if embeddings is None:
            raise ValueError("encoder result contains no embeddings")
        prompt = recv_req.input_ids if recv_req.input_ids is not None else recv_req.text
        mm_inputs = self._mm_processor.get_mm_data(
            prompt,
            embeddings,
            **{key: value for key, value in result.items() if key != "embeddings"},
        )
        recv_req.mm_inputs = mm_inputs
        recv_req.input_ids = mm_inputs.input_ids
        recv_req.radix_input_ids = mm_inputs.radix_input_ids
        recv_req.need_wait_for_mm_inputs = False

    def _abort_encoder_request(self: Scheduler, recv_req, error_msg: str) -> None:
        logger.error("Encoder request failed. rid=%s error=%s", recv_req.rid, error_msg)
        output = AbortReq(rid=recv_req.rid, aborted_message=error_msg)
        if self._comm_backend is not None:
            self._comm_backend.send_pyobj(output)
        else:
            self.send_to_tokenizer.send_pyobj(output)

    def _remove_encoder_waiting(self: Scheduler, rid: str) -> None:
        pending = self.encoder_waiting.pop(rid)
        pending.close()

    def _cancel_encoder_requests(self: Scheduler, recv_req: AbortReq) -> None:
        for rid in list(self.encoder_waiting):
            if recv_req.abort_all or rid.startswith(recv_req.rid):
                self._remove_encoder_waiting(rid)
                output = AbortReq(rid=rid, aborted_message=recv_req.aborted_message)
                if self._comm_backend is not None:
                    self._comm_backend.send_pyobj(output)
                else:
                    self.send_to_tokenizer.send_pyobj(output)
