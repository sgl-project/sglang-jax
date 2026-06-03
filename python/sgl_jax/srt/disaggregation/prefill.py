"""Prefill-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

from http import HTTPStatus
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from functools import partial

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVSender,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


# Bucket page counts to bound XLA's per-shape compile pool.
_KV_GATHER_PAGE_BUCKETS = (1, 2, 4, 8, 16, 32, 64)


def _pad_to_page_bucket(num_pages: int) -> int:
    for b in _KV_GATHER_PAGE_BUCKETS:
        if b >= num_pages:
            return b
    return _KV_GATHER_PAGE_BUCKETS[-1]


@partial(jax.jit, static_argnames=("out_sharding",))
def _jit_gather_all_layers(buffers, page_indices, out_sharding):
    """Gather ``page_indices`` from every per-layer KV buffer in one jit."""

    return [
        buf.at[page_indices].get(out_sharding=out_sharding) for buf in buffers
    ]


@dataclass
class PrefillBookkeeping:
    """Per-request prefill-side state tracked by the Mixin."""

    req_id: str
    sender: JaxTransferKVSender
    # Optional callback the scheduler runs when this entry reaches a
    # terminal state — used to release ``req_to_token_pool`` and any
    # owned KV indices.
    on_terminal: object | None = None


class PrefillBootstrapQueue:
    """Tracks senders pending decoder ack. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, PrefillBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(
        self,
        req_id: str,
        sender: JaxTransferKVSender,
        on_terminal=None,
    ) -> None:
        with self._lock:
            if req_id in self._entries:
                raise ValueError(
                    f"PrefillBootstrapQueue already tracks "
                    f"req_id={req_id!r}"
                )
            self._entries[req_id] = PrefillBookkeeping(
                req_id=req_id, sender=sender, on_terminal=on_terminal
            )

    def drain_terminal(self) -> list[PrefillBookkeeping]:
        """Remove and return entries that reached SUCCESS or FAILED."""

        terminal: list[PrefillBookkeeping] = []
        with self._lock:
            for req_id, entry in list(self._entries.items()):
                state = entry.sender.poll()
                if state in (KVPoll.SUCCESS, KVPoll.FAILED):
                    terminal.append(entry)
                    del self._entries[req_id]
        return terminal

    def snapshot_states(self) -> dict[str, KVPoll]:
        with self._lock:
            return {
                rid: entry.sender.poll()
                for rid, entry in self._entries.items()
            }


class SchedulerDisaggregationPrefillMixin:
    """Mixin for PD prefill mode on Scheduler."""

    disagg_kv_manager: JaxTransferKVManager
    disagg_prefill_queue: PrefillBootstrapQueue
    disagg_use_d2h_staging: bool

    def event_loop_normal_disagg_prefill(self) -> None:
        """Prefill-only event loop."""

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()  # type: ignore[attr-defined]
                if self._comm_backend is not None  # type: ignore[attr-defined]
                else self.recv_requests()  # type: ignore[attr-defined]
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)  # type: ignore[attr-defined]
            self.process_input_requests(recv_reqs)  # type: ignore[attr-defined]

            if self._engine_paused:  # type: ignore[attr-defined]
                continue

            batch = self.get_next_batch_to_run()  # type: ignore[attr-defined]
            self.cur_batch = batch  # type: ignore[attr-defined]

            if batch:
                result = self.run_batch(batch)  # type: ignore[attr-defined]
                self.process_prefill_chunk(batch, result)
            else:
                self.send_kv_chunk()
                self.new_token_ratio = self.init_new_token_ratio  # type: ignore[attr-defined]
                if self._comm_backend is not None:  # type: ignore[attr-defined]
                    self._comm_backend.wait_for_new_requests(0.001)  # type: ignore[attr-defined]

            self.send_kv_chunk()
            self.last_batch = batch  # type: ignore[attr-defined]

    def process_prefill_chunk(self, batch, result) -> None:
        """Extract KV for PD reqs and hand off to sender."""

        pd_reqs = [
            req
            for req in batch.reqs
            if getattr(req, "bootstrap_room", None) is not None
        ]
        if not pd_reqs:
            self.process_batch_result(batch, result)  # type: ignore[attr-defined]
            return

        self.set_next_batch_sampling_info_done(batch)  # type: ignore[attr-defined]

        for req in batch.reqs:
            if req.bootstrap_room is None:
                continue
            req_id = req.rid
            if req_id in self.disagg_prefill_queue._entries:
                continue
            try:
                device_kv = self._extract_req_kv(req)
            except Exception:
                logger.exception(
                    "failed to extract KV for req_id=%s; skipping send",
                    req_id,
                )
                continue
            self._maybe_log_prefill_extract_debug(
                req,
                device_kv,
                use_d2h_staging=self.disagg_use_d2h_staging,
            )
            sender = self.disagg_kv_manager.create_sender(req_id)
            sender.init(
                kv_indices=None,
                transfer_id=(
                    getattr(req, "disagg_transfer_id", None) or req_id
                ),
            )
            sender.attach_payload(
                {"kv": device_kv},
                use_d2h_staging=self.disagg_use_d2h_staging,
            )
            sender.send()

            def _on_terminal(req_obj=req, sender_obj=sender):
                self._on_prefill_transfer_terminal(req_obj, sender_obj)

            self.disagg_prefill_queue.add(
                req_id, sender, on_terminal=_on_terminal
            )

    def send_kv_chunk(self) -> None:
        """Reap senders that reached SUCCESS / FAILED."""

        terminal = self.disagg_prefill_queue.drain_terminal()
        for entry in terminal:
            on_terminal = entry.on_terminal
            if on_terminal is None:
                continue
            try:
                on_terminal()
            except Exception:
                logger.exception(
                    "on_terminal for req_id=%s raised; continuing",
                    entry.req_id,
                )

    # ------------------------------------------------------------------
    # Overridable / test-friendly hooks
    # ------------------------------------------------------------------

    def _extract_req_kv(self, req: Req):
        """Gather prefilled KV from the paged pool for ``req``.

        Returns shape ``(layer_num, padded_pages, page_size, ...)``.
        """

        req_to_token = self.req_to_token_pool.req_to_token  # type: ignore[attr-defined]
        kv_pool = (
            self.token_to_kv_pool_allocator.get_kvcache()  # type: ignore[attr-defined]
        )
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        padded_pages = _pad_to_page_bucket(num_pages)
        page_id_source = (
            req_to_token[
                req.req_pool_idx,
                : padded_pages * page_size : page_size,
            ]
        )
        import numpy as _np
        from jax.sharding import NamedSharding as _NamedSharding
        from jax.sharding import PartitionSpec as _P

        # Indices must be placed on the same mesh as the KV pool.
        idx_sharding = _NamedSharding(kv_pool.mesh, _P(None))
        page_indices = jax.device_put(
            _np.asarray(page_id_source) // page_size,
            idx_sharding,
        )
        # out_sharding describes the gather output, not the pool.
        pool_pspec = kv_pool.kv_sharding.spec
        gather_pspec = _P(None, *pool_pspec[1:])
        gather_out_sharding = _NamedSharding(kv_pool.mesh, gather_pspec)
        layer_buffers = [
            kv_pool.get_kv_buffer(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        layer_kvs = _jit_gather_all_layers(
            layer_buffers, page_indices, gather_out_sharding
        )
        return jnp.stack(layer_kvs, axis=0)

    def _release_prefill_req_resources(self, req: Req) -> None:
        """Release req resources. Delegates to cache_finished_req."""

        if hasattr(self, "cache_finished_req"):
            self.cache_finished_req(req)  # type: ignore[attr-defined]

    def _on_prefill_transfer_terminal(
        self, req: Req, sender: JaxTransferKVSender
    ) -> None:
        try:
            if sender.poll() == KVPoll.SUCCESS:
                self._finish_prefill_only_success(req)
            else:
                self._finish_prefill_only_failure(req, sender)
        finally:
            if hasattr(sender, "clear"):
                sender.clear()
            self._release_prefill_req_resources(req)

    def _finish_prefill_only_success(self, req: Req) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_LENGTH

        req.finished_reason = FINISH_LENGTH(length=0)
        req.output_ids = []
        req.finished_len = 0
        if hasattr(self, "stream_output"):
            self.stream_output(  # type: ignore[attr-defined]
                [req],
                getattr(req, "return_logprob", False),
                getattr(req, "return_output_logprob_only", False),
            )

    def _finish_prefill_only_failure(
        self, req: Req, sender: JaxTransferKVSender
    ) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT

        error_message = (
            f"Prefill transfer failed for req_id={getattr(req, 'rid', None)!r} "
            f"bootstrap_room={getattr(req, 'bootstrap_room', None)!r}"
        )
        try:
            sender.failure_exception()
        except Exception as exc:  # noqa: BLE001
            error_message = f"{error_message}: {exc}"
        req.finished_reason = FINISH_ABORT(
            error_message,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "PDTransferError",
        )
        req.output_ids = []
        if hasattr(self, "stream_output"):
            self.stream_output(  # type: ignore[attr-defined]
                [req],
                getattr(req, "return_logprob", False),
                getattr(req, "return_output_logprob_only", False),
            )

    def _maybe_log_prefill_extract_debug(self, req: Req, kv, **meta) -> None:
        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            kv_debug_enabled,
        )

        if not kv_debug_enabled(getattr(req, "rid", None)):
            return

        snapshot = build_kv_debug_snapshot(kv)
        logger.warning(
            "PD-KV-DEBUG prefill_extract req_id=%s shape=%s dtype=%s "
            "sharding=%s digest=%s sample=%s meta=%s",
            req.rid,
            snapshot.shape,
            snapshot.dtype,
            snapshot.sharding,
            snapshot.global_digest,
            snapshot.sample_page_digests(),
            meta,
        )
