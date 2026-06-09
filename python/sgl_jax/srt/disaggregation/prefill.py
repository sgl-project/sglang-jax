"""Prefill-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVSender,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


# Bucket page counts to bound XLA's per-shape compile pool.
# Largest bucket (512 pages × 128 tokens/page) covers 64k-token prompts.
_KV_GATHER_PAGE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def _pad_to_page_bucket(num_pages: int) -> int:
    for b in _KV_GATHER_PAGE_BUCKETS:
        if b >= num_pages:
            return b
    # Beyond the largest predefined bucket: round up to a multiple of it so we
    # never truncate KV, while keeping the set of compiled shapes bounded.
    largest = _KV_GATHER_PAGE_BUCKETS[-1]
    return ((num_pages + largest - 1) // largest) * largest


@partial(jax.jit, static_argnames=("out_sharding",))
def _jit_gather_one_layer(buf, page_indices, out_sharding):
    """Gather ``page_indices`` from a single per-layer KV buffer.

    Split from the original all-layers-in-one-jit to avoid XLA compile-time
    OOM when layer_num * per_layer_size exceeds HBM (e.g. 36 layers × 684 MB
    = 24.6 GB input → ~45 GB compile footprint on v6e-1).
    Per-layer compile footprint is ~1.2 GB regardless of layer count.
    """
    return buf.at[page_indices].get(out_sharding=out_sharding)


def _jit_gather_all_layers(buffers, page_indices, out_sharding):
    """Gather ``page_indices`` from every per-layer KV buffer.

    Dispatches per-layer jit calls. Each call compiles independently with
    ~1.2 GB footprint. The 36 kernel launches add ~1.8ms total overhead
    (negligible vs transfer + E2E latency).
    """
    return [
        _jit_gather_one_layer(buf, page_indices, out_sharding) for buf in buffers
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
                raise ValueError(f"PrefillBootstrapQueue already tracks " f"req_id={req_id!r}")
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

    def abort_matching(self, rid_prefix: str, abort_all: bool) -> list[PrefillBookkeeping]:
        out: list[PrefillBookkeeping] = []
        with self._lock:
            for req_id in list(self._entries):
                if abort_all or req_id.startswith(rid_prefix):
                    out.append(self._entries.pop(req_id))
        return out


class SchedulerDisaggregationPrefillMixin:
    """Mixin for PD prefill mode on Scheduler."""

    disagg_kv_manager: JaxTransferKVManager
    disagg_prefill_queue: PrefillBootstrapQueue
    disagg_use_d2h_staging: bool

    def event_loop_normal_disagg_prefill(self: Scheduler) -> None:
        """Prefill-only event loop."""

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)
            self.process_input_requests(recv_reqs)

            if self._engine_paused:
                continue

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_prefill_chunk(batch, result)
            else:
                self.send_kv_chunk()
                self.new_token_ratio = self.init_new_token_ratio
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.send_kv_chunk()
            self.last_batch = batch

    def process_prefill_chunk(self: Scheduler, batch, result) -> None:
        """Extract KV for PD reqs and hand off to sender."""

        pd_reqs = [req for req in batch.reqs if req.bootstrap_room is not None]
        if not pd_reqs:
            self.process_batch_result(batch, result)
            return

        self.set_next_batch_sampling_info_done(batch)

        for req in batch.reqs:
            if req.bootstrap_room is None:
                continue
            req_id = req.rid
            if req_id in self.disagg_prefill_queue._entries:
                continue
            try:
                device_kv = self._extract_req_kv(req)
            except Exception as exc:
                logger.exception(
                    "failed to extract KV for req_id=%s; aborting",
                    req_id,
                )
                self._abort_prefill_req(
                    req,
                    f"KV extraction failed for req_id={req_id!r}: {exc}",
                    metric_reason="kv_extraction",
                )
                continue
            sender = None
            try:
                self._maybe_log_prefill_extract_debug(
                    req,
                    device_kv,
                    use_d2h_staging=self.disagg_use_d2h_staging,
                )
                sender = self.disagg_kv_manager.create_sender(req_id)
                sender.init(
                    kv_indices=None,
                    transfer_id=req.disagg_transfer_id or req_id,
                )
                sender.attach_payload(
                    {"kv": device_kv},
                    use_d2h_staging=self.disagg_use_d2h_staging,
                )
                self._pd_mark_time(req, "transfer_start")
                sender.send()
            except Exception as exc:
                logger.exception(
                    "sender init/send failed for req_id=%s; aborting",
                    req_id,
                )
                if sender is not None:
                    with suppress(Exception):
                        sender.abort()
                    with suppress(Exception):
                        sender.clear()
                self._abort_prefill_req(
                    req,
                    f"Prefill sender failed for req_id={req_id!r}: {exc}",
                    metric_reason="sender_init",
                )
                continue

            def _on_terminal(req_obj=req, sender_obj=sender):
                self._on_prefill_transfer_terminal(req_obj, sender_obj)

            self.disagg_prefill_queue.add(req_id, sender, on_terminal=_on_terminal)

    def send_kv_chunk(self: Scheduler) -> None:
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

    def _pd_mark_time(self: Scheduler, req: Req, name: str) -> None:
        """Record a PD lifecycle mark on ``req`` (no-op unless enabled)."""

        if not getattr(self.server_args, "enable_request_time_stats_logging", False):
            return
        from sgl_jax.srt.disaggregation.req_time_stats import TimeStats

        ts = req.pd_time_stats
        if ts is None:
            ts = TimeStats("prefill")
            req.pd_time_stats = ts
        ts.mark(name)

    def _extract_req_kv(self: Scheduler, req: Req):
        """Gather prefilled KV from the paged pool for ``req``.

        Returns shape ``(layer_num, padded_pages, page_size, ...)``.
        """

        req_to_token = self.req_to_token_pool.req_to_token
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        padded_pages = _pad_to_page_bucket(num_pages)
        page_id_source = req_to_token[
            req.req_pool_idx,
            : padded_pages * page_size : page_size,
        ]
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
        layer_kvs = _jit_gather_all_layers(layer_buffers, page_indices, gather_out_sharding)
        return jnp.stack(layer_kvs, axis=0)

    def _release_prefill_req_resources(self: Scheduler, req: Req) -> None:
        """Release prefill-side KV and request-pool resources."""

        from sgl_jax.srt.mem_cache.common import release_kv_cache

        release_kv_cache(req, self.tree_cache)

    def _record_prefill_transfer_failure(self, reason: str) -> None:
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_FAILURES_TOTAL,
            )

            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="prefill").inc()

    def _stream_prefill_req(self: Scheduler, req: Req) -> None:
        self.stream_output(
            [req],
            req.return_logprob,
            req.return_output_logprob_only,
        )

    def _abort_prefill_req(
        self: Scheduler,
        req: Req,
        message: str,
        *,
        metric_reason: str,
    ) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT

        self._record_prefill_transfer_failure(metric_reason)
        req.finished_reason = FINISH_ABORT(
            message,
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "PDTransferError",
        )
        req.output_ids = []
        self._stream_prefill_req(req)
        self._release_prefill_req_resources(req)

    def _on_prefill_transfer_terminal(
        self: Scheduler, req: Req, sender: JaxTransferKVSender
    ) -> None:
        try:
            if sender.poll() == KVPoll.SUCCESS:
                self._finish_prefill_only_success(req)
            else:
                self._finish_prefill_only_failure(req, sender)
        finally:
            self._pd_mark_time(req, "transfer_done")
            from sgl_jax.srt.disaggregation.req_time_stats import maybe_log_time_stats

            maybe_log_time_stats(
                req.pd_time_stats,
                req_id=req.rid,
                enabled=getattr(
                    self.server_args, "enable_request_time_stats_logging", False
                ),
            )
            sender.clear()
            self._release_prefill_req_resources(req)

    def _finish_prefill_only_success(self: Scheduler, req: Req) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_LENGTH

        req.finished_reason = FINISH_LENGTH(length=0)
        req.output_ids = []
        req.finished_len = 0
        self._stream_prefill_req(req)

    def _finish_prefill_only_failure(
        self: Scheduler, req: Req, sender: JaxTransferKVSender
    ) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT

        error_message = (
            f"Prefill transfer failed for req_id={req.rid!r} "
            f"bootstrap_room={req.bootstrap_room!r}"
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
        self._stream_prefill_req(req)

    def _maybe_log_prefill_extract_debug(self, req: Req, kv, **meta) -> None:
        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            kv_debug_enabled,
        )

        if not kv_debug_enabled(req.rid):
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
