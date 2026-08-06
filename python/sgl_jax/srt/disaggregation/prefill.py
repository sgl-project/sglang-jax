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

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll, KVSender
from sgl_jax.srt.disaggregation.base.transfer import (
    PrefillTransferContext,
    TransferBackend,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _batch_reqs(batch) -> tuple[Req, ...]:
    if batch is None:
        return ()
    reqs_info = getattr(batch, "reqs_info", None)
    if reqs_info is None:
        return tuple(getattr(batch, "reqs", ()) or ())
    return tuple(req for info in reqs_info for req in (info.reqs or ()))


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


def _globalize_rank_local_page_ids(
    page_ids,
    *,
    dp_rank: int,
    dp_size: int,
    total_pages: int,
):
    """Preserve each DP shard's padding page when indexing the global page axis."""

    import numpy as np

    ids = np.asarray(page_ids)
    dp_rank = int(dp_rank)
    dp_size = int(dp_size)
    total_pages = int(total_pages)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank={dp_rank} is outside [0, {dp_size})")
    if total_pages <= 0 or total_pages % dp_size:
        raise ValueError(
            f"global KV page count {total_pages} is not divisible by dp_size={dp_size}"
        )
    pages_per_rank = total_pages // dp_size
    if np.any(ids < 0) or np.any(ids >= pages_per_rank):
        raise ValueError(
            f"rank-local KV page IDs must be in [0, {pages_per_rank}); got {ids.tolist()}"
        )
    return ids + dp_rank * pages_per_rank


@partial(jax.jit, static_argnames=("out_sharding",))
def _jit_gather_one_layer(buf, page_indices, out_sharding):
    """Gather ``page_indices`` from a single per-layer KV buffer.

    Gathered per layer to cap the XLA compile-time HBM footprint (~1.2 GB
    per layer regardless of layer count).
    """
    return buf.at[page_indices].get(out_sharding=out_sharding)


def _jit_gather_all_layers(buffers, page_indices, out_sharding):
    """Gather ``page_indices`` from every per-layer KV buffer.

    Dispatches an independent per-layer jit call so each compiles separately.
    """
    return [_jit_gather_one_layer(buf, page_indices, out_sharding) for buf in buffers]


def _global_to_local_shard(arr: jax.Array) -> jax.Array:
    """View this host's addressable shards of a globally-sharded ``arr`` as a
    fully-addressable array on a 1-D local mesh. Assumes ``arr`` is sharded
    along exactly one dimension (the KV head axis). Zero-copy.
    """
    import numpy as _np
    from jax.sharding import Mesh as _Mesh
    from jax.sharding import NamedSharding as _NamedSharding
    from jax.sharding import PartitionSpec as _P

    spec = arr.sharding.spec
    sharded_dims = [i for i, s in enumerate(spec) if s is not None]
    if len(sharded_dims) != 1:
        raise ValueError(f"_global_to_local_shard expects exactly one sharded dim, got spec={spec}")
    sd = sharded_dims[0]
    ldev = jax.local_devices()
    shards = [s.data for s in arr.addressable_shards]
    lshape = tuple(
        shards[0].shape[i] * len(ldev) if i == sd else shards[0].shape[i] for i in range(arr.ndim)
    )
    lmesh = _Mesh(_np.asarray(ldev), ("_local",))
    lspec = _P(*("_local" if i == sd else None for i in range(arr.ndim)))
    return jax.make_array_from_single_device_arrays(lshape, _NamedSharding(lmesh, lspec), shards)


def local_kv_spec_for_pool(kv_pool, layer_num: int, padded_pages: int) -> jax.ShapeDtypeStruct:
    """Build the ShapeDtypeStruct that D should pull on a multi-host process:
    this host's 1/nproc slice of the stacked KV, on a 1-D local mesh.
    """
    import numpy as _np
    from jax.sharding import Mesh as _Mesh
    from jax.sharding import NamedSharding as _NamedSharding
    from jax.sharding import PartitionSpec as _P

    pool_pspec = kv_pool.kv_sharding.spec
    per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
    gshape = (layer_num, padded_pages) + per_layer_tail
    gspec = (None, None) + tuple(pool_pspec[1:])
    sharded_dims = [i for i, s in enumerate(gspec) if s is not None]
    if len(sharded_dims) != 1:
        raise ValueError(f"expected one sharded dim in KV spec, got {gspec}")
    sd = sharded_dims[0]
    ldev = jax.local_devices()
    nproc = jax.process_count()
    lshape = tuple(gshape[i] // nproc if i == sd else gshape[i] for i in range(len(gshape)))
    lmesh = _Mesh(_np.asarray(ldev), ("_local",))
    lspec = _P(*("_local" if i == sd else None for i in range(len(gshape))))
    return jax.ShapeDtypeStruct(lshape, kv_pool.dtype, sharding=_NamedSharding(lmesh, lspec))


@dataclass
class PrefillBookkeeping:
    """Per-request prefill-side state tracked by the Mixin."""

    req_id: str
    sender: KVSender
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
        sender: KVSender,
        on_terminal=None,
    ) -> None:
        with self._lock:
            if req_id in self._entries:
                raise ValueError(f"PrefillBootstrapQueue already tracks req_id={req_id!r}")
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

    def cancel_matching(self, rid_prefix: str, abort_all: bool) -> list[PrefillBookkeeping]:
        """Return matching entries without releasing their owned KV pages."""

        with self._lock:
            return [
                entry
                for req_id, entry in self._entries.items()
                if abort_all or req_id.startswith(rid_prefix)
            ]


class SchedulerDisaggregationPrefillMixin:
    """Mixin for PD prefill mode on Scheduler."""

    disagg_kv_manager: TransferBackend
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
                batch_reqs = _batch_reqs(batch)
                for req in batch_reqs:
                    if req.bootstrap_room is not None:
                        self._pd_mark_time(req, "forward_start")
                result = self.run_batch(batch)
                self.process_prefill_chunk(batch, result)
            else:
                self.send_kv_chunk()
                self.new_token_ratio = self.init_new_token_ratio
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.send_kv_chunk()
            # PD reqs are finished and released inside process_prefill_chunk;
            # do not merge them into running_batch.
            batch_reqs = _batch_reqs(batch)
            self.last_batch = (
                None if batch and any(r.bootstrap_room is not None for r in batch_reqs) else batch
            )

    def process_prefill_chunk(self: Scheduler, batch, result) -> None:
        """Extract KV for PD reqs and hand off to sender."""

        batch_reqs = _batch_reqs(batch)
        pd_reqs = [req for req in batch_reqs if req.bootstrap_room is not None]
        if not pd_reqs:
            self.process_batch_result(batch, result)
            return

        for req in pd_reqs:
            self._pd_mark_time(req, "forward_done")

        self.set_next_batch_sampling_info_done(batch)

        chunked_now = tuple(r for r in getattr(self, "chunked_reqs", ()) if r is not None)
        ready_to_transfer = [
            req
            for req in pd_reqs
            if not any(req is chunked_req for chunked_req in chunked_now)
            and req.rid not in self.disagg_prefill_queue._entries
        ]
        if ready_to_transfer:
            kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
            self.disagg_kv_manager.prepare_prefill_batch(kv_pool.kv_buffer)
        for req in batch_reqs:
            if req.bootstrap_room is None:
                continue
            req_id = req.rid
            is_mid_chunk = any(req is cr for cr in chunked_now)
            if is_mid_chunk:
                # Still mid-chunk: KV is incomplete, and releasing the
                # req_pool_idx here would leak the slot the next chunk
                # round re-allocates. Both engines transfer only after the final
                # chunk; transfer/forward overlap is intentionally out of scope.
                assert req.is_chunked > 0
                req.is_chunked -= 1
                continue
            if req_id in self.disagg_prefill_queue._entries:
                continue
            try:
                req.disagg_host_buffer_id = self.disagg_kv_manager.reserve_prefill_buffer(
                    getattr(req, "disagg_host_buffer_id", None)
                )
                transfer = self.disagg_kv_manager.start_prefill(
                    PrefillTransferContext(
                        req_id=req_id,
                        transfer_id=req.disagg_transfer_id or req_id,
                        bootstrap_room=req.bootstrap_room,
                        dp_rank=int(req.dp_rank),
                        buffer_id=req.disagg_host_buffer_id,
                        payload_factory=lambda req_obj=req: {"kv": self._extract_req_kv(req_obj)},
                        block_ids_factory=lambda req_obj=req: self._extract_req_block_ids(req_obj),
                        on_payload=lambda payload, req_obj=req: (
                            self._maybe_log_prefill_extract_debug(
                                req_obj,
                                payload["kv"],
                                use_d2h_staging=self.disagg_use_d2h_staging,
                            )
                        ),
                        on_ready=lambda req_obj=req: self._pd_mark_time(req_obj, "transfer_start"),
                    )
                )
            except Exception as exc:
                logger.exception("prefill transfer setup failed for req_id=%s", req_id)
                self._abort_prefill_req(
                    req,
                    f"Prefill sender failed for req_id={req_id!r}: {exc}",
                    metric_reason="sender_init",
                )
                continue

            if transfer.release_device_kv:
                self._release_prefill_kv_pool(req)

            def _on_terminal(req_obj=req, sender_obj=transfer.sender):
                self._on_prefill_transfer_terminal(req_obj, sender_obj)

            self.disagg_prefill_queue.add(req_id, transfer.sender, on_terminal=_on_terminal)

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
            role = getattr(self.server_args, "disaggregation_mode", "prefill")
            ts = TimeStats(role)
            req.pd_time_stats = ts
        ts.mark(name)

    def _extract_req_block_ids(self: Scheduler, req: Req) -> list[int]:
        from sgl_jax.srt.disaggregation.base.transfer import slots_to_page_ids

        req_to_token = self.req_to_token_pool.req_to_token
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        slot_source = req_to_token[
            req.req_pool_idx,
            :seqlen,
        ]
        return list(slots_to_page_ids(slot_source, page_size, seqlen))

    def _extract_req_kv(self: Scheduler, req: Req):
        """Gather prefilled KV from the paged pool for ``req``.

        Returns a per-layer list of ``(padded_pages, page_size, ...)`` arrays.
        """

        req_to_token = self.req_to_token_pool.req_to_token
        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        padded_pages = _pad_to_page_bucket(num_pages)
        # Only the first num_pages slots of req_to_token are written for this
        # req; the bucket-padding region holds whatever the previous occupant
        # left. Slice the real pages and zero-pad so device_put's cross-process
        # assert_equal sees identical indices on every NP.
        page_id_source = req_to_token[
            req.req_pool_idx,
            : num_pages * page_size : page_size,
        ]
        import numpy as _np
        from jax.sharding import NamedSharding as _NamedSharding
        from jax.sharding import PartitionSpec as _P

        page_ids = _np.asarray(page_id_source) // page_size
        if padded_pages > num_pages:
            page_ids = _np.concatenate(
                [page_ids, _np.zeros(padded_pages - num_pages, dtype=page_ids.dtype)]
            )
        layer_buffers = [
            kv_pool.get_kv_buffer(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        if not layer_buffers:
            raise ValueError("cannot gather debug KV from an empty layer range")
        page_ids = _globalize_rank_local_page_ids(
            page_ids,
            dp_rank=int(getattr(req, "dp_rank", 0) or 0),
            dp_size=int(getattr(kv_pool, "dp_size", 1)),
            total_pages=int(layer_buffers[0].shape[0]),
        )
        idx_sharding = _NamedSharding(kv_pool.mesh, _P(None))
        page_indices = jax.device_put(page_ids, idx_sharding)
        # Page indices are replicated; preserve pool sharding on the remaining axes.
        pool_pspec = kv_pool.kv_sharding.spec
        gather_pspec = _P(None, *pool_pspec[1:])
        gather_out_sharding = _NamedSharding(kv_pool.mesh, gather_pspec)
        layer_kvs = _jit_gather_all_layers(layer_buffers, page_indices, gather_out_sharding)
        if jax.process_count() > 1:
            # Multi-host: expose only this host's TP shard as a fully-addressable
            # local-mesh array; each P host registers its 1/nproc slice and the
            # matching D host (same jax_process_index) pulls exactly that slice.
            stacked = jnp.stack(layer_kvs, axis=0)
            stacked.block_until_ready()
            return _global_to_local_shard(stacked)
        # Single-host: return the per-layer list. The D2H staging path
        # (copy_from_device) consumes a ``list[jax.Array]``.
        return layer_kvs

    def _release_prefill_kv_pool(self: Scheduler, req: Req) -> None:
        """Release the prefill device KV cache + request-pool slot.

        Idempotent: ``release_kv_cache`` no-ops once ``req.req_pool_idx`` is
        cleared, so calling this both at staged D2H completion and again at the
        terminal callback is safe.
        """

        from sgl_jax.srt.mem_cache.common import release_kv_cache

        release_kv_cache(req, self.tree_cache)

    def _release_prefill_req_resources(self: Scheduler, req: Req) -> None:
        """Release prefill-side KV and request-pool resources."""

        self._release_prefill_kv_pool(req)
        self._release_prefill_host_buffer(req)

    def _release_prefill_host_buffer(self: Scheduler, req: Req) -> None:
        buffer_id = getattr(req, "disagg_host_buffer_id", None)
        if buffer_id is None:
            return
        req.disagg_host_buffer_id = None
        mgr = getattr(self, "disagg_kv_manager", None)
        pool = mgr.host_pool if mgr is not None else None
        if pool is not None:
            with suppress(Exception):
                pool.release(buffer_id)

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
        self: Scheduler,
        req: Req,
        sender: KVSender,
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
                enabled=getattr(self.server_args, "enable_request_time_stats_logging", False),
            )
            sender.clear()
            self._release_prefill_req_resources(req)

    def _finish_prefill_only_success(self: Scheduler, req: Req) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_LENGTH

        req.finished_reason = FINISH_LENGTH(length=0)
        req.output_ids = []
        req.finished_len = 0
        self._stream_prefill_req(req)

    def _finish_prefill_only_failure(self: Scheduler, req: Req, sender: KVSender) -> None:
        from sgl_jax.srt.managers.schedule_batch import FINISH_ABORT

        error_message = (
            f"Prefill transfer failed for req_id={req.rid!r} bootstrap_room={req.bootstrap_room!r}"
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
