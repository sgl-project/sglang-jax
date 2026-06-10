"""Decode-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVReceiver,
    PMetadata,
)

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


@partial(
    jax.jit,
    static_argnames=(
        "page_size",
        "kv_partition_axis",
        "attention_data_partition_axis",
        "mesh",
    ),
    donate_argnames=("kv_cache",),
)
def _jit_write_one_layer(
    layer_kv,
    loc,
    kv_cache,
    page_size,
    kv_partition_axis,
    attention_data_partition_axis,
    mesh,
):
    """One-layer PD KV write, wrapped in a module-level ``jax.jit``.

    ``update_fused_kv_cache_vectorized`` builds its ``jax.shard_map`` as a nested
    closure that is recreated on every call, so an eager call never hits JAX's
    compilation cache — the ~9s Pallas write kernel recompiles per layer per
    request and trips the scheduler watchdog. Wrapping the call in this stable
    module-level ``jax.jit`` makes the trace cache hit on shape + static args, so
    the kernel compiles once per write shape and is reused thereafter.
    """
    from sgl_jax.srt.mem_cache.memory_pool import _set_fused_kv_buffer

    total_tokens = loc.shape[0]
    fused_sharding = NamedSharding(
        mesh,
        PartitionSpec(
            attention_data_partition_axis,
            None,
            kv_partition_axis,
            None,
            None,
        ),
    )
    fused = jax.lax.reshape(
        layer_kv,
        (total_tokens, 1) + tuple(layer_kv.shape[2:]),
        out_sharding=fused_sharding,
    )
    return _set_fused_kv_buffer(
        fused_kv=fused,
        loc=loc,
        kv_cache=kv_cache,
        page_size=page_size,
        kv_partition_axis=kv_partition_axis,
        attention_data_partition_axis=attention_data_partition_axis,
        mesh=mesh,
    )


@dataclass
class DecodeBookkeeping:
    """Per-request decode-side state."""

    req_id: str
    req: Req
    receiver: JaxTransferKVReceiver | None = None
    # Indices into the paged pool reserved for this request.
    kv_indices: object | None = None
    # Whether the receiver has been initialized + poll started.
    started: bool = False


class DecodePreallocQueue:
    """Requests awaiting KV pull start. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(f"DecodePreallocQueue already tracks " f"req_id={entry.req_id!r}")
            self._entries[entry.req_id] = entry

    def pop_all(self) -> list[DecodeBookkeeping]:
        with self._lock:
            out = list(self._entries.values())
            self._entries.clear()
            return out

    def abort_matching(self, rid_prefix: str, abort_all: bool) -> list[DecodeBookkeeping]:
        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid in list(self._entries):
                if abort_all or rid.startswith(rid_prefix):
                    out.append(self._entries.pop(rid))
        return out


class DecodeTransferQueue:
    """Receivers in TRANSFERRING; polled each tick."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(f"DecodeTransferQueue already tracks " f"req_id={entry.req_id!r}")
            self._entries[entry.req_id] = entry

    def drain_terminal(self) -> list[DecodeBookkeeping]:
        """Return entries whose receiver reached SUCCESS or FAILED."""

        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid, entry in list(self._entries.items()):
                assert entry.receiver is not None
                state = entry.receiver.poll()
                if state in (KVPoll.SUCCESS, KVPoll.FAILED):
                    out.append(entry)
                    del self._entries[rid]
        return out

    def abort_matching(self, rid_prefix: str, abort_all: bool) -> list[DecodeBookkeeping]:
        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid in list(self._entries):
                if abort_all or rid.startswith(rid_prefix):
                    out.append(self._entries.pop(rid))
        return out


class SchedulerDisaggregationDecodeMixin:
    """Mixin for PD decode mode on Scheduler."""

    disagg_kv_manager: JaxTransferKVManager
    disagg_bootstrap_client: BootstrapClient
    disagg_prealloc_queue: DecodePreallocQueue
    disagg_transfer_queue: DecodeTransferQueue

    def event_loop_normal_disagg_decode(self: Scheduler) -> None:
        """Decode event loop."""

        wd = self.disagg_decode_watchdog
        wd.start()

        while True:
            wd.beat("recv_requests")
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)
            wd.beat("process_input_requests")
            self.process_input_requests_disagg_decode(recv_reqs)

            if self._engine_paused:
                continue

            wd.beat("process_decode_queue")
            self.process_decode_queue()

            wd.beat("get_next_batch")
            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                wd.beat("run_batch")
                result = self.run_batch(batch)
                wd.beat("process_batch_result")
                self.process_batch_result(batch, result)
            else:
                wd.beat("idle")
                # Skip check_memory / check_tree_cache for PD decode.
                self.new_token_ratio = self.init_new_token_ratio
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.last_batch = batch

    def _decode_backlog_snapshot(self: Scheduler) -> str:
        """One-line backlog snapshot for the watchdog stall report.

        Cheap reads only; never raises (the watchdog suppresses, but a
        clean string is more useful in the log than a swallowed error).
        """

        prealloc = len(self.disagg_prealloc_queue or ())
        transfer = len(self.disagg_transfer_queue or ())
        try:
            ns, nr = self.disagg_kv_manager.inflight_count()
        except Exception:
            ns, nr = (-1, -1)
        try:
            kv_avail = self.token_to_kv_pool_allocator.available_size()
        except Exception:
            kv_avail = -1
        running = len(self.running_batch.reqs) if self.running_batch is not None else 0
        return (
            f"prealloc_q={prealloc} transfer_q={transfer} "
            f"inflight_send={ns} inflight_recv={nr} "
            f"kv_avail={kv_avail} running_reqs={running}"
        )

    def process_input_requests_disagg_decode(self: Scheduler, recv_reqs) -> None:
        """Decode-mode request intake. PD reqs are extracted from
        waiting_queue and routed to the prealloc queue.
        """

        self.process_input_requests(recv_reqs)

        recv_pd_rids = {
            getattr(r, "rid", None)
            for r in recv_reqs
            if getattr(r, "bootstrap_room", None) is not None
        }
        if not recv_pd_rids:
            return

        pd_reqs = self._extract_pd_reqs_from_waiting_queue(recv_pd_rids)
        for req in pd_reqs:
            try:
                from sgl_jax.srt.disaggregation.common.metrics import time_phase

                self._pd_mark_time(req, "bootstrap_start")
                with time_phase("bootstrap", "decode"):
                    p_info = self.disagg_bootstrap_client.get_prefill_info(req.bootstrap_room)
                self._pd_mark_time(req, "bootstrap_done")
            except Exception:
                logger.exception(
                    "bootstrap lookup failed for req_id=%s "
                    "bootstrap_room=%s; releasing resources",
                    req.rid,
                    req.bootstrap_room,
                )
                self._record_decode_transfer_failure("bootstrap_lookup")
                self._abort_decode_request(req, "bootstrap_lookup")
                continue

            try:
                from sgl_jax.srt.disaggregation.bootstrap import check_prefill_compat

                check_prefill_compat(
                    p_info,
                    local_page_size=self.server_args.page_size,
                    local_kv_dtype=self.server_args.kv_cache_dtype,
                )
            except ValueError as exc:
                logger.error(
                    "prefill/decode KV layout mismatch for req_id=%s: %s",
                    req.rid,
                    exc,
                )
                self._record_decode_transfer_failure("config_mismatch")
                self._abort_decode_request(req, "config_mismatch")
                continue

            kv_indices = None
            try:
                self._pd_mark_time(req, "prealloc_entry")
                kv_indices = self._prealloc_decode_kv_indices(req)
                receiver = self.disagg_kv_manager.create_receiver(req.rid)
                spec = self._build_kv_spec_for_req(req)
                receiver.init(
                    PMetadata(
                        remote_addr=(f"{p_info['host']}:{p_info['transfer_port']}"),
                        uuid=req.disagg_transfer_id or req.rid,
                        specs={"kv": spec},
                        p_side_channel_host=str(p_info["host"]),
                        p_side_channel_port=int(p_info["side_channel_port"]),
                    )
                )
            except Exception:
                logger.exception(
                    "failed to set up KVReceiver for req_id=%s",
                    req.rid,
                )
                self._record_decode_transfer_failure("receiver_init")
                # Release any slots we allocated before the failure.
                if kv_indices is not None:
                    self._release_decode_kv_indices(kv_indices)
                self._abort_decode_request(req, "receiver_init")
                continue

            entry = DecodeBookkeeping(
                req_id=req.rid,
                req=req,
                receiver=receiver,
                kv_indices=kv_indices,
                started=True,
            )
            self._pd_mark_time(req, "transfer_entry")
            self.disagg_prealloc_queue.add(entry)

    def _extract_pd_reqs_from_waiting_queue(self: Scheduler, rids: set) -> list[Req]:
        """Extract PD reqs from waiting_queue by rid set."""

        out: list[Req] = []
        queue = self.waiting_queue
        survivors = []
        for req in queue:
            if req.rid in rids and req.bootstrap_room is not None:
                out.append(req)
            else:
                survivors.append(req)
        queue.clear()
        queue.extend(survivors)
        return out

    def process_decode_queue(self: Scheduler) -> None:
        """Drive prealloc -> transfer -> ready transitions."""

        for entry in self.disagg_prealloc_queue.pop_all():
            self.disagg_transfer_queue.add(entry)

        for entry in self.disagg_transfer_queue.drain_terminal():
            assert entry.receiver is not None
            state = entry.receiver.poll()
            if state == KVPoll.SUCCESS:
                try:
                    kv_result = entry.receiver.result
                    kv = kv_result["kv"] if kv_result else None
                    self._maybe_log_decode_pull_debug(entry.req, kv)
                    self._write_kv_to_pool(entry.req, entry.kv_indices, kv)
                    self._record_decode_transfer_bytes(kv)
                    self._enqueue_for_decode(entry.req)
                    self._pd_mark_time(entry.req, "first_token")
                    from sgl_jax.srt.disaggregation.req_time_stats import (
                        maybe_log_time_stats,
                    )

                    maybe_log_time_stats(
                        entry.req.pd_time_stats,
                        req_id=entry.req_id,
                        enabled=getattr(
                            self.server_args,
                            "enable_request_time_stats_logging",
                            False,
                        ),
                    )
                except Exception:
                    logger.exception(
                        "failed to install KV / enqueue decode for "
                        "req_id=%s; releasing resources",
                        entry.req_id,
                    )
                    if entry.kv_indices is not None:
                        self._release_decode_kv_indices(entry.kv_indices)
                    self._abort_decode_request(entry.req, "kv_writeback")
            else:
                logger.warning(
                    "KVReceiver for req_id=%s reached %s; releasing "
                    "resources and aborting request",
                    entry.req_id,
                    state.value,
                )
                self._record_decode_transfer_failure("receiver_terminal_failed")
                if entry.kv_indices is not None:
                    self._release_decode_kv_indices(entry.kv_indices)
                self._abort_decode_request(entry.req, "receiver_terminal_failed")

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
            role = getattr(self.server_args, "disaggregation_mode", "decode")
            ts = TimeStats(role)
            req.pd_time_stats = ts
        ts.mark(name)

    def _prealloc_decode_kv_indices(self: Scheduler, req: Req):
        """Reserve page-aligned KV slots in the paged pool for ``req``."""

        seqlen = len(req.origin_input_ids)
        allocator = self.token_to_kv_pool_allocator
        if allocator is None:
            return None
        page_size = allocator.page_size
        page_aligned = ((seqlen + page_size - 1) // page_size) * page_size
        return allocator.alloc(page_aligned)

    def _release_decode_kv_indices(self: Scheduler, kv_indices) -> None:
        """Release KV indices back to the allocator."""

        if kv_indices is None:
            return
        allocator = self.token_to_kv_pool_allocator
        if allocator is not None:
            try:
                allocator.free(kv_indices)
            except Exception:
                logger.exception("failed to free kv_indices=%r", kv_indices)

    def _build_kv_spec_for_req(self: Scheduler, req: Req) -> list[jax.ShapeDtypeStruct]:
        """Build per-layer ShapeDtypeStructs matching P's KV layout."""

        from sgl_jax.srt.disaggregation.prefill import _pad_to_page_bucket

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        padded_pages = _pad_to_page_bucket(num_pages)
        per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
        shape = (padded_pages, *per_layer_tail)
        sharding = kv_pool.kv_sharding
        return [
            jax.ShapeDtypeStruct(shape, kv_pool.dtype, sharding=sharding)
            for _ in range(kv_pool.layer_num)
        ]

    def _write_kv_to_pool(self: Scheduler, req: Req, kv_indices, kv: jax.Array) -> None:
        """Write pulled KV into the local paged pool (in place)."""

        if kv_indices is None:
            raise RuntimeError(
                f"_write_kv_to_pool: kv_indices is None for req "
                f"{req.rid!r}; allocator may have OOM'd"
            )

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_size = kv_pool.page_size
        seqlen = len(req.origin_input_ids)
        num_pages = (seqlen + page_size - 1) // page_size
        kv_indices_np = (
            np.asarray(kv_indices) if not isinstance(kv_indices, np.ndarray) else kv_indices
        )
        padded_pages = kv[0].shape[0]
        # page_ids_padded is only consumed by the debug verifier below, which is
        # a no-op unless SGL_JAX_PD_DEBUG_KV is set. The write itself is
        # token-level via ``loc``, so skip this numpy work on the production path.
        from sgl_jax.srt.disaggregation.debug_utils import kv_debug_enabled

        if kv_debug_enabled(req.rid):
            page_ids_np = kv_indices_np[::page_size] // page_size
            page_ids_np = page_ids_np[:num_pages]
            if num_pages < padded_pages:
                pad = np.full(
                    padded_pages - num_pages, page_ids_np[-1], dtype=page_ids_np.dtype
                )
                page_ids_padded = np.concatenate([page_ids_np, pad])
            else:
                page_ids_padded = page_ids_np
        else:
            page_ids_padded = None

        # Write via the same in-place Pallas kernel the forward path uses
        # (``update_fused_kv_cache_vectorized`` with ``input_output_aliases``).
        # Unlike ``.at[page_ids].set(...)`` — whose scatter XLA:TPU refuses to
        # alias, forcing a fresh full-layer (~884 MB) buffer that OOMs a v6e
        # single chip and wedges the decode loop — this kernel updates the pool
        # in place with a footprint proportional to the tokens written.
        # ``loc`` is per-token absolute pool slots; -1 marks padding tokens that
        # are skipped, so no tail-repeat payload duplication is needed.
        total_tokens = padded_pages * page_size
        loc_np = np.full(total_tokens, -1, dtype=np.int32)
        loc_np[:seqlen] = kv_indices_np[:seqlen]
        loc = jax.device_put(
            jnp.asarray(loc_np),
            NamedSharding(
                kv_pool.mesh, PartitionSpec(kv_pool.attention_data_partition_axis)
            ),
        )
        # Each layer write goes through the module-level ``_jit_write_one_layer``
        # so the Pallas write kernel compiles once per shape and caches, instead
        # of recompiling per layer per request (which trips the watchdog).
        for i, layer_id in enumerate(
            range(kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num)
        ):
            layer_idx = layer_id - kv_pool.start_layer
            kv_pool.kv_buffer[layer_idx] = _jit_write_one_layer(
                kv[i],
                loc,
                kv_pool.kv_buffer[layer_idx],
                page_size,
                kv_pool.kv_partition_axis,
                kv_pool.attention_data_partition_axis,
                kv_pool.mesh,
            )
        # Set prefix_indices to all-but-last so extend_input_len=1.
        valid_slots = kv_indices_np[:seqlen]
        if len(valid_slots) >= 1:
            req.prefix_indices = valid_slots[:-1]
        else:
            req.prefix_indices = valid_slots
        req.last_matched_prefix_len = len(req.prefix_indices)
        req._pd_skip_prefix_match = True
        req._pd_prealloc_kv_indices = kv_indices_np
        # Make sure fill_ids is set so the scheduler doesn't re-derive
        # an empty prefill chunk.
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)
        self._maybe_verify_decode_writeback_debug(req, kv_pool, page_ids_padded, kv)

    def _enqueue_for_decode(self: Scheduler, req: Req) -> None:
        """Put ``req`` into the scheduler's decode-ready queue."""

        if req not in self.waiting_queue:
            self.waiting_queue.append(req)

    def _release_decode_req_resources(self: Scheduler, req: Req) -> None:
        """Best-effort release of req_to_token_pool slot. Does NOT
        call cache_finished_req (req never went through prefill).
        """

        if req.req_pool_idx is None:
            return
        try:
            self.req_to_token_pool.free(req)
        except Exception:
            logger.exception(
                "failed to free req_to_token_pool slot %d for req_id=%s",
                req.req_pool_idx,
                req.rid,
            )

    def _abort_decode_request(self: Scheduler, req: Req, reason: str) -> None:
        """Release resources AND send AbortReq back to tokenizer."""

        self._release_decode_req_resources(req)
        try:
            from sgl_jax.srt.managers.io_struct import AbortReq

            abort_out = AbortReq(rid=req.rid)
            if self._comm_backend is not None:
                self._comm_backend.send_pyobj(abort_out)
            else:
                self.send_to_tokenizer.send_pyobj(abort_out)
        except Exception:
            logger.exception(
                "failed to send AbortReq for req_id=%s (reason=%s)",
                req.rid,
                reason,
            )

    def _record_decode_transfer_failure(self, reason: str) -> None:
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_FAILURES_TOTAL,
            )

            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()

    def _record_decode_transfer_bytes(self, kv) -> None:
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            if not kv:
                return
            leaves = jax.tree.leaves(kv)
            total = int(sum(int(x.nbytes) for x in leaves))
            PD_TRANSFER_BYTES_TOTAL.labels(direction="h2d", role="decode").inc(total)

    def _maybe_log_decode_pull_debug(self, req: Req, kv) -> None:
        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            kv_debug_enabled,
        )

        if not kv_debug_enabled(req.rid):
            return

        snapshot = build_kv_debug_snapshot(kv)
        logger.warning(
            "PD-KV-DEBUG decode_pull req_id=%s shape=%s dtype=%s "
            "sharding=%s digest=%s sample=%s",
            req.rid,
            snapshot.shape,
            snapshot.dtype,
            snapshot.sharding,
            snapshot.global_digest,
            snapshot.sample_page_digests(),
        )

    def _maybe_verify_decode_writeback_debug(self, req: Req, kv_pool, page_ids_padded, kv) -> None:
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.disaggregation.debug_utils import (
            build_kv_debug_snapshot,
            count_kv_debug_mismatches,
            find_first_kv_debug_mismatch,
            kv_debug_enabled,
        )
        from sgl_jax.srt.disaggregation.prefill import _jit_gather_all_layers

        if not kv_debug_enabled(req.rid):
            return

        page_ids_jax = jax.device_put(
            page_ids_padded,
            NamedSharding(kv_pool.mesh, PartitionSpec(None)),
        )
        gather_out_sharding = NamedSharding(
            kv_pool.mesh,
            PartitionSpec(None, *kv_pool.kv_sharding.spec[1:]),
        )
        layer_buffers = [
            kv_pool.get_kv_buffer(layer_id)
            for layer_id in range(
                kv_pool.start_layer,
                kv_pool.start_layer + kv_pool.layer_num,
            )
        ]
        readback = jnp.stack(
            _jit_gather_all_layers(
                layer_buffers,
                page_ids_jax,
                gather_out_sharding,
            ),
            axis=0,
        )

        expected = build_kv_debug_snapshot(kv)
        actual = build_kv_debug_snapshot(readback)
        mismatch_count = count_kv_debug_mismatches(expected, actual)
        first_mismatch = find_first_kv_debug_mismatch(expected, actual)

        logger.warning(
            "PD-KV-DEBUG decode_writeback req_id=%s expected_digest=%s "
            "readback_digest=%s mismatch_count=%d first_mismatch=%s "
            "expected_sample=%s readback_sample=%s page_ids=%s",
            req.rid,
            expected.global_digest,
            actual.global_digest,
            mismatch_count,
            first_mismatch,
            expected.sample_page_digests(),
            actual.sample_page_digests(),
            page_ids_padded.tolist(),
        )
