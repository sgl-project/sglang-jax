"""Decode-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll, KVReceiver
from sgl_jax.srt.disaggregation.base.transfer import (
    AdmissionState,
    DecodeTransferContext,
    TransferBackend,
)
from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient, PrefillInfoCache
from sgl_jax.srt.disaggregation.common.capacity import per_rank_inflight_limit
from sgl_jax.srt.mem_cache.memory_pool import write_kv_layer

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


def _batch_req_count(batch) -> int:
    if batch is None:
        return 0
    reqs_info = getattr(batch, "reqs_info", None)
    if reqs_info is None:
        return len(getattr(batch, "reqs", ()) or ())
    return sum(len(info.reqs) if info.reqs else 0 for info in reqs_info)


def _batch_req_count_for_dp(batch, dp_rank: int) -> int:
    if batch is None:
        return 0
    reqs_info = getattr(batch, "reqs_info", None)
    if reqs_info is not None:
        info = reqs_info[dp_rank]
        return len(info.reqs) if info.reqs else 0
    return sum(
        1
        for req in (getattr(batch, "reqs", ()) or ())
        if (getattr(req, "dp_rank", None) if getattr(req, "dp_rank", None) is not None else 0)
        == dp_rank
    )


def _request_dp_rank(req, dp_size: int, *, prefill: bool = False) -> int:
    field = "disagg_prefill_dp_rank" if prefill else "dp_rank"
    value = getattr(req, field, None)
    if value is None:
        if dp_size == 1:
            return 0
        raise ValueError(f"PD request {req.rid!r} is missing {field} for dp_size={dp_size}")
    value = int(value)
    if not 0 <= value < dp_size:
        raise ValueError(f"PD request {req.rid!r} has {field}={value} outside [0, {dp_size})")
    return value


@dataclass
class DecodeBookkeeping:
    """Per-request decode-side state."""

    req_id: str
    req: Req
    receiver: KVReceiver | None = None
    # Indices into the paged pool reserved for this request.
    kv_indices: object | None = None
    # Whether the receiver has been initialized + poll started.
    started: bool = False
    # Set by _drain_transfer_queue_synced on multi-host so downstream
    # does not re-poll (a poll() that raised would re-raise and desync).
    synced_state: KVPoll | None = None
    # Prefill-side info from bootstrap, stashed at intake so KV alloc +
    # receiver setup can be deferred to the capacity-gated admission step.
    p_info: dict | None = None
    cancelled: bool = False
    created_at: float = field(default_factory=time.monotonic)


class DecodePreallocQueue:
    """PD reqs awaiting capacity-gated KV alloc. FIFO, thread-safe.

    Entries enter at intake holding only ``p_info`` (no KV indices, no
    receiver). The decode loop's admission gate pops them in FIFO order
    once the paged pool has room; reqs that don't fit stay queued and are
    retried next tick (deferral, never abort).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, DecodeBookkeeping] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def add(self, entry: DecodeBookkeeping) -> None:
        with self._lock:
            if entry.req_id in self._entries:
                raise ValueError(f"DecodePreallocQueue already tracks req_id={entry.req_id!r}")
            self._entries[entry.req_id] = entry

    def items_fifo(self) -> list[DecodeBookkeeping]:
        """FIFO snapshot for the admission gate (does not remove)."""

        with self._lock:
            return list(self._entries.values())

    def remove(self, req_id: str) -> None:
        """Drop an admitted (or failed) entry by id."""

        with self._lock:
            self._entries.pop(req_id, None)

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
                raise ValueError(f"DecodeTransferQueue already tracks req_id={entry.req_id!r}")
            self._entries[entry.req_id] = entry

    def count_by_rank(self, dp_size: int) -> list[int]:
        counts = [0] * dp_size
        with self._lock:
            for entry in self._entries.values():
                counts[_request_dp_rank(entry.req, dp_size)] += 1
        return counts

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

    def cancel_matching(self, rid_prefix: str, abort_all: bool) -> list[DecodeBookkeeping]:
        """Mark matching transfers cancelled while retaining their KV pages."""

        out: list[DecodeBookkeeping] = []
        with self._lock:
            for rid, entry in self._entries.items():
                if abort_all or rid.startswith(rid_prefix):
                    entry.cancelled = True
                    out.append(entry)
        return out


class SchedulerDisaggregationDecodeMixin:
    """Mixin for PD decode mode on Scheduler."""

    disagg_kv_manager: TransferBackend
    disagg_bootstrap_client: BootstrapClient
    disagg_prefill_info_cache: PrefillInfoCache
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
            kv_avail = [
                self.token_to_kv_pool_allocator.available_size(dp_rank)
                for dp_rank in range(self.dp_size)
            ]
        except Exception:
            kv_avail = -1
        running = _batch_req_count(self.running_batch)
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
        new_pd_reqs = self._extract_pd_reqs_from_waiting_queue(recv_pd_rids) if recv_pd_rids else []

        # Retry reqs deferred on a previous tick because no prefill was
        # registered yet (bootstrap cache miss). They go ahead of new reqs so
        # FIFO ordering is preserved across deferrals.
        pending = self._pd_pending_bootstrap
        self._pd_pending_bootstrap = []
        pd_reqs = pending + new_pd_reqs
        if not pd_reqs:
            return

        for req in pd_reqs:
            try:
                from sgl_jax.srt.disaggregation.common.metrics import time_phase

                prefill_dp_rank = _request_dp_rank(req, self.dp_size, prefill=True)
                self._pd_mark_time(req, "bootstrap_start")
                with time_phase("bootstrap", "decode"):
                    if jax.process_count() > 1:
                        # Multi-host caches the matched peer after the first
                        # lookup, so this does no per-request network I/O.
                        p_info = self._pick_prefill_peer_for_this_host(prefill_dp_rank)
                    else:
                        # Local cache resolution (sglang-style): a warm cache
                        # does zero network I/O, so this no longer blocks the
                        # event loop.
                        p_info = self.disagg_prefill_info_cache.pick_for_room(
                            req.bootstrap_room, prefill_dp_rank
                        )
                self._pd_mark_time(req, "bootstrap_done")
            except Exception:
                logger.exception(
                    "bootstrap lookup failed for req_id=%s bootstrap_room=%s; releasing resources",
                    req.rid,
                    req.bootstrap_room,
                )
                self._record_decode_transfer_failure("bootstrap_lookup")
                self._abort_decode_request(req, "bootstrap_lookup")
                continue

            if p_info is None:
                # No prefill registered yet (or the rate-limited refresh was
                # skipped this tick). Defer and retry next tick — never abort.
                self._pd_pending_bootstrap.append(req)
                continue

            req.disagg_peer_process_index = int(p_info.get("jax_process_index", 0))

            try:
                from sgl_jax.srt.disaggregation.bootstrap import (
                    check_prefill_compat,
                    resolve_kv_dtype_name,
                )

                local_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                manager = getattr(self, "disagg_kv_manager", None)
                check_prefill_compat(
                    p_info,
                    local_page_size=self.server_args.page_size,
                    local_kv_dtype=resolve_kv_dtype_name(local_kv_pool.dtype),
                    expected_transfer_engine=(None if manager is None else manager.engine_name),
                    expected_dp_rank=prefill_dp_rank,
                    expected_dp_size=self.dp_size,
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

            # KV alloc + receiver setup are deferred to the capacity-gated
            # admission step (process_decode_queue). At intake the entry holds
            # only p_info and consumes no paged-pool slots, so a backlog of
            # waiting reqs cannot exhaust decode KV cache.
            entry = DecodeBookkeeping(
                req_id=req.rid,
                req=req,
                p_info=p_info,
            )
            self._pd_mark_time(req, "prealloc_entry")
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

        self._admit_decode_prealloc()

        for entry in self._drain_transfer_queue_synced():
            assert entry.receiver is not None
            state = entry.synced_state
            if state is None:
                try:
                    state = entry.receiver.poll()
                except Exception:
                    logger.exception("receiver.poll() raised for req_id=%s", entry.req_id)
                    state = KVPoll.FAILED
            if entry.cancelled:
                if entry.kv_indices is not None:
                    self._release_decode_kv_indices(entry.kv_indices, entry.req.dp_rank)
                continue
            if state == KVPoll.SUCCESS:
                try:
                    entry.receiver.commit(
                        lambda kv, req=entry.req, indices=entry.kv_indices: (
                            self._install_received_kv(req, indices, kv)
                        )
                    )
                    self._set_decode_bookkeeping(entry.req, entry.kv_indices)
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
                        "failed to install KV / enqueue decode for req_id=%s; releasing resources",
                        entry.req_id,
                    )
                    if entry.kv_indices is not None:
                        self._release_decode_kv_indices(entry.kv_indices, entry.req.dp_rank)
                    self._abort_decode_request(
                        entry.req,
                        "kv_writeback",
                        cleanup_transfer=False,
                    )
            else:
                logger.warning(
                    "KVReceiver for req_id=%s reached %s; releasing resources and aborting request",
                    entry.req_id,
                    state.value,
                )
                self._record_decode_transfer_failure("receiver_terminal_failed")
                if entry.kv_indices is not None:
                    self._release_decode_kv_indices(entry.kv_indices, entry.req.dp_rank)
                self._abort_decode_request(
                    entry.req,
                    "receiver_terminal_failed",
                    cleanup_transfer=False,
                )

    def _pick_prefill_peer_for_this_host(self: Scheduler, dp_rank: int) -> dict[str, object]:
        """Multi-host: find the P host whose jax_process_index matches ours.
        That host's local KV shard is exactly the slice this D host needs.
        Requires P/D to have the same nproc (same-TP constraint).
        """
        peers = getattr(self, "_disagg_prefill_peers", None)
        if peers is None:
            peers = self._disagg_prefill_peers = {}
        if dp_rank in peers:
            return peers[dp_rank]
        my_pidx = jax.process_index()
        my_nproc = jax.process_count()
        all_p = self.disagg_bootstrap_client.list_prefills()
        for p in all_p:
            if (
                int(p.get("jax_process_index", -1)) == my_pidx
                and int(p.get("system_dp_rank", 0)) == dp_rank
            ):
                if int(p.get("jax_process_count", 0)) != my_nproc:
                    raise RuntimeError(
                        f"P/D process_count mismatch: P={p.get('jax_process_count')} "
                        f"D={my_nproc}. Per-host shard transfer requires same nproc."
                    )
                peers[dp_rank] = p
                return p
        raise RuntimeError(
            f"no prefill host with jax_process_index={my_pidx}, dp_rank={dp_rank} "
            "registered (got "
            f"{[(p.get('host'), p.get('jax_process_index'), p.get('system_dp_rank')) for p in all_p]})"
        )

    def _drain_transfer_queue_synced(self: Scheduler) -> list[DecodeBookkeeping]:
        """On multi-host, only drain entries whose receiver has reached a
        terminal state on every NP — _write_kv_to_pool issues a cross-host
        jit and all NPs must enter it for the same set of reqs.
        """
        if jax.process_count() <= 1:
            return self.disagg_transfer_queue.drain_terminal()
        from sgl_jax.srt.disaggregation.common.multihost_sync import (
            synced_terminal_rooms,
        )

        with self.disagg_transfer_queue._lock:
            entries = list(self.disagg_transfer_queue._entries.values())
        success, failed = synced_terminal_rooms(
            entries,
            poll_fn=lambda e: e.receiver.poll(),
            room_fn=lambda e: getattr(e.req, "bootstrap_room", None),
        )
        if not success and not failed:
            return []
        out: list[DecodeBookkeeping] = []
        with self.disagg_transfer_queue._lock:
            for rid, e in list(self.disagg_transfer_queue._entries.items()):
                room = getattr(e.req, "bootstrap_room", None)
                if room in failed:
                    self.disagg_transfer_queue._entries.pop(rid, None)
                    with suppress(Exception):
                        e.receiver.fail(reason="peer_np_failed")
                    e.synced_state = KVPoll.FAILED
                    out.append(e)
                elif room in success:
                    self.disagg_transfer_queue._entries.pop(rid, None)
                    e.synced_state = KVPoll.SUCCESS
                    out.append(e)
        return out

    def _admit_decode_prealloc(self: Scheduler) -> None:
        """Admit transfers without consuming non-retractable Decode headroom."""

        allocator = self.token_to_kv_pool_allocator
        if allocator is None:
            return

        page_size = allocator.page_size
        reserved_per = self.server_args.disaggregation_num_reserved_decode_tokens
        max_inflight = self.server_args.disaggregation_max_inflight_transfers
        per_rank_inflight = per_rank_inflight_limit(max_inflight, self.dp_size)
        n_transfer = len(self.disagg_transfer_queue)
        admitted = 0
        transfer_per_dp = self.disagg_transfer_queue.count_by_rank(self.dp_size)
        admitted_per_dp = [0] * self.dp_size
        capacity_blocked_ranks: set[int] = set()

        for entry in self.disagg_prealloc_queue.items_fifo():
            decode_dp_rank = _request_dp_rank(entry.req, self.dp_size)
            prefill_dp_rank = _request_dp_rank(entry.req, self.dp_size, prefill=True)
            # Pulled KV buffers remain outside paged-pool accounting until scatter.
            if max_inflight > 0 and (n_transfer + admitted) >= max_inflight:
                break
            # Preserve FIFO within a blocked rank without stalling other ranks.
            if decode_dp_rank in capacity_blocked_ranks:
                continue
            if (
                per_rank_inflight > 0
                and transfer_per_dp[decode_dp_rank] + admitted_per_dp[decode_dp_rank]
                >= per_rank_inflight
            ):
                continue
            seqlen = len(entry.req.origin_input_ids)
            page_aligned = ((seqlen + page_size - 1) // page_size) * page_size
            n_running = _batch_req_count_for_dp(self.running_batch, decode_dp_rank)
            reserved = reserved_per * (
                n_running + transfer_per_dp[decode_dp_rank] + admitted_per_dp[decode_dp_rank]
            )
            if page_aligned + reserved > allocator.available_size(decode_dp_rank):
                capacity_blocked_ranks.add(decode_dp_rank)
                continue

            kv_indices = allocator.alloc(page_aligned, dp_rank=decode_dp_rank)
            if kv_indices is None:
                capacity_blocked_ranks.add(decode_dp_rank)
                continue

            try:
                admission = self.disagg_kv_manager.try_start_decode(
                    DecodeTransferContext(
                        req_id=entry.req.rid,
                        transfer_id=entry.req.disagg_transfer_id or entry.req.rid,
                        bootstrap_room=entry.req.bootstrap_room,
                        decode_dp_rank=decode_dp_rank,
                        prefill_dp_rank=prefill_dp_rank,
                        peer_info=entry.p_info or {},
                        kv_indices=kv_indices,
                        page_size=page_size,
                        prompt_tokens=len(entry.req.origin_input_ids),
                        spec_factory=lambda req=entry.req: self._build_kv_spec_for_req(req),
                        direct_commit=lambda expected, req=entry.req, indices=kv_indices: (
                            self._commit_direct_received_kv(req, indices, expected)
                        ),
                    )
                )
            except Exception:
                logger.exception(
                    "failed to set up KVReceiver for req_id=%s",
                    entry.req.rid,
                )
                self._record_decode_transfer_failure("receiver_init")
                self._release_decode_kv_indices(kv_indices, decode_dp_rank)
                self.disagg_prealloc_queue.remove(entry.req_id)
                self._abort_decode_request(entry.req, "receiver_init")
                continue

            if admission.state == AdmissionState.DEFERRED:
                self._release_decode_kv_indices(kv_indices, decode_dp_rank)
                timeout_s = self.server_args.disaggregation_pull_timeout_seconds
                if timeout_s > 0 and time.monotonic() - entry.created_at >= timeout_s:
                    self.disagg_prealloc_queue.remove(entry.req_id)
                    self._record_decode_transfer_failure("metadata_timeout")
                    self._abort_decode_request(entry.req, "metadata_timeout")
                continue

            receiver = admission.receiver
            if receiver is None:
                raise RuntimeError("admitted transfer is missing a receiver")

            entry.kv_indices = kv_indices
            entry.receiver = receiver
            entry.started = True
            self._pd_mark_time(entry.req, "transfer_entry")
            self.disagg_prealloc_queue.remove(entry.req_id)
            self.disagg_transfer_queue.add(entry)
            admitted += 1
            admitted_per_dp[decode_dp_rank] += 1

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

    def _release_decode_kv_indices(self: Scheduler, kv_indices, dp_rank: int | None) -> None:
        """Release KV indices back to the allocator."""

        if kv_indices is None:
            return
        allocator = self.token_to_kv_pool_allocator
        if allocator is not None:
            try:
                rank = 0 if dp_rank is None and self.dp_size == 1 else int(dp_rank)
                allocator.free(kv_indices, dp_rank=rank)
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
        if jax.process_count() > 1:
            from sgl_jax.srt.disaggregation.prefill import local_kv_spec_for_pool

            return local_kv_spec_for_pool(kv_pool, kv_pool.layer_num, padded_pages)
        per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
        shape = (padded_pages, *per_layer_tail)
        sharding = kv_pool.kv_sharding
        return [
            jax.ShapeDtypeStruct(shape, kv_pool.dtype, sharding=sharding)
            for _ in range(kv_pool.layer_num)
        ]

    def _install_received_kv(self: Scheduler, req: Req, kv_indices, kv) -> None:
        self._maybe_log_decode_pull_debug(req, kv)
        self._write_kv_to_pool(req, kv_indices, kv)
        self._record_decode_transfer_bytes(kv)

    def _commit_direct_received_kv(
        self: Scheduler,
        req: Req,
        kv_indices,
        expected_debug: dict[str, object] | None,
    ) -> None:
        """Record a backend that wrote the destination KV pages directly."""

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        num_pages = (len(req.origin_input_ids) + kv_pool.page_size - 1) // kv_pool.page_size
        total = 0
        for layer in kv_pool.kv_buffer:
            total += num_pages * (int(layer.nbytes) // int(layer.shape[0]))
        with suppress(Exception):
            from sgl_jax.srt.disaggregation.common.metrics import (
                PD_TRANSFER_BYTES_TOTAL,
            )

            PD_TRANSFER_BYTES_TOTAL.labels(direction="h2d", role="decode").inc(total)
        self._maybe_verify_direct_receive_debug(req, kv_indices, expected_debug)

    def _write_kv_to_pool(self: Scheduler, req: Req, kv_indices, kv: jax.Array) -> None:
        """Write pulled KV into the local paged pool (in place)."""

        if kv_indices is None:
            raise RuntimeError(
                f"_write_kv_to_pool: kv_indices is None for req "
                f"{req.rid!r}; allocator may have OOM'd"
            )

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        if jax.process_count() > 1 and kv.is_fully_addressable:
            # Pulled KV is this host's local shard on a 1-D local mesh.
            # Assemble it into the global pool sharding (zero-copy: each NP
            # contributes its own addressable_shards).
            pool_pspec = kv_pool.kv_sharding.spec
            stacked_spec = PartitionSpec(None, None, *pool_pspec[1:])
            gsh = NamedSharding(kv_pool.mesh, stacked_spec)
            per_layer_tail = kv_pool.kv_buffer[0].shape[1:]
            gshape = (kv.shape[0], kv.shape[1]) + per_layer_tail
            kv = jax.make_array_from_single_device_arrays(
                gshape, gsh, [s.data for s in kv.addressable_shards]
            )
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
                pad = np.full(padded_pages - num_pages, page_ids_np[-1], dtype=page_ids_np.dtype)
                page_ids_padded = np.concatenate([page_ids_np, pad])
            else:
                page_ids_padded = page_ids_np
        else:
            page_ids_padded = None

        # Write via the in-place Pallas kernel (``update_fused_kv_cache_vectorized``
        # with ``input_output_aliases``), so the footprint scales with the tokens
        # written. ``loc`` is per-token absolute pool slots; -1 marks padding
        # tokens that are skipped.
        total_tokens = padded_pages * page_size
        loc_np = np.full(total_tokens, -1, dtype=np.int32)
        loc_np[:seqlen] = kv_indices_np[:seqlen]
        loc = jax.device_put(
            jnp.asarray(loc_np),
            NamedSharding(kv_pool.mesh, PartitionSpec(kv_pool.attention_data_partition_axis)),
        )

        for i, layer_id in enumerate(
            range(kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num)
        ):
            layer_idx = layer_id - kv_pool.start_layer
            kv_pool.kv_buffer[layer_idx] = write_kv_layer(
                kv[i],
                loc,
                kv_pool.kv_buffer[layer_idx],
                page_size,
                kv_pool.kv_partition_axis,
                kv_pool.attention_data_partition_axis,
                kv_pool.mesh,
            )
        self._maybe_verify_decode_writeback_debug(req, kv_pool, page_ids_padded, kv)

    def _set_decode_bookkeeping(self: Scheduler, req: Req, kv_indices) -> None:
        if kv_indices is None:
            raise RuntimeError(f"missing KV indices for req_id={req.rid!r}")
        kv_indices_np = (
            np.asarray(kv_indices) if not isinstance(kv_indices, np.ndarray) else kv_indices
        )
        seqlen = len(req.origin_input_ids)
        valid_slots = kv_indices_np[:seqlen]
        if len(valid_slots) >= 1:
            req.prefix_indices = valid_slots[:-1]
        else:
            req.prefix_indices = valid_slots
        req.last_matched_prefix_len = len(req.prefix_indices)
        req._pd_skip_prefix_match = True
        req._pd_prealloc_kv_indices = kv_indices_np
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)

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

    def _abort_decode_request(
        self: Scheduler,
        req: Req,
        reason: str,
        *,
        cleanup_transfer: bool = True,
    ) -> None:
        """Release resources AND send AbortReq back to tokenizer."""

        manager = getattr(self, "disagg_kv_manager", None)
        room = getattr(req, "bootstrap_room", None)
        if manager is not None and cleanup_transfer:
            try:
                prefill_dp_rank = _request_dp_rank(req, self.dp_size, prefill=True)
            except Exception:
                logger.warning(
                    "cannot resolve Prefill DP rank while cleaning up aborted "
                    "req_id=%s bootstrap_room=%s reason=%s; transfer metadata "
                    "will expire by TTL",
                    req.rid,
                    room,
                    reason,
                    exc_info=True,
                )
            else:
                try:
                    manager.cleanup_transfer(
                        room,
                        jax_process_index=getattr(req, "disagg_peer_process_index", None),
                        prefill_dp_rank=prefill_dp_rank,
                    )
                except Exception:
                    logger.warning(
                        "failed to clean up transfer metadata for req_id=%s "
                        "bootstrap_room=%s prefill_dp_rank=%s reason=%s",
                        req.rid,
                        room,
                        prefill_dp_rank,
                        reason,
                        exc_info=True,
                    )
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
            "PD-KV-DEBUG decode_pull req_id=%s shape=%s dtype=%s sharding=%s digest=%s sample=%s",
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

    def _maybe_verify_direct_receive_debug(
        self: Scheduler,
        req: Req,
        kv_indices,
        expected_metadata: dict[str, object] | None,
    ) -> None:
        from jax.sharding import NamedSharding, PartitionSpec

        from sgl_jax.srt.disaggregation.base.transfer import slots_to_page_ids
        from sgl_jax.srt.disaggregation.debug_utils import (
            KVDebugSnapshot,
            build_kv_debug_snapshot,
            count_kv_debug_mismatches,
            find_first_kv_debug_mismatch,
            kv_debug_enabled,
        )
        from sgl_jax.srt.disaggregation.prefill import _jit_gather_all_layers

        if not kv_debug_enabled(req.rid):
            return

        kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
        page_ids = np.asarray(
            slots_to_page_ids(kv_indices, kv_pool.page_size, len(req.origin_input_ids)),
            dtype=np.int32,
        )
        page_ids_jax = jax.device_put(
            page_ids,
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
            _jit_gather_all_layers(layer_buffers, page_ids_jax, gather_out_sharding),
            axis=0,
        )
        actual = build_kv_debug_snapshot(readback)
        if expected_metadata is None:
            logger.warning(
                "PD-KV-DEBUG raiden_direct req_id=%s digest=%s sample=%s "
                "expected=missing page_ids=%s",
                req.rid,
                actual.global_digest,
                actual.sample_page_digests(),
                page_ids.tolist(),
            )
            return

        expected = KVDebugSnapshot(
            shape=tuple(int(value) for value in expected_metadata.get("shape", ())),
            dtype=str(expected_metadata.get("dtype", "")),
            sharding="prefill",
            global_digest=str(expected_metadata.get("global_digest", "")),
            page_digests=tuple(
                tuple(str(digest) for digest in row)
                for row in expected_metadata.get("page_digests", ())
            ),
        )
        mismatch_count = count_kv_debug_mismatches(expected, actual)
        first_mismatch = find_first_kv_debug_mismatch(expected, actual)
        logger.warning(
            "PD-KV-DEBUG raiden_direct req_id=%s expected_digest=%s "
            "readback_digest=%s mismatch_count=%d first_mismatch=%s "
            "expected_sample=%s readback_sample=%s page_ids=%s",
            req.rid,
            expected.global_digest,
            actual.global_digest,
            mismatch_count,
            first_mismatch,
            expected.sample_page_digests(),
            actual.sample_page_digests(),
            page_ids.tolist(),
        )
