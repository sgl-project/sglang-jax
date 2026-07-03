"""Decode-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.disaggregation.base.kv_manager import KVPoll
from sgl_jax.srt.disaggregation.bootstrap import BootstrapClient, PrefillInfoCache
from sgl_jax.srt.disaggregation.jax_transfer.conn import (
    JaxTransferKVManager,
    JaxTransferKVReceiver,
    PMetadata,
)
from sgl_jax.srt.mem_cache.memory_pool import write_kv_layer

if TYPE_CHECKING:
    from sgl_jax.srt.managers.schedule_batch import Req
    from sgl_jax.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


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
    # Set by _drain_transfer_queue_synced on multi-host so downstream
    # does not re-poll (a poll() that raised would re-raise and desync).
    synced_state: KVPoll | None = None
    # Prefill-side info from bootstrap, stashed at intake so KV alloc +
    # receiver setup can be deferred to the capacity-gated admission step.
    p_info: dict | None = None


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
                raise ValueError(f"DecodePreallocQueue already tracks " f"req_id={entry.req_id!r}")
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

                self._pd_mark_time(req, "bootstrap_start")
                with time_phase("bootstrap", "decode"):
                    if jax.process_count() > 1:
                        # Multi-host caches the matched peer after the first
                        # lookup, so this does no per-request network I/O.
                        p_info = self._pick_prefill_peer_for_this_host(
                            dp_rank=getattr(req, "dp_rank", 0)
                        )
                    else:
                        # Local cache resolution (sglang-style): a warm cache
                        # does zero network I/O, so this no longer blocks the
                        # event loop.
                        p_info = self.disagg_prefill_info_cache.pick_for_room(
                            req.bootstrap_room, dp_rank=getattr(req, "dp_rank", 0)
                        )
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

            if p_info is None:
                # No prefill registered yet (or the rate-limited refresh was
                # skipped this tick). Defer and retry next tick — never abort.
                self._pd_pending_bootstrap.append(req)
                continue

            try:
                from sgl_jax.srt.disaggregation.bootstrap import (
                    check_prefill_compat,
                    resolve_kv_dtype_name,
                )

                local_kv_pool = self.token_to_kv_pool_allocator.get_kvcache()
                check_prefill_compat(
                    p_info,
                    local_page_size=self.server_args.page_size,
                    local_kv_dtype=resolve_kv_dtype_name(local_kv_pool.dtype),
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
            if state == KVPoll.SUCCESS:
                try:
                    if self.disagg_kv_manager.use_raiden:
                        # raiden landed the KV directly into D's device pool
                        # blocks and the decode bookkeeping was set at admission,
                        # so skip the path-A pull-result scatter (_write_kv_to_pool).
                        # NEEDS-TPU-VERIFICATION: confirm the blocks are correct
                        # and no write-back is required.
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
                        continue
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

    def _pick_prefill_peer_for_this_host(
        self: Scheduler, dp_rank: int | None = None
    ) -> dict[str, object]:
        """Multi-host: find the P host whose jax_process_index matches ours.
        That host's local KV shard is exactly the slice this D host needs.
        Requires P/D to have the same nproc (same-TP constraint).
        """
        cache = getattr(self, "_disagg_prefill_peers", {})
        if dp_rank is not None and dp_rank in cache:
            return cache[dp_rank]
        if dp_rank is None and getattr(self, "_disagg_prefill_peer", None) is not None:
            return self._disagg_prefill_peer
        my_pidx = jax.process_index()
        my_nproc = jax.process_count()
        all_p = self.disagg_bootstrap_client.list_prefills()
        for p in all_p:
            if int(p.get("jax_process_index", -1)) == my_pidx:
                if dp_rank is not None and int(p.get("system_dp_rank", 0)) != dp_rank:
                    continue
                if int(p.get("jax_process_count", 0)) != my_nproc:
                    raise RuntimeError(
                        f"P/D process_count mismatch: P={p.get('jax_process_count')} "
                        f"D={my_nproc}. Per-host shard transfer requires same nproc."
                    )
                if dp_rank is not None:
                    cache[dp_rank] = p
                    self._disagg_prefill_peers = cache
                else:
                    self._disagg_prefill_peer = p
                return p
        raise RuntimeError(
            f"no prefill host with jax_process_index={my_pidx} "
            f"{'and dp_rank=' + str(dp_rank) if dp_rank is not None else ''} registered "
            f"(got {[(p.get('host'), p.get('jax_process_index'), p.get('system_dp_rank')) for p in all_p]})"
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
        """Capacity-gated FIFO admission of preallocated PD reqs.

        Pops reqs from the prealloc queue into the transfer queue only while
        the paged pool has room, reserving ``num_reserved_decode_tokens`` of
        headroom per in-flight/running request so a running decode step can
        always alloc its next token even when every other req is mid-transfer
        (transfer-queue reqs cannot be retracted). KV indices are allocated
        here, not at intake. Reqs that don't fit stay queued and retry next
        tick — deferral, never abort.
        """

        allocator = self.token_to_kv_pool_allocator
        if allocator is None:
            return

        page_size = allocator.page_size
        reserved_per = self.server_args.disaggregation_num_reserved_decode_tokens
        max_inflight = self.server_args.disaggregation_max_inflight_transfers
        n_running = len(self.running_batch.reqs) if self.running_batch is not None else 0
        n_transfer = len(self.disagg_transfer_queue)
        admitted = 0

        for entry in self.disagg_prealloc_queue.items_fifo():
            # In-flight transfer cap: each admitted transfer holds a pulled KV
            # destination buffer on decode HBM (untracked by the paged-pool
            # budget below) until it is scattered. Stop admitting once the cap
            # is reached so a burst of concurrent requests cannot allocate that
            # many transient buffers at once and OOM. Excess reqs stay queued
            # and retry next tick (deferral, never abort).
            if max_inflight > 0 and (n_transfer + admitted) >= max_inflight:
                break
            seqlen = len(entry.req.origin_input_ids)
            page_aligned = ((seqlen + page_size - 1) // page_size) * page_size
            reserved = reserved_per * (n_running + n_transfer + admitted)
            if page_aligned + reserved > allocator.available_size():
                # Insufficient capacity: defer this and all later (FIFO) reqs.
                break

            kv_indices = allocator.alloc(page_aligned)
            if kv_indices is None:
                # Budget check should prevent this; treat a surprise shortfall
                # as transient and retry next tick rather than abort.
                break

            if self.disagg_kv_manager.use_raiden:
                admitted_raiden = self._admit_one_raiden(entry, kv_indices, page_size)
                if admitted_raiden is None:
                    # P hasn't published this req's block metadata yet (bootstrap
                    # 404). Free the slot we just allocated and leave the entry in
                    # the prealloc queue to retry next tick (deferral, not abort).
                    self._release_decode_kv_indices(kv_indices)
                    continue
                if admitted_raiden is False:
                    # Setup failed and the request was already aborted inside the
                    # helper; move on.
                    continue
                admitted += 1
                continue

            try:
                receiver = self.disagg_kv_manager.create_receiver(entry.req.rid)
                spec = self._build_kv_spec_for_req(entry.req)
                p_info = entry.p_info
                receiver.init(
                    PMetadata(
                        remote_addr=(f"{p_info['host']}:{p_info['transfer_port']}"),
                        uuid=entry.req.disagg_transfer_id or entry.req.rid,
                        specs={"kv": spec},
                        p_side_channel_host=str(p_info["host"]),
                        p_side_channel_port=int(p_info["side_channel_port"]),
                    )
                )
            except Exception:
                logger.exception(
                    "failed to set up KVReceiver for req_id=%s",
                    entry.req.rid,
                )
                self._record_decode_transfer_failure("receiver_init")
                self._release_decode_kv_indices(kv_indices)
                self.disagg_prealloc_queue.remove(entry.req_id)
                self._abort_decode_request(entry.req, "receiver_init")
                continue

            entry.kv_indices = kv_indices
            entry.receiver = receiver
            entry.started = True
            self._pd_mark_time(entry.req, "transfer_entry")
            self.disagg_prealloc_queue.remove(entry.req_id)
            self.disagg_transfer_queue.add(entry)
            admitted += 1

    def _admit_one_raiden(self: Scheduler, entry, kv_indices, page_size: int):
        """raiden admission for a single prealloc entry.

        Returns:
          * ``True``  -- admitted to the transfer queue.
          * ``None``  -- P's per-request block metadata not yet published
            (bootstrap 404); caller should defer (free kv_indices, retry).
          * ``False`` -- setup failed and the request was aborted here.
        """

        import numpy as np

        req = entry.req
        try:
            info = self.disagg_bootstrap_client.get_transfer_info(req.bootstrap_room)
        except Exception:
            logger.exception(
                "raiden get_transfer_info raised for room=%s",
                req.bootstrap_room,
            )
            return None
        if info is None:
            # Not published yet -> defer.
            return None

        try:
            import json as _json

            # Chunked transfer: P publishes one entry per chunk. Admit as soon as
            # chunk 0 exists; the receiver discovers + starts the rest as P
            # produces them (transfer/compute overlap). The endpoint descriptor
            # is chunk-independent, so read it from the first available chunk.
            chunks = info.get("chunks", {}) or {}
            if not chunks:
                return None
            first_info = chunks[min(chunks)]
            endpoints_json = first_info.get("raiden_endpoints_json", "") or ""
            p_info = entry.p_info
            p_host = p_info["host"]
            # Producer's advertised base control port: prefer the port carried in
            # its endpoint descriptors, else the explicit control port field.
            p_endpoints = _json.loads(endpoints_json) if endpoints_json else None
            if p_endpoints:
                base_port = int(str(p_endpoints[0]["endpoint"]).rsplit(":", 1)[1])
            else:
                base_port = int(first_info.get("local_control_port", 0))

            # Shape remote_endpoint by the CONSUMER's local sub-manager count,
            # mirroring tpu-inference's _remote_endpoint. A single sub-manager
            # (TP=1 / single-chip) must get a plain "host:port" string; passing a
            # list-of-dict here hits start_read's broadcast overload and raiden
            # returns failed_recving immediately. Only >1 sub-managers use the
            # shard-matched list form (base_port + i per local endpoint).
            local_eps = self.disagg_kv_manager.raiden_wrapper.endpoints or []
            remote_endpoint: object
            if len(local_eps) <= 1:
                remote_endpoint = f"{p_host}:{base_port}"
            else:
                remote_endpoint = [
                    {
                        "endpoint": f"{p_host}:{base_port + i}",
                        "shards": list(ep["shards"]),
                    }
                    for i, ep in enumerate(local_eps)
                ]

            # Whole-prompt local device page ids (sequence order). The receiver
            # slices these per chunk via each chunk's chunk_page_offset, so the
            # local blocks line up one-to-one with P's per-chunk remote blocks.
            kv_indices_np = (
                np.asarray(kv_indices)
                if not isinstance(kv_indices, np.ndarray)
                else kv_indices
            )
            local_pages = tuple(int(p) for p in (kv_indices_np[::page_size] // page_size))

            receiver = self.disagg_kv_manager.create_receiver(req.rid)
            receiver.init(
                PMetadata(
                    remote_addr=(f"{p_info['host']}:{p_info['transfer_port']}"),
                    uuid=req.disagg_transfer_id or req.rid,
                    specs={},
                    p_side_channel_host=str(p_info["host"]),
                    p_side_channel_port=int(p_info["side_channel_port"]),
                    remote_endpoint=remote_endpoint,
                    bootstrap_room=req.bootstrap_room,
                    local_pages=local_pages,
                )
            )
        except Exception:
            logger.exception(
                "failed to set up raiden KVReceiver for req_id=%s",
                req.rid,
            )
            self._record_decode_transfer_failure("receiver_init")
            self._release_decode_kv_indices(kv_indices)
            self.disagg_prealloc_queue.remove(entry.req_id)
            self._abort_decode_request(req, "receiver_init")
            return False

        entry.kv_indices = kv_indices
        entry.receiver = receiver
        entry.started = True
        # raiden lands the KV straight into D's device pool blocks, so the
        # post-transfer Pallas write-back is skipped (see process_decode_queue).
        # Set the decode bookkeeping (prefix_indices / fill_ids) now so the req
        # is ready to enqueue on SUCCESS.
        self._raiden_set_decode_bookkeeping(req, kv_indices)
        self._pd_mark_time(req, "transfer_entry")
        self.disagg_prealloc_queue.remove(entry.req_id)
        self.disagg_transfer_queue.add(entry)
        return True

    def _raiden_set_decode_bookkeeping(self: Scheduler, req, kv_indices) -> None:
        """Set prefix_indices / fill_ids for a raiden-transferred req.

        raiden writes KV directly into the device pool, so unlike path-A there is
        no ``_write_kv_to_pool`` scatter. This mirrors the bookkeeping tail of
        ``_write_kv_to_pool`` without touching the pool tensors.
        NEEDS-TPU-VERIFICATION: confirm raiden's landed layout matches the
        allocator slot order so prefix_indices point at the correct pages.
        """

        import numpy as np

        kv_indices_np = (
            np.asarray(kv_indices)
            if not isinstance(kv_indices, np.ndarray)
            else kv_indices
        )
        seqlen = len(req.origin_input_ids)
        valid_slots = kv_indices_np[:seqlen]
        if len(valid_slots) > 0:
            req.prefix_indices = valid_slots[:-1]
        else:
            req.prefix_indices = valid_slots
        req.last_matched_prefix_len = len(req.prefix_indices)
        # Reuse the transferred KV as prefix (extend_input_len=1) instead of
        # re-prefilling the whole prompt on decode. Mirrors path-A's
        # _write_kv_to_pool; without this the raiden path re-runs a full prefill
        # (cached-token=0), wasting the transfer and leaking the prealloc pages.
        req._pd_skip_prefix_match = True
        req._pd_prealloc_kv_indices = kv_indices_np
        req.fill_ids = list(req.origin_input_ids) + list(req.output_ids)

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
