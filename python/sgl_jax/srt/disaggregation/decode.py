"""Decode-side scheduler Mixin for PD disaggregation."""

from __future__ import annotations

import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

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

        while True:
            recv_reqs = (
                self._comm_backend.recv_requests()
                if self._comm_backend is not None
                else self.recv_requests()
            )
            recv_reqs = self.select_dp_for_request(recv_reqs)
            self.process_input_requests_disagg_decode(recv_reqs)

            if self._engine_paused:
                continue

            self.process_decode_queue()

            batch = self.get_next_batch_to_run()
            self.cur_batch = batch

            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
            else:
                # Skip check_memory / check_tree_cache for PD decode.
                self.new_token_ratio = self.init_new_token_ratio
                if self._comm_backend is not None:
                    self._comm_backend.wait_for_new_requests(0.001)

            self.last_batch = batch

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

                with time_phase("bootstrap", "decode"):
                    if jax.process_count() > 1:
                        p_info = self._pick_prefill_peer_for_this_host()
                    else:
                        p_info = self.disagg_bootstrap_client.get_prefill_info(req.bootstrap_room)
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

            kv_indices = None
            try:
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

        for entry in self._drain_transfer_queue_synced():
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

    def _pick_prefill_peer_for_this_host(self: Scheduler) -> dict[str, object]:
        """Multi-host: find the P host whose jax_process_index matches ours.
        That host's local KV shard is exactly the slice this D host needs.
        Requires P/D to have the same nproc (same-TP constraint).
        """
        if getattr(self, "_disagg_prefill_peer", None) is not None:
            return self._disagg_prefill_peer
        my_pidx = jax.process_index()
        my_nproc = jax.process_count()
        all_p = self.disagg_bootstrap_client.list_prefills()
        for p in all_p:
            if int(p.get("jax_process_index", -1)) == my_pidx:
                if int(p.get("jax_process_count", 0)) != my_nproc:
                    raise RuntimeError(
                        f"P/D process_count mismatch: P={p.get('jax_process_count')} "
                        f"D={my_nproc}. Per-host shard transfer requires same nproc."
                    )
                self._disagg_prefill_peer = p
                return p
        raise RuntimeError(
            f"no prefill host with jax_process_index={my_pidx} registered "
            f"(got {[(p.get('host'), p.get('jax_process_index')) for p in all_p]})"
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
                    if e.receiver.poll() != KVPoll.FAILED:
                        with suppress(Exception):
                            e.receiver.fail(reason="peer_np_failed")
                    out.append(e)
                elif room in success:
                    self.disagg_transfer_queue._entries.pop(rid, None)
                    out.append(e)
        return out

    # ------------------------------------------------------------------
    # Overridable / test-friendly hooks
    # ------------------------------------------------------------------

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

    def _build_kv_spec_for_req(self: Scheduler, req: Req) -> jax.ShapeDtypeStruct:
        """Build ShapeDtypeStruct matching P's KV layout for the receiver."""

        from jax.sharding import NamedSharding, PartitionSpec

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
        shape = (kv_pool.layer_num, padded_pages) + per_layer_tail
        base_spec = kv_pool.kv_sharding.spec
        stacked_spec = PartitionSpec(None, *base_spec)
        sharding = NamedSharding(kv_pool.kv_sharding.mesh, stacked_spec)
        return jax.ShapeDtypeStruct(shape, kv_pool.dtype, sharding=sharding)

    def _write_kv_to_pool(self: Scheduler, req: Req, kv_indices, kv: jax.Array) -> None:
        """Scatter pulled KV into the local paged pool."""

        if kv_indices is None:
            raise RuntimeError(
                f"_write_kv_to_pool: kv_indices is None for req "
                f"{req.rid!r}; allocator may have OOM'd"
            )
        import numpy as np
        from jax.sharding import NamedSharding, PartitionSpec

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
        page_ids_np = kv_indices_np[::page_size] // page_size
        page_ids_np = page_ids_np[:num_pages]
        # Pad page ids to bucket size by repeating the last valid id.
        padded_pages = kv.shape[1]
        if num_pages < padded_pages:
            pad = np.full(padded_pages - num_pages, page_ids_np[-1], dtype=page_ids_np.dtype)
            page_ids_padded = np.concatenate([page_ids_np, pad])
            # Duplicate last valid page's payload so stale tail
            # rows don't overwrite the final real page.
            valid_prefix = jax.lax.slice_in_dim(
                kv,
                start_index=0,
                limit_index=num_pages,
                axis=1,
            )
            last_valid = jax.lax.dynamic_slice_in_dim(
                valid_prefix,
                start_index=num_pages - 1,
                slice_size=1,
                axis=1,
            )
            padded_tail = jnp.repeat(
                last_valid,
                padded_pages - num_pages,
                axis=1,
            )
            padded_tail = jax.device_put(
                padded_tail,
                valid_prefix.sharding,
            )
            kv = jnp.concatenate(
                [
                    valid_prefix,
                    padded_tail,
                ],
                axis=1,
            )
        else:
            page_ids_padded = page_ids_np
        idx_sharding = NamedSharding(kv_pool.mesh, PartitionSpec(None))
        page_ids_jax = jax.device_put(page_ids_padded, idx_sharding)
        for i, layer_id in enumerate(
            range(kv_pool.start_layer, kv_pool.start_layer + kv_pool.layer_num)
        ):
            layer_idx = layer_id - kv_pool.start_layer
            kv_pool.kv_buffer[layer_idx] = (
                kv_pool.kv_buffer[layer_idx]
                .at[page_ids_jax]
                .set(kv[i], out_sharding=kv_pool.kv_sharding)
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

            if kv is not None and hasattr(kv, "nbytes"):
                PD_TRANSFER_BYTES_TOTAL.labels(direction="h2d", role="decode").inc(int(kv.nbytes))

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
