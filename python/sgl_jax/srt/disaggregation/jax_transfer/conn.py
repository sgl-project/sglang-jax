"""Event-driven ``JaxTransferKVManager`` backend.

``JaxTransferKVManager`` extends :class:`CommonKVManager` with
the ``jax.experimental.transfer`` engine (via :class:`JaxTransferWrapper`).

``producer_handoff`` is the single prefill-side entry point. It selects
between direct device pulls and host staging, registers the payload with
the wrapper, and returns a cleanup hook for the sender.

Senders remain in ``TRANSFERRING`` until the decode side explicitly
acknowledges completion over the ZMQ side channel. Receivers send that
ack after a successful pull.
"""

from __future__ import annotations

import logging
import queue as _queue
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import jax

from sgl_jax.srt.disaggregation.base.kv_manager import (
    KVPoll,
    KVReceiver,
    KVSender,
    StateHolder,
)
from sgl_jax.srt.disaggregation.common.core import (
    CommonKVManager,
    TerminalTransferRecord,
)
from sgl_jax.srt.disaggregation.common.metrics import (
    PD_TRANSFER_FAILURES_TOTAL,
    time_phase,
)
from sgl_jax.srt.disaggregation.common.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.disaggregation.jax_transfer.wrapper import (
    JaxTransferWrapper,
    RaidenTransferWrapper,
    _uuid_to_int,
)
from sgl_jax.srt.mem_cache.host_kv_pool import HostKVPool

__all__ = [
    "JaxTransferKVManager",
    "JaxTransferKVReceiver",
    "JaxTransferKVSender",
    "PMetadata",
    "TerminalTransferRecord",
    "TransferStatus",
]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PMetadata:
    """Out-of-band metadata D needs to pull from P.

    ``p_side_channel_host`` / ``p_side_channel_port`` tell the receiver
    where to send the ``pull-done`` ack after the transfer completes.

    ``specs`` maps entry names to their shape/dtype so the receiver can
    construct sub-uuid pulls for each entry independently.

    raiden fields (Phase 0 raiden data plane): ``remote_endpoint`` is raiden's
    control endpoint descriptor (from P's ``get_local_endpoints()``, or a
    ``"host:control_port"`` string); ``remote_block_ids`` are P's device block
    (page) ids for this request; ``local_block_ids`` are D's pre-allocated
    device block ids. All None/empty on the path-A code path.
    """

    remote_addr: str
    uuid: str
    specs: dict[str, jax.ShapeDtypeStruct]
    p_side_channel_host: str
    p_side_channel_port: int
    remote_endpoint: object | None = None
    remote_block_ids: tuple[int, ...] | None = None
    local_block_ids: tuple[int, ...] | None = None


@dataclass
class TransferStatus:
    """Result of :meth:`JaxTransferKVManager.producer_handoff`.

    ``uuid`` is the wire-level base uuid for this request.
    ``sub_uuids`` lists all per-entry uuids registered with the
    backend (format: ``f"{uuid}:{entry_name}"``).
    ``on_done`` is currently a no-op for both path A and path B. The
    host-pool slot reserved for D2H staging is released exactly once by
    the scheduler's prefill-terminal callback (single-owner), NOT here.
    """

    uuid: str
    sub_uuids: tuple[str, ...]
    on_done: Callable[[], None]


class JaxTransferKVManager(CommonKVManager):
    """Concrete KV transfer manager for ``jax.experimental.transfer``.

    Extends :class:`CommonKVManager` (lifecycle, reaper, shutdown) with:
    * :class:`JaxTransferWrapper` — tensor publish / pull / release
    * :class:`ZmqPullNotifier` — pull-done side channel
    * (optional) :class:`HostKVPool` — path-A D2H staging
    """

    def __init__(
        self,
        wrapper: JaxTransferWrapper,
        zmq_notifier: ZmqPullNotifier,
        *,
        host_pool: HostKVPool | None = None,
        raiden_wrapper: RaidenTransferWrapper | None = None,
        bootstrap_client: object | None = None,
        ack_timeout_seconds: float = 60.0,
        pull_timeout_seconds: float = 30.0,
        reaper_interval_seconds: float = 5.0,
        pull_worker_count: int = 4,
    ) -> None:
        super().__init__(
            ack_timeout_seconds=ack_timeout_seconds,
            pull_timeout_seconds=pull_timeout_seconds,
            reaper_interval_seconds=reaper_interval_seconds,
        )
        self._wrapper = wrapper
        self._zmq_notifier = zmq_notifier
        self._host_pool = host_pool
        # raiden data plane (Phase 0). When ``_raiden_wrapper`` is set the KV
        # data plane is served by tpu-raiden instead of the path-A wrapper +
        # host pool + zmq ack. ``_bootstrap_client`` carries the per-request
        # block metadata P->D (raiden's start_read needs P's block ids).
        self._raiden_wrapper = raiden_wrapper
        self._bootstrap_client = bootstrap_client
        # raiden's ``poll_stats()`` is process-global (returns req_id lists for
        # all in-flight transfers). Poll it once per manager tick and cache the
        # cumulative done/failed sets so every sender/receiver ``poll()`` reads
        # membership without re-querying the engine. Guarded because senders,
        # receivers and the reaper all read it from different threads.
        self._raiden_poll_lock = threading.Lock()
        self._raiden_done_sending: set[str] = set()
        self._raiden_done_recving: set[str] = set()
        self._raiden_failed_recving: set[str] = set()
        # A pool of long-lived workers drains the pull queue and runs the
        # blocking ``wrapper.pull`` off the decode event-loop thread (on TPU
        # ``link.pull`` is a synchronous native call). ``pull_worker_count`` is
        # matched to the transfer engine's ``max_num_parallel_copies`` so
        # concurrent pulls run in parallel and a stalled pull only ties up one
        # worker until its ``timeout`` fires (no head-of-line blocking).
        self._pull_worker_count = max(1, int(pull_worker_count))
        self._pull_queue: _queue.Queue[JaxTransferKVReceiver | None] = _queue.Queue()
        self._pull_workers: list[threading.Thread] = []
        for i in range(self._pull_worker_count):
            t = threading.Thread(
                target=self._pull_worker_loop,
                name=f"jax-kv-pull-worker-{i}",
                daemon=True,
            )
            t.start()
            self._pull_workers.append(t)

    def enqueue_pull(self, receiver: JaxTransferKVReceiver) -> None:
        """Hand a TRANSFERRING receiver to the background pull worker.

        Non-blocking: the queue is unbounded and ``put`` returns at once,
        so the decode event loop never stalls behind a pull.
        """

        self._pull_queue.put(receiver)

    def _pull_worker_loop(self) -> None:
        while True:
            receiver = self._pull_queue.get()
            if receiver is None:
                return
            try:
                receiver._run_pull()
            except Exception:  # noqa: BLE001
                logger.exception("jax-kv-pull-worker: receiver pull crashed")

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @property
    def wrapper(self) -> JaxTransferWrapper:
        return self._wrapper

    @property
    def zmq_notifier(self) -> ZmqPullNotifier:
        return self._zmq_notifier

    @property
    def host_pool(self) -> HostKVPool | None:
        return self._host_pool

    @property
    def raiden_wrapper(self) -> RaidenTransferWrapper | None:
        return self._raiden_wrapper

    @property
    def use_raiden(self) -> bool:
        return self._raiden_wrapper is not None

    @property
    def bootstrap_client(self) -> object | None:
        return self._bootstrap_client

    def poll_raiden(self) -> None:
        """Refresh the cached raiden done/failed sets from the engine.

        Non-blocking. Idempotent per tick — sender/receiver ``poll()`` call
        this then read the cached membership. ``poll_stats()`` reports each
        req_id once when it transitions to done, so the sets accumulate.
        """

        if self._raiden_wrapper is None:
            return
        try:
            done_s, done_r, failed_r = self._raiden_wrapper.poll_stats()
        except Exception:  # noqa: BLE001
            logger.exception("raiden poll_stats() raised")
            return
        with self._raiden_poll_lock:
            self._raiden_done_sending.update(done_s)
            self._raiden_done_recving.update(done_r)
            self._raiden_failed_recving.update(failed_r)

    def raiden_sender_done(self, req_id: str) -> bool:
        with self._raiden_poll_lock:
            return req_id in self._raiden_done_sending

    def raiden_receiver_state(self, req_id: str) -> str | None:
        """Return ``"done"`` / ``"failed"`` / ``None`` for a receiver req_id."""

        with self._raiden_poll_lock:
            if req_id in self._raiden_failed_recving:
                return "failed"
            if req_id in self._raiden_done_recving:
                return "done"
            return None

    def raiden_forget(self, req_id: str) -> None:
        """Drop cached raiden state for a retired req_id (bounds the sets)."""

        with self._raiden_poll_lock:
            self._raiden_done_sending.discard(req_id)
            self._raiden_done_recving.discard(req_id)
            self._raiden_failed_recving.discard(req_id)

    # ------------------------------------------------------------------
    # KV-domain: prefill-side handoff (path A / path B)
    # ------------------------------------------------------------------

    def producer_handoff(
        self,
        uuid: str,
        payload: dict[str, jax.Array],
        *,
        use_d2h_staging: bool,
        buffer_id: int | None = None,
    ) -> TransferStatus:
        """Register ``payload`` entries for remote pull under sub-uuids.

        Each entry in ``payload`` is registered as
        ``f"{uuid}:{entry_name}"`` with the wrapper. Path A stages
        each entry to a host buffer first; path B registers HBM arrays
        directly. The returned :class:`TransferStatus` aggregates all
        sub-uuids and chains cleanup hooks.
        """

        if not payload:
            raise ValueError("payload must be a non-empty dict")
        for name in payload:
            if ":" in name:
                raise ValueError(f"entry name {name!r} must not contain ':'")

        sub_uuids: list[str] = []

        if use_d2h_staging:
            if self._host_pool is None:
                raise RuntimeError(
                    "use_d2h_staging=True requires a host_pool on the "
                    "manager; pass one via JaxTransferKVManager(..., host_pool=...)"
                )
            if buffer_id is None:
                raise RuntimeError(
                    "use_d2h_staging=True requires a reserved buffer_id "
                    "(reserved at admission in get_new_batch_prefill)"
                )
            pool = self._host_pool
            try:
                for name, arr_pytree in payload.items():
                    sub = f"{uuid}:{name}"
                    staged = pool.copy_from_device(arr_pytree, buffer_id)
                    self._wrapper.register_pull(sub, staged.array_pytree)
                    sub_uuids.append(sub)
            except Exception:
                # Roll back wrapper registrations only. The pool slot is owned
                # by the scheduler prefill-terminal callback, which releases it
                # during the abort that follows this raise.
                for sub_uuid in sub_uuids:
                    self._wrapper.release(sub_uuid)
                raise
            return TransferStatus(uuid=uuid, sub_uuids=tuple(sub_uuids), on_done=lambda: None)

        # path B: direct from HBM
        try:
            for name, arr in payload.items():
                sub = f"{uuid}:{name}"
                self._wrapper.register_pull(sub, arr)
                sub_uuids.append(sub)
        except Exception:
            for sub_uuid in sub_uuids:
                self._wrapper.release(sub_uuid)
            raise
        return TransferStatus(uuid=uuid, sub_uuids=tuple(sub_uuids), on_done=lambda: None)

    # ------------------------------------------------------------------
    # KV-domain: raiden prefill-side handoff (Phase 0)
    # ------------------------------------------------------------------

    def producer_register_read(
        self,
        req_id: str,
        uuid: str,
        block_ids: list[int],
        *,
        bootstrap_room: int | None = None,
        transfer_id: str | None = None,
    ) -> bool:
        """raiden producer handoff: mark ``block_ids`` readable and publish the
        per-request block metadata to bootstrap so D can pull.

        Returns raiden's ``register_read`` result (False = nothing to transfer).
        Unlike path-A there is no D2H staging and no HBM buffer to keep alive:
        raiden references the device pool blocks directly and pulls them on
        demand, so the sender holds no payload reference.
        """

        if self._raiden_wrapper is None:
            raise RuntimeError("producer_register_read requires a raiden_wrapper on the manager")
        needed = self._raiden_wrapper.register_read(req_id, _uuid_to_int(uuid), block_ids)
        logger.warning(
            "RAIDEN-P register_read req_id=%s uuid=%s uuid_int=%s room=%s nblocks=%d blocks=%s needed=%s",
            req_id,
            uuid,
            _uuid_to_int(uuid),
            bootstrap_room,
            len(block_ids),
            list(block_ids)[:8],
            needed,
        )
        # Publish the block layout + control endpoint for D. Even when
        # ``needed`` is False we register (empty transfers still resolve on the
        # decode side); an empty block list means D's start_read is a no-op.
        if self._bootstrap_client is not None and bootstrap_room is not None:
            import json as _json

            endpoints = self._raiden_wrapper.endpoints
            self._bootstrap_client.register_transfer(
                bootstrap_room,
                transfer_id or uuid,
                block_ids,
                raiden_endpoints_json=_json.dumps(endpoints) if endpoints is not None else "",
            )
        return needed

    # ------------------------------------------------------------------
    # ABC — factory methods
    # ------------------------------------------------------------------

    def create_sender(self, req_id: str) -> JaxTransferKVSender:
        sender = JaxTransferKVSender(self, req_id)
        self.register_sender(req_id, sender)
        return sender

    def create_receiver(self, req_id: str) -> JaxTransferKVReceiver:
        receiver = JaxTransferKVReceiver(self, req_id)
        self.register_receiver(req_id, receiver)
        return receiver


class JaxTransferKVSender(KVSender, StateHolder):
    """Prefill-side per-request handle.

    State machine:
      BOOTSTRAPPING -> WAITING_FOR_INPUT (init)
      WAITING_FOR_INPUT -> TRANSFERRING (send: producer_handoff +
                          register zmq callback)
      TRANSFERRING -> SUCCESS (zmq ack callback) | FAILED

    The sender does NOT block in ``send()``. Outer event loop polls
    :meth:`poll` until terminal.
    """

    def __init__(self, mgr: JaxTransferKVManager, req_id: str) -> None:
        StateHolder.__init__(self, KVPoll.BOOTSTRAPPING, role="prefill")
        self._mgr = mgr
        self._req_id = req_id
        self._transfer_id: str | None = None
        self._payload: dict[str, jax.Array] | None = None
        self._use_d2h_staging: bool | None = None
        self._buffer_id: int | None = None
        self._status: TransferStatus | None = None
        # raiden path: device block ids to register + bootstrap room for the
        # P->D block-metadata publish. ``_use_raiden`` mirrors the manager.
        self._use_raiden: bool = mgr.use_raiden
        self._block_ids: list[int] | None = None
        self._bootstrap_room: int | None = None
        self._state_lock = threading.Lock()
        self._ack_timer: object | None = None
        self._transfer_started_at: float | None = None

    @property
    def req_id(self) -> str:
        return self._req_id

    @property
    def uuid(self) -> str:
        return self._transfer_id or self._req_id

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    def init(self, kv_indices, transfer_id: str | None = None) -> None:  # noqa: ARG002
        with self._state_lock:
            self._transfer_id = transfer_id or self._req_id
            self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def attach_payload(
        self, payload: dict[str, jax.Array], *, use_d2h_staging: bool, buffer_id: int | None = None
    ) -> None:
        if self._payload is not None:
            raise RuntimeError(f"sender {self._req_id!r} payload already attached")
        if not payload:
            raise ValueError(f"sender {self._req_id!r} payload must be non-empty")
        self._payload = payload
        self._use_d2h_staging = use_d2h_staging
        self._buffer_id = buffer_id

    def attach_block_ids(self, block_ids: list[int], *, bootstrap_room: int | None) -> None:
        """raiden path: attach the device block (page) ids of this request +
        the bootstrap room used to publish them to D. No payload is captured —
        raiden reads the device pool blocks directly on pull."""

        if self._block_ids is not None:
            raise RuntimeError(f"sender {self._req_id!r} block_ids already attached")
        self._block_ids = list(block_ids)
        self._bootstrap_room = bootstrap_room

    def send(self) -> None:
        if self._use_raiden:
            self._send_raiden()
            return
        if self._payload is None:
            raise RuntimeError(
                f"sender {self._req_id!r} has no payload attached; "
                "call attach_payload() before send()"
            )
        assert self._use_d2h_staging is not None
        callback_uuid = self.uuid.encode("utf-8")
        with self._state_lock:
            # Register callback before producer_handoff so the ack can't
            # arrive between data registration and callback registration.
            self._mgr.zmq_notifier.register_callback(callback_uuid, self._on_ack)
            try:
                status = self._mgr.producer_handoff(
                    self.uuid,
                    self._payload,
                    use_d2h_staging=self._use_d2h_staging,
                    buffer_id=self._buffer_id,
                )
            except Exception:
                self._mgr.zmq_notifier.unregister_callback(callback_uuid)
                raise
            self._status = status
            if self._use_d2h_staging:
                # Staging copied the payload to host and registered the host
                # arrays for pull, so drop our ref to free the device gather
                # output's HBM now. Path B registers HBM arrays directly, so
                # those must stay alive until the ack.
                self._payload = None
            self._transition_to(KVPoll.TRANSFERRING)
            self._ack_timer = time_phase("ack", "prefill")
            self._ack_timer.__enter__()
            import time as _time

            self._transfer_started_at = _time.monotonic()

    def _send_raiden(self) -> None:
        """raiden send: register the request's device blocks as readable and
        publish their layout to D. No zmq callback — SUCCESS is driven by
        ``poll()`` reading ``poll_stats().done_sending``."""

        if self._block_ids is None:
            raise RuntimeError(
                f"sender {self._req_id!r} has no block_ids attached; "
                "call attach_block_ids() before send()"
            )
        with self._state_lock:
            self._mgr.producer_register_read(
                self._req_id,
                self.uuid,
                self._block_ids,
                bootstrap_room=self._bootstrap_room,
                transfer_id=self.uuid,
            )
            self._transition_to(KVPoll.TRANSFERRING)
            self._ack_timer = time_phase("ack", "prefill")
            self._ack_timer.__enter__()
            import time as _time

            self._transfer_started_at = _time.monotonic()

    def poll(self) -> KVPoll:
        if self._use_raiden:
            return self._poll_raiden()
        with self._state_lock:
            return self.state

    def _poll_raiden(self) -> KVPoll:
        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING:
                return self.state
        # Refresh the manager's cached raiden done set, then check membership.
        self._mgr.poll_raiden()
        if not self._mgr.raiden_sender_done(self._req_id):
            return KVPoll.TRANSFERRING
        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING:
                return self.state
            self._transition_to(KVPoll.SUCCESS)
            self._close_ack_timer()
            self._transfer_started_at = None
            self._mgr.record_terminal(
                self._req_id,
                role="prefill",
                transfer_id=self.uuid,
                state=KVPoll.SUCCESS,
                reason="raiden_done_sending",
            )
        self._mgr.raiden_forget(self._req_id)
        self._mgr._prune_sender(self._req_id)
        return KVPoll.SUCCESS

    def clear(self) -> None:
        self._mgr._clear_terminal_record(self._req_id, role="prefill")

    def abort(self) -> None:
        self.fail(reason="abort")

    def failure_exception(self) -> None:
        record = self._mgr.get_terminal_record(self._req_id, role="prefill")
        if record is None:
            raise RuntimeError(
                f"Prefill transfer has no terminal record for " f"req_id={self._req_id!r}"
            )
        if record.state != KVPoll.FAILED:
            raise RuntimeError(
                f"Prefill transfer did not fail for req_id="
                f"{self._req_id!r}; state={record.state.value}"
            )
        raise RuntimeError(
            f"Prefill transfer failed for req_id={self._req_id!r}: " f"{record.reason}"
        )

    def fail(self, *, reason: str = "sender_fail") -> None:
        callback_uuid = self.uuid.encode("utf-8")
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            claimed = self._mgr.zmq_notifier.unregister_callback(callback_uuid)
            if claimed is not None and self._status is not None:
                for sub_uuid in self._status.sub_uuids:
                    self._mgr.wrapper.release(sub_uuid)
                self._status.on_done()
            self._transition_to(KVPoll.FAILED)
            self._close_ack_timer()
            self._transfer_started_at = None
            self._mgr.zmq_notifier.mark_retired(
                callback_uuid,
                state=KVPoll.FAILED.value,
                reason=reason,
            )
            self._mgr.record_terminal(
                self._req_id,
                role="prefill",
                transfer_id=self.uuid,
                state=KVPoll.FAILED,
                reason=reason,
            )
        with suppress(Exception):
            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="prefill").inc()
        self._mgr._prune_sender(self._req_id)

    def _on_ack(self, _uuid_bytes: bytes) -> None:
        callback_uuid = self.uuid.encode("utf-8")
        try:
            with self._state_lock:
                try:
                    if self._status is not None:
                        for sub_uuid in self._status.sub_uuids:
                            self._mgr.wrapper.release(sub_uuid)
                        self._status.on_done()
                except Exception:
                    if self.state == KVPoll.TRANSFERRING:
                        self._transition_to(KVPoll.FAILED)
                        self._close_ack_timer()
                        self._transfer_started_at = None
                        self._mgr.zmq_notifier.mark_retired(
                            callback_uuid,
                            state=KVPoll.FAILED.value,
                            reason="ack_cleanup",
                        )
                        self._mgr.record_terminal(
                            self._req_id,
                            role="prefill",
                            transfer_id=self.uuid,
                            state=KVPoll.FAILED,
                            reason="ack_cleanup",
                        )
                        with suppress(Exception):
                            PD_TRANSFER_FAILURES_TOTAL.labels(
                                reason="ack_cleanup", role="prefill"
                            ).inc()
                    raise
                if self.state == KVPoll.TRANSFERRING:
                    self._transition_to(KVPoll.SUCCESS)
                    self._close_ack_timer()
                    self._transfer_started_at = None
                    self._mgr.zmq_notifier.mark_retired(
                        callback_uuid,
                        state=KVPoll.SUCCESS.value,
                        reason="ack",
                    )
                    self._mgr.record_terminal(
                        self._req_id,
                        role="prefill",
                        transfer_id=self.uuid,
                        state=KVPoll.SUCCESS,
                        reason="ack",
                    )
        finally:
            self._mgr._prune_sender(self._req_id)

    def _close_ack_timer(self) -> None:
        timer = self._ack_timer
        if timer is None:
            return
        self._ack_timer = None
        with suppress(Exception):
            timer.__exit__(None, None, None)


class JaxTransferKVReceiver(KVReceiver, StateHolder):
    """Decode-side per-request handle.

    State machine:
      BOOTSTRAPPING -> WAITING_FOR_INPUT (init with PMetadata)
      WAITING_FOR_INPUT -> TRANSFERRING (first poll triggers pull)
      TRANSFERRING -> SUCCESS (pull returns; ack sent to P) | FAILED
    """

    def __init__(self, mgr: JaxTransferKVManager, req_id: str) -> None:
        StateHolder.__init__(self, KVPoll.BOOTSTRAPPING, role="decode")
        self._mgr = mgr
        self._req_id = req_id
        self._metadata: PMetadata | None = None
        self._results: dict[str, jax.Array] | None = None
        self._pull_timer: object | None = None
        self._transfer_started_at: float | None = None
        # raiden path: mirrors the manager. ``_started_read`` guards the
        # one-shot ``start_read`` so a repeated poll never re-issues the pull.
        self._use_raiden: bool = mgr.use_raiden
        self._started_read: bool = False
        self._state_lock = threading.Lock()

    @property
    def req_id(self) -> str:
        return self._req_id

    @property
    def result(self) -> dict[str, jax.Array] | None:
        return self._results

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    def clear(self) -> None:
        self._mgr._clear_terminal_record(self._req_id, role="decode")

    def abort(self) -> None:
        self.fail(reason="abort")

    def failure_exception(self) -> None:
        record = self._mgr.get_terminal_record(self._req_id, role="decode")
        if record is None:
            raise RuntimeError(
                f"Decode transfer has no terminal record for " f"req_id={self._req_id!r}"
            )
        if record.state != KVPoll.FAILED:
            raise RuntimeError(
                f"Decode transfer did not fail for req_id="
                f"{self._req_id!r}; state={record.state.value}"
            )
        raise RuntimeError(
            f"Decode transfer failed for req_id={self._req_id!r}: " f"{record.reason}"
        )

    def fail(self, *, reason: str = "receiver_fail") -> None:
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            try:
                self._transition_to(KVPoll.FAILED)
            except ValueError:
                return
            self._close_pull_timer()
            self._transfer_started_at = None
            transfer_id = self._metadata.uuid if self._metadata is not None else self._req_id
            self._mgr.record_terminal(
                self._req_id,
                role="decode",
                transfer_id=transfer_id,
                state=KVPoll.FAILED,
                reason=reason,
            )
        with suppress(Exception):
            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()
        self._mgr._prune_receiver(self._req_id)

    def init(self, p_metadata: PMetadata) -> None:
        if not isinstance(p_metadata, PMetadata):
            raise TypeError(f"p_metadata must be PMetadata, got " f"{type(p_metadata).__name__}")
        self._metadata = p_metadata
        # Do NOT pre-connect here: the link is a native handle and must be
        # created and used on the same thread. ``_run_pull`` connects lazily
        # on the pull worker so connect+pull share that thread; pre-connecting
        # on the decode event-loop thread and pulling on the worker hangs the
        # native transfer.
        self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def poll(self) -> KVPoll:
        if self._use_raiden:
            return self._poll_raiden()
        state = self.state
        if state == KVPoll.WAITING_FOR_INPUT:
            if self._metadata is None:
                raise RuntimeError(
                    "JaxTransferKVReceiver.init() must be called "
                    "before poll(); state machine should have caught "
                    "this — please file a bug."
                )
            import time as _time

            with self._state_lock:
                if self.state != KVPoll.WAITING_FOR_INPUT:
                    return self.state
                self._transition_to(KVPoll.TRANSFERRING)
                self._pull_timer = time_phase("pull", "decode")
                self._pull_timer.__enter__()
                self._transfer_started_at = _time.monotonic()
            # Hand the blocking pull to the background worker. ``poll()`` stays
            # non-blocking; a later poll drives ``is_ready()`` -> ack -> SUCCESS
            # once the worker has stored the results.
            self._mgr.enqueue_pull(self)
            return self.state

        if state == KVPoll.TRANSFERRING:
            if self._results is None:
                return state
            if not all(
                leaf.is_ready() for r in self._results.values() for leaf in jax.tree.leaves(r)
            ):
                return state
            assert self._metadata is not None
            with self._state_lock:
                if self.state != KVPoll.TRANSFERRING:
                    return self.state
                try:
                    self._mgr.zmq_notifier.send_done(
                        self._metadata.uuid.encode("utf-8"),
                        self._metadata.p_side_channel_host,
                        self._metadata.p_side_channel_port,
                    )
                    self._transition_to(KVPoll.SUCCESS)
                    self._close_pull_timer()
                    self._transfer_started_at = None
                    self._mgr.record_terminal(
                        self._req_id,
                        role="decode",
                        transfer_id=self._metadata.uuid,
                        state=KVPoll.SUCCESS,
                        reason="ack_send",
                    )
                except Exception:
                    self._transition_to(KVPoll.FAILED)
                    self._close_pull_timer()
                    self._transfer_started_at = None
                    self._mgr.record_terminal(
                        self._req_id,
                        role="decode",
                        transfer_id=self._metadata.uuid,
                        state=KVPoll.FAILED,
                        reason="ack_send",
                    )
                    with suppress(Exception):
                        PD_TRANSFER_FAILURES_TOTAL.labels(reason="ack_send", role="decode").inc()
                    self._mgr._prune_receiver(self._req_id)
                    return self.state
            self._mgr._prune_receiver(self._req_id)
        return self.state

    def _poll_raiden(self) -> KVPoll:
        """raiden receiver drive: on the first WAITING_FOR_INPUT poll, issue
        ``start_read`` (P's blocks -> D's pre-allocated blocks) and move to
        TRANSFERRING. Subsequent polls read ``poll_stats().done_recving`` /
        ``failed_recving`` via the manager's cached sets. No ZMQ ack and no
        Pallas write-back: raiden lands the blocks straight into D's device KV
        pool, so SUCCESS is purely a completion signal.
        """

        state = self.state
        if state == KVPoll.WAITING_FOR_INPUT:
            if self._metadata is None:
                raise RuntimeError(
                    "JaxTransferKVReceiver.init() must be called before poll()"
                )
            import time as _time

            with self._state_lock:
                if self.state != KVPoll.WAITING_FOR_INPUT:
                    return self.state
                self._transition_to(KVPoll.TRANSFERRING)
                self._pull_timer = time_phase("pull", "decode")
                self._pull_timer.__enter__()
                self._transfer_started_at = _time.monotonic()
                if not self._started_read:
                    self._started_read = True
                    md = self._metadata
                    logger.warning(
                        "RAIDEN-D start_read req_id=%s uuid=%s uuid_int=%s remote_ep=%r "
                        "n_remote=%d remote=%s n_local=%d local=%s",
                        self._req_id,
                        md.uuid,
                        _uuid_to_int(md.uuid),
                        md.remote_endpoint,
                        len(md.remote_block_ids or ()),
                        list(md.remote_block_ids or ())[:8],
                        len(md.local_block_ids or ()),
                        list(md.local_block_ids or ())[:8],
                    )
                    try:
                        self._mgr.raiden_wrapper.start_read(
                            self._req_id,
                            _uuid_to_int(md.uuid),
                            md.remote_endpoint,
                            list(md.remote_block_ids or ()),
                            list(md.local_block_ids or ()),
                        )
                    except Exception:
                        self._transition_to(KVPoll.FAILED)
                        self._close_pull_timer()
                        self._transfer_started_at = None
                        self._mgr.record_terminal(
                            self._req_id,
                            role="decode",
                            transfer_id=md.uuid,
                            state=KVPoll.FAILED,
                            reason="raiden_start_read",
                        )
                        with suppress(Exception):
                            PD_TRANSFER_FAILURES_TOTAL.labels(
                                reason="raiden_start_read", role="decode"
                            ).inc()
                        self._mgr._prune_receiver(self._req_id)
                        return self.state
            return self.state

        if state == KVPoll.TRANSFERRING:
            self._mgr.poll_raiden()
            rstate = self._mgr.raiden_receiver_state(self._req_id)
            if rstate is None:
                return state
            assert self._metadata is not None
            with self._state_lock:
                if self.state != KVPoll.TRANSFERRING:
                    return self.state
                if rstate == "failed":
                    self._transition_to(KVPoll.FAILED)
                    self._close_pull_timer()
                    self._transfer_started_at = None
                    self._mgr.record_terminal(
                        self._req_id,
                        role="decode",
                        transfer_id=self._metadata.uuid,
                        state=KVPoll.FAILED,
                        reason="raiden_failed_recving",
                    )
                    with suppress(Exception):
                        PD_TRANSFER_FAILURES_TOTAL.labels(
                            reason="raiden_failed_recving", role="decode"
                        ).inc()
                    self._mgr.raiden_forget(self._req_id)
                    self._mgr._prune_receiver(self._req_id)
                    return self.state
                self._transition_to(KVPoll.SUCCESS)
                self._close_pull_timer()
                self._transfer_started_at = None
                self._mgr.record_terminal(
                    self._req_id,
                    role="decode",
                    transfer_id=self._metadata.uuid,
                    state=KVPoll.SUCCESS,
                    reason="raiden_done_recving",
                )
            self._mgr.raiden_forget(self._req_id)
            self._mgr._prune_receiver(self._req_id)
        return self.state

    def _run_pull(self) -> None:
        """Run the pull on a background worker thread, off the decode
        event-loop thread. Results are stored under ``_state_lock`` and only
        if the receiver is still TRANSFERRING; a reaper ``fail()`` that
        already moved the state to FAILED wins, and the late results are
        dropped.
        """

        assert self._metadata is not None
        try:
            results: dict[str, jax.Array] = {}
            for name, spec in self._metadata.specs.items():
                sub_uuid = f"{self._metadata.uuid}:{name}"
                results[name] = self._mgr.wrapper.pull(
                    sub_uuid,
                    spec,
                    remote_addr=self._metadata.remote_addr,
                )
        except Exception:
            with self._state_lock:
                if self.state != KVPoll.TRANSFERRING:
                    return
                self._transition_to(KVPoll.FAILED)
                self._close_pull_timer()
                self._transfer_started_at = None
                self._mgr.record_terminal(
                    self._req_id,
                    role="decode",
                    transfer_id=self._metadata.uuid,
                    state=KVPoll.FAILED,
                    reason="pull_init",
                )
            with suppress(Exception):
                PD_TRANSFER_FAILURES_TOTAL.labels(reason="pull_init", role="decode").inc()
            self._mgr._prune_receiver(self._req_id)
            return

        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING:
                # Reaper timed the transfer out while we were pulling; the
                # terminal state wins and the results are discarded.
                return
            self._results = results

    def _close_pull_timer(self) -> None:
        timer = self._pull_timer
        if timer is None:
            return
        self._pull_timer = None
        with suppress(Exception):
            timer.__exit__(None, None, None)
