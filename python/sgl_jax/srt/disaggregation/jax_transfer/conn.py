"""Stage 1 backend: ``JaxTransferKVManager`` over the wrapper.

Replaces the Stage 0 happy-path-only placeholder with an
event-driven path:

  * :meth:`JaxTransferKVManager.producer_handoff` is the single entry
    point for the prefill side. It dispatches between path A (D2H
    staging via :class:`QueueHostKVPool`) and path B (direct from HBM)
    based on the ``use_d2h_staging`` flag, registers the buffer with
    the wrapper, and returns a :class:`TransferStatus` carrying the
    ``on_done`` cleanup hook.
  * :class:`JaxTransferKVSender` no longer optimistically transitions
    to ``SUCCESS``. Instead it registers an ack callback on the ZMQ
    notifier and stays in ``TRANSFERRING`` until the decoder confirms
    via :meth:`ZmqPullNotifier.send_done`.
  * :class:`JaxTransferKVReceiver` calls
    :meth:`ZmqPullNotifier.send_done` after a successful pull, so the
    sender's ``on_done`` releases the buffer back to the host pool.

The Stage 0 ``_attach_kv_data_for_testing`` scaffolding is removed —
real data now flows through ``producer_handoff(uuid, device_kv, ...)``.

M5 / M7 carry-over from Stage 0 review:
  * ``wrapper._pending`` is now lock-protected (``register_pull`` from
    main thread, ``release`` from listener thread).
  * ``on_done`` callbacks remove the sender from
    ``_senders`` so the per-request dict does not grow unbounded.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import jax

from sgl_jax.srt.disaggregation.base.kv_manager import (
    KVManager,
    KVPoll,
    KVReceiver,
    KVSender,
    StateHolder,
)
from sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier import (
    ZmqPullNotifier,
)
from sgl_jax.srt.disaggregation.jax_transfer_wrapper import (
    JaxTransferWrapper,
)
from sgl_jax.srt.disaggregation.metrics import (
    PD_TRANSFER_FAILURES_TOTAL,
    PD_TRANSFER_INFLIGHT,
    time_phase,
)
from sgl_jax.srt.mem_cache.host_kv_pool import (
    HostKVPool,
    StagedData,
)


@dataclass(frozen=True)
class PMetadata:
    """Out-of-band metadata D needs to pull from P.

    Stage 0: produced by the test entry script and handed to the
    receiver directly. Stage 2 generates this from the bootstrap server.

    ``p_side_channel_host`` / ``p_side_channel_port`` tell the receiver
    where to send the ``pull-done`` ack after the transfer completes.
    """

    remote_addr: str
    uuid: str
    spec: jax.ShapeDtypeStruct
    p_side_channel_host: str
    p_side_channel_port: int


@dataclass
class TransferStatus:
    """Result of :meth:`JaxTransferKVManager.producer_handoff`.

    ``uuid`` is the wire-level uuid registered with the transfer
    wrapper. ``on_done`` is the cleanup hook the sender invokes when
    it receives the ZMQ ack (path A returns the host buffer to the
    pool; path B is a no-op).
    """

    uuid: str
    on_done: Callable[[], None]


@dataclass(frozen=True)
class TerminalTransferRecord:
    req_id: str
    role: str
    transfer_id: str
    state: KVPoll
    reason: str
    terminal_at: float


class JaxTransferKVManager(KVManager):
    """Process-level manager for the jax_transfer backend.

    Holds the wrapper, the ZMQ notifier, and (optionally) a
    :class:`QueueHostKVPool` for path A. Senders / receivers borrow
    these via :meth:`producer_handoff` and the receiver helpers.

    ``ack_timeout_seconds`` / ``pull_timeout_seconds`` set the orphan
    cleanup thresholds — once a sender has been TRANSFERRING longer
    than the ack timeout, or a receiver longer than the pull timeout,
    the background reaper (started via :meth:`start_reaper`) forces
    them into ``FAILED`` and releases their slots / buffers. Set both
    to a non-positive number to disable the reaper.
    """

    def __init__(
        self,
        wrapper: JaxTransferWrapper,
        zmq_notifier: ZmqPullNotifier,
        *,
        host_pool: Optional[HostKVPool] = None,
        ack_timeout_seconds: float = 60.0,
        pull_timeout_seconds: float = 30.0,
        reaper_interval_seconds: float = 5.0,
    ) -> None:
        self._wrapper = wrapper
        self._zmq_notifier = zmq_notifier
        self._host_pool = host_pool
        self._senders_lock = threading.Lock()
        self._receivers_lock = threading.Lock()
        self._senders: Dict[str, "JaxTransferKVSender"] = {}
        self._receivers: Dict[str, "JaxTransferKVReceiver"] = {}
        self._terminal_records_lock = threading.Lock()
        self._terminal_records: OrderedDict[
            Tuple[str, str], TerminalTransferRecord
        ] = OrderedDict()
        self._max_terminal_records = 4096
        self._ack_timeout_s = ack_timeout_seconds
        self._pull_timeout_s = pull_timeout_seconds
        self._reaper_interval_s = reaper_interval_seconds
        self._reaper_stop = threading.Event()
        self._reaper_thread: Optional[threading.Thread] = None

    @property
    def wrapper(self) -> JaxTransferWrapper:
        return self._wrapper

    @property
    def zmq_notifier(self) -> ZmqPullNotifier:
        return self._zmq_notifier

    @property
    def host_pool(self) -> Optional[HostKVPool]:
        return self._host_pool

    # ------------------------------------------------------------------
    # PD-side handoff
    # ------------------------------------------------------------------

    def producer_handoff(
        self,
        uuid: str,
        device_kv: jax.Array,
        *,
        use_d2h_staging: bool,
    ) -> TransferStatus:
        """Register ``device_kv`` for remote pull keyed by ``uuid``.

        Dispatches between path A (stage to host pool first, then
        register the host buffer) and path B (register HBM array
        directly). Returns a :class:`TransferStatus` whose ``on_done``
        must be wired into the ZMQ ack callback so the host buffer is
        returned to the pool when the receiver finishes pulling.
        """

        if use_d2h_staging:
            if self._host_pool is None:
                raise RuntimeError(
                    "use_d2h_staging=True requires a host_pool on the "
                    "manager; pass one via JaxTransferKVManager("
                    "..., host_pool=...)"
                )
            staged: StagedData = self._host_pool.copy_from_device(device_kv)
            self._wrapper.register_pull(uuid, staged.array)
            buffer_id = staged.buffer_id
            pool = self._host_pool

            def _on_done() -> None:
                pool.put_buffer(buffer_id)
                # wrapper.release happens in the sender's on_done
                # wrapper too — keep that single source of truth.
            return TransferStatus(uuid=uuid, on_done=_on_done)

        # path B: direct from HBM
        self._wrapper.register_pull(uuid, device_kv)
        return TransferStatus(uuid=uuid, on_done=lambda: None)

    # ------------------------------------------------------------------
    # ABC
    # ------------------------------------------------------------------

    def create_sender(self, req_id: str) -> "JaxTransferKVSender":
        with self._senders_lock:
            if req_id in self._senders:
                raise ValueError(
                    f"sender for req_id={req_id!r} already exists"
                )
            sender = JaxTransferKVSender(self, req_id)
            self._senders[req_id] = sender
        self._clear_terminal_record(req_id, role="prefill")
        try:
            PD_TRANSFER_INFLIGHT.labels(role="prefill").inc()
        except Exception:  # noqa: BLE001
            pass
        return sender

    def create_receiver(self, req_id: str) -> "JaxTransferKVReceiver":
        with self._receivers_lock:
            if req_id in self._receivers:
                raise ValueError(
                    f"receiver for req_id={req_id!r} already exists"
                )
            receiver = JaxTransferKVReceiver(self, req_id)
            self._receivers[req_id] = receiver
        self._clear_terminal_record(req_id, role="decode")
        try:
            PD_TRANSFER_INFLIGHT.labels(role="decode").inc()
        except Exception:  # noqa: BLE001
            pass
        return receiver

    # Internal: lifecycle cleanup (M7 carry-over from Stage 0 review).
    def _prune_sender(self, req_id: str) -> None:
        with self._senders_lock:
            removed = self._senders.pop(req_id, None)
        if removed is not None:
            try:
                PD_TRANSFER_INFLIGHT.labels(role="prefill").dec()
            except Exception:  # noqa: BLE001
                pass

    def _prune_receiver(self, req_id: str) -> None:
        with self._receivers_lock:
            removed = self._receivers.pop(req_id, None)
        if removed is not None:
            try:
                PD_TRANSFER_INFLIGHT.labels(role="decode").dec()
            except Exception:  # noqa: BLE001
                pass

    def _clear_terminal_record(self, req_id: str, *, role: str) -> None:
        key = (req_id, role)
        with self._terminal_records_lock:
            self._terminal_records.pop(key, None)

    def record_terminal(
        self,
        req_id: str,
        *,
        role: str,
        transfer_id: str,
        state: KVPoll,
        reason: str,
    ) -> None:
        key = (req_id, role)
        record = TerminalTransferRecord(
            req_id=req_id,
            role=role,
            transfer_id=transfer_id,
            state=state,
            reason=reason,
            terminal_at=time.monotonic(),
        )
        with self._terminal_records_lock:
            self._terminal_records[key] = record
            self._terminal_records.move_to_end(key)
            while len(self._terminal_records) > self._max_terminal_records:
                self._terminal_records.popitem(last=False)

    def get_terminal_record(
        self, req_id: str, *, role: str
    ) -> Optional[TerminalTransferRecord]:
        key = (req_id, role)
        with self._terminal_records_lock:
            record = self._terminal_records.get(key)
            if record is not None:
                self._terminal_records.move_to_end(key)
            return record

    # ------------------------------------------------------------------
    # Stage 4 H-B: orphan / timeout reaper
    # ------------------------------------------------------------------

    def start_reaper(self) -> None:
        """Start the background orphan reaper. Idempotent."""

        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return
        if self._ack_timeout_s <= 0 and self._pull_timeout_s <= 0:
            return
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            name="JaxTransferKVManager-Reaper",
            daemon=True,
        )
        self._reaper_thread.start()

    def stop_reaper(self) -> None:
        if self._reaper_thread is None:
            return
        self._reaper_stop.set()
        self._reaper_thread.join(timeout=self._reaper_interval_s + 1.0)
        self._reaper_thread = None

    def _reaper_loop(self) -> None:
        import time as _time

        while not self._reaper_stop.is_set():
            try:
                self.reap_once(_time.monotonic())
            except Exception:  # noqa: BLE001
                # Reaper must never crash the process.
                pass
            self._reaper_stop.wait(self._reaper_interval_s)

    def reap_once(self, now: float) -> Tuple[List[str], List[str]]:
        """Single pass of the reaper. Exposed for tests so they can
        drive deterministic time. Returns the lists of (sender_ids,
        receiver_ids) that were forced into FAILED on this pass.
        """

        timed_out_senders: List[str] = []
        timed_out_receivers: List[str] = []

        if self._ack_timeout_s > 0:
            with self._senders_lock:
                sender_snapshot = list(self._senders.items())
            for req_id, sender in sender_snapshot:
                if sender.transfer_started_at is None:
                    continue
                if now - sender.transfer_started_at < self._ack_timeout_s:
                    continue
                try:
                    sender.fail(reason="timeout")
                except Exception:  # noqa: BLE001
                    pass
                timed_out_senders.append(req_id)

        if self._pull_timeout_s > 0:
            with self._receivers_lock:
                receiver_snapshot = list(self._receivers.items())
            for req_id, receiver in receiver_snapshot:
                if receiver.transfer_started_at is None:
                    continue
                if now - receiver.transfer_started_at < self._pull_timeout_s:
                    continue
                try:
                    receiver.fail(reason="timeout")
                except Exception:  # noqa: BLE001
                    pass
                timed_out_receivers.append(req_id)

        return timed_out_senders, timed_out_receivers

    # ------------------------------------------------------------------
    # Stage 4 H-D: graceful shutdown
    # ------------------------------------------------------------------

    def inflight_count(self) -> Tuple[int, int]:
        """Return ``(num_senders, num_receivers)`` currently tracked."""

        with self._senders_lock:
            ns = len(self._senders)
        with self._receivers_lock:
            nr = len(self._receivers)
        return ns, nr

    def graceful_shutdown(
        self, drain_timeout_seconds: float = 30.0
    ) -> Tuple[int, int]:
        """Drain in-flight transfers, then abort any stragglers.

        Returns ``(num_aborted_senders, num_aborted_receivers)``.
        Callers should invoke this from a SIGTERM handler after
        un-registering from the bootstrap server, so that no new
        traffic arrives during the drain window.
        """

        import time as _time

        deadline = _time.monotonic() + max(0.0, drain_timeout_seconds)
        while _time.monotonic() < deadline:
            ns, nr = self.inflight_count()
            if ns == 0 and nr == 0:
                self.stop_reaper()
                return 0, 0
            _time.sleep(0.1)

        # Drain timeout exceeded — force-fail everything that remains.
        self.stop_reaper()
        with self._senders_lock:
            sender_snapshot = list(self._senders.values())
        aborted_s = 0
        for sender in sender_snapshot:
            try:
                sender.fail(reason="shutdown")
                aborted_s += 1
            except Exception:  # noqa: BLE001
                pass
        with self._receivers_lock:
            receiver_snapshot = list(self._receivers.values())
        aborted_r = 0
        for receiver in receiver_snapshot:
            try:
                receiver.fail(reason="shutdown")
                aborted_r += 1
            except Exception:  # noqa: BLE001
                pass
        return aborted_s, aborted_r


class JaxTransferKVSender(KVSender, StateHolder):
    """Prefill-side per-request handle.

    State machine (Stage 1):
      BOOTSTRAPPING -> WAITING_FOR_INPUT (init)
      WAITING_FOR_INPUT -> TRANSFERRING (send: producer_handoff +
                          register zmq callback)
      TRANSFERRING -> SUCCESS (zmq ack callback) | FAILED

    The sender does NOT block in ``send()``. Outer event loop polls
    :meth:`poll` until terminal.
    """

    def __init__(
        self, mgr: JaxTransferKVManager, req_id: str
    ) -> None:
        StateHolder.__init__(self, KVPoll.BOOTSTRAPPING, role="prefill")
        self._mgr = mgr
        self._req_id = req_id
        self._transfer_id: Optional[str] = None
        self._device_kv: Optional[jax.Array] = None
        self._use_d2h_staging: Optional[bool] = None
        self._status: Optional[TransferStatus] = None
        self._state_lock = threading.Lock()
        self._ack_timer: Optional[object] = None
        self._transfer_started_at: Optional[float] = None

    @property
    def req_id(self) -> str:
        return self._req_id

    @property
    def uuid(self) -> str:
        # Wire identity used by the transfer wrapper + side-channel ack.
        # Defaults to ``req_id`` for backward compatibility, but callers
        # can provide a per-attempt ``transfer_id`` to isolate late acks
        # when a logical request id is reused.
        return self._transfer_id or self._req_id

    @property
    def transfer_started_at(self) -> Optional[float]:
        """Monotonic timestamp when this sender entered TRANSFERRING,
        or ``None`` if not yet there or already past terminal. Used by
        :meth:`JaxTransferKVManager.reap_once` to detect orphans."""

        return self._transfer_started_at

    def init(
        self, kv_indices, transfer_id: Optional[str] = None
    ) -> None:  # noqa: D401, ARG002
        with self._state_lock:
            self._transfer_id = transfer_id or self._req_id
            self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def attach_payload(
        self, device_kv: jax.Array, *, use_d2h_staging: bool
    ) -> None:
        """Stage 1 hand-off point.

        Stage 0 used a ``_attach_kv_data_for_testing`` helper on the
        manager that all senders shared. Stage 1 routes per-request
        data through the sender itself: the scheduler (Stage 2) or the
        test harness supplies ``device_kv`` directly before
        :meth:`send`. ``use_d2h_staging`` decides path A vs path B.
        """

        if self._device_kv is not None:
            raise RuntimeError(
                f"sender {self._req_id!r} payload already attached"
            )
        self._device_kv = device_kv
        self._use_d2h_staging = use_d2h_staging

    def send(self) -> None:
        if self._device_kv is None:
            raise RuntimeError(
                f"sender {self._req_id!r} has no payload attached; "
                "call attach_payload() before send()"
            )
        assert self._use_d2h_staging is not None
        callback_uuid = self.uuid.encode("utf-8")
        # Hold the state lock around the entire registration sequence
        # so the listener's ``_on_ack`` (which also acquires this
        # lock) cannot race the TRANSFERRING transition. The callback
        # MUST be registered before ``producer_handoff`` (which calls
        # ``register_pull`` and makes the buffer pullable by the
        # decoder) so the decoder's ack always finds a callback in
        # the dict. Stage 1 code review identified this as the C1
        # race; verified reproducer in
        # ``test_kv_sender_event_driven.test_send_ack_race_safe``.
        with self._state_lock:
            self._mgr.zmq_notifier.register_callback(
                callback_uuid, self._on_ack
            )
            try:
                status = self._mgr.producer_handoff(
                    self.uuid,
                    self._device_kv,
                    use_d2h_staging=self._use_d2h_staging,
                )
            except Exception:
                # Roll back the callback if producer_handoff failed
                # before any ack could possibly arrive.
                self._mgr.zmq_notifier.unregister_callback(callback_uuid)
                raise
            self._status = status
            self._transition_to(KVPoll.TRANSFERRING)
            # Start the ack-phase timer right when the buffer becomes
            # pullable by the decoder. ``_on_ack`` observes the elapsed
            # time on terminal SUCCESS.
            self._ack_timer = time_phase("ack", "prefill")
            self._ack_timer.__enter__()
            import time as _time

            self._transfer_started_at = _time.monotonic()

    def poll(self) -> KVPoll:
        with self._state_lock:
            return self.state

    def clear(self) -> None:
        self._mgr._clear_terminal_record(self._req_id, role="prefill")

    def abort(self) -> None:
        self.fail(reason="abort")

    def failure_exception(self) -> None:
        record = self._mgr.get_terminal_record(
            self._req_id, role="prefill"
        )
        if record is None:
            raise RuntimeError(
                f"Prefill transfer has no terminal record for "
                f"req_id={self._req_id!r}"
            )
        if record.state != KVPoll.FAILED:
            raise RuntimeError(
                f"Prefill transfer did not fail for req_id="
                f"{self._req_id!r}; state={record.state.value}"
            )
        raise RuntimeError(
            f"Prefill transfer failed for req_id={self._req_id!r}: "
            f"{record.reason}"
        )

    def fail(self, *, reason: str = "sender_fail") -> None:
        """Test/scheduler hook: force the sender into FAILED.

        Uses :meth:`ZmqPullNotifier.unregister_callback`'s return
        value to decide cleanup ownership: if it returns the
        callback, the listener has not yet popped it and we own the
        cleanup; if it returns ``None``, ``_on_ack`` is in flight and
        will run cleanup itself. This avoids double-running
        ``status.on_done`` (which on path A would
        ``pool.put_buffer`` twice → ``RuntimeError: double free``).
        Stage 1 review C2.

        ``reason`` feeds the ``pd_transfer_failures_total`` metric
        label (default ``"sender_fail"``; reaper uses ``"timeout"``).
        """

        callback_uuid = self.uuid.encode("utf-8")
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            claimed = self._mgr.zmq_notifier.unregister_callback(
                callback_uuid
            )
            if claimed is not None:
                self._mgr.wrapper.release(self.uuid)
                if self._status is not None:
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
        try:
            PD_TRANSFER_FAILURES_TOTAL.labels(
                reason=reason, role="prefill"
            ).inc()
        except Exception:  # noqa: BLE001
            pass
        self._mgr._prune_sender(self._req_id)

    # Called by ZmqPullNotifier listener thread.
    def _on_ack(self, _uuid_bytes: bytes) -> None:
        callback_uuid = self.uuid.encode("utf-8")
        try:
            with self._state_lock:
                # Cleanup under the lock so a concurrent ``fail()``
                # cannot also fire ``on_done`` (Stage 1 review C2 /
                # I1). If cleanup raises, transition to FAILED rather
                # than leaving the sender wedged in TRANSFERRING.
                try:
                    self._mgr.wrapper.release(self.uuid)
                    if self._status is not None:
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
                        try:
                            PD_TRANSFER_FAILURES_TOTAL.labels(
                                reason="ack_cleanup", role="prefill"
                            ).inc()
                        except Exception:  # noqa: BLE001
                            pass
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
        try:
            timer.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass


class JaxTransferKVReceiver(KVReceiver, StateHolder):
    """Decode-side per-request handle.

    State machine (Stage 1):
      BOOTSTRAPPING -> WAITING_FOR_INPUT (init with PMetadata)
      WAITING_FOR_INPUT -> TRANSFERRING (first poll triggers pull)
      TRANSFERRING -> SUCCESS (pull returns; ack sent to P) | FAILED
    """

    def __init__(
        self, mgr: JaxTransferKVManager, req_id: str
    ) -> None:
        StateHolder.__init__(self, KVPoll.BOOTSTRAPPING, role="decode")
        self._mgr = mgr
        self._req_id = req_id
        self._metadata: Optional[PMetadata] = None
        self._result: Optional[jax.Array] = None
        self._pull_timer: Optional[object] = None
        self._transfer_started_at: Optional[float] = None
        # Stage 4 review I1: the reaper can call ``fail()`` from a
        # background thread while the main scheduler thread is mid-
        # ``poll()``. Mirror the sender's ``_state_lock`` so the
        # transition + cleanup pair stays atomic.
        self._state_lock = threading.Lock()

    @property
    def req_id(self) -> str:
        return self._req_id

    @property
    def result(self) -> Optional[jax.Array]:
        return self._result

    @property
    def transfer_started_at(self) -> Optional[float]:
        """Monotonic timestamp when this receiver entered
        TRANSFERRING. Used by the reaper to detect stuck pulls."""

        return self._transfer_started_at

    def clear(self) -> None:
        self._mgr._clear_terminal_record(self._req_id, role="decode")

    def abort(self) -> None:
        self.fail(reason="abort")

    def failure_exception(self) -> None:
        record = self._mgr.get_terminal_record(
            self._req_id, role="decode"
        )
        if record is None:
            raise RuntimeError(
                f"Decode transfer has no terminal record for "
                f"req_id={self._req_id!r}"
            )
        if record.state != KVPoll.FAILED:
            raise RuntimeError(
                f"Decode transfer did not fail for req_id="
                f"{self._req_id!r}; state={record.state.value}"
            )
        raise RuntimeError(
            f"Decode transfer failed for req_id={self._req_id!r}: "
            f"{record.reason}"
        )

    def fail(self, *, reason: str = "receiver_fail") -> None:
        """Force the receiver into FAILED. Used by the reaper for
        pull-timeout orphans."""

        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            try:
                self._transition_to(KVPoll.FAILED)
            except ValueError:
                return
            self._close_pull_timer()
            self._transfer_started_at = None
            transfer_id = (
                self._metadata.uuid if self._metadata is not None else self._req_id
            )
            self._mgr.record_terminal(
                self._req_id,
                role="decode",
                transfer_id=transfer_id,
                state=KVPoll.FAILED,
                reason=reason,
            )
        try:
            PD_TRANSFER_FAILURES_TOTAL.labels(
                reason=reason, role="decode"
            ).inc()
        except Exception:  # noqa: BLE001
            pass
        self._mgr._prune_receiver(self._req_id)

    def init(self, p_metadata: PMetadata) -> None:
        if not isinstance(p_metadata, PMetadata):
            raise TypeError(
                f"p_metadata must be PMetadata, got "
                f"{type(p_metadata).__name__}"
            )
        self._metadata = p_metadata
        self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def poll(self) -> KVPoll:
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
                # Reaper may have force-failed us between the read of
                # ``state`` above and entering the lock. If so, bail.
                if self.state != KVPoll.WAITING_FOR_INPUT:
                    return self.state
                self._transition_to(KVPoll.TRANSFERRING)
                self._pull_timer = time_phase("pull", "decode")
                self._pull_timer.__enter__()
                self._transfer_started_at = _time.monotonic()
                try:
                    self._result = self._mgr.wrapper.pull(
                        self._metadata.uuid,
                        self._metadata.spec,
                        remote_addr=self._metadata.remote_addr,
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
                        reason="pull_init",
                    )
                    try:
                        PD_TRANSFER_FAILURES_TOTAL.labels(
                            reason="pull_init", role="decode"
                        ).inc()
                    except Exception:  # noqa: BLE001
                        pass
                    self._mgr._prune_receiver(self._req_id)
                    raise
            return self.state

        if state == KVPoll.TRANSFERRING:
            # ``wrapper.pull`` returns a lazy ``jax.Array``; stay in
            # TRANSFERRING until the underlying transfer completes.
            if self._result is None:
                return state
            if not self._result.is_ready():
                return state
            assert self._metadata is not None
            with self._state_lock:
                # The reaper may have flipped us to FAILED while we
                # were waiting on ``is_ready``; skip the SUCCESS
                # transition in that case so we don't run an illegal
                # FAILED -> SUCCESS.
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
                    try:
                        PD_TRANSFER_FAILURES_TOTAL.labels(
                            reason="ack_send", role="decode"
                        ).inc()
                    except Exception:  # noqa: BLE001
                        pass
                    self._mgr._prune_receiver(self._req_id)
                    raise
            self._mgr._prune_receiver(self._req_id)
        return self.state

    def _close_pull_timer(self) -> None:
        timer = self._pull_timer
        if timer is None:
            return
        self._pull_timer = None
        try:
            timer.__exit__(None, None, None)
        except Exception:  # noqa: BLE001
            pass
