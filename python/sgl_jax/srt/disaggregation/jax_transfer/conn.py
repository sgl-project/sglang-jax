"""Event-driven ``JaxTransferKVManager`` backend.

``JaxTransferKVManager`` is a thin façade that composes a domain-agnostic
:class:`RequestTransportCore` (lifecycle, reaper, terminal records) with
a :class:`JaxTransferBackend` (``jax.experimental.transfer`` publish /
pull / release).

``producer_handoff`` is the single prefill-side entry point. It selects
between direct device pulls and host staging, registers the payload with
the transport backend, and returns a cleanup hook for the sender.

Senders remain in ``TRANSFERRING`` until the decode side explicitly
acknowledges completion over the ZMQ side channel. Receivers send that
ack after a successful pull.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass

import jax

from sgl_jax.srt.disaggregation.base.kv_manager import (
    KVManager,
    KVPoll,
    KVReceiver,
    KVSender,
    StateHolder,
)
from sgl_jax.srt.disaggregation.jax_transfer.zmq_notifier import ZmqPullNotifier
from sgl_jax.srt.disaggregation.jax_transfer_wrapper import JaxTransferWrapper
from sgl_jax.srt.disaggregation.metrics import (
    PD_TRANSFER_FAILURES_TOTAL,
    time_phase,
)
from sgl_jax.srt.disaggregation.transport.backend import (
    JaxTransferBackend,
    TransferBackend,
)
from sgl_jax.srt.disaggregation.transport.core import (
    RequestTransportCore,
    TerminalTransferRecord,
)
from sgl_jax.srt.mem_cache.host_kv_pool import HostKVPool, StagedData

__all__ = [
    "JaxTransferKVManager",
    "JaxTransferKVReceiver",
    "JaxTransferKVSender",
    "PMetadata",
    "TerminalTransferRecord",
    "TransferStatus",
]


@dataclass(frozen=True)
class PMetadata:
    """Out-of-band metadata D needs to pull from P.

    ``p_side_channel_host`` / ``p_side_channel_port`` tell the receiver
    where to send the ``pull-done`` ack after the transfer completes.

    ``specs`` maps entry names to their shape/dtype so the receiver can
    construct sub-uuid pulls for each entry independently.
    """

    remote_addr: str
    uuid: str
    specs: dict[str, jax.ShapeDtypeStruct]
    p_side_channel_host: str
    p_side_channel_port: int


@dataclass
class TransferStatus:
    """Result of :meth:`JaxTransferKVManager.producer_handoff`.

    ``uuid`` is the wire-level base uuid for this request.
    ``sub_uuids`` lists all per-entry uuids registered with the
    backend (format: ``f"{uuid}:{entry_name}"``).
    ``on_done`` is the composite cleanup hook (path A returns all
    host buffers to the pool; path B is a no-op).
    """

    uuid: str
    sub_uuids: tuple[str, ...]
    on_done: Callable[[], None]


class JaxTransferKVManager(KVManager):
    """Process-level KV transfer manager (façade).

    Composes:
    * :class:`RequestTransportCore` — request lifecycle, reaper, shutdown
    * :class:`TransferBackend` — tensor publish / pull / release
    * :class:`ZmqPullNotifier` — pull-done side channel
    * (optional) :class:`HostKVPool` — path-A D2H staging
    """

    def __init__(
        self,
        wrapper: JaxTransferWrapper,
        zmq_notifier: ZmqPullNotifier,
        *,
        host_pool: HostKVPool | None = None,
        ack_timeout_seconds: float = 60.0,
        pull_timeout_seconds: float = 30.0,
        reaper_interval_seconds: float = 5.0,
    ) -> None:
        self._wrapper = wrapper
        self._backend: TransferBackend = JaxTransferBackend(wrapper)
        self._zmq_notifier = zmq_notifier
        self._host_pool = host_pool
        self._core = RequestTransportCore(
            ack_timeout_seconds=ack_timeout_seconds,
            pull_timeout_seconds=pull_timeout_seconds,
            reaper_interval_seconds=reaper_interval_seconds,
        )

    # ------------------------------------------------------------------
    # Component access
    # ------------------------------------------------------------------

    @property
    def core(self) -> RequestTransportCore:
        return self._core

    @property
    def backend(self) -> TransferBackend:
        return self._backend

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
    def _senders(self) -> dict[str, JaxTransferKVSender]:
        return self._core._senders  # type: ignore[return-value]

    @property
    def _receivers(self) -> dict[str, JaxTransferKVReceiver]:
        return self._core._receivers  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # KV-domain: prefill-side handoff (path A / path B)
    # ------------------------------------------------------------------

    def producer_handoff(
        self,
        uuid: str,
        payload: dict[str, jax.Array],
        *,
        use_d2h_staging: bool,
    ) -> TransferStatus:
        """Register ``payload`` entries for remote pull under sub-uuids.

        Each entry in ``payload`` is registered as
        ``f"{uuid}:{entry_name}"`` with the backend. Path A stages
        each entry to a host buffer first; path B registers HBM arrays
        directly. The returned :class:`TransferStatus` aggregates all
        sub-uuids and chains cleanup hooks.
        """

        if not payload:
            raise ValueError("payload must be a non-empty dict")
        for name in payload:
            if ":" in name:
                raise ValueError(
                    f"entry name {name!r} must not contain ':'"
                )

        sub_uuids: list[str] = []

        if use_d2h_staging:
            if self._host_pool is None:
                raise RuntimeError(
                    "use_d2h_staging=True requires a host_pool on the "
                    "manager; pass one via JaxTransferKVManager("
                    "..., host_pool=...)"
                )
            pool = self._host_pool
            buffer_ids: list[int] = []
            for name, arr in payload.items():
                sub = f"{uuid}:{name}"
                staged: StagedData = pool.copy_from_device(arr)
                self._backend.register_pull(sub, staged.array)
                sub_uuids.append(sub)
                buffer_ids.append(staged.buffer_id)

            def _on_done() -> None:
                for _bid in buffer_ids:
                    pool.put_buffer(_bid)

            return TransferStatus(
                uuid=uuid, sub_uuids=tuple(sub_uuids), on_done=_on_done
            )

        # path B: direct from HBM
        for name, arr in payload.items():
            sub = f"{uuid}:{name}"
            self._backend.register_pull(sub, arr)
            sub_uuids.append(sub)
        return TransferStatus(
            uuid=uuid, sub_uuids=tuple(sub_uuids), on_done=lambda: None
        )

    # ------------------------------------------------------------------
    # ABC — factory methods
    # ------------------------------------------------------------------

    def create_sender(self, req_id: str) -> JaxTransferKVSender:
        sender = JaxTransferKVSender(self, req_id)
        self._core.register_sender(req_id, sender)
        return sender

    def create_receiver(self, req_id: str) -> JaxTransferKVReceiver:
        receiver = JaxTransferKVReceiver(self, req_id)
        self._core.register_receiver(req_id, receiver)
        return receiver

    # ------------------------------------------------------------------
    # Lifecycle delegation — keeps callers unchanged
    # ------------------------------------------------------------------

    def _prune_sender(self, req_id: str) -> None:
        self._core.prune_sender(req_id)

    def _prune_receiver(self, req_id: str) -> None:
        self._core.prune_receiver(req_id)

    def _clear_terminal_record(self, req_id: str, *, role: str) -> None:
        self._core.clear_terminal_record(req_id, role=role)

    def record_terminal(
        self,
        req_id: str,
        *,
        role: str,
        transfer_id: str,
        state: KVPoll,
        reason: str,
    ) -> None:
        self._core.record_terminal(
            req_id,
            role=role,
            transfer_id=transfer_id,
            state=state,
            reason=reason,
        )

    def get_terminal_record(
        self, req_id: str, *, role: str
    ) -> TerminalTransferRecord | None:
        return self._core.get_terminal_record(req_id, role=role)

    def start_reaper(self) -> None:
        self._core.start_reaper()

    def stop_reaper(self) -> None:
        self._core.stop_reaper()

    def reap_once(self, now: float) -> tuple[list[str], list[str]]:
        return self._core.reap_once(now)

    def inflight_count(self) -> tuple[int, int]:
        return self._core.inflight_count()

    def graceful_shutdown(
        self, drain_timeout_seconds: float = 30.0
    ) -> tuple[int, int]:
        return self._core.graceful_shutdown(drain_timeout_seconds)


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
        self._status: TransferStatus | None = None
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
        self, payload: dict[str, jax.Array], *, use_d2h_staging: bool
    ) -> None:
        if self._payload is not None:
            raise RuntimeError(f"sender {self._req_id!r} payload already attached")
        if not payload:
            raise ValueError(f"sender {self._req_id!r} payload must be non-empty")
        self._payload = payload
        self._use_d2h_staging = use_d2h_staging

    def send(self) -> None:
        if self._payload is None:
            raise RuntimeError(
                f"sender {self._req_id!r} has no payload attached; "
                "call attach_payload() before send()"
            )
        assert self._use_d2h_staging is not None
        callback_uuid = self.uuid.encode("utf-8")
        with self._state_lock:
            self._mgr.zmq_notifier.register_callback(callback_uuid, self._on_ack)
            try:
                status = self._mgr.producer_handoff(
                    self.uuid,
                    self._payload,
                    use_d2h_staging=self._use_d2h_staging,
                )
            except Exception:
                self._mgr.zmq_notifier.unregister_callback(callback_uuid)
                raise
            self._status = status
            self._transition_to(KVPoll.TRANSFERRING)
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
        record = self._mgr.get_terminal_record(self._req_id, role="prefill")
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
        callback_uuid = self.uuid.encode("utf-8")
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            claimed = self._mgr.zmq_notifier.unregister_callback(callback_uuid)
            if claimed is not None and self._status is not None:
                for sub_uuid in self._status.sub_uuids:
                    self._mgr.backend.release(sub_uuid)
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
                            self._mgr.backend.release(sub_uuid)
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
        with suppress(Exception):
            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()
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
                if self.state != KVPoll.WAITING_FOR_INPUT:
                    return self.state
                self._transition_to(KVPoll.TRANSFERRING)
                self._pull_timer = time_phase("pull", "decode")
                self._pull_timer.__enter__()
                self._transfer_started_at = _time.monotonic()
                try:
                    results: dict[str, jax.Array] = {}
                    for name, spec in self._metadata.specs.items():
                        sub_uuid = f"{self._metadata.uuid}:{name}"
                        results[name] = self._mgr.backend.pull(
                            sub_uuid,
                            spec,
                            remote_addr=self._metadata.remote_addr,
                        )
                    self._results = results
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
                    with suppress(Exception):
                        PD_TRANSFER_FAILURES_TOTAL.labels(
                            reason="pull_init", role="decode"
                        ).inc()
                    self._mgr._prune_receiver(self._req_id)
                    raise
            return self.state

        if state == KVPoll.TRANSFERRING:
            if self._results is None:
                return state
            if not all(r.is_ready() for r in self._results.values()):
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
                        PD_TRANSFER_FAILURES_TOTAL.labels(
                            reason="ack_send", role="decode"
                        ).inc()
                    self._mgr._prune_receiver(self._req_id)
                    raise
            self._mgr._prune_receiver(self._req_id)
        return self.state

    def _close_pull_timer(self) -> None:
        timer = self._pull_timer
        if timer is None:
            return
        self._pull_timer = None
        with suppress(Exception):
            timer.__exit__(None, None, None)
