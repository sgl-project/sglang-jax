"""Shared KV manager implementation for PD disaggregation backends.

``CommonKVManager`` extends the base :class:`KVManager` ABC with
sender / receiver registries, bounded terminal-record bookkeeping,
an orphan reaper, and graceful shutdown. It knows nothing about KV
caches, host staging, or JAX arrays — those concerns belong to
concrete backend subclasses (e.g. ``JaxTransferKVManager``).

Follows the same three-layer pattern as sglang:
``BaseKVManager`` → ``CommonKVManager`` → ``<Backend>KVManager``.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass

from sgl_jax.srt.disaggregation.base.kv_manager import KVManager, KVPoll
from sgl_jax.srt.disaggregation.common.metrics import PD_TRANSFER_INFLIGHT


@dataclass(frozen=True)
class TerminalTransferRecord:
    req_id: str
    role: str
    transfer_id: str
    state: KVPoll
    reason: str
    terminal_at: float


class CommonKVManager(KVManager):
    """Shared lifecycle manager for request-scoped transfers.

    Subclasses must implement :meth:`create_sender` and
    :meth:`create_receiver`. Participants (senders / receivers) must
    expose two duck-type members used by the reaper and graceful
    shutdown:

    * ``transfer_started_at -> float | None``
    * ``fail(*, reason: str) -> None``
    """

    def __init__(
        self,
        *,
        ack_timeout_seconds: float = 60.0,
        pull_timeout_seconds: float = 30.0,
        reaper_interval_seconds: float = 5.0,
    ) -> None:
        self._senders_lock = threading.Lock()
        self._receivers_lock = threading.Lock()
        self._senders: dict[str, object] = {}
        self._receivers: dict[str, object] = {}

        self._terminal_records_lock = threading.Lock()
        self._terminal_records: OrderedDict[
            tuple[str, str], TerminalTransferRecord
        ] = OrderedDict()
        self._max_terminal_records = 4096

        self._ack_timeout_s = ack_timeout_seconds
        self._pull_timeout_s = pull_timeout_seconds
        self._reaper_interval_s = reaper_interval_seconds
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Sender / receiver registry
    # ------------------------------------------------------------------

    def register_sender(self, req_id: str, sender: object) -> None:
        with self._senders_lock:
            if req_id in self._senders:
                raise ValueError(f"sender for req_id={req_id!r} already exists")
            self._senders[req_id] = sender
        self._clear_terminal_record(req_id, role="prefill")
        with suppress(Exception):
            PD_TRANSFER_INFLIGHT.labels(role="prefill").inc()

    def register_receiver(self, req_id: str, receiver: object) -> None:
        with self._receivers_lock:
            if req_id in self._receivers:
                raise ValueError(f"receiver for req_id={req_id!r} already exists")
            self._receivers[req_id] = receiver
        self._clear_terminal_record(req_id, role="decode")
        with suppress(Exception):
            PD_TRANSFER_INFLIGHT.labels(role="decode").inc()

    def _prune_sender(self, req_id: str) -> None:
        with self._senders_lock:
            removed = self._senders.pop(req_id, None)
        if removed is not None:
            with suppress(Exception):
                PD_TRANSFER_INFLIGHT.labels(role="prefill").dec()

    def _prune_receiver(self, req_id: str) -> None:
        with self._receivers_lock:
            removed = self._receivers.pop(req_id, None)
        if removed is not None:
            with suppress(Exception):
                PD_TRANSFER_INFLIGHT.labels(role="decode").dec()

    # ------------------------------------------------------------------
    # Terminal records
    # ------------------------------------------------------------------

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
    ) -> TerminalTransferRecord | None:
        key = (req_id, role)
        with self._terminal_records_lock:
            record = self._terminal_records.get(key)
            if record is not None:
                self._terminal_records.move_to_end(key)
            return record

    # ------------------------------------------------------------------
    # Orphan / timeout reaper
    # ------------------------------------------------------------------

    def start_reaper(self) -> None:
        if self._reaper_thread is not None and self._reaper_thread.is_alive():
            return
        if self._ack_timeout_s <= 0 and self._pull_timeout_s <= 0:
            return
        self._reaper_stop.clear()
        self._reaper_thread = threading.Thread(
            target=self._reaper_loop,
            name="CommonKVManager-Reaper",
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
        while not self._reaper_stop.is_set():
            with suppress(Exception):
                self.reap_once(time.monotonic())
            self._reaper_stop.wait(self._reaper_interval_s)

    def reap_once(self, now: float) -> tuple[list[str], list[str]]:
        """Single reaper pass. Returns ``(timed_out_senders, timed_out_receivers)``."""

        timed_out_senders: list[str] = []
        timed_out_receivers: list[str] = []

        if self._ack_timeout_s > 0:
            with self._senders_lock:
                sender_snapshot = list(self._senders.items())
            for req_id, sender in sender_snapshot:
                started = getattr(sender, "transfer_started_at", None)
                if started is None:
                    continue
                if now - started < self._ack_timeout_s:
                    continue
                with suppress(Exception):
                    sender.fail(reason="timeout")  # type: ignore[union-attr]
                timed_out_senders.append(req_id)

        if self._pull_timeout_s > 0:
            with self._receivers_lock:
                receiver_snapshot = list(self._receivers.items())
            for req_id, receiver in receiver_snapshot:
                started = getattr(receiver, "transfer_started_at", None)
                if started is None:
                    continue
                if now - started < self._pull_timeout_s:
                    continue
                with suppress(Exception):
                    receiver.fail(reason="timeout")  # type: ignore[union-attr]
                timed_out_receivers.append(req_id)

        return timed_out_senders, timed_out_receivers

    # ------------------------------------------------------------------
    # Graceful shutdown
    # ------------------------------------------------------------------

    def inflight_count(self) -> tuple[int, int]:
        with self._senders_lock:
            ns = len(self._senders)
        with self._receivers_lock:
            nr = len(self._receivers)
        return ns, nr

    def graceful_shutdown(
        self, drain_timeout_seconds: float = 30.0
    ) -> tuple[int, int]:
        """Drain in-flight transfers, then abort stragglers.

        Returns ``(num_aborted_senders, num_aborted_receivers)``.
        """

        deadline = time.monotonic() + max(0.0, drain_timeout_seconds)
        while time.monotonic() < deadline:
            ns, nr = self.inflight_count()
            if ns == 0 and nr == 0:
                self.stop_reaper()
                return 0, 0
            time.sleep(0.1)

        self.stop_reaper()
        with self._senders_lock:
            sender_snapshot = list(self._senders.values())
        aborted_s = 0
        for sender in sender_snapshot:
            with suppress(Exception):
                sender.fail(reason="shutdown")  # type: ignore[union-attr]
                aborted_s += 1
        with self._receivers_lock:
            receiver_snapshot = list(self._receivers.values())
        aborted_r = 0
        for receiver in receiver_snapshot:
            with suppress(Exception):
                receiver.fail(reason="shutdown")  # type: ignore[union-attr]
                aborted_r += 1
        return aborted_s, aborted_r
