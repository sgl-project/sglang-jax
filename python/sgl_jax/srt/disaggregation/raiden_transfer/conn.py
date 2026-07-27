"""Raiden-backed PD KV transfer manager and request handles."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

from sgl_jax.srt.disaggregation.base.kv_manager import (
    KVPoll,
    KVReceiver,
    KVSender,
    StateHolder,
)
from sgl_jax.srt.disaggregation.base.transfer import (
    DecodeAdmission,
    DecodeTransferContext,
    PrefillTransfer,
    PrefillTransferContext,
)
from sgl_jax.srt.disaggregation.common.core import CommonKVManager
from sgl_jax.srt.disaggregation.common.metrics import (
    PD_TRANSFER_FAILURES_TOTAL,
    time_phase,
)
from sgl_jax.srt.disaggregation.raiden_transfer.wrapper import RaidenTransferWrapper

logger = logging.getLogger(__name__)


def _uuid_to_int(value: str) -> int:
    """Keep Raiden IDs below JSON's exact-integer limit, as tpu-inference does."""

    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 50) - 1)


@dataclass(frozen=True)
class RaidenMetadata:
    uuid: str
    remote_endpoint: object
    remote_block_ids: tuple[int, ...]
    local_block_ids: tuple[int, ...]
    bootstrap_room: int | None
    jax_process_index: int = 0


class RaidenTransferKVManager(CommonKVManager):
    engine_name = "raiden"
    requires_host_staging = False
    host_pool = None

    def __init__(
        self,
        wrapper: RaidenTransferWrapper,
        bootstrap_client: object,
        *,
        ack_timeout_seconds: float = 60.0,
        pull_timeout_seconds: float = 30.0,
        reaper_interval_seconds: float = 5.0,
    ) -> None:
        super().__init__(
            ack_timeout_seconds=ack_timeout_seconds,
            pull_timeout_seconds=pull_timeout_seconds,
            reaper_interval_seconds=reaper_interval_seconds,
        )
        self.wrapper = wrapper
        self.bootstrap_client = bootstrap_client
        self._poll_lock = threading.Lock()
        self._done_sending: set[str] = set()
        self._done_receiving: set[str] = set()
        self._failed_receiving: set[str] = set()

    def reserve_prefill_buffer(self, existing: int | None) -> int | None:
        return existing

    def prefill_transport_metadata(self) -> dict[str, object]:
        endpoints = list(self.wrapper.endpoints or [])
        control_port = self.wrapper.control_port
        if endpoints:
            control_port = int(str(endpoints[0]["endpoint"]).rsplit(":", 1)[1])
        if not 0 < control_port <= 65535:
            raise RuntimeError("Raiden did not publish a valid control endpoint")
        return {
            "engine": self.engine_name,
            "local_control_port": control_port,
            "endpoints": endpoints,
        }

    def start_prefill(self, context: PrefillTransferContext) -> PrefillTransfer:
        sender = self.create_sender(context.req_id)
        try:
            jax.effects_barrier()
            sender.init(None, transfer_id=context.transfer_id)
            sender.attach_block_ids(
                context.block_ids_factory(),
                bootstrap_room=context.bootstrap_room,
            )
            if context.on_ready is not None:
                context.on_ready()
            sender.send()
        except Exception:
            with suppress(Exception):
                sender.abort()
            with suppress(Exception):
                sender.clear()
            raise
        return PrefillTransfer(sender=sender)

    def try_start_decode(self, context: DecodeTransferContext) -> DecodeAdmission:
        if context.bootstrap_room is None:
            raise ValueError("Raiden decode requires bootstrap_room")
        peer_process_index = int(context.peer_info.get("jax_process_index", 0))
        try:
            info = self.bootstrap_client.get_transfer_info(
                context.bootstrap_room,
                jax_process_index=peer_process_index,
            )
        except Exception as exc:  # transient bootstrap failure
            logger.warning(
                "Raiden metadata lookup failed for room=%s: %s",
                context.bootstrap_room,
                exc,
            )
            return DecodeAdmission.deferred("metadata_lookup")
        if info is None:
            return DecodeAdmission.deferred("metadata_pending")

        receiver: RaidenTransferKVReceiver | None = None
        try:
            if str(info.get("transfer_id", "")) != context.transfer_id:
                raise ValueError(
                    "Raiden transfer ID mismatch: "
                    f"expected={context.transfer_id!r}, got={info.get('transfer_id')!r}"
                )
            metadata = info.get("transport_metadata", info)
            if not isinstance(metadata, dict):
                raise TypeError("Raiden request transport_metadata must be a mapping")
            remote_block_ids = tuple(int(v) for v in metadata.get("remote_block_ids", ()))
            expected_blocks = (context.prompt_tokens + context.page_size - 1) // context.page_size
            if len(remote_block_ids) != expected_blocks:
                raise ValueError(
                    f"Raiden block count mismatch: expected={expected_blocks}, "
                    f"remote={len(remote_block_ids)}"
                )

            peer_metadata = context.peer_info.get("transport_metadata")
            if peer_metadata is None:
                peer_metadata = context.peer_info
            if not isinstance(peer_metadata, dict):
                raise TypeError("prefill transport_metadata must be a mapping")
            peer_endpoints = peer_metadata.get("endpoints")
            if peer_endpoints is None:
                serialized_endpoints = peer_metadata.get("raiden_endpoints_json", "")
                peer_endpoints = (
                    json.loads(str(serialized_endpoints)) if serialized_endpoints else []
                )
            if not isinstance(peer_endpoints, list):
                raise TypeError("Raiden endpoints must be a list")
            base_port = int(peer_metadata.get("local_control_port", 0))
            if peer_endpoints:
                base_port = int(str(peer_endpoints[0]["endpoint"]).rsplit(":", 1)[1])
            if not 0 < base_port <= 65535:
                raise ValueError(f"invalid Raiden control port: {base_port}")

            local_endpoints = list(self.wrapper.endpoints or [])
            if peer_endpoints and len(peer_endpoints) != len(local_endpoints):
                raise ValueError(
                    "Raiden endpoint topology mismatch: "
                    f"prefill={len(peer_endpoints)}, decode={len(local_endpoints)}"
                )
            peer_host = str(context.peer_info["host"])
            if len(local_endpoints) <= 1:
                remote_endpoint: object = f"{peer_host}:{base_port}"
            else:
                # The structured form keeps each local sub-manager matched to
                # the producer sub-manager serving the same shards.
                remote_endpoint = [
                    {
                        "endpoint": f"{peer_host}:{base_port + index}",
                        "shards": list(endpoint["shards"]),
                    }
                    for index, endpoint in enumerate(local_endpoints)
                ]

            kv_indices = np.asarray(context.kv_indices)
            local_pages = kv_indices[:: context.page_size] // context.page_size
            local_block_ids = tuple(int(v) for v in local_pages[: len(remote_block_ids)])
            if len(local_block_ids) != len(remote_block_ids):
                raise ValueError(
                    f"Raiden local block count mismatch: remote={len(remote_block_ids)}, "
                    f"local={len(local_block_ids)}"
                )

            receiver = self.create_receiver(context.req_id)
            receiver.init(
                RaidenMetadata(
                    uuid=context.transfer_id,
                    remote_endpoint=remote_endpoint,
                    remote_block_ids=remote_block_ids,
                    local_block_ids=local_block_ids,
                    bootstrap_room=context.bootstrap_room,
                    jax_process_index=peer_process_index,
                )
            )
            return DecodeAdmission.admitted(receiver)
        except Exception:
            if receiver is not None:
                with suppress(Exception):
                    receiver.fail(reason="receiver_init")
            raise

    def register_read(
        self,
        req_id: str,
        transfer_id: str,
        block_ids: list[int],
        bootstrap_room: int | None,
    ) -> bool:
        needed = self.wrapper.register_read(req_id, _uuid_to_int(transfer_id), block_ids)
        if bootstrap_room is not None:
            self.bootstrap_client.register_transfer(
                bootstrap_room,
                transfer_id,
                jax_process_index=jax.process_index(),
                transport_metadata={"remote_block_ids": list(block_ids)},
            )
        return needed

    def cleanup_transfer(
        self,
        bootstrap_room: int | None,
        *,
        jax_process_index: int | None = None,
    ) -> None:
        if bootstrap_room is None:
            return
        if jax_process_index is None:
            jax_process_index = jax.process_index()
        with suppress(Exception):
            self.bootstrap_client.pop_transfer(
                bootstrap_room,
                jax_process_index=jax_process_index,
            )

    def poll_engine(self) -> None:
        try:
            sent, received, failed = self.wrapper.poll_stats()
        except Exception:  # noqa: BLE001
            logger.exception("Raiden poll_stats() failed")
            return
        with self._poll_lock:
            self._done_sending.update(sent)
            self._done_receiving.update(received)
            self._failed_receiving.update(failed)

    def sender_done(self, req_id: str) -> bool:
        with self._poll_lock:
            return req_id in self._done_sending

    def receiver_state(self, req_id: str) -> str | None:
        with self._poll_lock:
            if req_id in self._failed_receiving:
                return "failed"
            if req_id in self._done_receiving:
                return "done"
            return None

    def forget(self, req_id: str) -> None:
        with self._poll_lock:
            self._done_sending.discard(req_id)
            self._done_receiving.discard(req_id)
            self._failed_receiving.discard(req_id)

    def create_sender(self, req_id: str) -> RaidenTransferKVSender:
        sender = RaidenTransferKVSender(self, req_id)
        self.register_sender(req_id, sender)
        return sender

    def create_receiver(self, req_id: str) -> RaidenTransferKVReceiver:
        receiver = RaidenTransferKVReceiver(self, req_id)
        self.register_receiver(req_id, receiver)
        return receiver


class RaidenTransferKVSender(KVSender, StateHolder):
    def __init__(self, manager: RaidenTransferKVManager, req_id: str) -> None:
        StateHolder.__init__(self, KVPoll.BOOTSTRAPPING, role="prefill")
        self._manager = manager
        self._req_id = req_id
        self._transfer_id: str | None = None
        self._block_ids: list[int] | None = None
        self._bootstrap_room: int | None = None
        self._state_lock = threading.Lock()
        self._timer: object | None = None
        self._transfer_started_at: float | None = None

    @property
    def uuid(self) -> str:
        return self._transfer_id or self._req_id

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    def init(self, kv_indices, transfer_id: str | None = None) -> None:  # noqa: ARG002
        self._transfer_id = transfer_id or self._req_id
        self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def attach_block_ids(self, block_ids: list[int], *, bootstrap_room: int | None) -> None:
        if self._block_ids is not None:
            raise RuntimeError(f"sender {self._req_id!r} is already configured")
        self._block_ids = list(block_ids)
        self._bootstrap_room = bootstrap_room

    def send(self) -> None:
        if self._block_ids is None:
            raise RuntimeError(f"sender {self._req_id!r} has no block IDs")
        with self._state_lock:
            self._manager.register_read(
                self.uuid,
                self.uuid,
                self._block_ids,
                self._bootstrap_room,
            )
            self._transition_to(KVPoll.TRANSFERRING)
            self._timer = time_phase("ack", "prefill")
            self._timer.__enter__()
            self._transfer_started_at = time.monotonic()

    def poll(self) -> KVPoll:
        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING:
                return self.state
        self._manager.poll_engine()
        if not self._manager.sender_done(self.uuid):
            return KVPoll.TRANSFERRING
        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING:
                return self.state
            self._transition_to(KVPoll.SUCCESS)
            self._finish_timer()
            self._transfer_started_at = None
            self._manager.record_terminal(
                self._req_id,
                role="prefill",
                transfer_id=self.uuid,
                state=KVPoll.SUCCESS,
                reason="raiden_done_sending",
            )
        self._manager.forget(self.uuid)
        self._manager._prune_sender(self._req_id)
        return KVPoll.SUCCESS

    def clear(self) -> None:
        self._manager._clear_terminal_record(self._req_id, role="prefill")

    def abort(self) -> None:
        self.fail(reason="abort")

    def failure_exception(self) -> None:
        record = self._manager.get_terminal_record(self._req_id, role="prefill")
        if record is None or record.state != KVPoll.FAILED:
            raise RuntimeError(f"Prefill transfer {self._req_id!r} has no failure record")
        raise RuntimeError(f"Prefill transfer failed for {self._req_id!r}: {record.reason}")

    def fail(self, *, reason: str = "sender_fail") -> None:
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            self._transition_to(KVPoll.FAILED)
            self._finish_timer()
            self._transfer_started_at = None
            self._manager.record_terminal(
                self._req_id,
                role="prefill",
                transfer_id=self.uuid,
                state=KVPoll.FAILED,
                reason=reason,
            )
        self._manager.forget(self.uuid)
        self._manager.cleanup_transfer(self._bootstrap_room)
        with suppress(Exception):
            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="prefill").inc()
        self._manager._prune_sender(self._req_id)

    def _finish_timer(self) -> None:
        timer = self._timer
        if timer is not None:
            self._timer = None
            with suppress(Exception):
                timer.__exit__(None, None, None)


class RaidenTransferKVReceiver(KVReceiver, StateHolder):
    def __init__(self, manager: RaidenTransferKVManager, req_id: str) -> None:
        StateHolder.__init__(self, KVPoll.BOOTSTRAPPING, role="decode")
        self._manager = manager
        self._req_id = req_id
        self._metadata: RaidenMetadata | None = None
        self._state_lock = threading.Lock()
        self._started = False
        self._timer: object | None = None
        self._transfer_started_at: float | None = None

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    def init(self, p_metadata) -> None:
        if not isinstance(p_metadata, RaidenMetadata):
            raise TypeError(f"expected RaidenMetadata, got {type(p_metadata).__name__}")
        self._metadata = p_metadata
        self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def poll(self) -> KVPoll:
        state = self.state
        if state == KVPoll.WAITING_FOR_INPUT:
            assert self._metadata is not None
            with self._state_lock:
                if self.state != KVPoll.WAITING_FOR_INPUT:
                    return self.state
                self._transition_to(KVPoll.TRANSFERRING)
                self._timer = time_phase("pull", "decode")
                self._timer.__enter__()
                self._transfer_started_at = time.monotonic()
                should_start = not self._started
                self._started = True
            if should_start:
                try:
                    self._manager.wrapper.start_read(
                        self._metadata.uuid,
                        _uuid_to_int(self._metadata.uuid),
                        self._metadata.remote_endpoint,
                        list(self._metadata.remote_block_ids),
                        list(self._metadata.local_block_ids),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Raiden start_read() failed for %s", self._req_id)
                    self.fail(reason="raiden_start_read")
            return self.state

        if state == KVPoll.TRANSFERRING:
            assert self._metadata is not None
            self._manager.poll_engine()
            remote_state = self._manager.receiver_state(self._metadata.uuid)
            if remote_state is None:
                return self.state
            if remote_state == "failed":
                self.fail(reason="raiden_failed_receiving")
                return self.state
            with self._state_lock:
                if self.state != KVPoll.TRANSFERRING:
                    return self.state
                self._transition_to(KVPoll.SUCCESS)
                self._finish_timer()
                self._transfer_started_at = None
                self._manager.record_terminal(
                    self._req_id,
                    role="decode",
                    transfer_id=self._metadata.uuid,
                    state=KVPoll.SUCCESS,
                    reason="raiden_done_receiving",
                )
            self._manager.forget(self._metadata.uuid)
            self._manager.cleanup_transfer(
                self._metadata.bootstrap_room,
                jax_process_index=self._metadata.jax_process_index,
            )
            self._manager._prune_receiver(self._req_id)
        return self.state

    def commit(self, install: Callable[[Any], None]) -> None:  # noqa: ARG002
        if self.state != KVPoll.SUCCESS:
            raise RuntimeError("cannot commit an incomplete Raiden receive")

    def clear(self) -> None:
        self._manager._clear_terminal_record(self._req_id, role="decode")

    def abort(self) -> None:
        self.fail(reason="abort")

    def failure_exception(self) -> None:
        record = self._manager.get_terminal_record(self._req_id, role="decode")
        if record is None or record.state != KVPoll.FAILED:
            raise RuntimeError(f"Decode transfer {self._req_id!r} has no failure record")
        raise RuntimeError(f"Decode transfer failed for {self._req_id!r}: {record.reason}")

    def fail(self, *, reason: str = "receiver_fail") -> None:
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return
            try:
                self._transition_to(KVPoll.FAILED)
            except ValueError:
                return
            self._finish_timer()
            self._transfer_started_at = None
            metadata = self._metadata
            transfer_id = metadata.uuid if metadata is not None else self._req_id
            self._manager.record_terminal(
                self._req_id,
                role="decode",
                transfer_id=transfer_id,
                state=KVPoll.FAILED,
                reason=reason,
            )
        self._manager.forget(transfer_id)
        self._manager.cleanup_transfer(
            metadata.bootstrap_room if metadata else None,
            jax_process_index=metadata.jax_process_index if metadata else None,
        )
        with suppress(Exception):
            PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()
        self._manager._prune_receiver(self._req_id)

    def _finish_timer(self) -> None:
        timer = self._timer
        if timer is not None:
            self._timer = None
            with suppress(Exception):
                timer.__exit__(None, None, None)
