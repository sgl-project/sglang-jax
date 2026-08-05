"""Raiden-backed PD KV transfer manager and request handles."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import jax

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
    slots_to_page_ids,
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


def _normalize_peer_endpoint(endpoint: object, peer_host: str) -> str:
    value = str(endpoint)
    try:
        host, port_text = value.rsplit(":", 1)
        port = int(port_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Raiden endpoint: {value!r}") from exc
    if not 0 < port <= 65535:
        raise ValueError(f"invalid Raiden endpoint port: {port}")
    host = host.strip("[]")
    if host in {"", "0.0.0.0", "127.0.0.1", "::", "::1", "localhost"}:
        host = peer_host
    if ":" in host:
        host = f"[{host}]"
    return f"{host}:{port}"


def _endpoint_descriptors(
    endpoints: object,
    *,
    peer_host: str,
) -> list[dict[str, object]]:
    if not isinstance(endpoints, list) or not endpoints:
        raise ValueError("Raiden peer did not publish endpoint descriptors")
    descriptors: list[dict[str, object]] = []
    for item in endpoints:
        if not isinstance(item, Mapping):
            raise TypeError("Raiden endpoint descriptor must be a mapping")
        shards = item.get("shards", [])
        if not isinstance(shards, list):
            raise TypeError("Raiden endpoint shards must be a list")
        descriptors.append(
            {
                "endpoint": _normalize_peer_endpoint(item.get("endpoint", ""), peer_host),
                "shards": [int(shard) for shard in shards],
            }
        )
    return descriptors


def _validate_endpoint_topology(
    peer_endpoints: list[dict[str, object]],
    local_endpoints: list[object],
) -> None:
    if not local_endpoints:
        raise RuntimeError("local Raiden engine did not publish endpoints")
    if len(peer_endpoints) == len(local_endpoints) == 1:
        return

    def _shards(items: list[object]) -> set[int]:
        out: set[int] = set()
        for item in items:
            if not isinstance(item, Mapping):
                raise TypeError("Raiden endpoint descriptor must be a mapping")
            values = item.get("shards", [])
            if not isinstance(values, list):
                raise TypeError("Raiden endpoint shards must be a list")
            out.update(int(value) for value in values)
        return out

    peer_shards = _shards(list(peer_endpoints))
    local_shards = _shards(local_endpoints)
    if not peer_shards or peer_shards != local_shards:
        raise ValueError(
            "Raiden endpoint shard topology mismatch: "
            f"prefill={sorted(peer_shards)}, decode={sorted(local_shards)}"
        )


def _debug_metadata(payload: dict[str, Any], num_blocks: int) -> dict[str, object]:
    from sgl_jax.srt.disaggregation.debug_utils import build_kv_debug_snapshot

    kv = payload.get("kv")
    if not isinstance(kv, (list, tuple)):
        raise TypeError("Raiden debug payload must contain per-layer KV arrays")
    snapshot = build_kv_debug_snapshot([layer[:num_blocks] for layer in kv])
    return {
        "shape": list(snapshot.shape),
        "dtype": snapshot.dtype,
        "global_digest": snapshot.global_digest,
        "page_digests": [list(row) for row in snapshot.page_digests],
    }


@dataclass(frozen=True)
class RaidenMetadata:
    uuid: str
    remote_endpoint: object
    remote_block_ids: tuple[int, ...]
    local_block_ids: tuple[int, ...]
    bootstrap_room: int | None
    jax_process_index: int = 0
    prefill_dp_rank: int = 0
    decode_dp_rank: int = 0
    direct_commit: Callable[[Mapping[str, object] | None], None] | None = None
    expected_debug: Mapping[str, object] | None = None


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

    def prepare_prefill_batch(self, kv_buffers: Any) -> None:
        # Raiden reads raw PJRT buffer aliases outside the XLA program, so the
        # serving layer must finish the donated KV writes before registration.
        jax.block_until_ready(kv_buffers)

    def prefill_transport_metadata(self, dp_rank: int = 0) -> dict[str, object]:
        try:
            endpoints = self.wrapper.endpoints_by_dp_rank[int(dp_rank)]
        except KeyError as exc:
            raise ValueError(
                f"Raiden has no endpoints for dp_rank={dp_rank}; "
                f"available={sorted(self.wrapper.endpoints_by_dp_rank)}"
            ) from exc
        if not endpoints:
            raise RuntimeError(f"Raiden did not publish endpoints for dp_rank={dp_rank}")
        control_port = int(str(endpoints[0]["endpoint"]).rsplit(":", 1)[1])
        if not 0 < control_port <= 65535:
            raise RuntimeError("Raiden did not publish a valid control endpoint")
        return {
            "engine": self.engine_name,
            "dp_rank": int(dp_rank),
            "dp_size": self.wrapper.dp_size,
            "local_control_port": control_port,
            "endpoints": endpoints,
        }

    def start_prefill(self, context: PrefillTransferContext) -> PrefillTransfer:
        sender = self.create_sender(context.req_id)
        try:
            sender.init(None, transfer_id=context.transfer_id)
            block_ids = context.block_ids_factory()
            sender.attach_block_ids(
                block_ids,
                bootstrap_room=context.bootstrap_room,
                dp_rank=context.dp_rank,
            )
            from sgl_jax.srt.disaggregation.debug_utils import kv_debug_enabled

            if kv_debug_enabled(context.req_id):
                payload = context.payload_factory()
                if context.on_payload is not None:
                    context.on_payload(payload)
                sender.attach_debug_metadata(_debug_metadata(payload, len(block_ids)))
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
                prefill_dp_rank=context.prefill_dp_rank,
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
            metadata_prefill_dp_rank = int(info.get("prefill_dp_rank", 0))
            if metadata_prefill_dp_rank != context.prefill_dp_rank:
                raise ValueError(
                    "Raiden transfer Prefill rank mismatch: "
                    f"expected={context.prefill_dp_rank}, got={metadata_prefill_dp_rank}"
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
            peer_dp_rank = int(
                peer_metadata.get("dp_rank", context.peer_info.get("system_dp_rank", 0))
            )
            if peer_dp_rank != context.prefill_dp_rank:
                raise ValueError(
                    "prefill endpoint rank mismatch: "
                    f"expected={context.prefill_dp_rank}, got={peer_dp_rank}"
                )
            peer_dp_size = int(peer_metadata.get("dp_size", self.wrapper.dp_size))
            if peer_dp_size != self.wrapper.dp_size:
                raise ValueError(
                    "Raiden DP topology mismatch: "
                    f"prefill={peer_dp_size}, decode={self.wrapper.dp_size}"
                )
            peer_host = str(context.peer_info["host"])
            peer_endpoints = _endpoint_descriptors(
                peer_metadata.get("endpoints"),
                peer_host=peer_host,
            )
            try:
                local_endpoints = list(self.wrapper.endpoints_by_dp_rank[context.decode_dp_rank])
            except KeyError as exc:
                raise ValueError(
                    f"Raiden has no Decode manager for dp_rank={context.decode_dp_rank}"
                ) from exc
            _validate_endpoint_topology(peer_endpoints, local_endpoints)
            remote_endpoint: object = (
                peer_endpoints[0]["endpoint"] if len(peer_endpoints) == 1 else peer_endpoints
            )

            local_block_ids = slots_to_page_ids(
                context.kv_indices,
                context.page_size,
                context.prompt_tokens,
            )
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
                    prefill_dp_rank=context.prefill_dp_rank,
                    decode_dp_rank=context.decode_dp_rank,
                    direct_commit=context.direct_commit,
                    expected_debug=(
                        metadata.get("kv_debug")
                        if isinstance(metadata.get("kv_debug"), Mapping)
                        else None
                    ),
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
        dp_rank: int = 0,
    ) -> bool:
        if not block_ids:
            raise ValueError("Raiden transfer requires at least one KV block")
        return self.wrapper.register_read(
            req_id,
            _uuid_to_int(transfer_id),
            block_ids,
            dp_rank=dp_rank,
        )

    def publish_transfer(
        self,
        transfer_id: str,
        block_ids: list[int],
        bootstrap_room: int | None,
        debug_metadata: Mapping[str, object] | None,
        dp_rank: int = 0,
    ) -> None:
        if bootstrap_room is None:
            return
        transport_metadata: dict[str, object] = {"remote_block_ids": list(block_ids)}
        if debug_metadata is not None:
            transport_metadata["kv_debug"] = dict(debug_metadata)
        self.bootstrap_client.register_transfer(
            bootstrap_room,
            transfer_id,
            jax_process_index=jax.process_index(),
            prefill_dp_rank=dp_rank,
            transport_metadata=transport_metadata,
        )

    def cleanup_transfer(
        self,
        bootstrap_room: int | None,
        *,
        jax_process_index: int | None = None,
        prefill_dp_rank: int = 0,
    ) -> None:
        if bootstrap_room is None:
            return
        if jax_process_index is None:
            jax_process_index = jax.process_index()
        with suppress(Exception):
            self.bootstrap_client.pop_transfer(
                bootstrap_room,
                jax_process_index=jax_process_index,
                prefill_dp_rank=prefill_dp_rank,
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

    def reap_once(self, now: float) -> tuple[list[str], list[str]]:
        """Request logical cancellation without releasing engine-owned pages."""

        timed_out_senders: list[str] = []
        timed_out_receivers: list[str] = []
        if self._ack_timeout_s > 0:
            with self._senders_lock:
                senders = list(self._senders.items())
            for req_id, sender in senders:
                started = getattr(sender, "transfer_started_at", None)
                if (
                    started is not None
                    and now - started >= self._ack_timeout_s
                    and sender.request_abort("timeout")
                ):
                    timed_out_senders.append(req_id)
        if self._pull_timeout_s > 0:
            with self._receivers_lock:
                receivers = list(self._receivers.items())
            for req_id, receiver in receivers:
                started = getattr(receiver, "transfer_started_at", None)
                if (
                    started is not None
                    and now - started >= self._pull_timeout_s
                    and receiver.request_abort("timeout")
                ):
                    timed_out_receivers.append(req_id)
        return timed_out_senders, timed_out_receivers

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
        self._dp_rank: int = 0
        self._state_lock = threading.Lock()
        self._timer: object | None = None
        self._transfer_started_at: float | None = None
        self._pending_failure_reason: str | None = None
        self._debug_metadata: Mapping[str, object] | None = None

    @property
    def uuid(self) -> str:
        return self._transfer_id or self._req_id

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    def init(self, kv_indices, transfer_id: str | None = None) -> None:
        del kv_indices
        self._transfer_id = transfer_id or self._req_id
        self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def attach_block_ids(
        self,
        block_ids: list[int],
        *,
        bootstrap_room: int | None,
        dp_rank: int = 0,
    ) -> None:
        if self._block_ids is not None:
            raise RuntimeError(f"sender {self._req_id!r} is already configured")
        self._block_ids = list(block_ids)
        self._bootstrap_room = bootstrap_room
        self._dp_rank = int(dp_rank)

    def attach_debug_metadata(self, metadata: Mapping[str, object]) -> None:
        self._debug_metadata = dict(metadata)

    def send(self) -> None:
        if self._block_ids is None:
            raise RuntimeError(f"sender {self._req_id!r} has no block IDs")
        with self._state_lock:
            needed = self._manager.register_read(
                self.uuid,
                self.uuid,
                self._block_ids,
                dp_rank=self._dp_rank,
            )
            self._transition_to(KVPoll.TRANSFERRING)
            if needed:
                self._timer = time_phase("ack", "prefill")
                self._timer.__enter__()
                self._transfer_started_at = time.monotonic()
        if not needed:
            # tpu-raiden@8756479 register_read documents False as "nothing to
            # transfer"; NotifyForRead does not use it for slot admission.
            self._finish(KVPoll.SUCCESS, "raiden_transfer_not_needed")
            return
        try:
            self._manager.publish_transfer(
                self.uuid,
                self._block_ids,
                self._bootstrap_room,
                self._debug_metadata,
                dp_rank=self._dp_rank,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Raiden transfer metadata publish failed for %s", self._req_id)
            self.request_abort("bootstrap_register")

    def poll(self) -> KVPoll:
        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING:
                return self.state
        self._manager.poll_engine()
        if not self._manager.sender_done(self.uuid):
            return KVPoll.TRANSFERRING
        # tpu-raiden@8756479 CompleteReadRaw reports send-side failures and
        # timeouts through done_sending; failed_recving is receiver-only.
        reason = self._pending_failure_reason
        return self._finish(
            KVPoll.FAILED if reason is not None else KVPoll.SUCCESS,
            reason or "raiden_done_sending",
        )

    def clear(self) -> None:
        self._manager._clear_terminal_record(self._req_id, role="prefill")

    def abort(self) -> None:
        self.request_abort("abort")

    def failure_exception(self) -> None:
        record = self._manager.get_terminal_record(self._req_id, role="prefill")
        if record is None or record.state != KVPoll.FAILED:
            raise RuntimeError(f"Prefill transfer {self._req_id!r} has no failure record")
        raise RuntimeError(f"Prefill transfer failed for {self._req_id!r}: {record.reason}")

    def fail(self, *, reason: str = "sender_fail") -> None:
        self.request_abort(reason)

    def request_abort(self, reason: str) -> bool:
        finish_now = False
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return False
            if self._pending_failure_reason is not None:
                return False
            self._pending_failure_reason = reason
            finish_now = self.state != KVPoll.TRANSFERRING
        if finish_now:
            self._finish(KVPoll.FAILED, reason)
        return True

    def _finish(self, state: KVPoll, reason: str) -> KVPoll:
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return self.state
            self._transition_to(state)
            self._finish_timer()
            self._transfer_started_at = None
            self._manager.record_terminal(
                self._req_id,
                role="prefill",
                transfer_id=self.uuid,
                state=state,
                reason=reason,
            )
        self._manager.forget(self.uuid)
        self._manager.cleanup_transfer(
            self._bootstrap_room,
            prefill_dp_rank=self._dp_rank,
        )
        if state == KVPoll.FAILED:
            with suppress(Exception):
                PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="prefill").inc()
        self._manager._prune_sender(self._req_id)
        return state

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
        self._pending_failure_reason: str | None = None

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
                        decode_dp_rank=self._metadata.decode_dp_rank,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Raiden start_read() failed for %s", self._req_id)
                    self._finish(KVPoll.FAILED, "raiden_start_read")
            return self.state

        if state == KVPoll.TRANSFERRING:
            assert self._metadata is not None
            self._manager.poll_engine()
            remote_state = self._manager.receiver_state(self._metadata.uuid)
            if remote_state is None:
                return self.state
            if remote_state == "failed":
                return self._finish(
                    KVPoll.FAILED,
                    self._pending_failure_reason or "raiden_failed_receiving",
                )
            reason = self._pending_failure_reason
            return self._finish(
                KVPoll.FAILED if reason is not None else KVPoll.SUCCESS,
                reason or "raiden_done_receiving",
            )
        return self.state

    def commit(self, install: Callable[[Any], None]) -> None:
        del install
        if self.state != KVPoll.SUCCESS:
            raise RuntimeError("cannot commit an incomplete Raiden receive")
        assert self._metadata is not None
        if self._metadata.direct_commit is not None:
            self._metadata.direct_commit(self._metadata.expected_debug)

    def clear(self) -> None:
        self._manager._clear_terminal_record(self._req_id, role="decode")

    def abort(self) -> None:
        self.request_abort("abort")

    def failure_exception(self) -> None:
        record = self._manager.get_terminal_record(self._req_id, role="decode")
        if record is None or record.state != KVPoll.FAILED:
            raise RuntimeError(f"Decode transfer {self._req_id!r} has no failure record")
        raise RuntimeError(f"Decode transfer failed for {self._req_id!r}: {record.reason}")

    def fail(self, *, reason: str = "receiver_fail") -> None:
        self.request_abort(reason)

    def request_abort(self, reason: str) -> bool:
        finish_now = False
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return False
            if self._pending_failure_reason is not None:
                return False
            self._pending_failure_reason = reason
            finish_now = self.state != KVPoll.TRANSFERRING
        if finish_now:
            self._finish(KVPoll.FAILED, reason)
        return True

    def _finish(self, state: KVPoll, reason: str) -> KVPoll:
        with self._state_lock:
            if self.state in (KVPoll.SUCCESS, KVPoll.FAILED):
                return self.state
            self._transition_to(state)
            self._finish_timer()
            self._transfer_started_at = None
            metadata = self._metadata
            transfer_id = metadata.uuid if metadata is not None else self._req_id
            self._manager.record_terminal(
                self._req_id,
                role="decode",
                transfer_id=transfer_id,
                state=state,
                reason=reason,
            )
        self._manager.forget(transfer_id)
        self._manager.cleanup_transfer(
            metadata.bootstrap_room if metadata else None,
            jax_process_index=metadata.jax_process_index if metadata else None,
            prefill_dp_rank=metadata.prefill_dp_rank if metadata else 0,
        )
        if state == KVPoll.FAILED:
            with suppress(Exception):
                PD_TRANSFER_FAILURES_TOTAL.labels(reason=reason, role="decode").inc()
        self._manager._prune_receiver(self._req_id)
        return state

    def _finish_timer(self) -> None:
        timer = self._timer
        if timer is not None:
            self._timer = None
            with suppress(Exception):
                timer.__exit__(None, None, None)
