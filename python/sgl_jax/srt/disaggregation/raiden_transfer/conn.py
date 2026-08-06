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
from sgl_jax.srt.disaggregation.common.capacity import CHUNK_TRANSFER_WINDOW
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


@dataclass(frozen=True)
class RaidenChunkedMetadata:
    base_uuid: str
    remote_endpoint: object
    local_block_ids: tuple[int, ...]
    bootstrap_room: int
    jax_process_index: int
    prefill_dp_rank: int
    decode_dp_rank: int
    initial_chunks: Mapping[int, Mapping[str, object]]
    known_num_chunks: int = 0
    direct_commit: Callable[[Mapping[str, object] | None], None] | None = None


def _normalize_transfer_bundle(
    info: Mapping[str, object],
) -> tuple[dict[int, Mapping[str, object]], int]:
    """Normalize v5 metadata and the pre-v5 flat shape used by test doubles."""

    raw_chunks = info.get("chunks")
    if raw_chunks is None:
        return {0: info}, 1
    if not isinstance(raw_chunks, Mapping):
        raise TypeError("Raiden transfer chunks must be a mapping")
    chunks: dict[int, Mapping[str, object]] = {}
    for raw_index, value in raw_chunks.items():
        if not isinstance(value, Mapping):
            raise TypeError("Raiden chunk metadata must be a mapping")
        chunks[int(raw_index)] = value
    num_chunks = int(info.get("num_chunks", 0) or 0)
    if num_chunks < 0:
        raise ValueError("Raiden num_chunks must be non-negative")
    return chunks, num_chunks


class RaidenTransferKVManager(CommonKVManager):
    engine_name = "raiden"
    requires_host_staging = False
    host_pool = None

    def __init__(
        self,
        wrapper: RaidenTransferWrapper,
        bootstrap_client: object,
        *,
        enable_chunk_prefill_transfer: bool = False,
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
        self.enable_chunk_prefill_transfer = bool(enable_chunk_prefill_transfer)
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
            chunks, known_num_chunks = _normalize_transfer_bundle(info)
            if not chunks:
                return DecodeAdmission.deferred("metadata_pending")
            first_chunk = chunks.get(0)
            if first_chunk is None:
                return DecodeAdmission.deferred("chunk_zero_pending")

            expected_first_id = (
                f"{context.transfer_id}#c0"
                if self.enable_chunk_prefill_transfer
                else context.transfer_id
            )
            if str(first_chunk.get("transfer_id", "")) != expected_first_id:
                raise ValueError(
                    "Raiden transfer ID mismatch: "
                    f"expected={expected_first_id!r}, got={first_chunk.get('transfer_id')!r}"
                )
            metadata_prefill_dp_rank = int(first_chunk.get("prefill_dp_rank", 0))
            if metadata_prefill_dp_rank != context.prefill_dp_rank:
                raise ValueError(
                    "Raiden transfer Prefill rank mismatch: "
                    f"expected={context.prefill_dp_rank}, got={metadata_prefill_dp_rank}"
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

            receiver = self.create_receiver(context.req_id)
            if self.enable_chunk_prefill_transfer:
                receiver.init(
                    RaidenChunkedMetadata(
                        base_uuid=context.transfer_id,
                        remote_endpoint=remote_endpoint,
                        local_block_ids=local_block_ids,
                        bootstrap_room=int(context.bootstrap_room),
                        jax_process_index=peer_process_index,
                        prefill_dp_rank=context.prefill_dp_rank,
                        decode_dp_rank=context.decode_dp_rank,
                        initial_chunks=chunks,
                        known_num_chunks=known_num_chunks,
                        direct_commit=context.direct_commit,
                    )
                )
            else:
                if known_num_chunks != 1 or set(chunks) != {0}:
                    raise ValueError(
                        "request-level Raiden transfer requires exactly one finalized chunk"
                    )
                metadata = first_chunk.get("transport_metadata", first_chunk)
                if not isinstance(metadata, Mapping):
                    raise TypeError("Raiden request transport_metadata must be a mapping")
                remote_block_ids = tuple(int(v) for v in metadata.get("remote_block_ids", ()))
                expected_blocks = (
                    context.prompt_tokens + context.page_size - 1
                ) // context.page_size
                if len(remote_block_ids) != expected_blocks:
                    raise ValueError(
                        f"Raiden block count mismatch: expected={expected_blocks}, "
                        f"remote={len(remote_block_ids)}"
                    )
                if len(local_block_ids) != len(remote_block_ids):
                    raise ValueError(
                        f"Raiden local block count mismatch: remote={len(remote_block_ids)}, "
                        f"local={len(local_block_ids)}"
                    )
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
        *,
        base_transfer_id: str | None = None,
        chunk_index: int = 0,
        num_chunks: int = 1,
        chunk_page_offset: int = 0,
    ) -> None:
        if bootstrap_room is None:
            return
        transport_metadata: dict[str, object] = {"remote_block_ids": list(block_ids)}
        if debug_metadata is not None:
            transport_metadata["kv_debug"] = dict(debug_metadata)
        self.bootstrap_client.register_transfer(
            bootstrap_room,
            transfer_id,
            base_transfer_id=base_transfer_id,
            jax_process_index=jax.process_index(),
            prefill_dp_rank=dp_rank,
            chunk_index=chunk_index,
            num_chunks=num_chunks,
            chunk_page_offset=chunk_page_offset,
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
        self._chunk_mode = False
        self._started_chunks: set[int] = set()
        self._num_chunks: int | None = None

    @property
    def uuid(self) -> str:
        return self._transfer_id or self._req_id

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    @property
    def has_pending_failure(self) -> bool:
        with self._state_lock:
            return self._pending_failure_reason is not None

    @property
    def has_started_chunks(self) -> bool:
        with self._state_lock:
            return bool(self._started_chunks)

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

    def send_chunk(
        self,
        chunk_index: int,
        block_ids: list[int],
        *,
        bootstrap_room: int,
        chunk_page_offset: int,
        is_final: bool,
        dp_rank: int = 0,
    ) -> None:
        """Register and publish one newly-computed prefill chunk."""

        if not self._manager.enable_chunk_prefill_transfer:
            raise RuntimeError("chunk transfer is disabled on this Raiden manager")
        chunk_index = int(chunk_index)
        if chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if not block_ids:
            raise ValueError("Raiden chunk transfer requires at least one KV block")
        child_id = f"{self.uuid}#c{chunk_index}"
        with self._state_lock:
            if self.state not in (KVPoll.WAITING_FOR_INPUT, KVPoll.TRANSFERRING):
                raise RuntimeError(f"cannot send chunk from terminal state {self.state.value}")
            if chunk_index in self._started_chunks:
                raise RuntimeError(f"chunk_index={chunk_index} was already registered")
            if self._num_chunks is not None:
                raise RuntimeError("cannot register a chunk after the final chunk")

        needed = self._manager.register_read(
            child_id,
            child_id,
            list(block_ids),
            dp_rank=int(dp_rank),
        )
        if not needed:
            raise RuntimeError(f"Raiden declined non-empty chunk transfer for {child_id!r}")

        with self._state_lock:
            self._chunk_mode = True
            self._bootstrap_room = int(bootstrap_room)
            self._dp_rank = int(dp_rank)
            self._started_chunks.add(chunk_index)
            if is_final:
                self._num_chunks = chunk_index + 1
            if self.state == KVPoll.WAITING_FOR_INPUT:
                self._transition_to(KVPoll.TRANSFERRING)
                self._timer = time_phase("ack", "prefill")
                self._timer.__enter__()
            # Timeout each producer-to-consumer progress interval rather than
            # the whole long prompt. Otherwise a healthy multi-chunk request
            # can exceed ack_timeout solely because later chunks are computing.
            self._transfer_started_at = time.monotonic()

        try:
            self._manager.publish_transfer(
                child_id,
                list(block_ids),
                int(bootstrap_room),
                None,
                dp_rank=int(dp_rank),
                base_transfer_id=self.uuid,
                chunk_index=chunk_index,
                num_chunks=(chunk_index + 1 if is_final else 0),
                chunk_page_offset=int(chunk_page_offset),
            )
        except Exception:
            logger.exception(
                "Raiden chunk metadata publish failed for req_id=%s chunk=%d",
                self._req_id,
                chunk_index,
            )
            self.request_abort("bootstrap_register")
            raise

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
            chunk_mode = self._chunk_mode
            started_chunks = tuple(sorted(self._started_chunks))
            num_chunks = self._num_chunks
            pending_failure = self._pending_failure_reason
        self._manager.poll_engine()
        if chunk_mode:
            child_ids = [f"{self.uuid}#c{k}" for k in started_chunks]
            if not all(self._manager.sender_done(child_id) for child_id in child_ids):
                return KVPoll.TRANSFERRING
            if pending_failure is not None:
                return self._finish(KVPoll.FAILED, pending_failure)
            if num_chunks is None or started_chunks != tuple(range(num_chunks)):
                return KVPoll.TRANSFERRING
            return self._finish(KVPoll.SUCCESS, "raiden_chunks_done_sending")
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
            forget_ids = (
                [f"{self.uuid}#c{k}" for k in self._started_chunks]
                if self._chunk_mode
                else [self.uuid]
            )
        for transfer_id in forget_ids:
            self._manager.forget(transfer_id)
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
        self._metadata: RaidenMetadata | RaidenChunkedMetadata | None = None
        self._state_lock = threading.Lock()
        self._started = False
        self._started_chunks: set[int] = set()
        self._chunk_records: dict[int, Mapping[str, object]] = {}
        self._known_num_chunks: int | None = None
        self._timer: object | None = None
        self._transfer_started_at: float | None = None
        self._pending_failure_reason: str | None = None

    @property
    def transfer_started_at(self) -> float | None:
        return self._transfer_started_at

    def init(self, p_metadata) -> None:
        if not isinstance(p_metadata, (RaidenMetadata, RaidenChunkedMetadata)):
            raise TypeError(
                "expected RaidenMetadata or RaidenChunkedMetadata, "
                f"got {type(p_metadata).__name__}"
            )
        self._metadata = p_metadata
        if isinstance(p_metadata, RaidenChunkedMetadata):
            self._chunk_records = {
                int(index): value for index, value in p_metadata.initial_chunks.items()
            }
            self._known_num_chunks = (
                int(p_metadata.known_num_chunks) if p_metadata.known_num_chunks > 0 else None
            )
        self._transition_to(KVPoll.WAITING_FOR_INPUT)

    def poll(self) -> KVPoll:
        if isinstance(self._metadata, RaidenChunkedMetadata):
            return self._poll_chunked()
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

    def _poll_chunked(self) -> KVPoll:
        metadata = self._metadata
        assert isinstance(metadata, RaidenChunkedMetadata)
        state = self.state
        if state == KVPoll.WAITING_FOR_INPUT:
            with self._state_lock:
                if self.state != KVPoll.WAITING_FOR_INPUT:
                    return self.state
                self._transition_to(KVPoll.TRANSFERRING)
                self._timer = time_phase("pull", "decode")
                self._timer.__enter__()
                self._transfer_started_at = time.monotonic()
            self._discover_and_start_chunks(metadata)
            state = self.state

        if state != KVPoll.TRANSFERRING:
            return state

        with self._state_lock:
            pending_failure = self._pending_failure_reason
        if pending_failure is None:
            self._discover_and_start_chunks(metadata)

        self._manager.poll_engine()
        with self._state_lock:
            started_chunks = tuple(sorted(self._started_chunks))
            num_chunks = self._known_num_chunks
            pending_failure = self._pending_failure_reason

        states = {
            chunk_index: self._manager.receiver_state(f"{metadata.base_uuid}#c{chunk_index}")
            for chunk_index in started_chunks
        }
        if any(value == "failed" for value in states.values()):
            self.request_abort("raiden_failed_receiving")
            pending_failure = self._pending_failure_reason
        all_started_terminal = all(value in ("done", "failed") for value in states.values())
        if pending_failure is not None:
            if all_started_terminal:
                return self._finish(KVPoll.FAILED, pending_failure)
            return KVPoll.TRANSFERRING
        if num_chunks is None or started_chunks != tuple(range(num_chunks)):
            return KVPoll.TRANSFERRING
        if not all(value == "done" for value in states.values()):
            return KVPoll.TRANSFERRING
        try:
            self._validate_complete_chunk_layout(metadata, num_chunks)
        except (TypeError, ValueError):
            logger.exception("Raiden chunk layout validation failed for %s", self._req_id)
            self.request_abort("chunk_layout")
            return self._finish(KVPoll.FAILED, "chunk_layout")
        return self._finish(KVPoll.SUCCESS, "raiden_chunks_done_receiving")

    def _discover_and_start_chunks(self, metadata: RaidenChunkedMetadata) -> None:
        with self._state_lock:
            if self.state != KVPoll.TRANSFERRING or self._pending_failure_reason is not None:
                return
            complete = self._known_num_chunks is not None and self._started_chunks == set(
                range(self._known_num_chunks)
            )
        if complete:
            return

        try:
            info = self._manager.bootstrap_client.get_transfer_info(
                metadata.bootstrap_room,
                jax_process_index=metadata.jax_process_index,
                prefill_dp_rank=metadata.prefill_dp_rank,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Raiden chunk metadata lookup failed for room=%s: %s",
                metadata.bootstrap_room,
                exc,
            )
            return
        if info is not None:
            chunks, known_num_chunks = _normalize_transfer_bundle(info)
            conflict_reason: str | None = None
            with self._state_lock:
                for chunk_index, record in chunks.items():
                    existing = self._chunk_records.get(chunk_index)
                    if existing is not None and existing != record:
                        conflict_reason = "chunk_metadata_conflict"
                        break
                    self._chunk_records[chunk_index] = record
                if conflict_reason is None and known_num_chunks:
                    if self._known_num_chunks not in (None, known_num_chunks):
                        conflict_reason = "chunk_count_conflict"
                    elif any(index >= known_num_chunks for index in self._chunk_records):
                        conflict_reason = "chunk_index_out_of_range"
                    else:
                        self._known_num_chunks = known_num_chunks
            if conflict_reason is not None:
                self.request_abort(conflict_reason)
                return

        while True:
            with self._state_lock:
                started_chunks = tuple(self._started_chunks)
                candidates = sorted(set(self._chunk_records) - self._started_chunks)
                if not candidates or self._pending_failure_reason is not None:
                    return
                chunk_index = candidates[0]
                record = self._chunk_records[chunk_index]
            started_states = [
                self._manager.receiver_state(f"{metadata.base_uuid}#c{index}")
                for index in started_chunks
            ]
            if any(state == "failed" for state in started_states):
                self.request_abort("raiden_failed_receiving")
                return
            active_pulls = sum(state is None for state in started_states)
            if active_pulls >= CHUNK_TRANSFER_WINDOW:
                return
            try:
                expected_id = f"{metadata.base_uuid}#c{chunk_index}"
                if str(record.get("transfer_id", "")) != expected_id:
                    raise ValueError(
                        f"chunk transfer ID mismatch: expected={expected_id!r}, "
                        f"got={record.get('transfer_id')!r}"
                    )
                if int(record.get("chunk_index", chunk_index)) != chunk_index:
                    raise ValueError("chunk_index does not match its registry key")
                if int(record.get("prefill_dp_rank", 0)) != metadata.prefill_dp_rank:
                    raise ValueError("chunk Prefill DP rank mismatch")
                transport = record.get("transport_metadata", record)
                if not isinstance(transport, Mapping):
                    raise TypeError("chunk transport_metadata must be a mapping")
                remote_block_ids = [int(value) for value in transport.get("remote_block_ids", ())]
                if not remote_block_ids:
                    raise ValueError("Raiden chunk metadata has no remote blocks")
                page_offset = int(record.get("chunk_page_offset", 0))
                local_block_ids = list(
                    metadata.local_block_ids[page_offset : page_offset + len(remote_block_ids)]
                )
                if len(local_block_ids) != len(remote_block_ids):
                    raise ValueError(
                        "Raiden chunk local block range is out of bounds: "
                        f"offset={page_offset}, remote={len(remote_block_ids)}, "
                        f"local_total={len(metadata.local_block_ids)}"
                    )
                self._validate_chunk_ranges(chunk_index, page_offset, len(remote_block_ids))
                self._manager.wrapper.start_read(
                    expected_id,
                    _uuid_to_int(expected_id),
                    metadata.remote_endpoint,
                    remote_block_ids,
                    local_block_ids,
                    decode_dp_rank=metadata.decode_dp_rank,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Raiden start_read failed for req_id=%s chunk=%d",
                    self._req_id,
                    chunk_index,
                )
                self.request_abort("raiden_start_read")
                return
            with self._state_lock:
                self._started_chunks.add(chunk_index)
                # Bound time since the latest started pull, not total prompt
                # duration. The reaper still catches a producer that stops
                # publishing after this chunk.
                self._transfer_started_at = time.monotonic()

    def _validate_chunk_ranges(self, chunk_index: int, page_offset: int, num_pages: int) -> None:
        if page_offset < 0:
            raise ValueError("chunk_page_offset must be non-negative")
        new_end = page_offset + num_pages
        for other_index, record in self._chunk_records.items():
            if other_index == chunk_index:
                continue
            transport = record.get("transport_metadata", record)
            if not isinstance(transport, Mapping):
                continue
            other_pages = len(transport.get("remote_block_ids", ()))
            if other_pages <= 0:
                continue
            other_start = int(record.get("chunk_page_offset", 0))
            other_end = other_start + other_pages
            if max(page_offset, other_start) < min(new_end, other_end):
                raise ValueError(
                    f"Raiden chunk page ranges overlap: chunk={chunk_index}, "
                    f"other={other_index}"
                )

    def _validate_complete_chunk_layout(
        self,
        metadata: RaidenChunkedMetadata,
        num_chunks: int,
    ) -> None:
        expected_page_offset = 0
        seen_remote_blocks: set[int] = set()
        for chunk_index in range(num_chunks):
            record = self._chunk_records[chunk_index]
            page_offset = int(record.get("chunk_page_offset", 0))
            if page_offset != expected_page_offset:
                raise ValueError(
                    "Raiden chunk page ranges must cover the prompt contiguously: "
                    f"chunk={chunk_index}, expected_offset={expected_page_offset}, "
                    f"actual_offset={page_offset}"
                )
            transport = record.get("transport_metadata", record)
            if not isinstance(transport, Mapping):
                raise TypeError("chunk transport_metadata must be a mapping")
            remote_blocks = [int(value) for value in transport.get("remote_block_ids", ())]
            if not remote_blocks:
                raise ValueError(f"Raiden chunk {chunk_index} has no remote blocks")
            duplicate_blocks = seen_remote_blocks.intersection(remote_blocks)
            if duplicate_blocks:
                raise ValueError(
                    "Raiden remote blocks repeat across chunks: "
                    f"chunk={chunk_index}, duplicates={sorted(duplicate_blocks)}"
                )
            seen_remote_blocks.update(remote_blocks)
            expected_page_offset += len(remote_blocks)
        if expected_page_offset != len(metadata.local_block_ids):
            raise ValueError(
                "Raiden chunks do not cover all decode prompt pages: "
                f"covered={expected_page_offset}, expected={len(metadata.local_block_ids)}"
            )

    def commit(self, install: Callable[[Any], None]) -> None:
        del install
        if self.state != KVPoll.SUCCESS:
            raise RuntimeError("cannot commit an incomplete Raiden receive")
        assert self._metadata is not None
        if self._metadata.direct_commit is not None:
            expected_debug = (
                self._metadata.expected_debug
                if isinstance(self._metadata, RaidenMetadata)
                else None
            )
            self._metadata.direct_commit(expected_debug)

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
            transfer_id = (
                metadata.uuid
                if isinstance(metadata, RaidenMetadata)
                else (
                    metadata.base_uuid
                    if isinstance(metadata, RaidenChunkedMetadata)
                    else self._req_id
                )
            )
            self._manager.record_terminal(
                self._req_id,
                role="decode",
                transfer_id=transfer_id,
                state=state,
                reason=reason,
            )
            forget_ids = (
                [f"{metadata.base_uuid}#c{k}" for k in self._started_chunks]
                if isinstance(metadata, RaidenChunkedMetadata)
                else [transfer_id]
            )
        for child_id in forget_ids:
            self._manager.forget(child_id)
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
