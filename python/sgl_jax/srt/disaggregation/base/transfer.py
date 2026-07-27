"""Engine-neutral request contexts for PD KV transfer backends."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from sgl_jax.srt.disaggregation.base.kv_manager import KVReceiver, KVSender


@dataclass(frozen=True)
class PrefillTransferContext:
    req_id: str
    transfer_id: str
    bootstrap_room: int | None
    buffer_id: int | None
    payload_factory: Callable[[], dict[str, Any]]
    block_ids_factory: Callable[[], list[int]]
    on_payload: Callable[[dict[str, Any]], None] | None = None
    on_ready: Callable[[], None] | None = None


@dataclass(frozen=True)
class PrefillTransfer:
    sender: KVSender
    release_device_kv: bool = False


@dataclass(frozen=True)
class DecodeTransferContext:
    req_id: str
    transfer_id: str
    bootstrap_room: int | None
    peer_info: Mapping[str, object]
    kv_indices: Any
    page_size: int
    prompt_tokens: int
    spec_factory: Callable[[], Any]


class AdmissionState(enum.Enum):
    ADMITTED = "admitted"
    DEFERRED = "deferred"


@dataclass(frozen=True)
class DecodeAdmission:
    state: AdmissionState
    receiver: KVReceiver | None = None
    reason: str | None = None

    @classmethod
    def admitted(cls, receiver: KVReceiver) -> DecodeAdmission:
        return cls(AdmissionState.ADMITTED, receiver=receiver)

    @classmethod
    def deferred(cls, reason: str) -> DecodeAdmission:
        return cls(AdmissionState.DEFERRED, reason=reason)


class TransferBackend(Protocol):
    engine_name: str
    requires_host_staging: bool
    host_pool: Any

    def reserve_prefill_buffer(self, existing: int | None) -> int | None: ...

    def start_prefill(self, context: PrefillTransferContext) -> PrefillTransfer: ...

    def try_start_decode(self, context: DecodeTransferContext) -> DecodeAdmission: ...

    def prefill_transport_metadata(self) -> dict[str, object]: ...

    def cleanup_transfer(self, bootstrap_room: int | None) -> None: ...

    def inflight_count(self) -> tuple[int, int]: ...

    def start_reaper(self) -> None: ...

    def graceful_shutdown(self, drain_timeout_seconds: float = 30.0) -> None: ...
