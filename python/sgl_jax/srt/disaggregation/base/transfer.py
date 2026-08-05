"""Engine-neutral request contexts for PD KV transfer backends."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from sgl_jax.srt.disaggregation.base.kv_manager import KVReceiver, KVSender


def slots_to_page_ids(slots: Any, page_size: int, token_count: int) -> tuple[int, ...]:
    """Validate page-backed token slots and return their ordered page IDs."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if token_count <= 0:
        raise ValueError(f"token_count must be positive, got {token_count}")
    values = np.asarray(slots).reshape(-1)
    if values.size < token_count:
        raise ValueError(
            f"KV slot count is smaller than token count: slots={values.size}, tokens={token_count}"
        )
    values = values[:token_count].astype(np.int64, copy=False)
    pages: list[int] = []
    for offset in range(0, token_count, page_size):
        chunk = values[offset : min(offset + page_size, token_count)]
        first = int(chunk[0])
        if first < 0 or first % page_size != 0:
            raise ValueError(
                f"KV page starts at a non-aligned slot: slot={first}, page_size={page_size}"
            )
        expected = np.arange(first, first + chunk.size, dtype=np.int64)
        if not np.array_equal(chunk, expected):
            raise ValueError(
                f"KV slots are not contiguous within page {first // page_size}: "
                f"got={chunk.tolist()}"
            )
        pages.append(first // page_size)
    if len(set(pages)) != len(pages):
        raise ValueError(f"KV page IDs must be unique, got={pages}")
    return tuple(pages)


@dataclass(frozen=True)
class PrefillTransferContext:
    req_id: str
    transfer_id: str
    bootstrap_room: int | None
    dp_rank: int
    buffer_id: int | None
    payload_factory: Callable[[], dict[str, Any]]
    block_ids_factory: Callable[[], list[int]]
    on_payload: Callable[[dict[str, Any]], None] | None = None
    on_ready: Callable[[], None] | None = None


@dataclass(frozen=True)
class PrefillTransfer:
    sender: KVSender
    # True only when the backend retained another complete copy. Direct-HBM
    # engines keep the allocator pages owned until the sender is terminal.
    release_device_kv: bool = False


@dataclass(frozen=True)
class DecodeTransferContext:
    req_id: str
    transfer_id: str
    bootstrap_room: int | None
    decode_dp_rank: int
    prefill_dp_rank: int
    peer_info: Mapping[str, object]
    kv_indices: Any
    page_size: int
    prompt_tokens: int
    spec_factory: Callable[[], Any]
    direct_commit: Callable[[Mapping[str, object] | None], None] | None = None


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

    def prepare_prefill_batch(self, kv_buffers: Any) -> None: ...

    def start_prefill(self, context: PrefillTransferContext) -> PrefillTransfer: ...

    def try_start_decode(self, context: DecodeTransferContext) -> DecodeAdmission: ...

    def prefill_transport_metadata(self, dp_rank: int = 0) -> dict[str, object]: ...

    def cleanup_transfer(
        self,
        bootstrap_room: int | None,
        *,
        jax_process_index: int | None = None,
        prefill_dp_rank: int = 0,
    ) -> None: ...

    def inflight_count(self) -> tuple[int, int]: ...

    def start_reaper(self) -> None: ...

    def graceful_shutdown(self, drain_timeout_seconds: float = 30.0) -> None: ...
