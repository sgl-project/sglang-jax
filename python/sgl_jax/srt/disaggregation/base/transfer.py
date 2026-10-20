"""Engine-neutral request contexts for PD KV transfer backends."""

from __future__ import annotations

import enum
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NotRequired, Protocol, TypedDict, cast

import numpy as np

from sgl_jax.srt.disaggregation.base.kv_manager import KVReceiver, KVSender
from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStateSpec


class TransferPayload(TypedDict):
    """Structured PD transfer payload dictionary.

    * ``"kv"``: Per-layer list of KV page tensors (or stacked local shard on multi-host).
    * ``"recurrent_state"``: Optional per-layer list of temporal recurrent state tensors,
      shape per layer ``(num_heads, head_dim, head_dim)``.
    * ``"conv_state"``: Optional per-layer list (or nested list) of 1D conv state tensors,
      shape per layer ``(proj_size, conv_kernel_size - 1)``.
    """

    kv: Any
    recurrent_state: NotRequired[Any]
    conv_state: NotRequired[Any]


PayloadFactory = Callable[[], TransferPayload]
SpecFactory = Callable[[], TransferPayload | dict[str, Any] | Any]


def chunk_transfer_id(base_transfer_id: str, chunk_index: int) -> str:
    """Return the v5 wire ID for one chunk of a logical PD transfer."""

    if not base_transfer_id:
        raise ValueError("base_transfer_id must be non-empty")
    chunk_index = int(chunk_index)
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    return f"{base_transfer_id}#c{chunk_index}"


def parse_chunk_transfer_id(transfer_id: str) -> tuple[str, int]:
    """Parse a v5 chunk wire ID into ``(base_transfer_id, chunk_index)``."""

    base_transfer_id, separator, raw_index = str(transfer_id).rpartition("#c")
    if not separator or not base_transfer_id or not raw_index:
        raise ValueError(f"invalid chunk transfer_id={transfer_id!r}")
    try:
        chunk_index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"invalid chunk transfer_id={transfer_id!r}") from exc
    if chunk_index < 0:
        raise ValueError(f"invalid chunk transfer_id={transfer_id!r}")
    return base_transfer_id, chunk_index


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


def recurrent_slot_to_global_idx(
    slot_local_idx: int,
    *,
    dp_rank: int,
    dp_size: int = 1,
    slots_per_rank: int,
    allow_dummy: bool = False,
) -> int:
    """Convert a rank-local recurrent state slot into a global ``RecurrentStatePool`` index.

    Follows the DP-aware memory model:
    ``global_slot_idx = slot_local_idx + dp_rank * (slots_per_rank + 1)``,
    where ``+ 1`` accounts for each DP rank's reserved dummy slot ``0``.
    """

    dp_rank = int(dp_rank)
    dp_size = int(dp_size)
    slots_per_rank = int(slots_per_rank)
    slot_local_idx = int(slot_local_idx)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank={dp_rank} is outside [0, {dp_size})")
    if slots_per_rank <= 0:
        raise ValueError(f"slots_per_rank must be positive, got {slots_per_rank}")
    min_slot = 0 if allow_dummy else 1
    if not min_slot <= slot_local_idx <= slots_per_rank:
        raise ValueError(
            f"rank-local recurrent slot_local_idx={slot_local_idx} is outside "
            f"[{min_slot}, {slots_per_rank}]"
        )
    return slot_local_idx + dp_rank * (slots_per_rank + 1)


def _extract_conv_leaf_shapes(layer_entry: Any) -> list[tuple[int, ...]]:
    if isinstance(layer_entry, (list, tuple)):
        if not layer_entry:
            raise ValueError("conv_state layer entry must not be empty")
        return [tuple(int(d) for d in item.shape) for item in layer_entry]
    return [tuple(int(d) for d in layer_entry.shape)]


def validate_transfer_payload(
    payload: TransferPayload | Mapping[str, Any],
    *,
    recurrent_state_spec: RecurrentStateSpec | None = None,
) -> TransferPayload:
    """Validate a prefill/decode transfer payload dictionary and optional recurrent state shapes."""

    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("transfer payload must be a non-empty mapping")
    if "kv" not in payload or payload["kv"] is None:
        raise ValueError("transfer payload must contain a non-null 'kv' entry")
    for key in payload:
        if ":" in str(key):
            raise ValueError(f"payload entry name {key!r} must not contain ':'")

    has_rec = "recurrent_state" in payload and payload["recurrent_state"] is not None
    has_conv = "conv_state" in payload and payload["conv_state"] is not None
    if has_rec != has_conv:
        raise ValueError(
            "transfer payload must include both 'recurrent_state' and 'conv_state' "
            f"together (got recurrent_state={has_rec}, conv_state={has_conv})"
        )

    if recurrent_state_spec is not None:
        if not has_rec or not has_conv:
            raise ValueError(
                "hybrid model requires 'recurrent_state' and 'conv_state' in transfer payload"
            )
        num_layers = int(recurrent_state_spec.get("num_layers", 0) or 0)
        expected_rec_shape = tuple(
            int(d) for d in (recurrent_state_spec.get("recurrent_shape") or ())
        )
        expected_conv_shape = tuple(int(d) for d in (recurrent_state_spec.get("conv_shape") or ()))

        rec_list = payload["recurrent_state"]
        conv_list = payload["conv_state"]
        if not isinstance(rec_list, (list, tuple)) or not isinstance(conv_list, (list, tuple)):
            raise TypeError("'recurrent_state' and 'conv_state' must be per-layer sequences")
        if num_layers > 0 and len(rec_list) != num_layers:
            raise ValueError(
                f"recurrent_state layer count mismatch: expected={num_layers}, got={len(rec_list)}"
            )
        if num_layers > 0 and len(conv_list) != num_layers:
            raise ValueError(
                f"conv_state layer count mismatch: expected={num_layers}, got={len(conv_list)}"
            )
        if expected_rec_shape:
            for layer_idx, tensor in enumerate(rec_list):
                actual_shape = tuple(int(d) for d in tensor.shape)
                if actual_shape != expected_rec_shape:
                    raise ValueError(
                        f"recurrent_state layer {layer_idx} shape mismatch: "
                        f"expected={expected_rec_shape}, got={actual_shape}"
                    )
        if expected_conv_shape:
            for layer_idx, entry in enumerate(conv_list):
                for actual_shape in _extract_conv_leaf_shapes(entry):
                    if actual_shape != expected_conv_shape:
                        raise ValueError(
                            f"conv_state layer {layer_idx} shape mismatch: "
                            f"expected={expected_conv_shape}, got={actual_shape}"
                        )

    return cast(TransferPayload, dict(payload))


def normalize_decode_specs(spec_or_mapping: Any) -> dict[str, Any]:
    """Normalize ``DecodeTransferContext.spec_factory()`` output into a named spec dict."""

    if isinstance(spec_or_mapping, Mapping):
        if "kv" not in spec_or_mapping or spec_or_mapping["kv"] is None:
            raise ValueError("decode spec mapping must contain a non-null 'kv' entry")
        return dict(spec_or_mapping)
    if spec_or_mapping is None:
        raise ValueError("decode spec_factory returned None")
    return {"kv": spec_or_mapping}


@dataclass(frozen=True)
class PrefillTransferContext:
    req_id: str
    transfer_id: str
    bootstrap_room: int | None
    dp_rank: int
    buffer_id: int | None
    payload_factory: PayloadFactory
    block_ids_factory: Callable[[], list[int]]
    on_payload: Callable[[TransferPayload], None] | None = None
    on_ready: Callable[[], None] | None = None
    recurrent_slot_idx: int | None = None
    recurrent_state_spec: RecurrentStateSpec | None = None


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
    spec_factory: SpecFactory
    direct_commit: Callable[[Mapping[str, object] | None], None] | None = None
    recurrent_slot_idx: int | None = None
    recurrent_state_spec: RecurrentStateSpec | None = None


@dataclass(frozen=True)
class DecodeMetadataContext:
    req_id: str
    transfer_id: str
    bootstrap_room: int | None
    prefill_dp_rank: int
    peer_info: Mapping[str, object]
    recurrent_state_spec: RecurrentStateSpec | None = None


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

    def poll_decode_metadata(self, context: DecodeMetadataContext) -> bool: ...

    def try_start_decode(self, context: DecodeTransferContext) -> DecodeAdmission: ...

    def prefill_transport_metadata(self, dp_rank: int = 0) -> dict[str, object]: ...

    def cleanup_transfer(
        self,
        bootstrap_room: int | None,
        *,
        jax_process_index: int | None = None,
        prefill_dp_rank: int = 0,
        expected_transfer_id: str | None = None,
    ) -> None: ...

    def inflight_count(self) -> tuple[int, int]: ...

    def start_reaper(self) -> None: ...

    def graceful_shutdown(self, drain_timeout_seconds: float = 30.0) -> None: ...
