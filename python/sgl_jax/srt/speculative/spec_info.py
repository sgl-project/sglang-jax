from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from typing import Any, ClassVar, Protocol, Self, runtime_checkable

import jax
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput

logger = logging.getLogger(__name__)


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    NEXTN = auto()
    DFLASH = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_eagle(self):
        """Whether the algorithm uses EAGLE-style speculative state."""
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.NEXTN,
        )

    def is_eagle_family(self):
        return self in (SpeculativeAlgorithm.EAGLE, SpeculativeAlgorithm.EAGLE3)

    def is_nextn(self):
        return self == SpeculativeAlgorithm.NEXTN

    def is_dflash(self):
        return self == SpeculativeAlgorithm.DFLASH

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "NEXTN": SpeculativeAlgorithm.NEXTN,
            "DFLASH": SpeculativeAlgorithm.DFLASH,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]


class SpecRelayPolicy(Enum):
    """How a field participates when state is represented by relay indices."""

    DROP = auto()
    SCATTER_ONLY = auto()
    KEEP = auto()


class SpecConcatPolicy(Enum):
    """How per-rank optional values are normalized during concatenation."""

    REQUIRE_ALL = auto()
    DROP_IF_PARTIAL = auto()
    EMPTY_IS_NONE = auto()


@dataclass(frozen=True)
class SpecStateField:
    """Layout policy for one per-request speculative draft-state field."""

    name: str
    relay: SpecRelayPolicy = SpecRelayPolicy.SCATTER_ONLY
    concat: SpecConcatPolicy = SpecConcatPolicy.REQUIRE_ALL
    required_for_scatter: bool = False
    required_for_split: bool = False
    data_sharded: bool = False
    preserve_compact_on_host: bool = False


@dataclass(frozen=True)
class SpecStateLayout:
    """Declarative layout used by common draft-state batch transitions."""

    name: str
    fields: tuple[SpecStateField, ...]
    static_fields: tuple[str, ...] = ()
    relay_indicator: str = "future_indices"
    ensure_host_before_split: bool = False


@runtime_checkable
class SpecDraftState(Protocol):
    """Scheduler-facing contract for persistent speculative draft state."""

    STATE_LAYOUT: ClassVar[SpecStateLayout]
    allocate_lens: np.ndarray | None
    future_indices: np.ndarray | None

    def new_tokens_required_next_decode(self, requests, page_size: int) -> int: ...
    def prepare_for_decode(self, schedule_batch: Any) -> None: ...
    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None: ...
    def trim_to_length(self, n: int) -> None: ...
    def merge_batch(self, other: Self) -> None: ...

    def scatter_to_dp_slots(
        self,
        selector: np.ndarray,
        total_bs: int,
        *,
        mesh: Any = None,
        host_state_scatter: bool = False,
    ) -> Self: ...

    def split_per_rank(self, real_bs_per_dp: list[int]) -> list[Self | None]: ...

    @classmethod
    def concat_per_rank(cls, per_rank: list[Self]) -> Self: ...


@runtime_checkable
class SpecVerifyInput(Protocol):
    """Minimal contract for transient target-verification input."""

    draft_token: jax.Array


SpecForwardInput = SpecDraftState | SpecVerifyInput

# Backward-compatible name for callers that imported the old, unused protocol.
SpecInput = SpecDraftState


class SpecDraftStateMixin:
    """Template implementation for mechanical DP draft-state transitions."""

    STATE_LAYOUT: ClassVar[SpecStateLayout]

    def trim_to_length(self, n: int) -> None:
        self._ensure_host()
        for field in self.STATE_LAYOUT.fields:
            value = getattr(self, field.name, None)
            if value is not None and len(value) != n:
                setattr(self, field.name, value[:n])

    def scatter_to_dp_slots(
        self,
        selector: np.ndarray,
        total_bs: int,
        *,
        mesh=None,
        host_state_scatter: bool = False,
    ):
        """Scatter compact request state into DP-padded request slots."""
        layout = self.STATE_LAYOUT
        relay_state = getattr(self, layout.relay_indicator, None) is not None
        kwargs = {field_name: getattr(self, field_name) for field_name in layout.static_fields}

        for field in layout.fields:
            value = getattr(self, field.name, None)
            if relay_state and field.relay is SpecRelayPolicy.DROP:
                kwargs[field.name] = None
                continue
            if host_state_scatter and field.preserve_compact_on_host:
                kwargs[field.name] = value
                continue
            if value is None:
                if field.required_for_scatter:
                    raise ValueError(
                        f"{layout.name} state field {field.name!r} is missing before DP scatter."
                    )
                kwargs[field.name] = None
                continue

            array = np.asarray(value)
            if array.shape[0] != len(selector):
                if field.required_for_scatter:
                    raise ValueError(
                        f"{layout.name} state length does not match real request "
                        "slots before DP scatter: "
                        f"field={field.name}, state_bs={array.shape[0]}, "
                        f"real_bs={len(selector)}."
                    )
                kwargs[field.name] = None
                continue

            padded = np.zeros((total_bs,) + array.shape[1:], dtype=array.dtype)
            padded[selector] = array
            if field.data_sharded and not host_state_scatter and mesh is not None:
                padded = jax.device_put(padded, NamedSharding(mesh, P("data")))
            kwargs[field.name] = padded

        return type(self)(**kwargs)

    def split_per_rank(self, real_bs_per_dp: list[int]):
        """Split rank-major compact state into one state object per DP rank."""
        layout = self.STATE_LAYOUT
        relay_state = getattr(self, layout.relay_indicator, None) is not None
        if layout.ensure_host_before_split and not relay_state:
            self._ensure_host()

        missing = [
            field.name
            for field in layout.fields
            if field.required_for_split and getattr(self, field.name, None) is None
        ]
        if missing and not relay_state:
            field_states = {
                field.name: self._field_state(getattr(self, field.name, None))
                for field in layout.fields
            }
            raise RuntimeError(
                f"Cannot split incomplete {layout.name} draft state "
                f"missing={missing}, field_states={field_states}, "
                f"real_bs_per_dp={real_bs_per_dp}"
            )

        result = []
        offset = 0
        for size in real_bs_per_dp:
            if size == 0:
                result.append(None)
                continue
            end = offset + size
            kwargs = {field_name: getattr(self, field_name) for field_name in layout.static_fields}
            for field in layout.fields:
                value = getattr(self, field.name, None)
                if relay_state and field.relay is not SpecRelayPolicy.KEEP:
                    kwargs[field.name] = None
                else:
                    kwargs[field.name] = None if value is None else value[offset:end]
            result.append(type(self)(**kwargs))
            offset = end
        return result

    @classmethod
    def concat_per_rank(cls, per_rank: list):
        """Concatenate per-rank state into rank-major compact state."""
        layout = cls.STATE_LAYOUT
        relay_state = any(
            getattr(state, layout.relay_indicator, None) is not None for state in per_rank
        )
        if relay_state and not all(
            getattr(state, layout.relay_indicator, None) is not None for state in per_rank
        ):
            raise ValueError(
                f"{layout.name} overlap concat requires {layout.relay_indicator} "
                "on every nonempty rank."
            )

        kwargs = {
            field_name: getattr(per_rank[0], field_name) for field_name in layout.static_fields
        }
        for field in layout.fields:
            values = [getattr(state, field.name, None) for state in per_rank]
            if relay_state and field.relay is not SpecRelayPolicy.KEEP:
                kwargs[field.name] = None
                continue
            if field.concat is SpecConcatPolicy.EMPTY_IS_NONE and not any(
                value is not None and value.shape[0] > 0 for value in values
            ):
                kwargs[field.name] = None
                continue

            nonnull = [value for value in values if value is not None]
            if not nonnull:
                kwargs[field.name] = None
                continue
            if field.concat is SpecConcatPolicy.DROP_IF_PARTIAL and len(nonnull) != len(per_rank):
                kwargs[field.name] = None
                continue
            assert len(nonnull) == len(per_rank), (
                f"{layout.name} concat field {field.name!r} is None on "
                f"{len(per_rank) - len(nonnull)}/{len(per_rank)} nonempty rank(s); "
                "all-or-nothing required"
            )
            if len(nonnull) == 1:
                kwargs[field.name] = nonnull[0]
            elif isinstance(nonnull[0], np.ndarray):
                kwargs[field.name] = np.concatenate(nonnull, axis=0)
            else:
                kwargs[field.name] = np.concatenate(
                    [np.asarray(value) for value in nonnull],
                    axis=0,
                )
        return cls(**kwargs)

    @staticmethod
    def _field_state(value):
        if value is None:
            return None
        shape = getattr(value, "shape", None)
        return shape if shape is not None else len(value)


def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token_pool,
    start_offsets,
    end_offsets,
    out_cache_loc,
):
    """Assign newly allocated KV slots to each request on the host."""
    start_offsets = np.asarray(start_offsets, dtype=np.int32)
    end_offsets = np.asarray(end_offsets, dtype=np.int32)
    out_cache_loc = np.asarray(out_cache_loc, dtype=np.int32)

    lengths = end_offsets - start_offsets
    total = int(np.sum(lengths))
    assert total == out_cache_loc.shape[0], (
        "not all allocated cache locations were assigned to req_to_token_pool: "
        f"assigned={total}, allocated={out_cache_loc.shape[0]}"
    )
    if total == 0:
        return

    row_indices = np.repeat(req_pool_indices, lengths)
    block_starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
    local_offsets = np.arange(total) - np.repeat(block_starts, lengths)
    col_indices = local_offsets + np.repeat(start_offsets, lengths)
    req_to_token_pool.req_to_token[row_indices, col_indices] = out_cache_loc


def detect_nan(logits_output: LogitsProcessorOutput):
    logits = logits_output.next_token_logits
    if jax.numpy.any(jax.numpy.isnan(logits)):
        logger.error("Detected errors during sampling! NaN in the logits.")
        raise ValueError("Detected errors during sampling! NaN in the logits.")
