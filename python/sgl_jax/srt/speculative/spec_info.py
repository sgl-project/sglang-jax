from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from enum import IntEnum, auto
from typing import Protocol, runtime_checkable

import jax
import numpy as np

from sgl_jax.srt.layers.logits_processor import LogitsProcessorOutput

logger = logging.getLogger(__name__)


@runtime_checkable
class SpecInput(Protocol):
    """Common interface for speculative-decode state passed through
    ``ModelWorkerBatch.spec_info`` (#1053 P1-5a data contract).

    Separates three token counts that the scheduler / KV allocator / verify
    path each need but which differ under spec decode:

    - **logical** — tokens the scheduler advances request output by
      (= accepted count incl. bonus). Host scalar/array.
    - **allocated** — KV slots already pre-allocated this round (for trimming
      over-allocation on finished reqs). Host array.
    - **verify** — flattened token count target verify will forward (drives
      verify attention metadata + DP token accounting). Host scalar.

    Implementations MUST NOT hold worker/runner/pool/future/callback handles
    in pytree children (these would enter the JIT cache key). Device arrays
    (``topk_p``, ``hidden_states``, ``draft_token``, ...) stay on device;
    lengths/indices stay host-side ``np.ndarray``.

    DP layout (Route 1, target+draft both DP): all per-request fields use
    DP-padded order — section ``[dp_rank*per_dp_bs : dp_rank*per_dp_bs+real_bs]``.
    Padding slots MUST NOT participate in valid state updates.
    """

    def is_draft_input(self) -> bool: ...
    def is_verify_input(self) -> bool: ...

    def get_spec_adjust_token_coefficient(self) -> int:
        """Multiplier for scheduler new-token budgeting (e.g. draft_token_num)."""
        ...

    def get_logical_token_num(self, bs: int) -> np.ndarray:
        """Per-request host int32 ``(bs,)``; callers sum for batch totals."""
        ...

    def get_allocated_token_num(self) -> np.ndarray | None: ...
    def get_verify_token_num(self, bs: int) -> int: ...

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None: ...
    def merge_batch(self, other: SpecInput) -> None: ...


@runtime_checkable
class SpecDraftState(SpecInput, Protocol):
    """Persistent speculative state that crosses scheduler DP boundaries."""

    def scatter_to_dp_slots(
        self,
        selector: np.ndarray,
        total_bs: int,
        *,
        mesh=None,
        legacy_host_scatter: bool = False,
    ) -> SpecDraftState: ...

    def split_per_rank(self, real_bs_per_dp: list[int]) -> list[SpecDraftState | None]: ...

    @classmethod
    def concat_per_rank(
        cls, per_rank: Sequence[SpecDraftState | None]
    ) -> SpecDraftState | None: ...


def scatter_spec_state_fields(
    state,
    selector: np.ndarray,
    total_bs: int,
    *,
    per_request_fields: Sequence[str],
    static_fields: Mapping[str, object],
    required_fields: Sequence[str] = (),
    data_sharded_fields: Sequence[str] = (),
    passthrough_fields: Sequence[str] = (),
    mesh=None,
):
    """Scatter compact per-request state into scheduler DP-padded slots."""
    selector = np.asarray(selector, dtype=np.int32)
    required = set(required_fields)
    data_sharded = set(data_sharded_fields)
    passthrough = set(passthrough_fields)
    kwargs = dict(static_fields)

    for field in per_request_fields:
        value = getattr(state, field, None)
        if field in passthrough:
            kwargs[field] = value
            continue
        if value is None:
            if field in required:
                raise ValueError(f"Spec state field {field!r} is missing before DP scatter.")
            kwargs[field] = None
            continue

        array = np.asarray(value)
        if array.shape[0] != len(selector):
            if field in required:
                raise ValueError(
                    "Spec state length does not match real request slots before DP scatter: "
                    f"field={field}, state_bs={array.shape[0]}, real_bs={len(selector)}."
                )
            kwargs[field] = None
            continue

        padded = np.zeros((int(total_bs),) + array.shape[1:], dtype=array.dtype)
        padded[selector] = array
        if field in data_sharded and mesh is not None:
            from jax.sharding import NamedSharding
            from jax.sharding import PartitionSpec as P

            padded = jax.device_put(padded, NamedSharding(mesh, P("data")))
        kwargs[field] = padded

    return type(state)(**kwargs)


def split_spec_state_fields(
    state,
    real_bs_per_dp: Sequence[int],
    *,
    per_request_fields: Sequence[str],
    static_fields: Mapping[str, object],
    omitted_fields: Sequence[str] = (),
) -> list:
    """Split compact rank-major state into one state object per DP rank."""
    omitted = set(omitted_fields)
    out = []
    offset = 0
    for rank_bs in real_bs_per_dp:
        rank_bs = int(rank_bs)
        if rank_bs == 0:
            out.append(None)
            continue

        end = offset + rank_bs
        kwargs = dict(static_fields)
        for field in per_request_fields:
            value = getattr(state, field, None)
            kwargs[field] = None if field in omitted or value is None else value[offset:end]
        out.append(type(state)(**kwargs))
        offset = end
    return out


def concat_spec_state_fields(
    states: Sequence,
    *,
    per_request_fields: Sequence[str],
    static_fields: Mapping[str, object],
    allow_partial_fields: Sequence[str] = (),
    empty_or_none_fields: Sequence[str] = (),
):
    """Concatenate per-rank state into compact rank-major request order."""
    if not states:
        return None

    state_type = type(states[0])
    if any(type(state) is not state_type for state in states):
        raise TypeError("Cannot concatenate speculative states of different types.")

    allow_partial = set(allow_partial_fields)
    empty_or_none = set(empty_or_none_fields)
    kwargs = dict(static_fields)
    for field in per_request_fields:
        values = [getattr(state, field, None) for state in states]
        if field in empty_or_none:
            materialized = [
                value for value in values if value is not None and int(value.shape[0]) > 0
            ]
            if not materialized:
                kwargs[field] = None
                continue

        nonnull = [value for value in values if value is not None]
        if not nonnull:
            kwargs[field] = None
            continue
        if field in allow_partial and len(nonnull) != len(states):
            kwargs[field] = None
            continue
        if len(nonnull) != len(states):
            raise ValueError(
                f"Spec state field {field!r} is None on "
                f"{len(states) - len(nonnull)}/{len(states)} nonempty rank(s); "
                "all-or-nothing required."
            )
        if len(nonnull) == 1:
            kwargs[field] = nonnull[0]
            continue
        kwargs[field] = np.concatenate([np.asarray(value) for value in nonnull], axis=0)

    return state_type(**kwargs)


class SpecDraftStateMixin:
    """Shared DP layout implementation for persistent speculative state."""

    _dp_fields: Sequence[str] = ()
    _dp_required_scatter_fields: Sequence[str] = ()
    _dp_required_split_fields: Sequence[str] = ()
    _dp_data_sharded_fields: Sequence[str] = ()
    _dp_legacy_passthrough_fields: Sequence[str] = ()
    _dp_allow_partial_fields: Sequence[str] = ()
    _dp_empty_or_none_fields: Sequence[str] = ()
    _dp_future_retained_fields: Sequence[str] = ()

    def _dp_static_fields(self) -> dict[str, object]:
        fields = {"capture_hidden_mode": self.capture_hidden_mode}
        if hasattr(self, "block_size"):
            fields["block_size"] = self.block_size
        return fields

    def scatter_to_dp_slots(
        self,
        selector: np.ndarray,
        total_bs: int,
        *,
        mesh=None,
        legacy_host_scatter: bool = False,
    ):
        return scatter_spec_state_fields(
            self,
            selector,
            total_bs,
            per_request_fields=self._dp_fields,
            static_fields=self._dp_static_fields(),
            required_fields=self._dp_required_scatter_fields,
            data_sharded_fields=() if legacy_host_scatter else self._dp_data_sharded_fields,
            passthrough_fields=(self._dp_legacy_passthrough_fields if legacy_host_scatter else ()),
            mesh=mesh,
        )

    def split_per_rank(self, real_bs_per_dp: list[int]) -> list:
        has_future_indices = getattr(self, "future_indices", None) is not None
        if getattr(self, "pending_draft_extend_result", None) is not None:
            self.resolve_pending_draft_extend_result()
        if not has_future_indices and self._dp_required_split_fields:
            ensure_host = getattr(self, "_ensure_host", None)
            if ensure_host is not None:
                ensure_host()
            missing = [
                field
                for field in self._dp_required_split_fields
                if getattr(self, field, None) is None
            ]
            if missing:
                field_states = {
                    field: getattr(getattr(self, field, None), "shape", None)
                    for field in self._dp_fields
                }
                raise RuntimeError(
                    f"{type(self).__name__} is incomplete without pending result; "
                    f"missing={missing}, field_states={field_states}, "
                    f"real_bs_per_dp={real_bs_per_dp}"
                )

        omitted_fields = ()
        if has_future_indices:
            retained = set(self._dp_future_retained_fields)
            omitted_fields = tuple(field for field in self._dp_fields if field not in retained)
        return split_spec_state_fields(
            self,
            real_bs_per_dp,
            per_request_fields=self._dp_fields,
            static_fields=self._dp_static_fields(),
            omitted_fields=omitted_fields,
        )

    @classmethod
    def concat_per_rank(cls, per_rank: Sequence):
        states = [state for state in per_rank if state is not None]
        if not states:
            return None

        has_future_indices = any(
            getattr(state, "future_indices", None) is not None for state in states
        )
        if has_future_indices and not all(
            getattr(state, "future_indices", None) is not None for state in states
        ):
            raise ValueError(
                "Every nonempty rank must carry future_indices on the relay-buffer path."
            )
        if not has_future_indices:
            for state in states:
                resolver = getattr(state, "resolve_pending_draft_extend_result", None)
                if resolver is not None:
                    resolver()

        return concat_spec_state_fields(
            states,
            per_request_fields=states[0]._dp_fields,
            static_fields=states[0]._dp_static_fields(),
            allow_partial_fields=states[0]._dp_allow_partial_fields,
            empty_or_none_fields=states[0]._dp_empty_or_none_fields,
        )


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    NEXTN = auto()
    STANDALONE = auto()
    DFLASH = auto()

    def is_none(self):
        return self == SpeculativeAlgorithm.NONE

    def is_eagle(self):
        return self in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.NEXTN,
        )

    def is_eagle3(self):
        return self == SpeculativeAlgorithm.EAGLE3

    def is_nextn(self):
        return self == SpeculativeAlgorithm.NEXTN

    def is_standalone(self):
        return self == SpeculativeAlgorithm.STANDALONE

    def is_dflash(self):
        return self == SpeculativeAlgorithm.DFLASH

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "NEXTN": SpeculativeAlgorithm.NEXTN,
            "STANDALONE": SpeculativeAlgorithm.STANDALONE,
            "DFLASH": SpeculativeAlgorithm.DFLASH,
            None: SpeculativeAlgorithm.NONE,
        }
        if name is not None:
            name = name.upper()
        return name_map[name]


def detect_nan(logits_output: LogitsProcessorOutput):
    logits = logits_output.next_token_logits
    if jax.numpy.any(jax.numpy.isnan(logits)):
        logger.error("Detected errors during sampling! NaN in the logits.")
        raise ValueError("Detected errors during sampling! NaN in the logits.")
