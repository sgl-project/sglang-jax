from __future__ import annotations

import logging
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


class SpeculativeAlgorithm(IntEnum):
    NONE = auto()
    EAGLE = auto()
    EAGLE3 = auto()
    NEXTN = auto()
    STANDALONE = auto()

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

    @staticmethod
    def from_string(name: str):
        name_map = {
            "EAGLE": SpeculativeAlgorithm.EAGLE,
            "EAGLE3": SpeculativeAlgorithm.EAGLE3,
            "NEXTN": SpeculativeAlgorithm.NEXTN,
            "STANDALONE": SpeculativeAlgorithm.STANDALONE,
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
