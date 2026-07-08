from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode


def compute_dflash_accept_len_and_bonus(
    candidates: jax.Array,
    target_predict: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Greedy DFlash chain verification.

    ``candidates[:, 0]`` is the already verified seed token. Draft proposals are
    accepted while ``candidates[:, 1:]`` matches the target predictions for the
    previous positions. The bonus token is the target prediction at the first
    unaccepted position, or the final target prediction when all draft proposals
    are accepted.
    """

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={candidates.shape}.")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates, got "
            f"{target_predict.shape} vs {candidates.shape}."
        )

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1)
    row_ids = jnp.arange(candidates.shape[0], dtype=jnp.int32)
    bonus = target_predict[row_ids, accept_len]
    return accept_len.astype(jnp.int32), bonus.astype(jnp.int32)


@dataclass
class DFlashDraftInput:
    """Host-side DFlash state carried between decode iterations."""

    verified_id: jax.Array | np.ndarray
    target_hidden: jax.Array | None
    ctx_lens: np.ndarray
    draft_seq_lens: np.ndarray
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    def is_draft_input(self) -> bool:
        return True

    def is_verify_input(self) -> bool:
        return False

    def get_spec_adjust_token_coefficient(self) -> int:
        return 1

    def get_logical_token_num(self, bs: int) -> np.ndarray:
        return np.ones((bs,), dtype=np.int32)

    def get_allocated_token_num(self) -> np.ndarray | None:
        return None

    def get_verify_token_num(self, bs: int) -> int:
        return 0

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None:
        new_indices = np.asarray(new_indices, dtype=np.int32)
        self.verified_id = np.asarray(self.verified_id)[new_indices]
        self.ctx_lens = np.asarray(self.ctx_lens, dtype=np.int32)[new_indices]
        self.draft_seq_lens = np.asarray(self.draft_seq_lens, dtype=np.int32)[new_indices]

        if self.target_hidden is None:
            return
        old_lens = np.asarray(self.ctx_lens, dtype=np.int32)
        if np.sum(old_lens) == 0:
            self.target_hidden = self.target_hidden[:0]
            return
        raise NotImplementedError(
            "DFlashDraftInput.filter_batch with non-empty target_hidden is not "
            "implemented in stage2; DFLASH is restricted to dp_size=1/no retraction."
        )

    def merge_batch(self, other: "DFlashDraftInput") -> None:
        self.verified_id = np.concatenate(
            [np.asarray(self.verified_id), np.asarray(other.verified_id)], axis=0
        )
        self.ctx_lens = np.concatenate([self.ctx_lens, other.ctx_lens], axis=0)
        self.draft_seq_lens = np.concatenate(
            [self.draft_seq_lens, other.draft_seq_lens], axis=0
        )
        if self.target_hidden is None:
            self.target_hidden = other.target_hidden
        elif other.target_hidden is not None:
            self.target_hidden = jnp.concatenate([self.target_hidden, other.target_hidden], axis=0)


@register_pytree_node_class
@dataclass
class DFlashVerifyInput:
    """JIT-visible target verify input for a fixed DFlash block."""

    draft_token: jax.Array
    positions: jax.Array
    draft_token_num: int
    custom_mask: jax.Array | None = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    def is_draft_input(self) -> bool:
        return False

    def is_verify_input(self) -> bool:
        return True

    def get_spec_adjust_token_coefficient(self) -> int:
        return int(self.draft_token_num)

    def get_logical_token_num(self, bs: int) -> np.ndarray:
        return np.full((bs,), int(self.draft_token_num), dtype=np.int32)

    def get_allocated_token_num(self) -> np.ndarray | None:
        return None

    def get_verify_token_num(self, bs: int) -> int:
        return int(bs) * int(self.draft_token_num)

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None:
        raise NotImplementedError("DFlashVerifyInput is built per verify step and is not filtered.")

    def merge_batch(self, other: "DFlashVerifyInput") -> None:
        raise NotImplementedError("DFlashVerifyInput is built per verify step and is not merged.")

    def tree_flatten(self):
        children = (self.draft_token, self.positions, self.custom_mask)
        aux_data = {
            "draft_token_num": int(self.draft_token_num),
            "capture_hidden_mode": self.capture_hidden_mode,
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            draft_token=children[0],
            positions=children[1],
            custom_mask=children[2],
            draft_token_num=aux_data["draft_token_num"],
            capture_hidden_mode=aux_data["capture_hidden_mode"],
        )

    def verify_greedy(self, target_predict: jax.Array) -> tuple[jax.Array, jax.Array]:
        candidates = self.draft_token.reshape((-1, int(self.draft_token_num)))
        target_predict = target_predict.reshape(candidates.shape)
        return compute_dflash_accept_len_and_bonus(candidates, target_predict)
