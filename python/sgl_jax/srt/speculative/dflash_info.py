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


def build_dflash_draft_block(
    verified_id: np.ndarray | jax.Array,
    mask_token_id: int,
    target_prefix_lens: np.ndarray | jax.Array,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed-size DFlash draft block inputs for one decode step.

    Token 0 of each row is the already-verified seed token; tokens ``1..block_size-1``
    are the draft mask token. Positions are absolute target positions
    ``seq_len + arange(block_size)`` so RoPE matches the target KV cache.

    Returns ``(block_ids, positions)``, both host ``np.ndarray`` of shape
    ``(bs, block_size)`` and int32.
    """
    verified_id = np.asarray(verified_id, dtype=np.int32)
    target_prefix_lens = np.asarray(target_prefix_lens, dtype=np.int32)
    if verified_id.ndim != 1:
        raise ValueError(f"verified_id must be 1D, got shape={verified_id.shape}.")
    if target_prefix_lens.shape != verified_id.shape:
        raise ValueError(
            "target_prefix_lens must match verified_id, got "
            f"{target_prefix_lens.shape} vs {verified_id.shape}."
        )
    bs = int(verified_id.shape[0])
    block_size = int(block_size)

    block_ids = np.full((bs, block_size), int(mask_token_id), dtype=np.int32)
    block_ids[:, 0] = verified_id
    positions = target_prefix_lens[:, None] + np.arange(block_size, dtype=np.int32)[None, :]
    return block_ids, positions.astype(np.int32)


def dflash_committed_slices(
    ctx_lens: np.ndarray,
    draft_seq_lens: np.ndarray,
    is_prefill: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-request ``(start, length)`` of committed tokens in the KV cache.

    - Prefill: the new-prompt span occupies ``[prefix_len : prefix_len + extend_len]``,
      where ``draft_seq_lens`` is the cached prefix length and ``ctx_lens`` the new
      prompt token count.
    - Decode: the committed tokens occupy the tail ``[new_len - accept_len : new_len]``,
      where ``draft_seq_lens`` is the new total length and ``ctx_lens`` the number
      committed this step.
    """
    ctx_lens = np.asarray(ctx_lens, dtype=np.int32)
    draft_seq_lens = np.asarray(draft_seq_lens, dtype=np.int32)
    if is_prefill:
        starts = draft_seq_lens.copy()
    else:
        starts = draft_seq_lens - ctx_lens
    return starts.astype(np.int32), ctx_lens.copy()


def dflash_greedy_verify_outputs(
    draft_token: jax.Array,
    target_predict: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Greedy DFlash verify → scheduler-facing decode outputs.

    ``draft_token[:, 0]`` is the seed (already-emitted verified id). The tokens
    committed this step are the accepted draft proposals plus one bonus token,
    which are exactly ``target_predict[i, 0 : accept_len_draft[i] + 1]``.

    Returns:
    - ``accept_lens_out``: ``(bs,)`` emitted token count per request
      (= accepted drafts + 1 bonus). This is what the scheduler advances output
      length by and what ``resolve_spec_decode_token_ids`` slices with.
    - ``next_token_ids_flat``: ``(bs * block_size,)`` — ``target_predict``
      flattened; the scheduler keeps the first ``accept_lens_out[i]`` per row.
    - ``new_verified_id``: ``(bs,)`` next-step seed (= bonus token).
    - ``accept_len_draft``: ``(bs,)`` number of accepted draft proposals.
    """
    if draft_token.ndim != 2:
        raise ValueError(f"draft_token must be 2D, got shape={draft_token.shape}.")
    accept_len_draft, bonus = compute_dflash_accept_len_and_bonus(
        draft_token, target_predict
    )
    accept_lens_out = (accept_len_draft + 1).astype(jnp.int32)
    next_token_ids_flat = target_predict.reshape(-1).astype(jnp.int32)
    new_verified_id = bonus.astype(jnp.int32)
    return accept_lens_out, next_token_ids_flat, new_verified_id, accept_len_draft


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

    def prepare_for_verify(self, model_worker_batch, page_size: int) -> None:
        """Wire the drafted block into ``model_worker_batch`` for target verify.

        The scheduler has already allocated ``out_cache_loc`` for ``bs * block``
        verify tokens via ``get_spec_model_worker_batch``. Here we install the
        drafted token ids / positions and request full aux-hidden capture so the
        verify forward yields the target features DFlash needs for the next block.
        """
        model_worker_batch.input_ids = np.asarray(
            jax.device_get(self.draft_token), dtype=np.int32
        ).reshape(-1)
        model_worker_batch.positions = np.asarray(
            jax.device_get(self.positions), dtype=np.int32
        ).reshape(-1)
        model_worker_batch.spec_info_padded = self
        model_worker_batch.capture_hidden_mode = self.capture_hidden_mode

    def verify(
        self, target_logits: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Greedy-verify a target forward output.

        ``target_logits`` has shape ``[bs * block_size, vocab]``. Returns the
        scheduler-facing decode outputs from :func:`dflash_greedy_verify_outputs`:
        ``(accept_lens_out, next_token_ids_flat, new_verified_id, accept_len_draft)``.
        """
        target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
        candidates = self.draft_token.reshape((-1, int(self.draft_token_num)))
        target_predict = target_predict.reshape(candidates.shape)
        return dflash_greedy_verify_outputs(candidates, target_predict)
