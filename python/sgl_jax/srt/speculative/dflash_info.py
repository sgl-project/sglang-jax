from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode


# @haifeng Maybe common!
def compute_dflash_accept_len_and_bonus(
    candidates: jax.Array,
    target_predict: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Greedy DFlash chain verification.

    match: candidates[:, 1:] == target_predict[:, :-1]
    """

    if candidates.ndim != 2:
        raise ValueError(f"candidates must be 2D, got shape={candidates.shape}.")
    if target_predict.shape != candidates.shape:
        raise ValueError(
            "target_predict must have the same shape as candidates, got "
            f"{target_predict.shape} vs {candidates.shape}."
        )

    mesh = getattr(jax.typeof(target_predict).sharding, "mesh", None)
    if mesh is not None:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        replicated_2d = NamedSharding(mesh, P(None, None))
        candidates = jax.sharding.reshard(candidates, replicated_2d)
        target_predict = jax.sharding.reshard(target_predict, replicated_2d)

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1)
    row_ids = jnp.arange(candidates.shape[0], dtype=jnp.int32)
    flat_idx = row_ids * candidates.shape[1] + accept_len
    tp_flat = target_predict.reshape(-1)
    if mesh is None:
        bonus = jnp.take(tp_flat, flat_idx)
    else:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        bonus = tp_flat.at[flat_idx].get(out_sharding=NamedSharding(mesh, P(None)))
    return accept_len.astype(jnp.int32), bonus.astype(jnp.int32)


def build_dflash_draft_block(
    verified_id: np.ndarray | jax.Array,
    mask_token_id: int,
    target_prefix_lens: np.ndarray | jax.Array,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the fixed-size DFlash draft block inputs for one decode step.

    - block: [verified_id, mask_token_id, mask_token_id, ...]
    - position: [target_prefix_lens, target_prefix_lens + 1, ..., target_prefix_lens + block_size - 1]
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


def compute_new_kv_slices(
    ctx_lens: np.ndarray,
    draft_seq_lens: np.ndarray,
    is_prefill: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-request ``(start, length)`` of committed tokens in the KV cache.

    - Prefill: [prefix_len : prefix_len + extend_len]
    - Decode: [new_len - accept_len : new_len]
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
    - ``next_token_ids``: ``(bs * block_size,)`` — ``target_predict``
      flattened; the scheduler keeps the first ``accept_lens_out[i]`` per row.
    - ``verified_id``: ``(bs,)`` next-step seed (= bonus token).
    - ``accept_len_draft``: ``(bs,)`` number of accepted draft proposals.
    """
    if draft_token.ndim != 2:
        raise ValueError(f"draft_token must be 2D, got shape={draft_token.shape}.")
    accept_len_draft, bonus = compute_dflash_accept_len_and_bonus(draft_token, target_predict)
    accept_lens_out = (accept_len_draft + 1).astype(jnp.int32)
    next_token_ids = target_predict.reshape(-1).astype(jnp.int32)
    verified_id = bonus.astype(jnp.int32)
    return accept_lens_out, next_token_ids, verified_id, accept_len_draft


def dflash_greedy_verify_impl(
    draft_token: jax.Array,
    target_logits: jax.Array,
    *,
    draft_token_num: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure JAX target-logits argmax and greedy DFlash verification."""
    candidates = draft_token.reshape((-1, int(draft_token_num)))
    target_predict_flat = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
    mesh = getattr(jax.typeof(target_predict_flat).sharding, "mesh", None)
    target_predict = target_predict_flat.reshape(candidates.shape)
    if mesh is not None:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        replicated_2d = NamedSharding(mesh, P(None, None))
        candidates = jax.sharding.reshard(candidates, replicated_2d)
        target_predict = jax.sharding.reshard(target_predict, replicated_2d)

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len_draft = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1)
    row_ids = jnp.arange(candidates.shape[0], dtype=jnp.int32)
    flat_idx = row_ids * candidates.shape[1] + accept_len_draft
    target_predict_flat = target_predict.reshape(-1).astype(jnp.int32)
    if mesh is None:
        bonus = jnp.take(target_predict_flat, flat_idx)
    else:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        bonus = target_predict_flat.at[flat_idx].get(out_sharding=NamedSharding(mesh, P(None)))

    accept_lens_out = (accept_len_draft + 1).astype(jnp.int32)
    return accept_lens_out, target_predict_flat, bonus, accept_len_draft.astype(jnp.int32)


@functools.partial(jax.jit, static_argnames=("draft_token_num",))
def dflash_greedy_verify(
    draft_token: jax.Array,
    target_logits: jax.Array,
    *,
    draft_token_num: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fused target-logits argmax and greedy DFlash verification."""
    return dflash_greedy_verify_impl(
        draft_token,
        target_logits,
        draft_token_num=draft_token_num,
    )


def gather_dflash_accepted_hidden_padded(
    target_hidden: jax.Array,
    accept_lens: jax.Array,
    *,
    block_size: int,
    hidden_out_sharding=None,
    index_out_sharding=None,
) -> jax.Array:
    """Gather accepted target hidden states into DFlash draft-extend layout.

    Decode target hidden is row-major ``[bs, block_size, hidden]``. Draft KV
    materialization expects a fixed ``bs * block_size`` shape with valid tokens
    concatenated by request and trailing dummy rows padded from row 0.
    """
    hidden_size = target_hidden.shape[-1]
    flat_hidden = target_hidden.reshape((-1, hidden_size))
    accept_lens = accept_lens.astype(jnp.int32)
    mesh = getattr(jax.typeof(accept_lens).sharding, "mesh", None)
    if mesh is not None:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        accept_lens = jax.sharding.reshard(accept_lens, NamedSharding(mesh, P(None)))
    max_tokens = accept_lens.shape[0] * int(block_size)
    ends = jnp.cumsum(accept_lens)
    starts = ends - accept_lens
    slot_ids = jnp.arange(max_tokens, dtype=jnp.int32)
    row_ids = jnp.sum(slot_ids[:, None] >= ends[None, :], axis=1).astype(jnp.int32)
    row_ids = jnp.minimum(row_ids, accept_lens.shape[0] - 1)
    if index_out_sharding is None:
        row_starts = starts.at[row_ids].get()
    else:
        row_starts = starts.at[row_ids].get(out_sharding=index_out_sharding)
    col_ids = slot_ids - row_starts
    gather_ids = row_ids * int(block_size) + col_ids
    gather_ids = jnp.where(slot_ids < ends[-1], gather_ids, 0)
    if hidden_out_sharding is None:
        return flat_hidden.at[gather_ids].get()
    return flat_hidden.at[gather_ids].get(out_sharding=hidden_out_sharding)


@dataclass
class DFlashDraftInput:
    """Host-side DFlash state carried between decode iterations."""

    verified_id: jax.Array | np.ndarray = None
    target_hidden: jax.Array | None = None
    ctx_lens: np.ndarray = None
    draft_seq_lens: np.ndarray = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.FULL

    def __init__(
        self,
        *,
        verified_id=None,
        target_hidden=None,
        ctx_lens=None,
        draft_seq_lens=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        block_size=16,
        **_kwargs,
    ):
        self.verified_id = verified_id
        self.target_hidden = target_hidden
        self.ctx_lens = ctx_lens
        self.draft_seq_lens = draft_seq_lens
        self.capture_hidden_mode = capture_hidden_mode
        self.block_size = int(block_size)

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

    def _ensure_host(self) -> None:
        for f in ("verified_id", "ctx_lens", "draft_seq_lens"):
            v = getattr(self, f, None)
            if v is not None and hasattr(v, "copy_to_host_async"):
                v.copy_to_host_async()
        for f in ("verified_id", "ctx_lens", "draft_seq_lens"):
            v = getattr(self, f, None)
            if v is not None:
                setattr(self, f, np.asarray(v, dtype=np.int32))

    def trim_to_length(self, n: int) -> None:
        self._ensure_host()
        for f in ("verified_id", "ctx_lens", "draft_seq_lens"):
            v = getattr(self, f, None)
            if v is not None and len(v) != n:
                setattr(self, f, np.asarray(v, dtype=np.int32)[:n])
        if self.target_hidden is not None and self.target_hidden.shape[0] not in (0, n):
            self.target_hidden = self.target_hidden[:n]

    def new_tokens_required_next_decode(self, requests, page_size: int) -> int:
        total = 0
        block_size = int(self.block_size)
        for req in requests:
            cur = int(req.kv_allocated_len)
            nxt = max(cur, int(req.kv_committed_len) + block_size)
            total += ((nxt + page_size - 1) // page_size) * page_size - (
                (cur + page_size - 1) // page_size
            ) * page_size
        return total

    def filter_batch(self, new_indices: np.ndarray, has_been_filtered: bool = True) -> None:
        self._ensure_host()
        new_indices = np.asarray(new_indices, dtype=np.int32)
        if has_been_filtered and len(new_indices) == len(self.verified_id):
            idx = slice(0, len(new_indices))
        else:
            idx = new_indices
        self.verified_id = np.asarray(self.verified_id, dtype=np.int32)[idx]
        self.ctx_lens = np.asarray(self.ctx_lens, dtype=np.int32)[idx]
        self.draft_seq_lens = np.asarray(self.draft_seq_lens, dtype=np.int32)[idx]

        if self.target_hidden is None:
            return
        old_lens = np.asarray(self.ctx_lens, dtype=np.int32)
        if np.sum(old_lens) == 0:
            self.target_hidden = self.target_hidden[:0]
            return
        raise NotImplementedError(
            "DFlashDraftInput.filter_batch with non-empty target_hidden is not "
            "implemented; DFLASH is restricted to dp_size=1/no retraction."
        )

    def resolve_pending_draft_extend_result(self):
        pass

    # alloc the kv cache for draft block
    def prepare_for_decode(self, schedule_batch) -> None:
        from sgl_jax.srt.managers.schedule_batch import get_last_loc
        from sgl_jax.srt.mem_cache.common import (
            alloc_paged_token_slots_extend,
            alloc_token_slots,
        )
        from sgl_jax.srt.speculative.eagle_util import assign_req_to_token_pool

        block_size = self.block_size
        page_size = schedule_batch.token_to_kv_pool_allocator.page_size

        for dp_rank, info in enumerate(schedule_batch.reqs_info):
            if info.seq_lens is None or len(info.seq_lens) == 0:
                continue

            reqs = info.reqs

            old_r = np.asarray([req.kv_allocated_len for req in reqs], dtype=np.int32)
            committed_r = np.asarray([req.kv_committed_len for req in reqs], dtype=np.int32)
            self._align_to_reqs(reqs, committed_r)
            new_r = np.maximum(old_r, committed_r + block_size)
            ext_r = int((new_r - old_r).sum())

            if ext_r > 0 and page_size == 1:
                ocl_r = alloc_token_slots(schedule_batch.tree_cache, ext_r, dp_rank=dp_rank)
                assign_req_to_token_pool(
                    info.req_pool_indices,
                    schedule_batch.req_to_token_pool,
                    old_r,
                    new_r,
                    ocl_r,
                )
            elif ext_r > 0:
                last_loc_r = get_last_loc(
                    schedule_batch.req_to_token_pool.req_to_token,
                    info.req_pool_indices,
                    old_r,
                )
                ocl_r = alloc_paged_token_slots_extend(
                    schedule_batch.tree_cache,
                    old_r,
                    new_r,
                    last_loc_r,
                    int((new_r - old_r).sum()),
                    dp_rank=dp_rank,
                )
                assign_req_to_token_pool(
                    info.req_pool_indices,
                    schedule_batch.req_to_token_pool,
                    old_r,
                    new_r,
                    ocl_r,
                )

            req_to_token = schedule_batch.req_to_token_pool.req_to_token
            verify_locs = []
            for i, req in enumerate(reqs):
                rp = int(info.req_pool_indices[i])
                c = int(committed_r[i])
                verify_locs.append(np.asarray(req_to_token[rp, c : c + block_size], dtype=np.int32))
            info.out_cache_loc = (
                np.concatenate(verify_locs) if verify_locs else np.empty(0, dtype=np.int32)
            )

            for req, allocated_len in zip(reqs, new_r):
                req.decode_batch_idx += 1
                req.kv_allocated_len = int(allocated_len)
                req.kv_committed_len += 1

            info.seq_lens_sum = np.sum(info.seq_lens).item()

    def _align_to_reqs(self, reqs, committed_lens: np.ndarray) -> None:
        state_bs = int(np.asarray(self.draft_seq_lens, dtype=np.int32).shape[0])
        bs = len(reqs)
        if state_bs == bs:
            return

        verified_id = np.asarray(self.verified_id, dtype=np.int32)
        ctx_lens = np.asarray(self.ctx_lens, dtype=np.int32)
        draft_seq_lens = np.asarray(self.draft_seq_lens, dtype=np.int32)
        if state_bs > bs:
            self.verified_id = verified_id[:bs]
            self.ctx_lens = ctx_lens[:bs]
            self.draft_seq_lens = draft_seq_lens[:bs]
            return

        missing_reqs = reqs[state_bs:bs]
        missing_verified = np.asarray(
            [
                req.output_ids[-1] if len(req.output_ids) > 0 else req.origin_input_ids[-1]
                for req in missing_reqs
            ],
            dtype=np.int32,
        )
        self.verified_id = np.concatenate([verified_id, missing_verified], axis=0)
        self.ctx_lens = np.concatenate(
            [ctx_lens, np.zeros((bs - state_bs,), dtype=np.int32)], axis=0
        )
        self.draft_seq_lens = np.concatenate(
            [draft_seq_lens, committed_lens[state_bs:bs].astype(np.int32)], axis=0
        )

    def merge_batch(self, other: DFlashDraftInput) -> None:
        self._ensure_host()
        other._ensure_host()
        self.verified_id = np.concatenate(
            [np.asarray(self.verified_id), np.asarray(other.verified_id)], axis=0
        )
        self.ctx_lens = np.concatenate([self.ctx_lens, other.ctx_lens], axis=0)
        self.draft_seq_lens = np.concatenate([self.draft_seq_lens, other.draft_seq_lens], axis=0)
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
    input_ids_host: np.ndarray | None = None
    positions_host: np.ndarray | None = None
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

    def merge_batch(self, other: DFlashVerifyInput) -> None:
        raise NotImplementedError("DFlashVerifyInput is built per verify step and is not merged.")

    def resolve_pending_draft_extend_result(self):
        pass

    def tree_flatten(self):
        children = (
            self.draft_token,
            self.positions,
            self.custom_mask,
        )
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
            input_ids_host=None,
            positions_host=None,
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
        if self.input_ids_host is None:
            model_worker_batch.input_ids = np.asarray(
                jax.device_get(self.draft_token), dtype=np.int32
            ).reshape(-1)
        else:
            model_worker_batch.input_ids = np.asarray(self.input_ids_host, dtype=np.int32).reshape(
                -1
            )
        if self.positions_host is None:
            model_worker_batch.positions = np.asarray(
                jax.device_get(self.positions), dtype=np.int32
            ).reshape(-1)
        else:
            model_worker_batch.positions = np.asarray(self.positions_host, dtype=np.int32).reshape(
                -1
            )
        model_worker_batch.spec_info_padded = self
        model_worker_batch.capture_hidden_mode = self.capture_hidden_mode

    # @haifeng Need modify
    def verify(self, target_logits: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Greedy-verify a target forward output.

        ``target_logits`` has shape ``[bs * block_size, vocab]``. Returns the
        scheduler-facing decode outputs from :func:`dflash_greedy_verify_outputs`:
        ``(accept_lens_out, next_token_ids, verified_id, accept_len_draft)``.
        """
        return dflash_greedy_verify(
            self.draft_token,
            target_logits,
            draft_token_num=int(self.draft_token_num),
        )
