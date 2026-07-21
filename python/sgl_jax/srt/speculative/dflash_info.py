from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.model_executor.forward_batch_info import CaptureHiddenMode


def _mask_draft_kv_writes(
    cache_loc: jax.Array,
    accept_lens: jax.Array,
    active_mask: jax.Array,
) -> jax.Array:
    """Mask unaccepted and padded draft-KV writes inside jit_draft_extend."""
    tokens_per_row = cache_loc.shape[0] // accept_lens.shape[0]
    cache_rows = cache_loc.reshape((-1, tokens_per_row))
    accept_rows = accept_lens[:, None]
    active_rows = active_mask[:, None]
    token_offsets = jnp.arange(tokens_per_row, dtype=jnp.int32)[None, :]
    mesh = getattr(jax.typeof(cache_loc).sharding, "mesh", None)
    if mesh is not None and not getattr(mesh, "empty", False):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        row_sharding = NamedSharding(mesh, P("data", None))
        replicated_2d = NamedSharding(mesh, P(None, None))
        cache_rows = jax.sharding.reshard(cache_rows, row_sharding)
        accept_rows = jax.sharding.reshard(accept_rows, row_sharding)
        active_rows = jax.sharding.reshard(active_rows, row_sharding)
        token_offsets = jax.sharding.reshard(token_offsets, replicated_2d)
    write_mask = active_rows & (token_offsets < accept_rows)
    masked_cache_loc = jnp.where(
        write_mask,
        cache_rows,
        jnp.int32(-1),
    ).reshape(-1)
    cache_sharding = jax.typeof(cache_loc).sharding
    if isinstance(cache_sharding, jax.sharding.NamedSharding) and not cache_sharding.mesh.empty:
        masked_cache_loc = jax.sharding.reshard(masked_cache_loc, cache_sharding)
    return masked_cache_loc


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


# TODO: Share greedy chain verification through common speculative helpers.
def dflash_greedy_verify(
    draft_token: jax.Array,
    target_logits: jax.Array,
    *,
    draft_token_num: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Pure JAX target-logits argmax and greedy DFlash verification."""
    candidates = draft_token.reshape((-1, int(draft_token_num)))
    target_predict_flat = jnp.argmax(target_logits, axis=-1).astype(jnp.int32)
    mesh = getattr(jax.typeof(target_predict_flat).sharding, "mesh", None)
    if mesh is not None and getattr(mesh, "empty", False):
        mesh = None
    target_predict = target_predict_flat.reshape(candidates.shape)
    if mesh is not None:
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        data_2d = NamedSharding(mesh, P("data", None))
        candidates = jax.sharding.reshard(candidates, data_2d)
        target_predict = jax.sharding.reshard(target_predict, data_2d)

    matches = candidates[:, 1:] == target_predict[:, :-1]
    accept_len_draft = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1)
    target_predict_flat = target_predict.reshape(-1).astype(jnp.int32)
    if mesh is None:
        bonus = jnp.take_along_axis(
            target_predict,
            accept_len_draft[:, None],
            axis=1,
        ).reshape(-1)
    else:

        def _select_local_bonus(local_predict, local_accept_len):
            return jnp.take_along_axis(
                local_predict,
                local_accept_len[:, None],
                axis=1,
            ).reshape(-1)

        bonus = jax.shard_map(
            _select_local_bonus,
            mesh=mesh,
            in_specs=(P("data", None), P("data")),
            out_specs=P("data"),
        )(target_predict, accept_len_draft)

    accept_lens_out = (accept_len_draft + 1).astype(jnp.int32)
    return accept_lens_out, target_predict_flat, bonus, accept_len_draft.astype(jnp.int32)


@dataclass
class DFlashDraftInput:
    """Host-side DFlash state carried between decode iterations."""

    verified_id: jax.Array | np.ndarray = None
    target_hidden: jax.Array | None = None
    ctx_lens: np.ndarray = None
    draft_seq_lens: np.ndarray = None
    allocate_lens: np.ndarray = None
    reservation_base_lens: np.ndarray = None
    future_indices: np.ndarray = None
    block_size: int = 16
    capture_hidden_mode = CaptureHiddenMode.FULL

    def _ensure_host(self) -> None:
        fields = (
            "verified_id",
            "ctx_lens",
            "draft_seq_lens",
            "allocate_lens",
            "reservation_base_lens",
            "future_indices",
        )
        for f in fields:
            v = getattr(self, f, None)
            if v is not None and hasattr(v, "copy_to_host_async"):
                v.copy_to_host_async()
        for f in fields:
            v = getattr(self, f, None)
            if v is not None:
                setattr(self, f, np.asarray(v, dtype=np.int32))

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
        if self.future_indices is not None:
            old_bs = len(self.future_indices)
            selected = (
                np.arange(len(new_indices), dtype=np.int32)
                if has_been_filtered and len(new_indices) == old_bs
                else new_indices
            )
            for field in ("future_indices", "allocate_lens", "reservation_base_lens"):
                value = getattr(self, field, None)
                if value is not None:
                    setattr(self, field, np.asarray(value, dtype=np.int32)[selected])
            return

        old_verified_id = np.asarray(self.verified_id, dtype=np.int32)
        old_ctx_lens = np.asarray(self.ctx_lens, dtype=np.int32)
        old_draft_seq_lens = np.asarray(self.draft_seq_lens, dtype=np.int32)
        old_bs = len(old_verified_id)
        if has_been_filtered and len(new_indices) == old_bs:
            selected = np.arange(len(new_indices), dtype=np.int32)
        else:
            selected = new_indices

        self.verified_id = old_verified_id[selected]
        self.ctx_lens = old_ctx_lens[selected]
        self.draft_seq_lens = old_draft_seq_lens[selected]
        for field in ("allocate_lens", "reservation_base_lens"):
            value = getattr(self, field, None)
            if value is not None:
                setattr(self, field, np.asarray(value, dtype=np.int32)[selected])

        if self.target_hidden is not None and self.target_hidden.shape[0] != 0:
            raise ValueError("DFLASH target_hidden must be materialized before filtering.")
        self.target_hidden = None

    def prepare_for_decode(self, schedule_batch) -> None:
        # TODO: Share KV slot reservation and req_to_token_pool updates
        # with EAGLE through common speculative helpers in the next PR.
        from sgl_jax.srt.managers.schedule_batch import get_last_loc
        from sgl_jax.srt.mem_cache.common import (
            alloc_paged_token_slots_extend,
            alloc_token_slots,
        )
        from sgl_jax.srt.speculative.eagle_util import assign_req_to_token_pool

        block_size = self.block_size
        page_size = schedule_batch.token_to_kv_pool_allocator.page_size
        reserve_tokens = block_size * (2 if schedule_batch.enable_overlap else 1)

        self._align_dp_state_to_reqs(schedule_batch)
        allocate_lens = []
        reservation_base_lens = []

        for dp_rank, info in enumerate(schedule_batch.reqs_info):
            if info.seq_lens is None or len(info.seq_lens) == 0:
                continue

            reqs = info.reqs

            old_r = np.asarray([req.kv_allocated_len for req in reqs], dtype=np.int32)
            committed_r = np.asarray([req.kv_committed_len for req in reqs], dtype=np.int32)
            new_r = np.maximum(old_r, committed_r + reserve_tokens)
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
                verify_locs.append(
                    np.asarray(req_to_token[rp, c : c + reserve_tokens], dtype=np.int32)
                )
            info.out_cache_loc = (
                np.concatenate(verify_locs) if verify_locs else np.empty(0, dtype=np.int32)
            )
            allocate_lens.append(new_r)
            reservation_base_lens.append(committed_r)

            for req, allocated_len in zip(reqs, new_r):
                req.decode_batch_idx += 1
                req.kv_allocated_len = int(allocated_len)
                req.kv_committed_len += 1

            info.seq_lens_sum = np.sum(info.seq_lens).item()

        self.allocate_lens = (
            np.concatenate(allocate_lens) if allocate_lens else np.empty((0,), dtype=np.int32)
        )
        self.reservation_base_lens = (
            np.concatenate(reservation_base_lens)
            if reservation_base_lens
            else np.empty((0,), dtype=np.int32)
        )

    def _align_dp_state_to_reqs(self, schedule_batch) -> None:
        """Align each rank's state independently, then rebuild rank-major state."""
        if self.future_indices is not None:
            expected = sum(len(info.reqs or []) for info in schedule_batch.reqs_info)
            if len(self.future_indices) != expected:
                raise ValueError(
                    "DFLASH relay state does not match the decode requests: "
                    f"future_indices={len(self.future_indices)}, requests={expected}."
                )
            return

        rank_states = []
        for info in schedule_batch.reqs_info:
            reqs = info.reqs or []
            if not reqs:
                continue

            rank_state = info.spec_info
            if not isinstance(rank_state, DFlashDraftInput):
                rank_state = DFlashDraftInput(
                    verified_id=np.empty((0,), dtype=np.int32),
                    target_hidden=None,
                    ctx_lens=np.empty((0,), dtype=np.int32),
                    draft_seq_lens=np.empty((0,), dtype=np.int32),
                    block_size=self.block_size,
                )
            committed_lens = np.asarray([req.kv_committed_len for req in reqs], dtype=np.int32)
            rank_state._align_to_reqs(reqs, committed_lens)
            rank_states.append(rank_state)

        if not rank_states:
            self.verified_id = np.empty((0,), dtype=np.int32)
            self.ctx_lens = np.empty((0,), dtype=np.int32)
            self.draft_seq_lens = np.empty((0,), dtype=np.int32)
            self.target_hidden = None
            return

        self.verified_id = np.concatenate(
            [np.asarray(state.verified_id, dtype=np.int32) for state in rank_states]
        )
        self.ctx_lens = np.concatenate(
            [np.asarray(state.ctx_lens, dtype=np.int32) for state in rank_states]
        )
        self.draft_seq_lens = np.concatenate(
            [np.asarray(state.draft_seq_lens, dtype=np.int32) for state in rank_states]
        )

        hidden_parts = [state.target_hidden for state in rank_states]
        if all(hidden is None for hidden in hidden_parts):
            self.target_hidden = None
        elif all(hidden is not None and hidden.shape[0] == 0 for hidden in hidden_parts):
            self.target_hidden = hidden_parts[0][:0]
        else:
            raise ValueError(
                "DFLASH target_hidden must be materialized before DP decode preparation."
            )

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
        if self.future_indices is not None or other.future_indices is not None:
            if self.future_indices is None or other.future_indices is None:
                raise ValueError("DFLASH overlap merge requires future_indices on both batches.")
            self.future_indices = np.concatenate(
                [self.future_indices, other.future_indices], axis=0
            )
            for field in ("allocate_lens", "reservation_base_lens"):
                lhs = getattr(self, field, None)
                rhs = getattr(other, field, None)
                setattr(
                    self,
                    field,
                    None if lhs is None or rhs is None else np.concatenate([lhs, rhs], axis=0),
                )
            self.verified_id = None
            self.ctx_lens = None
            self.draft_seq_lens = None
            self.target_hidden = None
            return

        self.verified_id = np.concatenate(
            [np.asarray(self.verified_id), np.asarray(other.verified_id)], axis=0
        )
        self.ctx_lens = np.concatenate([self.ctx_lens, other.ctx_lens], axis=0)
        self.draft_seq_lens = np.concatenate([self.draft_seq_lens, other.draft_seq_lens], axis=0)
        for field in ("allocate_lens", "reservation_base_lens"):
            lhs = getattr(self, field, None)
            rhs = getattr(other, field, None)
            setattr(
                self,
                field,
                None if lhs is None or rhs is None else np.concatenate([lhs, rhs], axis=0),
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
    draft_token_num: int
    custom_mask = None

    def tree_flatten(self):
        return (self.draft_token,), {"draft_token_num": int(self.draft_token_num)}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            draft_token=children[0],
            draft_token_num=aux_data["draft_token_num"],
        )
