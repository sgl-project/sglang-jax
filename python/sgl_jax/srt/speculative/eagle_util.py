from __future__ import annotations

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.speculative.build_eagle_tree_structure_kernel import (
    build_eagle_tree_structure,
)


@functools.partial(
    jax.jit, static_argnames=["num_verify_tokens", "batch_size", "speculative_num_steps"]
)
def build_tree_kernel_efficient_preprocess(
    verified_id: jax.Array,
    scores: jax.Array,
    tokens: jax.Array,
    parents: jax.Array,
    num_verify_tokens: int,
    batch_size: int,
    speculative_num_steps: int,
):
    # score_list   (bs, 1 + (step - 1) * topk  , eagle_topk)
    # token_list   (bs, topk + (step - 1) * topk * topk)
    # parents_list (bs, topk + 1 + (step - 1) * topk)
    # Concatenate score_list along dim=1 and flatten from dim=1 onwards
    # b, n, topk; n = 1 + (num_steps-1) * self.topk
    score_tensor = scores
    score_tensor = score_tensor.reshape(score_tensor.shape[0], -1)

    # Concatenate token lists: b, (self.topk + (num_steps-1) * self.topk)
    ss_token_list = tokens

    # Get top scores and indices
    _, top_scores_index = jax.lax.top_k(score_tensor, num_verify_tokens - 1)
    top_scores_index = jnp.sort(top_scores_index, axis=-1)

    # Gather draft tokens using the top indices
    draft_tokens = jnp.take_along_axis(ss_token_list, top_scores_index, axis=1)
    # assert draft_tokens.shape == (batch_size, verified_id.shape[0])
    draft_tokens = jnp.concatenate(
        [jnp.expand_dims(verified_id, axis=1), draft_tokens], axis=1
    ).flatten()

    # Build parent list
    if speculative_num_steps > 1:
        parent_list = parents
    else:
        parent_list = jnp.full((batch_size, 1), -1, dtype=jnp.int32)

    return parent_list, top_scores_index, draft_tokens


def _extract_parent_branch_indices(
    parents_entry: np.ndarray, step_index: int, topk: int
) -> np.ndarray:
    if step_index == 0:
        raise ValueError("Step index 0 has no parents")
    offset = topk if step_index == 1 else topk**2 * (step_index - 1) + topk
    raw = parents_entry.astype(np.int64) - offset
    parent_indices = np.floor_divide(raw, topk)
    parent_indices = np.clip(parent_indices, 0, topk - 1).astype(np.int32)
    return parent_indices


def build_tree_mask_for_draft_decode(
    seq_lens: jax.Array | np.ndarray,
    topk: int,
    speculative_step_id: int,
    parents_list: Sequence[jax.Array],
) -> jax.Array:
    """
    Build flattened custom mask for draft decode that respects branch ancestry.

    Args:
        seq_lens: Sequence lengths (prompt+accepted) for each request.
        topk: Number of speculative branches processed in parallel.
        speculative_step_id: Current speculative step (0-indexed).
        parents_list: List of parent index tensors produced by ``select_top_k_tokens``.

    Returns:
        Flattened boolean mask concatenating ``topk`` rows per request.
    """

    if topk <= 0:
        raise ValueError("topk must be positive")

    seq_lens_np = np.asarray(seq_lens, dtype=np.int32)
    bs = seq_lens_np.shape[0]
    if speculative_step_id + 1 > len(parents_list):
        raise ValueError("parents_list must contain at least speculative_step_id + 1 entries")

    # Precompute ancestry mapping: path[step, bid, branch]
    ancestry = np.zeros((speculative_step_id + 1, bs, topk), dtype=np.int32)
    ancestry[speculative_step_id] = np.broadcast_to(np.arange(topk, dtype=np.int32), (bs, topk))

    for step in range(speculative_step_id, 0, -1):
        parents_entry = np.asarray(parents_list[step])
        parent_indices = _extract_parent_branch_indices(parents_entry, step, topk)
        for bid in range(bs):
            child_branch_ids = ancestry[step, bid]
            ancestry[step - 1, bid] = parent_indices[bid, child_branch_ids]

    masks: list[np.ndarray] = []
    for bid in range(bs):
        seq_len = int(seq_lens_np[bid])
        kv_len = seq_len + (speculative_step_id + 1) * topk
        mask = np.zeros((topk, kv_len), dtype=np.bool_)
        mask[:, :seq_len] = True

        for branch in range(topk):
            for step in range(speculative_step_id + 1):
                branch_idx = ancestry[step, bid, branch]
                position = seq_len + step * topk + branch_idx
                mask[branch, position] = True

        masks.append(mask.reshape(-1))

    if not masks:
        return jnp.zeros((0,), dtype=jnp.bool_)

    concatenated = np.concatenate(masks)
    return jnp.asarray(concatenated, dtype=jnp.int32)


def build_chain_verify_inputs(
    verified_id: np.ndarray,
    token_list: np.ndarray,
    seq_lens: np.ndarray,
    num_verify_tokens: int,
    batch_size: int,
) -> np.ndarray:
    """Build verify inputs for topk=1 (linear chain) without tree mask.

    Returns a single ``(5, bs*n)`` int32 buffer packing all 5 outputs
    [draft_tokens, positions, retrive_index, retrive_next_token,
    retrive_next_sibling] so the caller can do **one** ``device_put``
    instead of five — under multi-host setup each independent P() replicated
    output triggers a separate allgather (~1.5ms each).

    When topk=1 the draft tree is a simple chain, so causal attention is
    equivalent to the tree mask.
    """
    n = num_verify_tokens
    bs = batch_size
    out = np.empty((5, bs * n), dtype=np.int32)

    # row 0: draft_tokens (bs*n,)
    out[0].reshape(bs, n)[:, 0] = verified_id
    out[0].reshape(bs, n)[:, 1:] = token_list[:, : n - 1]

    # row 1: positions (bs*n,) = seq_lens[bid] + tid
    tid_range = np.arange(n, dtype=np.int32)
    out[1] = (seq_lens.astype(np.int32)[:, None] + tid_range[None, :]).reshape(-1)

    # row 2: retrive_index (bs, n) flattened: bid*n + tid
    out[2] = np.arange(bs * n, dtype=np.int32)

    # row 3: retrive_next_token (bs, n) flattened: chain → tid+1, last is -1
    next_token_row = np.empty(n, dtype=np.int32)
    next_token_row[: n - 1] = np.arange(1, n, dtype=np.int32)
    next_token_row[n - 1] = -1
    out[3] = np.broadcast_to(next_token_row, (bs, n)).reshape(-1)

    # row 4: retrive_next_sibling: chain has no siblings, all -1
    out[4].fill(-1)

    return out


@functools.partial(jax.jit, static_argnames=["num_verify_tokens", "batch_size"])
def build_chain_verify_inputs_device(
    verified_id: jax.Array,
    token_list: jax.Array,
    seq_lens: jax.Array,
    num_verify_tokens: int,
    batch_size: int,
) -> jax.Array:
    """Build verify inputs for topk=1 linear chains on device."""
    n = num_verify_tokens
    bs = batch_size
    tid_range = jnp.arange(n, dtype=jnp.int32)
    draft_tokens = jnp.concatenate(
        [verified_id.astype(jnp.int32)[:, None], token_list[:, : n - 1].astype(jnp.int32)],
        axis=1,
    ).reshape(bs * n)
    positions = (seq_lens.astype(jnp.int32)[:, None] + tid_range[None, :]).reshape(bs * n)
    retrive_index = jnp.arange(bs * n, dtype=jnp.int32)
    retrive_next_token = jnp.broadcast_to(
        jnp.concatenate([jnp.arange(1, n, dtype=jnp.int32), jnp.array([-1], dtype=jnp.int32)]),
        (bs, n),
    ).reshape(bs * n)
    retrive_next_sibling = jnp.full((bs * n,), -1, dtype=jnp.int32)
    return jnp.stack(
        [draft_tokens, positions, retrive_index, retrive_next_token, retrive_next_sibling],
        axis=0,
    )


def build_tree_kernel_efficient(
    verified_id: jax.Array,
    score_list: jax.Array,
    token_list: jax.Array,
    parents_list: jax.Array,
    seq_lens: jax.Array,
    seq_lens_sum: jax.Array,
    topk: int,
    num_verify_tokens: int,
    max_seq_len_per_req: int,
    batch_size: int,
    speculative_num_steps: int,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """JAX implementation of build_tree_kernel_efficient.

    Args:
        verified_id: Verified token IDs from previous step
        score_list: List of score tensors from draft model
        token_list: List of token tensors from draft model
        parents_list: List of parent index tensors
        seq_lens: Sequence lengths
        seq_lens_sum: Sum of sequence lengths
        topk: Number of top-k candidates
        num_verify_tokens: Number of tokens to verify
        max_seq_len_per_req: Maximum allowed sequence length per request (static bound)

    Returns:
        tuple of (tree_mask, positions, retrive_index, retrive_next_token,
                 retrive_next_sibling, draft_tokens)
    """
    rep = NamedSharding(mesh, P())
    verified_id, score_list, token_list, parents_list, seq_lens = jax.device_put(
        (verified_id, score_list, token_list, parents_list, seq_lens), rep
    )
    parent_list, top_scores_index, draft_tokens = build_tree_kernel_efficient_preprocess(
        verified_id,
        score_list,
        token_list,
        parents_list,
        num_verify_tokens,
        batch_size,
        speculative_num_steps,
    )

    with jax.set_mesh(mesh):
        tree_mask, positions, retrive_index, retrive_next_token, retrive_next_sibling = (
            build_eagle_tree_structure(
                parent_list=parent_list,
                selected_index=top_scores_index,
                verified_seq_len=seq_lens,
                draft_token_num=num_verify_tokens,
                topk=topk,
                seq_lens_sum=seq_lens_sum,
                max_context_len=max_seq_len_per_req,
                tree_mask_mode=0,  # FULL_MASK
            )
        )

    return (
        tree_mask,
        positions,
        retrive_index,
        retrive_next_token,
        retrive_next_sibling,
        draft_tokens,
    )


def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token_pool,
    start_offsets,
    end_offsets,
    out_cache_loc,
):
    # Ensure inputs are numpy arrays (CPU) to avoid JAX sync overhead
    start_offsets = np.asarray(start_offsets, dtype=np.int32)
    end_offsets = np.asarray(end_offsets, dtype=np.int32)
    out_cache_loc = np.asarray(out_cache_loc, dtype=np.int32)

    out_cache_lens = end_offsets - start_offsets
    repeats = out_cache_lens
    total_elements = np.sum(repeats)

    assert total_elements == out_cache_loc.shape[0], (
        f"not all allocate cache loc is assigned to req_token_pool, it's may lead to mem leak, assigned {total_elements}, allocate {out_cache_loc.shape[0]}"
    )

    if total_elements == 0:
        return

    # 1. Row indices: repeat req_pool_indices
    row_indices = np.repeat(req_pool_indices, repeats)

    # 2. Col indices: generate ranges
    block_starts = np.concatenate(([0], np.cumsum(repeats)[:-1]))
    shifts = np.repeat(block_starts, repeats)
    col_indices = np.arange(total_elements) - shifts + np.repeat(start_offsets, repeats)

    # 3. Assign
    req_to_token_pool.req_to_token[row_indices, col_indices] = out_cache_loc
