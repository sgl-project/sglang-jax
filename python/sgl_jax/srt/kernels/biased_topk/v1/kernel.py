"""Sort-free top-k routing for TPU.

The plain path returns the same values and ids as ``jax.lax.top_k`` for finite
f32 router scores. The biased path selects with
``router_logits + correction_bias`` while returning pre-bias weights. Tokens
occupy the TPU lane dimension in the VMEM compute layout so expert reductions
avoid cross-lane permutation.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.experimental.pallas as pl
import jax.numpy as jnp

from sgl_jax.srt.kernels.biased_topk.tuned_block_sizes import get_tuned_bt

NEG_INF = -jnp.inf
SAFE_AUTO_BT = 2048


def get_interpret() -> bool:
    return os.environ.get("PALLAS_INTERPRET", "").strip().lower() in ("1", "true")


def _safe_auto_block_tokens(batch_size: int) -> int | None:
    if batch_size <= SAFE_AUTO_BT:
        return batch_size
    for candidate in range(SAFE_AUTO_BT, 0, -128):
        if batch_size % candidate == 0:
            return candidate
    return None


def _select_topk(
    logits,  # [E, BT] f32, values returned to the caller
    scores,  # [E, BT] f32, values used for selection
    weights_ref,  # [topk, BT] f32
    ids_ref,  # [topk, BT] i32
    *,
    topk: int,
    num_experts: int,
):
    block_tokens = logits.shape[1]

    expert_iota = jax.lax.broadcasted_iota(
        jnp.int32,
        (num_experts, block_tokens),
        0,
    )
    row_iota = jax.lax.broadcasted_iota(jnp.int32, (topk, block_tokens), 0)
    # Derive the loop carry from ``scores`` so explicit shard-map axis types
    # are preserved across the fori_loop carry boundary on JAX 0.10+.
    ids_init = jnp.full_like(scores[:topk], -1, dtype=jnp.int32)
    weights_init = jnp.zeros_like(scores[:topk], dtype=jnp.float32)

    def select_one(k, carry):
        current_scores, ids, weights = carry
        max_score = jnp.max(current_scores, axis=0, keepdims=True)
        selected_id = jnp.min(
            jnp.where(current_scores == max_score, expert_iota, num_experts),
            axis=0,
            keepdims=True,
        )
        selected = expert_iota == selected_id
        selected_weight = jnp.sum(
            jnp.where(selected, logits, 0.0),
            axis=0,
            keepdims=True,
        )
        write_row = row_iota == k
        ids = jnp.where(write_row, selected_id.astype(jnp.int32), ids)
        weights = jnp.where(write_row, selected_weight.astype(jnp.float32), weights)
        current_scores = jnp.where(selected, NEG_INF, current_scores)
        return current_scores, ids, weights

    _, ids, weights = jax.lax.fori_loop(
        0,
        topk,
        select_one,
        (scores, ids_init, weights_init),
        unroll=True,
    )
    weights_ref[...] = weights
    ids_ref[...] = ids


def _biased_topk_kernel(
    logits_ref,  # [BT, E] f32, pre-bias
    bias_ref,  # [E] f32
    weights_ref,  # [topk, BT] f32
    ids_ref,  # [topk, BT] i32
    *,
    topk: int,
    num_experts: int,
):
    logits = logits_ref[...].astype(jnp.float32).T  # [E, BT]
    scores = logits + bias_ref[...].astype(jnp.float32)[:, None]
    _select_topk(
        logits,
        scores,
        weights_ref,
        ids_ref,
        topk=topk,
        num_experts=num_experts,
    )


def _topk_kernel(
    logits_ref,  # [BT, E] f32
    weights_ref,  # [topk, BT] f32
    ids_ref,  # [topk, BT] i32
    *,
    topk: int,
    num_experts: int,
):
    logits = logits_ref[...].astype(jnp.float32).T  # [E, BT]
    _select_topk(
        logits,
        logits,
        weights_ref,
        ids_ref,
        topk=topk,
        num_experts=num_experts,
    )


def _resolve_block_tokens(
    router_logits: jax.Array,
    *,
    topk: int,
    block_tokens: int | str,
) -> tuple[int, int, int]:
    if router_logits.ndim != 2:
        raise ValueError(f"router_logits must be rank 2, got shape={router_logits.shape}")
    batch_size, num_experts = router_logits.shape
    if num_experts % 128 != 0:
        raise ValueError(f"num_experts must be divisible by 128, got {num_experts}")
    if not 1 <= topk <= num_experts:
        raise ValueError(f"topk must be in [1, {num_experts}], got {topk}")
    if block_tokens == "auto":
        block_tokens = get_tuned_bt(batch_size, num_experts, topk)
        if block_tokens is None:
            block_tokens = _safe_auto_block_tokens(batch_size)
        if block_tokens is None:
            raise ValueError(
                f"no VMEM-safe block_tokens for batch_size={batch_size}; "
                "fall back to jax.lax.top_k"
            )
    block_tokens = int(block_tokens)
    if not 1 <= block_tokens <= batch_size:
        raise ValueError(f"block_tokens must be in [1, {batch_size}], got {block_tokens}")
    if batch_size % block_tokens != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by block_tokens={block_tokens}"
        )
    return batch_size, num_experts, block_tokens


def biased_topk_pallas(
    router_logits: jax.Array,
    correction_bias: jax.Array,
    *,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Select biased top-k expert ids and return their pre-bias weights."""
    batch_size, num_experts, block_tokens = _resolve_block_tokens(
        router_logits,
        topk=topk,
        block_tokens=block_tokens,
    )
    if correction_bias.shape != (num_experts,):
        raise ValueError(
            "correction_bias must have shape "
            f"({num_experts},), got shape={correction_bias.shape}"
        )
    if interpret is None:
        interpret = get_interpret()

    kernel = functools.partial(
        _biased_topk_kernel,
        topk=topk,
        num_experts=num_experts,
    )
    weights_t, ids_t = pl.pallas_call(
        kernel,
        grid=(batch_size // block_tokens,),
        in_specs=[
            pl.BlockSpec((block_tokens, num_experts), lambda i: (i, 0)),
            pl.BlockSpec((num_experts,), lambda i: (0,)),
        ],
        out_specs=[
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, batch_size), jnp.float32),
            jax.ShapeDtypeStruct((topk, batch_size), jnp.int32),
        ],
        interpret=interpret,
        name="biased-topk",
    )(
        router_logits.astype(jnp.float32),
        correction_bias.astype(jnp.float32),
    )
    return weights_t.T, ids_t.T


def topk_pallas(
    router_logits: jax.Array,
    *,
    topk: int,
    block_tokens: int | str = "auto",
    interpret: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return the top-k values and ids of a finite f32 routing matrix."""
    batch_size, num_experts, block_tokens = _resolve_block_tokens(
        router_logits,
        topk=topk,
        block_tokens=block_tokens,
    )
    if interpret is None:
        interpret = get_interpret()

    kernel = functools.partial(
        _topk_kernel,
        topk=topk,
        num_experts=num_experts,
    )
    weights_t, ids_t = pl.pallas_call(
        kernel,
        grid=(batch_size // block_tokens,),
        in_specs=[
            pl.BlockSpec((block_tokens, num_experts), lambda i: (i, 0)),
        ],
        out_specs=[
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
            pl.BlockSpec((topk, block_tokens), lambda i: (0, i)),
        ],
        out_shape=[
            jax.ShapeDtypeStruct((topk, batch_size), jnp.float32),
            jax.ShapeDtypeStruct((topk, batch_size), jnp.int32),
        ],
        interpret=interpret,
        name="topk",
    )(router_logits.astype(jnp.float32))
    return weights_t.T, ids_t.T
