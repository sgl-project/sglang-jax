"""Greedy linear-chain verification for the Frozen-KV MTP draft loop.

Gemma 4's assistant does not own a draft KV cache.  After target verification,
the next assistant proposal therefore starts from the target hidden state at
the last accepted row and its corresponding token. The fused worker performs
that row selection and relay publication in the same target-verify program.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("draft_token_num",))
def verify_frozen_kv_mtp_chain_greedy(
    draft_tokens: jax.Array,
    target_logits: jax.Array,
    *,
    draft_token_num: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Greedily verify a top-k-one Frozen-KV MTP chain on its native layout.

    Unlike generic EAGLE tree verification, Gemma 4 Frozen-KV MTP is launched
    with one candidate per depth, so its verify graph is a linear chain.  That
    means acceptance is local to each request:

    ``target_argmax[i] == draft_tokens[i + 1]``.

    Keeping this expression in JAX lets the target logits retain their model
    output sharding.  In particular, callers must not first turn logits and
    hidden states into ``P()``-replicated arrays solely to enter the generic
    tree Pallas kernel.  The returned token layout is the scheduler's standard
    ``[slot0 candidates..., slot1 candidates...]`` layout, and every accepted
    length includes the bonus target token, matching ``EagleVerifyInput``.

    This deliberately supports only the linear ``topk == 1`` contract.  A
    branching EAGLE tree has different traversal semantics and continues to
    use the generic verifier.
    """
    draft_token_num = int(draft_token_num)
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")
    if draft_tokens.ndim != 1:
        raise ValueError("Frozen-KV draft_tokens must be rank 1")
    if target_logits.ndim != 2:
        raise ValueError("Frozen-KV target_logits must have shape [rows, vocab]")
    if draft_tokens.shape[0] != target_logits.shape[0]:
        raise ValueError(
            "Frozen-KV draft token and target-logit rows must match: "
            f"{draft_tokens.shape[0]} vs {target_logits.shape[0]}."
        )
    if draft_tokens.shape[0] % draft_token_num:
        raise ValueError(
            "Frozen-KV target rows must be divisible by draft_token_num: "
            f"rows={draft_tokens.shape[0]}, draft_token_num={draft_token_num}."
        )

    candidates = draft_tokens.reshape((-1, draft_token_num)).astype(jnp.int32)
    # ``argmax`` is lowered against the target program's existing layout.  On
    # TP this may contain the necessary vocab reduction, but it must not force
    # the *whole* [rows, vocab] tensor through a host-orchestrated P() replica.
    target_predict = jnp.argmax(target_logits, axis=-1).astype(jnp.int32).reshape(candidates.shape)
    # The target's flat candidate rows are commonly sharded over ``data``.
    # After a [rows] -> [request, depth] reshape, that layout can place the
    # *depth* axis on the mesh (e.g. the c1 precompile shape is [1, 4@data]).
    # Prefix acceptance reduces over depth, so make depth local before cumprod.
    # This transfers only tiny token-ID vectors, never [rows, vocab] logits or
    # [rows, hidden] target states.  For c1/c2 the request dimension cannot be
    # partitioned over TP, so the small vectors are replicated instead.
    sharding = getattr(jax.typeof(target_predict).sharding, "mesh", None)
    if sharding is not None and not getattr(sharding, "empty", False):
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        data_size = int(sharding.shape.get("data", 1))
        request_spec = (
            P("data", None)
            if "data" in sharding.shape and candidates.shape[0] % data_size == 0
            else P()
        )
        request_sharding = NamedSharding(sharding, request_spec)
        candidates = jax.sharding.reshard(candidates, request_sharding)
        target_predict = jax.sharding.reshard(target_predict, request_sharding)
    matches = target_predict[:, :-1] == candidates[:, 1:]
    accepted_draft = jnp.sum(jnp.cumprod(matches.astype(jnp.int32), axis=1), axis=1)
    accept_lengths = (accepted_draft + 1).astype(jnp.int32)

    rows = jnp.arange(draft_token_num, dtype=jnp.int32)[None, :]
    bases = jnp.arange(candidates.shape[0], dtype=jnp.int32)[:, None] * draft_token_num
    accept_index = jnp.where(rows < accept_lengths[:, None], bases + rows, -1)
    return target_predict.reshape(-1), accept_lengths, accept_index.reshape(-1)
