"""Device-resident seed state for the Frozen-KV MTP draft loop.

Gemma 4's assistant does not own a draft KV cache.  After target verification,
the next assistant proposal therefore starts from the target hidden state at
the last accepted row and its corresponding token.  This module describes
that hand-off without depending on EAGLE's draft-extension state or on the
serving scheduler.

The scheduler still owns request admission, slot padding, and batch buckets.
``select_after_verify`` only adapts the scheduler's padded verification layout
to one seed row per live request.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


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


@register_pytree_node_class
@dataclass(frozen=True)
class FrozenKvMtpSeedState:
    """One next-draft seed per live request.

    All fields are request-aligned.  ``bonus_token`` and ``target_hidden`` are
    device values; the length and identity arrays are carried alongside them
    so merge/filter operations cannot accidentally detach page metadata from
    the corresponding seed.
    """

    bonus_token: jax.Array
    target_hidden: jax.Array
    committed_lens: jax.Array
    allocate_lens: jax.Array
    request_indices: jax.Array
    valid_mask: jax.Array

    def __post_init__(self):
        """Reject state whose request-aligned fields have drifted apart.

        The verify-to-draft handoff is device-resident, so values cannot be
        checked eagerly.  Static leading dimensions are nevertheless known
        during tracing and are enough to catch the dangerous case where a
        token/hidden seed is paired with another request's length or page
        metadata.
        """
        fields = (
            ("target_hidden", self.target_hidden),
            ("committed_lens", self.committed_lens),
            ("allocate_lens", self.allocate_lens),
            ("request_indices", self.request_indices),
            ("valid_mask", self.valid_mask),
        )
        if self.bonus_token.ndim != 1:
            raise ValueError("Frozen-KV bonus_token must be rank 1")
        if self.target_hidden.ndim < 2:
            raise ValueError("Frozen-KV target_hidden must have a batch dimension")
        batch_size = self.bonus_token.shape[0]
        for name, value in fields:
            if value.ndim == 0 or value.shape[0] != batch_size:
                raise ValueError(
                    f"Frozen-KV seed field {name!r} has {value.shape}; "
                    f"expected leading dimension {batch_size}"
                )

    def tree_flatten(self):
        return (
            (
                self.bonus_token,
                self.target_hidden,
                self.committed_lens,
                self.allocate_lens,
                self.request_indices,
                self.valid_mask,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _aux_data, children):
        return cls(*children)

    @property
    def batch_size(self) -> int:
        return self.bonus_token.shape[0]

    def merge(self, other: FrozenKvMtpSeedState) -> FrozenKvMtpSeedState:
        """Append another prefilled/running state in scheduler order."""
        if not isinstance(other, FrozenKvMtpSeedState):
            raise TypeError(f"expected FrozenKvMtpSeedState, got {type(other).__name__}")
        if self.target_hidden.ndim != other.target_hidden.ndim:
            raise ValueError("Frozen-KV seed hidden-state ranks must match")
        if self.target_hidden.shape[1:] != other.target_hidden.shape[1:]:
            raise ValueError("Frozen-KV seed hidden-state shapes must match")
        return FrozenKvMtpSeedState(
            bonus_token=jnp.concatenate((self.bonus_token, other.bonus_token), axis=0),
            target_hidden=jnp.concatenate((self.target_hidden, other.target_hidden), axis=0),
            committed_lens=jnp.concatenate((self.committed_lens, other.committed_lens), axis=0),
            allocate_lens=jnp.concatenate((self.allocate_lens, other.allocate_lens), axis=0),
            request_indices=jnp.concatenate((self.request_indices, other.request_indices), axis=0),
            valid_mask=jnp.concatenate((self.valid_mask, other.valid_mask), axis=0),
        )

    def filter(self, indices: jax.Array) -> FrozenKvMtpSeedState:
        """Keep request rows after completion/retraction/filtering."""
        indices = jnp.asarray(indices, dtype=jnp.int32)
        return FrozenKvMtpSeedState(
            bonus_token=jnp.take(self.bonus_token, indices, axis=0),
            target_hidden=jnp.take(self.target_hidden, indices, axis=0),
            committed_lens=jnp.take(self.committed_lens, indices, axis=0),
            allocate_lens=jnp.take(self.allocate_lens, indices, axis=0),
            request_indices=jnp.take(self.request_indices, indices, axis=0),
            valid_mask=jnp.take(self.valid_mask, indices, axis=0),
        )

    def scatter(self, selector: jax.Array, total_size: int) -> FrozenKvMtpSeedState:
        """Place compact request seeds into scheduler-padded slots.

        This is metadata transport only; bucket selection remains owned by the
        scheduler. Invalid padded slots are marked false and never consumed by
        the Frozen worker.
        """
        selector = jnp.asarray(selector, dtype=jnp.int32)
        total_size = int(total_size)
        if selector.ndim != 1 or selector.shape[0] != self.batch_size:
            raise ValueError("Frozen-KV seed selector must match seed batch size")
        zeros_token = jnp.zeros((total_size,), dtype=self.bonus_token.dtype)
        zeros_hidden = jnp.zeros(
            (total_size,) + self.target_hidden.shape[1:], dtype=self.target_hidden.dtype
        )
        zeros_int = jnp.zeros((total_size,), dtype=self.committed_lens.dtype)
        zeros_bool = jnp.zeros((total_size,), dtype=bool)
        return FrozenKvMtpSeedState(
            bonus_token=zeros_token.at[selector].set(self.bonus_token),
            target_hidden=zeros_hidden.at[selector].set(self.target_hidden),
            committed_lens=zeros_int.at[selector].set(self.committed_lens),
            allocate_lens=zeros_int.at[selector].set(self.allocate_lens),
            request_indices=zeros_int.at[selector].set(self.request_indices),
            valid_mask=zeros_bool.at[selector].set(self.valid_mask),
        )


@partial(jax.jit, static_argnames=("rows_per_request",))
def select_after_verify(
    verified_tokens: jax.Array,
    target_hidden: jax.Array,
    slot_selector: jax.Array,
    accept_lengths: jax.Array,
    committed_lens: jax.Array,
    allocate_lens: jax.Array,
    request_indices: jax.Array,
    *,
    rows_per_request: int,
) -> FrozenKvMtpSeedState:
    """Select one accepted target row per request from padded verify output.

    ``slot_selector`` maps compact live-request order to padded scheduler slots.
    For a row layout of ``rows_per_request`` candidates per slot, the accepted
    row is ``slot * rows_per_request + accept_length - 1``.  The operation is
    pure JAX and can therefore run inside the device-side verify-to-draft
    hand-off.  Invalid acceptance lengths are represented by ``valid_mask``
    while indexing a clipped row to keep the compiled program memory-safe.
    """
    rows_per_request = int(rows_per_request)
    if rows_per_request <= 0:
        raise ValueError("rows_per_request must be positive")

    verified_tokens = jnp.asarray(verified_tokens)
    target_hidden = jnp.asarray(target_hidden)
    slot_selector = jnp.asarray(slot_selector, dtype=jnp.int32)
    accept_lengths = jnp.asarray(accept_lengths, dtype=jnp.int32)
    committed_lens = jnp.asarray(committed_lens, dtype=jnp.int32)
    allocate_lens = jnp.asarray(allocate_lens, dtype=jnp.int32)
    request_indices = jnp.asarray(request_indices, dtype=jnp.int32)

    if verified_tokens.ndim != 1:
        raise ValueError("Frozen-KV verified_tokens must be rank 1")
    if target_hidden.ndim < 2 or target_hidden.shape[0] != verified_tokens.shape[0]:
        raise ValueError("Frozen-KV verify tokens and hidden rows must be aligned")
    request_count = slot_selector.shape[0]
    for name, value in (
        ("accept_lengths", accept_lengths),
        ("committed_lens", committed_lens),
        ("allocate_lens", allocate_lens),
        ("request_indices", request_indices),
    ):
        if value.ndim != 1 or value.shape[0] != request_count:
            raise ValueError(
                f"Frozen-KV {name} must have one entry per selected request "
                f"(got {value.shape}, expected ({request_count},))"
            )
    if target_hidden.shape[0] < rows_per_request:
        raise ValueError("Frozen-KV verify rows are smaller than one request bucket")

    row_offset = accept_lengths - 1
    valid_mask = (row_offset >= 0) & (row_offset < rows_per_request)
    safe_offset = jnp.clip(row_offset, 0, rows_per_request - 1)
    flat_rows = slot_selector * rows_per_request + safe_offset

    def gather_selected_rows(value: jax.Array) -> jax.Array:
        """Gather compact seed rows while retaining a legal mesh layout.

        JAX cannot infer whether this cross-row gather should remain
        partitioned or become replicated.  The answer depends on the compact
        request bucket, not on the large padded source.  State it explicitly
        for both the selected token and its corresponding hidden row.
        """
        value_sharding = jax.typeof(value).sharding
        mesh = getattr(value_sharding, "mesh", None)
        if mesh is None or getattr(mesh, "empty", False):
            return jnp.take(value, flat_rows, axis=0)
        from jax.sharding import NamedSharding
        from jax.sharding import PartitionSpec as P

        data_size = int(mesh.shape.get("data", 1))
        # A c1/c2 request bucket cannot be partitioned across TP.  Replicate
        # its selected *seed rows* (not the full candidate hidden tensor).
        # Larger buckets preserve request-row data sharding for the relay.
        if "data" in mesh.shape and request_count % data_size == 0:
            selected_spec = P("data", *([None] * (value.ndim - 1)))
        else:
            selected_spec = P()
        return value.at[flat_rows].get(out_sharding=NamedSharding(mesh, selected_spec))

    return FrozenKvMtpSeedState(
        bonus_token=gather_selected_rows(verified_tokens),
        target_hidden=gather_selected_rows(target_hidden),
        committed_lens=committed_lens,
        allocate_lens=allocate_lens,
        request_indices=request_indices,
        valid_mask=valid_mask,
    )
