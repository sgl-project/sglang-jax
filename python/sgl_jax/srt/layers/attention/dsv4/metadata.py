"""M2.1 -- DeepSeek-V4 attention metadata and address derivation.

Everything here is host-side NumPy on purpose. C2 hands over request lengths and
positions as host arrays, and C1's allocator is a host-side page ledger, so the
derivation is plain array arithmetic that a test can drive with fixed inputs and
no device. Only the finished arrays cross to the device.

Two invariants shape the whole module:

**Allocated capacity is not the same thing as a visible entry.** A page reserved
for compressed history exists long before the group that fills it completes.
Reading a "capacity" count as if it were data is how a query ends up attending to
an uninitialised record, so the two are separate fields (`capacity_entries` vs
`visible_entries_*`) and never derived from each other.

**Per-query causality is the kernel's job, not this module's.** A chunk's early
queries must not see compressed groups completed by its later tokens. This module
publishes per-request counts plus each query's absolute `position`; the attention
kernel narrows that per query. Doing it here would mean materialising a
per-query-per-group structure for no benefit. `visible_groups_for_positions` is
the reference rule the kernel has to honour -- a group is visible once complete,
`(position + 1) // ratio`, and the compressed set is a *union* with the sliding
window rather than a complement.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
from jax.tree_util import register_pytree_node_class

__all__ = [
    "DeepseekV4AttentionMetadata",
    "DeepseekV4RatioMetadata",
    "boundary_capacity",
    "complete_groups",
    "derive_attention_metadata",
    "visible_groups_for_positions",
]

# Fill for the non-index payload of an unused boundary slot. Consumers must gate
# on ``boundary_valid_mask`` rather than testing for this value.
#
# Note it is NOT used for ``boundary_token_indices``: JAX wraps negative indices,
# so a -1 gather index would quietly read the *last* live token instead of
# reading nothing. That array is padded with ``num_tokens`` -- genuinely out of
# range, so `mode="drop"`/`"fill"` behaves as intended.
INERT_BOUNDARY = -1


def complete_groups(consumed: np.ndarray, ratio: int) -> np.ndarray:
    """How many compression groups are fully complete after `consumed` tokens.

    Group ``g`` covers positions ``[g*ratio, (g+1)*ratio - 1]``, so it completes
    exactly when the token at position ``(g+1)*ratio - 1`` has been consumed.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return np.asarray(consumed, np.int64) // ratio


def visible_groups_for_positions(positions: np.ndarray, ratio: int, window_size=None) -> np.ndarray:
    """Compressed groups a query at each position may attend to: ``(position+1)//ratio``.

    A group is visible once it is **complete**, and that is the only condition.
    The compressed set is a *union* with the sliding window, not a complement:
    at ratio == window_size == 128, a query at position 255 attends both to window
    tokens 128..255 and to the compressed record of group 1, which covers those
    same tokens. The overlap is deliberate.

    This matches what the HCA path in `kernels/hca` actually implements --
    `attention.py` masks with `key_positions < (chunk_positions + 1) // 128` and
    derives `compressed_entries = (positions + 1) // ratio - 1` -- and it is the
    same rule as `dsv4.indexer.visible_entries_for_query`, which delegates here so
    there is one definition.

    `window_size` is accepted and ignored; it is retained so existing callers keep
    working, and because an earlier version of this function wrongly used it (see
    below).

    Earlier this computed ``(position - window_size + 1) // ratio``, i.e. only
    groups lying entirely older than the window. That was wrong: it hides the most
    recent complete group from every query, and at ratio == window_size it is off
    by exactly one group for any position that is not a group boundary. It was
    stated as "the rule M2.4 must implement", so it would have been propagated
    into the attention kernel as a silent numerical mismatch.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return (np.asarray(positions, np.int64) + 1) // ratio


def boundary_capacity(num_tokens: int, num_requests: int, ratio: int) -> int:
    """Padded length of the boundary arrays for a step.

    Depends only on the padded token and request counts, never on how many
    boundaries this particular step happens to hit -- otherwise every change in
    the boundary count would be a new compiled program. ``num_tokens // ratio``
    covers a dense run and ``+ num_requests`` covers one straddling group per
    request.
    """
    if num_tokens < 0 or num_requests < 0 or ratio <= 0:
        raise ValueError("invalid boundary capacity inputs")
    return max(1, num_tokens // ratio + num_requests)


@register_pytree_node_class
@dataclass(frozen=True)
class DeepseekV4RatioMetadata:
    """The compression-ratio-dependent half, one instance per ratio in use.

    Flash 0731 uses two: ratio 4 (CSA layers, which also drive the indexer cache)
    and ratio 128 (HCA layers). Ratio-0 layers have no compressed history and use
    none of this.

    Boundary arrays are padded to `boundary_capacity` with `INERT_BOUNDARY`.
    """

    # -- compression events: where a group completes inside this step's tokens --
    # Padded slots are inert: `boundary_valid_mask` is False, the token index is
    # `num_tokens` (out of range, so no wrap), and the rest is INERT_BOUNDARY.
    boundary_valid_mask: jax.Array  # [cap] gate every use on this
    boundary_token_indices: jax.Array  # [cap] index into the query axis
    boundary_group_ids: jax.Array  # [cap] which group id completed
    boundary_request_ids: jax.Array  # [cap] which request it belongs to
    boundary_write_entries: jax.Array  # [cap] flat compressed-entry address
    boundary_state_slots: jax.Array  # [cap] request slot whose state feeds it

    # -- what a reader may actually look at ------------------------------------
    # Complete groups before this step's tokens are applied, and after. Reads
    # that precede the write use `before`; the next step uses `after`.
    visible_entries_before: jax.Array  # [B]
    visible_entries_after: jax.Array  # [B]
    # What the allocation could hold. Always >= visible_entries_after. NOT data.
    capacity_entries: jax.Array  # [B]

    ratio: int

    def tree_flatten(self):
        return (
            (
                self.boundary_valid_mask,
                self.boundary_token_indices,
                self.boundary_group_ids,
                self.boundary_request_ids,
                self.boundary_write_entries,
                self.boundary_state_slots,
                self.visible_entries_before,
                self.visible_entries_after,
                self.capacity_entries,
            ),
            (self.ratio,),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        (ratio,) = aux
        return cls(*children, ratio=ratio)

    @property
    def num_boundaries(self) -> int:
        """Padded capacity, not the live count -- the live count is data."""
        return int(self.boundary_token_indices.shape[0])


@register_pytree_node_class
@dataclass(frozen=True)
class DeepseekV4AttentionMetadata:
    """One DP rank's V4 attention metadata for a single forward step.

    Per-query arrays are length `num_tokens` (the padded token axis); per-request
    arrays are length `num_requests` (the padded batch axis). Padded query slots
    carry `valid_token_mask == False` and padded requests `request_valid_mask ==
    False`; neither is given a plausible-looking address, so a stray write is
    dropped rather than corrupting slot zero.
    """

    # -- per query token -------------------------------------------------------
    query_request_ids: jax.Array  # [T] which request each token belongs to
    query_positions: jax.Array  # [T] absolute position within that request
    valid_token_mask: jax.Array  # [T]
    swa_write_loc: jax.Array  # [T] physical SWA slot
    history_write_loc: jax.Array  # [T] rank-local original token slot

    # -- per request -----------------------------------------------------------
    q_lens: jax.Array  # [B] query tokens contributed this step
    cu_q_lens: jax.Array  # [B+1]
    prefix_lens: jax.Array  # [B] tokens already consumed before this step
    seq_lens: jax.Array  # [B] prefix_lens + q_lens
    request_slots: jax.Array  # [B] ReqToTokenPool slot, indexes the state pool
    state_init_mask: jax.Array  # [B] reset state before consuming this slot
    request_valid_mask: jax.Array  # [B]

    # -- ratio-dependent -------------------------------------------------------
    c4: DeepseekV4RatioMetadata
    c128: DeepseekV4RatioMetadata

    # -- static ----------------------------------------------------------------
    page_size: int
    window_size: int

    def tree_flatten(self):
        return (
            (
                self.query_request_ids,
                self.query_positions,
                self.valid_token_mask,
                self.swa_write_loc,
                self.history_write_loc,
                self.q_lens,
                self.cu_q_lens,
                self.prefix_lens,
                self.seq_lens,
                self.request_slots,
                self.state_init_mask,
                self.request_valid_mask,
                self.c4,
                self.c128,
            ),
            (self.page_size, self.window_size),
        )

    @classmethod
    def tree_unflatten(cls, aux, children):
        page_size, window_size = aux
        return cls(*children, page_size=page_size, window_size=window_size)

    def ratio(self, ratio: int) -> DeepseekV4RatioMetadata:
        """Look up the ratio-dependent half by compression ratio."""
        if ratio == 4:
            return self.c4
        if ratio == 128:
            return self.c128
        raise ValueError(f"no V4 metadata for compression ratio {ratio}")

    @property
    def num_tokens(self) -> int:
        return int(self.query_positions.shape[0])

    @property
    def num_requests(self) -> int:
        return int(self.q_lens.shape[0])


def _derive_ratio(
    *,
    ratio: int,
    positions: np.ndarray,
    query_request_ids: np.ndarray,
    valid_token_mask: np.ndarray,
    history_write_loc: np.ndarray,
    prefix_lens: np.ndarray,
    seq_lens: np.ndarray,
    request_slots: np.ndarray,
    request_valid_mask: np.ndarray,
    pages_per_request: np.ndarray,
    page_size: int,
    num_tokens: int,
    num_requests: int,
) -> DeepseekV4RatioMetadata:
    if page_size % ratio:
        raise ValueError(
            f"page_size={page_size} must be a multiple of ratio={ratio}; otherwise a "
            "compression group straddles two pages and loc//ratio is not a single address"
        )

    # A group completes at the token that fills it. `loc // ratio` is then the
    # flat compressed-entry address (C1's contract), and it is the same for every
    # token in the group precisely because page_size % ratio == 0.
    completes = valid_token_mask & (((positions + 1) % ratio) == 0)
    hits = np.flatnonzero(completes).astype(np.int32)

    cap = boundary_capacity(num_tokens, num_requests, ratio)
    if hits.size > cap:  # pragma: no cover - capacity is derived to make this impossible
        raise AssertionError(f"{hits.size} boundaries exceed derived capacity {cap}")

    def pad(values, fill):
        out = np.full((cap,), fill, np.int32)
        out[: values.size] = values
        return out

    boundary_valid = np.arange(cap) < hits.size
    # `num_tokens` is out of range on the query axis; -1 would wrap under JAX
    # indexing and silently gather the last live token.
    boundary_tokens = pad(hits, num_tokens)
    boundary_groups = pad((positions[hits] // ratio).astype(np.int32), INERT_BOUNDARY)
    boundary_requests = pad(query_request_ids[hits].astype(np.int32), INERT_BOUNDARY)
    boundary_entries = pad((history_write_loc[hits] // ratio).astype(np.int32), INERT_BOUNDARY)
    boundary_slots = pad(request_slots[query_request_ids[hits]].astype(np.int32), INERT_BOUNDARY)

    # Visible vs capacity. `visible_*` counts groups that have actually been
    # produced; `capacity_entries` counts the slots the allocation reserved. The
    # second is always >= the first and is not readable data.
    entries_per_page = page_size // ratio
    visible_before = np.where(request_valid_mask, complete_groups(prefix_lens, ratio), 0)
    visible_after = np.where(request_valid_mask, complete_groups(seq_lens, ratio), 0)
    capacity = np.where(request_valid_mask, pages_per_request * entries_per_page, 0)
    if np.any(capacity < visible_after):
        raise ValueError(
            "compressed capacity is below the number of groups this step completes; "
            "the allocator did not reserve enough history pages"
        )

    return DeepseekV4RatioMetadata(
        boundary_valid_mask=boundary_valid,
        boundary_token_indices=boundary_tokens,
        boundary_group_ids=boundary_groups,
        boundary_request_ids=boundary_requests,
        boundary_write_entries=boundary_entries,
        boundary_state_slots=boundary_slots,
        visible_entries_before=visible_before.astype(np.int32),
        visible_entries_after=visible_after.astype(np.int32),
        capacity_entries=capacity.astype(np.int32),
        ratio=ratio,
    )


def derive_attention_metadata(
    *,
    q_lens,
    prefix_lens,
    positions,
    request_slots,
    history_write_loc,
    swa_write_loc,
    pages_per_request,
    page_size: int,
    window_size: int,
    state_init_mask=None,
    num_tokens: int | None = None,
) -> DeepseekV4AttentionMetadata:
    """Derive one step's attention metadata from what C2/C3 and C1 already know.

    Args:
      q_lens: [B] query tokens per request this step; 0 for an inactive request.
      prefix_lens: [B] tokens already consumed before this step.
      positions: [T] absolute position of each query token. The first
        ``sum(q_lens)`` entries are the live tokens in request order; the rest is
        padding and is not read.
      request_slots: [B] ReqToTokenPool slot per request, which is also the
        compressor-state index (C1.2 binds them; there is no separate state
        allocator).
      history_write_loc: [T] rank-local original token slot from
        ``alloc_extend``/``alloc_decode``.
      swa_write_loc: [T] physical SWA slot for the same token.
      pages_per_request: [B] history pages the allocation currently holds.
      page_size: 128 or 256.
      window_size: sliding-window length.
      state_init_mask: [B] requests whose state must be reset before use (first
        execution, or a recompute from zero after retract). Defaults to all False.
      num_tokens: padded token-axis length; defaults to ``len(positions)``.

    Batch order is irrelevant to correctness: every per-request quantity is
    carried by value and state is addressed by `request_slots`, never by position
    in the batch.
    """
    q_lens = np.asarray(q_lens, np.int64)
    prefix_lens = np.asarray(prefix_lens, np.int64)
    positions = np.asarray(positions, np.int64)
    request_slots = np.asarray(request_slots, np.int64)
    history_write_loc = np.asarray(history_write_loc, np.int64)
    swa_write_loc = np.asarray(swa_write_loc, np.int64)
    pages_per_request = np.asarray(pages_per_request, np.int64)

    if page_size not in (128, 256):
        raise ValueError(f"V4 page_size must be 128 or 256, got {page_size}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    for name, arr in (
        ("q_lens", q_lens),
        ("prefix_lens", prefix_lens),
        ("request_slots", request_slots),
        ("pages_per_request", pages_per_request),
    ):
        if arr.ndim != 1 or arr.shape != q_lens.shape:
            raise ValueError(f"{name} must be 1-D with the same length as q_lens")
    if np.any(q_lens < 0) or np.any(prefix_lens < 0) or np.any(pages_per_request < 0):
        raise ValueError("lengths and page counts must be non-negative")

    num_requests = int(q_lens.size)
    live = int(q_lens.sum())
    padded_tokens = int(num_tokens if num_tokens is not None else positions.size)
    if padded_tokens < live:
        raise ValueError(f"token axis of {padded_tokens} cannot hold {live} query tokens")
    for name, arr in (
        ("positions", positions),
        ("history_write_loc", history_write_loc),
        ("swa_write_loc", swa_write_loc),
    ):
        if arr.ndim != 1 or arr.size < padded_tokens:
            raise ValueError(f"{name} must be 1-D with at least {padded_tokens} entries")
    positions = positions[:padded_tokens]
    history_write_loc = history_write_loc[:padded_tokens]
    swa_write_loc = swa_write_loc[:padded_tokens]

    request_valid_mask = q_lens > 0
    seq_lens = prefix_lens + q_lens
    valid_token_mask = np.arange(padded_tokens) < live
    query_request_ids = np.zeros((padded_tokens,), np.int64)
    query_request_ids[:live] = np.repeat(np.arange(num_requests, dtype=np.int64), q_lens)

    # The positions C3 supplies must be the contiguous run each request is
    # actually consuming. A mismatch here silently shifts every derived address,
    # so it is a hard error rather than something the kernel discovers.
    expected = np.concatenate(
        [np.arange(p, p + n) for p, n in zip(prefix_lens.tolist(), q_lens.tolist()) if n]
        or [np.empty(0, np.int64)]
    )
    if not np.array_equal(positions[:live], expected):
        raise ValueError("positions must be prefix_lens..seq_lens-1 per request, in request order")

    if state_init_mask is None:
        state_init_mask = np.zeros((num_requests,), bool)
    else:
        state_init_mask = np.asarray(state_init_mask, bool)
        if state_init_mask.shape != q_lens.shape:
            raise ValueError("state_init_mask must be 1-D with the same length as q_lens")
        if np.any(state_init_mask & (prefix_lens > 0)):
            raise ValueError(
                "state may only be initialised for a request starting from zero; a "
                "continuing chunk must keep the state it accumulated"
            )

    cu_q_lens = np.concatenate((np.zeros((1,), np.int64), np.cumsum(q_lens)))

    shared = dict(
        positions=positions,
        query_request_ids=query_request_ids,
        valid_token_mask=valid_token_mask,
        history_write_loc=history_write_loc,
        prefix_lens=prefix_lens,
        seq_lens=seq_lens,
        request_slots=request_slots,
        request_valid_mask=request_valid_mask,
        pages_per_request=pages_per_request,
        page_size=page_size,
        num_tokens=padded_tokens,
        num_requests=num_requests,
    )

    return DeepseekV4AttentionMetadata(
        query_request_ids=query_request_ids.astype(np.int32),
        query_positions=positions.astype(np.int32),
        valid_token_mask=valid_token_mask,
        swa_write_loc=np.where(valid_token_mask, swa_write_loc, -1).astype(np.int32),
        history_write_loc=np.where(valid_token_mask, history_write_loc, -1).astype(np.int32),
        q_lens=q_lens.astype(np.int32),
        cu_q_lens=cu_q_lens.astype(np.int32),
        prefix_lens=prefix_lens.astype(np.int32),
        seq_lens=seq_lens.astype(np.int32),
        request_slots=request_slots.astype(np.int32),
        state_init_mask=state_init_mask,
        request_valid_mask=request_valid_mask,
        c4=_derive_ratio(ratio=4, **shared),
        c128=_derive_ratio(ratio=128, **shared),
        page_size=page_size,
        window_size=window_size,
    )
