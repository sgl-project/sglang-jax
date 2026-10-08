"""M2.4 -- SWA / CSA / HCA numerical attention.

The three V4 layer types are one formula. They differ only in which compressed
records a query is allowed to see:

===========  ==========================================================
layer        admitted compressed records
===========  ==========================================================
SWA-only     none (ratio 0 layers have no compressed history)
HCA (128)    every completed group
CSA (4)      the indexer's top-k among the completed groups (M2.3)
===========  ==========================================================

The keys are the **union** of the sliding window and the admitted records, not a
partition of history: at ratio == window == 128 a query at position 255 attends to
window tokens 128..255 *and* to the record of group 1, which covers those same
tokens. Confirmed against `kernels/hca/attention.py` and the independent fp32
oracle in `test/srt/kernels/hca/oracle.py`; see #352 for what happens when this is
guessed instead of read.

MLA shape: one tensor serves as both K and V, so the output is
``probs @ keys`` over the same rows the scores were computed from.

The attention sink is a phantom key that contributes to the **denominator only**::

    shift = max(scores.max(over keys), sink)      # per head
    probs = exp(scores - shift)
    out   = (probs @ keys) / (probs.sum(over keys) + exp(sink - shift))

Writing it as an extra key in the numerator would be wrong -- it has no value
vector. The `shift` taking the sink into account is what keeps `exp(sink - shift)`
from overflowing when the sink dominates.

This module never touches compressor state. The SWA cache write is a separate
function so that separation is structural rather than a comment: M2.2 owns state,
M2.4 owns the window.
"""

from __future__ import annotations

import os

import jax.numpy as jnp

# Fused CSA kernel tiling (queries x heads rows per block, keys per tile); the
# defaults are the measured v7x choice, the envs exist for A/B sweeps.
_CSA_FUSED_BLOCK_Q = int(os.environ.get("DSV4_CSA_FUSED_BLOCK_Q", "256"))
_CSA_FUSED_BLOCK_K = int(os.environ.get("DSV4_CSA_FUSED_BLOCK_K", "1024"))
# ``DSV4_PAGED_KV_WRITE=1``: prefill-sized window KV writes go through the page-run
# DMA writer instead of an XLA scatter (decode buckets keep the scatter).
_PAGED_KV_WRITE = os.environ.get("DSV4_PAGED_KV_WRITE", "1") == "1"
_PAGED_KV_WRITE_MIN_TOKENS = int(os.environ.get("DSV4_PAGED_KV_WRITE_MIN_TOKENS", "256"))
# Queries per program on the sparse CSA path (the block's selected-unit union is fetched once).
_CSA_SPARSE_QUERY_BLOCK = int(os.environ.get("DSV4_CSA_SPARSE_QUERY_BLOCK", 0))  # 0 = auto


def _csa_query_block(num_queries: int, local_heads: int) -> int:
    """Queries per program for the blocked kernel: keep QB*H <= 512 rows so the f32
    output block, the q block and the union membership table fit scoped VMEM
    (QB=256 with 8 local heads overflowed on v7x)."""
    qb = _CSA_SPARSE_QUERY_BLOCK or max(8, (512 // max(1, local_heads)) // 8 * 8)
    return int(min(qb, max(8, -(-num_queries // 8) * 8)))


__all__ = [
    "admissible_mask",
    "csa_sparse_attention",
    "dsv4_dense_attention",
    "update_window_kv",
]

_NEG_INF = jnp.finfo(jnp.float32).min


def packed_membership(selected, num_entries):
    """``[T, E]`` bool: entry ``e`` is one of row ``t``'s ``selected`` ([T, K], -1 = unused).

    Builds ``[T, ceil(E/32)]`` uint32 words by OR-reducing, over K, each selection's
    single bit placed in its word (a fused reduction over a logical [T, K, W] tensor,
    W = E/32), then expands each word's 32 bits back to the entry axis. Exact: no
    threshold, no approximation, out-of-range selections are ignored.
    """
    selected = jnp.asarray(selected)
    num_words = (num_entries + 31) // 32
    valid = (selected >= 0) & (selected < num_entries)
    word = jnp.where(valid, selected >> 5, num_words).astype(jnp.int32)  # invalid -> no lane
    bit = jnp.left_shift(jnp.uint32(1), (selected & 31).astype(jnp.uint32))
    lanes = jnp.arange(num_words, dtype=jnp.int32)[None, None, :]
    placed = jnp.where(word[:, :, None] == lanes, bit[:, :, None], jnp.uint32(0))  # [T, K, W]
    words = jnp.bitwise_or.reduce(placed, axis=1)  # [T, W]
    expanded = jnp.broadcast_to(words[:, :, None], (words.shape[0], num_words, 32))
    expanded = expanded.reshape(words.shape[0], num_words * 32)[:, :num_entries]
    shifts = (jnp.arange(num_entries, dtype=jnp.int32) & 31).astype(jnp.uint32)[None, :]
    return (jnp.right_shift(expanded, shifts) & jnp.uint32(1)) != 0


def admissible_mask(
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    window_size: int,
    ratio: int,
    selected_entries=None,
    selected_mask=None,
):
    """Which keys each query may attend to.

    Returns ``(window_mask [T, W], compressed_mask [T, E])``.

    A window row is admissible when it belongs to the query's request and sits
    inside ``(position - window_size, position]`` -- causal on the right, window on
    the left.

    A compressed record is admissible when it belongs to the query's request and its
    group is **complete** at the query: ``entry_id < (position + 1) // ratio``. That
    is the same rule as `dsv4.metadata.visible_groups_for_positions` and
    `dsv4.indexer.visible_entries_for_query`. `ratio == 0` admits nothing.

    `selected_entries`, when given, is ``[T, k]`` of **row indices into the
    compressed key array** (exactly what `dsv4.ref.indexer.csa_indexer_topk_ref` returns for
    the same row ordering), with -1 for unused slots. Selection intersects the
    completeness rule rather than replacing it, so a stale or over-eager selection
    still cannot reach an unwritten group.
    """
    qpos = jnp.asarray(query_positions)[:, None]
    qreq = jnp.asarray(query_request_ids)[:, None]
    valid = jnp.asarray(valid_token_mask, bool)[:, None]

    wpos = jnp.asarray(window_positions)[None, :]
    wreq = jnp.asarray(window_request_ids)[None, :]
    window_mask = valid & (wreq == qreq) & (wpos <= qpos) & (wpos > qpos - window_size)

    entry_ids = jnp.asarray(compressed_entry_ids)[None, :]
    creq = jnp.asarray(compressed_request_ids)[None, :]
    if ratio <= 0:
        compressed_mask = jnp.zeros(window_mask.shape[:1] + entry_ids.shape[1:], bool)
    else:
        complete = entry_ids < ((qpos + 1) // ratio)
        compressed_mask = valid & (creq == qreq) & complete

    if selected_entries is not None:
        selected = jnp.asarray(selected_entries)
        num_entries = entry_ids.shape[1]
        # The [T, E] membership of the indexer's top-k rows is built by reducing a
        # logical [T, K, E/32] tensor of one-bit words (see ``packed_membership``).
        chosen = packed_membership(selected, num_entries)
        compressed_mask = compressed_mask & chosen
    if selected_mask is not None:
        # ``[T, E]`` membership already in gathered-row coordinates (the indexer's
        # threshold path); intersects the completeness rule like ``selected_entries``.
        compressed_mask = compressed_mask & jnp.asarray(selected_mask, bool)

    return window_mask, compressed_mask


def dsv4_dense_attention(
    q,
    window_kv,
    compressed_kv,
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    attention_sink,
    softmax_scale: float,
    window_size: int,
    ratio: int,
    selected_entries=None,
    selected_mask=None,
):
    """Attention output for one V4 layer.

    Args:
      q: ``[T, H, D]``.
      window_kv: ``[W, D]`` sliding-window KV rows (MLA: K and V are the same).
      compressed_kv: ``[E, D]`` compressed records gathered for this step.
      attention_sink: ``[H]`` per-head sink logit.
      ratio: 0 for SWA-only layers, 4 for CSA, 128 for HCA.
      selected_entries: CSA top-k row indices, or None to admit every completed
        record (HCA) / nothing (ratio 0).

    Returns:
      ``[T, H, D]`` float32. Padded query rows are zero.
    """
    q = jnp.asarray(q, jnp.float32)
    window_kv = jnp.asarray(window_kv, jnp.float32)
    compressed_kv = jnp.asarray(compressed_kv, jnp.float32)
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, D], got {q.shape}")
    if window_kv.ndim != 2 or window_kv.shape[-1] != q.shape[-1]:
        raise ValueError(f"window_kv must be [W, {q.shape[-1]}], got {window_kv.shape}")
    if compressed_kv.ndim != 2 or compressed_kv.shape[-1] != q.shape[-1]:
        raise ValueError(f"compressed_kv must be [E, {q.shape[-1]}], got {compressed_kv.shape}")
    sink = jnp.asarray(attention_sink, jnp.float32)
    if sink.shape != (q.shape[1],):
        raise ValueError(f"attention_sink must be [H] = [{q.shape[1]}], got {sink.shape}")

    window_mask, compressed_mask = admissible_mask(
        query_positions=query_positions,
        query_request_ids=query_request_ids,
        valid_token_mask=valid_token_mask,
        window_positions=window_positions,
        window_request_ids=window_request_ids,
        compressed_entry_ids=compressed_entry_ids,
        compressed_request_ids=compressed_request_ids,
        window_size=window_size,
        ratio=ratio,
        selected_entries=selected_entries,
        selected_mask=selected_mask,
    )

    keys = jnp.concatenate((window_kv, compressed_kv), axis=0)  # [W+E, D]
    mask = jnp.concatenate((window_mask, compressed_mask), axis=1)  # [T, W+E]

    scores = jnp.einsum("thd,kd->thk", q, keys, preferred_element_type=jnp.float32)
    scores = scores * softmax_scale
    scores = jnp.where(mask[:, None, :], scores, _NEG_INF)

    # The sink participates in the shift so `exp(sink - shift)` cannot overflow when
    # the sink is the largest logit, which is exactly when it matters.
    shift = jnp.maximum(jnp.max(scores, axis=-1), sink[None, :])[..., None]
    probs = jnp.where(mask[:, None, :], jnp.exp(scores - shift), 0.0)
    denominator = jnp.sum(probs, axis=-1, keepdims=True) + jnp.exp(sink[None, :, None] - shift)
    out = jnp.einsum("thk,kd->thd", probs, keys, preferred_element_type=jnp.float32)
    out = out / denominator
    return jnp.where(jnp.asarray(valid_token_mask, bool)[:, None, None], out, 0.0)


def csa_fused_attention(
    q,
    window_kv,
    compressed_kv,
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    attention_sink,
    softmax_scale: float,
    window_size: int,
    ratio: int,
    selected_entries=None,
    selected_mask=None,
    interpret: bool = False,
):
    """`dsv4_dense_attention` computed by the fused flash-style kernel.

    Same admissibility as the dense path (`admissible_mask`, including the
    indexer's top-k membership); the kernel streams key tiles with an online
    softmax and skips tiles no query of the block may attend, instead of three
    full passes over a ``[T, H, W+E]`` f32 score tensor.
    """
    from sgl_jax.srt.kernels.dsv4.csa_flash_attention import csa_flash_attention

    q = jnp.asarray(q)
    window_mask, compressed_mask = admissible_mask(
        query_positions=query_positions,
        query_request_ids=query_request_ids,
        valid_token_mask=valid_token_mask,
        window_positions=window_positions,
        window_request_ids=window_request_ids,
        compressed_entry_ids=compressed_entry_ids,
        compressed_request_ids=compressed_request_ids,
        window_size=window_size,
        ratio=ratio,
        selected_entries=selected_entries,
        selected_mask=selected_mask,
    )
    keys = jnp.concatenate((jnp.asarray(window_kv), jnp.asarray(compressed_kv)), axis=0)
    mask = jnp.concatenate((window_mask, compressed_mask), axis=1)
    out = csa_flash_attention(
        q,
        keys,
        mask,
        jnp.asarray(attention_sink, jnp.float32),
        sm_scale=float(softmax_scale),
        block_q=_CSA_FUSED_BLOCK_Q,
        block_k=_CSA_FUSED_BLOCK_K,
        interpret=interpret,
    )
    return jnp.where(jnp.asarray(valid_token_mask, bool)[:, None, None], out, 0.0)


def update_window_kv(window_kv, new_kv, write_loc, valid_mask):
    """Scatter this step's KV into the sliding-window cache.

    Args:
      window_kv: ``[W, D]`` cache.
      new_kv: ``[T, D]`` rows for this step's tokens.
      write_loc: ``[T]`` destination row per token; padded tokens must carry a
        negative or out-of-range value.
      valid_mask: ``[T]``.

    Invalid rows are sent past the end of the cache and dropped rather than to row
    zero -- a duplicate write to row zero would race with whatever really lives
    there. Compressor state is untouched; that belongs to M2.2.
    """
    window_kv = jnp.asarray(window_kv)
    new_kv = jnp.asarray(new_kv, window_kv.dtype)
    loc = jnp.asarray(write_loc)
    keep = jnp.asarray(valid_mask, bool) & (loc >= 0) & (loc < window_kv.shape[0])
    if _PAGED_KV_WRITE and new_kv.shape[0] >= _PAGED_KV_WRITE_MIN_TOKENS and window_kv.ndim == 2:
        # Prefill chunks write thousands of page-contiguous rows; XLA's scatter does
        # them one row at a time (1.2 ms per layer for 8K on v7x).
        from sgl_jax.srt.kernels.dsv4.paged_row_write import paged_row_write

        return paged_row_write(window_kv, new_kv, loc, keep)
    loc = jnp.where(keep, loc, window_kv.shape[0])
    return window_kv.at[loc].set(new_kv, mode="drop")


def csa_sparse_attention(
    q,
    window_kv,
    compressed_kv,
    *,
    query_positions,
    query_request_ids,
    valid_token_mask,
    window_positions,
    window_request_ids,
    compressed_entry_ids,
    compressed_request_ids,
    attention_sink,
    softmax_scale: float,
    window_size: int,
    ratio: int,
    selected_entries,
    interpret: bool = False,
):
    """`dsv4_dense_attention` for the CSA path, attending only to the selected records.

    The dense path scores every one of the ``E`` gathered records and masks the
    non-selected ones, so its cost grows with the history; this path hands the
    ``kernels/dsa`` gathered-attention kernel one unit table made of the ``E``
    records followed by the ``W`` window rows, with per-query unit lists
    ``[selected (completeness/request filtered, else -1) | admissible window rows]``.
    Causality is enforced by that filtering, so the kernel's own positional bound
    is disabled (query position == last unit). The attention sink is applied
    afterwards from the kernel's log-sum-exp: ``out * L / (L + exp(sink))``.
    """
    from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import (
        sparse_mla_attention_qblock,
    )

    q = jnp.asarray(q)
    window_kv = jnp.asarray(window_kv)
    compressed_kv = jnp.asarray(compressed_kv)
    if q.ndim != 3:
        raise ValueError(f"q must be [T, H, D], got {q.shape}")
    T, H, D = q.shape
    W, E = window_kv.shape[0], compressed_kv.shape[0]
    if window_kv.shape[-1] != D or compressed_kv.shape[-1] != D:
        raise ValueError("window/compressed KV must share the query head dim")
    if ratio <= 0:
        raise ValueError("csa_sparse_attention requires a positive compression ratio")
    sink = jnp.asarray(attention_sink, jnp.float32)
    if sink.shape != (H,):
        raise ValueError(f"attention_sink must be [H] = [{H}], got {sink.shape}")

    qpos = jnp.asarray(query_positions, jnp.int32)[:, None]
    qreq = jnp.asarray(query_request_ids, jnp.int32)[:, None]
    valid = jnp.asarray(valid_token_mask, bool)[:, None]
    wpos = jnp.asarray(window_positions, jnp.int32)[None, :]
    wreq = jnp.asarray(window_request_ids, jnp.int32)[None, :]
    window_mask = valid & (wreq == qreq) & (wpos <= qpos) & (wpos > qpos - window_size)
    window_units = jnp.where(window_mask, E + jnp.arange(W, dtype=jnp.int32)[None, :], -1)

    sel = jnp.asarray(selected_entries, jnp.int32)
    entry_ids = jnp.asarray(compressed_entry_ids, jnp.int32)
    creq = jnp.asarray(compressed_request_ids, jnp.int32)
    safe = jnp.clip(sel, 0, max(E - 1, 0))
    complete = entry_ids[safe] < ((qpos + 1) // ratio)
    ok = (sel >= 0) & (sel < E) & valid & (creq[safe] == qreq) & complete
    selected_units = jnp.where(ok, sel, -1)

    indices = jnp.concatenate((selected_units, window_units), axis=1)  # [T, K+W]
    units = jnp.concatenate((compressed_kv, window_kv), axis=0)  # [E+W, D]
    kdt = units.dtype if units.dtype in (jnp.bfloat16, jnp.float32) else jnp.bfloat16
    units = units.astype(kdt)
    positions = jnp.full((1, T), E + W - 1, jnp.int32)  # kernel bound disabled
    out, lse = sparse_mla_attention_qblock(
        q.astype(kdt)[None],
        units[None],
        indices[None],
        positions,
        kv_lora_rank=D,
        read_block=1,
        query_block=_csa_query_block(T, H),
        sm_scale=float(softmax_scale),
        return_lse=True,
        interpret=interpret,
    )
    out, lse = out[0], lse[0]  # [T, H, D], [T, H]
    # L / (L + exp(sink)) == 1 / (1 + exp(sink - lse)); lse == -inf (nothing attended) -> 0.
    keep = 1.0 / (1.0 + jnp.exp(sink[None, :] - lse))
    out = out * keep[..., None]
    return jnp.where(valid[:, :, None], out, 0.0)
