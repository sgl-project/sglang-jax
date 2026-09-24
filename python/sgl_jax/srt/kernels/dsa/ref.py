"""JNP reference implementations for DSA sparse attention.

These are correctness oracles for the Pallas kernels in this directory, and
also serve as the Phase-A e2e path (``--dsa-use-pallas=false``). All functions
are jit-compatible with static shapes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

_NEG_INF = float("-inf")


def build_index_share_map(
    indexer_types: list[str] | None,
    skip_offset: int,
    num_layers: int,
) -> tuple[dict[int, int], dict[int, int], int]:
    """Static IndexShare layer→slot mapping.

    Returns:
      full_slot: layer_id → slot_id for layers with indexer_type == "full"
      src_slot:  layer_id → slot_id whose topk this layer consumes
                 (its own slot if full, nearest preceding full's slot if shared)
      num_full:  number of full layers (== number of indexer_key buffers)
    """
    if indexer_types is None:
        indexer_types = ["full"] * num_layers
    assert len(indexer_types) == num_layers

    full_slot: dict[int, int] = {}
    src_slot: dict[int, int] = {}
    last_slot = -1
    for layer_id, itype in enumerate(indexer_types):
        if itype == "full":
            last_slot = len(full_slot)
            full_slot[layer_id] = last_slot
            src_slot[layer_id] = last_slot
        elif itype == "shared":
            assert last_slot >= 0, f"layer {layer_id} is shared but no preceding full"
            src_slot[layer_id] = last_slot
        else:
            raise ValueError(f"unknown indexer_type {itype!r} at layer {layer_id}")
    return full_slot, src_slot, len(full_slot)


def prefill_score_temp_bytes(num_full_layers: int, chunk_tokens: int, context_len: int) -> int:
    """Estimated HLO temporaries for the jnp reference indexer scorer.

    The reference scorer materializes one f32 ``[chunk_tokens, context_len]``
    token-score buffer per full layer during sparse prefill, and XLA keeps
    them alive together. Observed within ~2% of the RESOURCE_EXHAUSTED report
    at 135k context (22 full layers x chunk 8192 -> ~98.5G).
    """
    return num_full_layers * chunk_tokens * context_len * 4


def suggest_chunked_prefill_size(
    num_full_layers: int, context_len: int, budget_bytes: int
) -> int | None:
    """Largest power-of-two ``chunked_prefill_size`` (capped at the context
    length) whose reference-scorer temporaries fit ``budget_bytes``. Returns
    ``None`` when even the smallest chunk (256) does not fit."""
    chunk, best = 256, None
    while (
        chunk <= context_len
        and prefill_score_temp_bytes(num_full_layers, chunk, context_len) <= budget_bytes
    ):
        best = chunk
        chunk *= 2
    return best


@functools.partial(jax.jit, static_argnames=("k", "pages_per_seq", "one_token_per_seq"))
def streamindex_topk_ref(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    one_token_per_seq: bool = False,
) -> jax.Array:
    """Reference lightning-indexer top-k.

    ``one_token_per_seq=True`` (decode: T == num_seqs, token i belongs to
    seq i) scores only each iteration's own query row — O(S * max_kv) total
    instead of O(S * T * max_kv); see ``streamindex_page_topk_ref``.

    Scores are ``sum_h relu(q_h · k) * w_h`` per DSA semantics. Because ReLU
    sits before the head sum, the naive ``[T, H, max_kv]`` intermediate OOMs
    at chunked-prefill sizes; instead we accumulate over the H heads (H=32)
    with a ``fori_loop`` so peak memory is a single ``[T, max_kv]`` buffer,
    then take ``top_k`` once via XLA (Pallas TPU has no ``top_k`` lowering).

    Args:
      q:            f[T, H, D]  indexer query heads
      weights:      f[T, H]     per-head mixing weights
      cache_kv:     f[P, page_size, D]  paged indexer keys
      seq_lens:     i32[S]      kv length per sequence
      page_indices: i32[N_pages]  packed; seq i's pages start at cu_kv_lens[i]//page_size
      cu_q_lens:    i32[S+1]
      cu_kv_lens:   i32[S+1]    cumsum of page-aligned kv lens (page_indices stride)
      distribution: i32[3]      (decode_end, prefill_end, num_seqs)
      k:            top-k budget
      pages_per_seq: static, page_indices stride

    Returns:
      i32[T, k]  top-k kv positions per query token; -1 for padding.
    """
    T, H, D = q.shape
    page_size = cache_kv.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]

    w = weights.astype(jnp.float32)
    out = jnp.full((T, k), -1, dtype=jnp.int32)

    if one_token_per_seq:

        def body_decode(seq_id, out):
            kv_len = seq_lens[seq_id]
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            keys = cache_kv[seq_pages].reshape(max_kv, D)
            q_i = jax.lax.dynamic_slice_in_dim(q, seq_id, 1, axis=0)  # [1, H, D]
            w_i = jax.lax.dynamic_slice_in_dim(w, seq_id, 1, axis=0)  # [1, H]
            s = jnp.einsum("thd,kd->thk", q_i, keys, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", w_i, jax.nn.relu(s))  # [1, max_kv]
            mask = (jnp.arange(max_kv) < kv_len)[None, :]
            scores = jnp.where(mask, scores, _NEG_INF)
            cand_v, cand_i = jax.lax.approx_max_k(
                scores, k, recall_target=0.70, aggregate_to_topk=False
            )
            vals, sel = jax.lax.top_k(cand_v, k)
            idx = jnp.take_along_axis(cand_i, sel, axis=-1)
            n_valid = mask.sum(-1, keepdims=True)
            idx = jnp.where((jnp.arange(k)[None, :] < n_valid) & (vals > _NEG_INF), idx, -1)
            return jax.lax.dynamic_update_slice_in_dim(out, idx, seq_id, axis=0)

        return jax.lax.fori_loop(0, num_seqs, body_decode, out)

    def body(seq_id, out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        kv_len = seq_lens[seq_id]
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        keys = cache_kv[seq_pages].reshape(max_kv, D)

        q_pos = jnp.arange(T)
        kv_pos = jnp.arange(max_kv)
        in_seq_q = (q_pos >= q_start) & (q_pos < q_end)
        abs_q = kv_len - (q_end - q_start) + (q_pos - q_start)
        mask = in_seq_q[:, None] & (kv_pos[None, :] < kv_len) & (kv_pos[None, :] <= abs_q[:, None])

        if T * H * max_kv <= 1 << 26:
            s = jnp.einsum("thd,kd->thk", q, keys, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", w, jax.nn.relu(s))
        else:

            def h_step(h, acc):
                q_h = jax.lax.dynamic_index_in_dim(q, h, axis=1, keepdims=False)
                w_h = jax.lax.dynamic_index_in_dim(w, h, axis=1, keepdims=True)
                s_h = jnp.einsum("td,kd->tk", q_h, keys, preferred_element_type=jnp.float32)
                return acc + jax.nn.relu(s_h) * w_h

            scores = jax.lax.fori_loop(0, H, h_step, jnp.zeros((T, max_kv), jnp.float32))
        scores = jnp.where(mask, scores, _NEG_INF)
        # Two-stage approx topk: aggregate_to_topk=False skips the internal
        # sort (returns L>=k unsorted candidates) which is the dominant cost;
        # a second top_k over the small candidate set is cheap. On v7x @128K:
        # 0.35ms vs 0.57ms (agg=True) vs 1.35ms (exact). recall ~95%.
        cand_v, cand_i = jax.lax.approx_max_k(
            scores, k, recall_target=0.70, aggregate_to_topk=False
        )
        vals, sel = jax.lax.top_k(cand_v, k)
        idx = jnp.take_along_axis(cand_i, sel, axis=-1)
        n_valid = mask.sum(-1, keepdims=True)
        # approx_max_k with recall<1 may return -inf-scored (out-of-range)
        # candidates inside the top n_valid slots on TPU; guard both by rank
        # and by value so downstream sparse_mla never gathers stale kv slots.
        idx = jnp.where((jnp.arange(k)[None, :] < n_valid) & (vals > _NEG_INF), idx, -1)
        return jnp.where(in_seq_q[:, None], idx, out)

    return jax.lax.fori_loop(0, num_seqs, body, out)


@functools.partial(jax.jit, static_argnames=("k_pages", "pages_per_seq", "one_token_per_seq"))
def streamindex_page_topk_ref(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k_pages: int,
    pages_per_seq: int,
    one_token_per_seq: bool = False,
) -> jax.Array:
    """Page-level lightning-indexer: max-pool token scores within each page,
    then top-k over pages_per_seq page scores.

    ``one_token_per_seq=True`` (decode batches: T == num_seqs, token i belongs
    to seq i) switches to a fast path where each loop iteration scores only
    its own single query row — O(S * max_kv) total instead of the general
    path's O(S * T * max_kv), which collapses under multi-request decode
    (each iteration would score the full [T, max_kv] and mask all but one row).

    Vs the token-level path (``streamindex_topk_ref`` + page union, which
    saturates k_pages_max at long context): the page budget is exact, the
    top_k runs over [T, pages_per_seq] instead of [T, max_kv] (page_size×
    smaller), and sparse-MLA cost becomes a true O(k_pages) flat.

    Returns:
      i32[T, k_pages]  seq-local page ids per query token; -1 for padding.
    """
    T, H, D = q.shape
    page_size = cache_kv.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]

    w = weights.astype(jnp.float32)
    out = jnp.full((T, k_pages), -1, dtype=jnp.int32)

    if one_token_per_seq:

        def body_decode(seq_id, out):
            kv_len = seq_lens[seq_id]
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            keys = cache_kv[seq_pages].reshape(max_kv, D)
            q_i = jax.lax.dynamic_slice_in_dim(q, seq_id, 1, axis=0)  # [1, H, D]
            w_i = jax.lax.dynamic_slice_in_dim(w, seq_id, 1, axis=0)  # [1, H]
            s = jnp.einsum("thd,kd->thk", q_i, keys, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", w_i, jax.nn.relu(s))  # [1, max_kv]
            # decode: query is the last token (abs pos kv_len-1), so the causal
            # bound coincides with the kv_len bound
            mask = (jnp.arange(max_kv) < kv_len)[None, :]
            scores = jnp.where(mask, scores, _NEG_INF)
            page_scores = scores.reshape(1, pages_per_seq, page_size).max(axis=-1)
            vals, pidx = jax.lax.top_k(page_scores, k_pages)
            pidx = jnp.where(vals > _NEG_INF, pidx, -1)
            return jax.lax.dynamic_update_slice_in_dim(out, pidx, seq_id, axis=0)

        return jax.lax.fori_loop(0, num_seqs, body_decode, out)

    def body(seq_id, out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        kv_len = seq_lens[seq_id]
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        keys = cache_kv[seq_pages].reshape(max_kv, D)

        q_pos = jnp.arange(T)
        kv_pos = jnp.arange(max_kv)
        in_seq_q = (q_pos >= q_start) & (q_pos < q_end)
        abs_q = kv_len - (q_end - q_start) + (q_pos - q_start)
        mask = in_seq_q[:, None] & (kv_pos[None, :] < kv_len) & (kv_pos[None, :] <= abs_q[:, None])

        if T * H * max_kv <= 1 << 26:
            s = jnp.einsum("thd,kd->thk", q, keys, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", w, jax.nn.relu(s))
        else:

            def h_step(h, acc):
                q_h = jax.lax.dynamic_index_in_dim(q, h, axis=1, keepdims=False)
                w_h = jax.lax.dynamic_index_in_dim(w, h, axis=1, keepdims=True)
                s_h = jnp.einsum("td,kd->tk", q_h, keys, preferred_element_type=jnp.float32)
                return acc + jax.nn.relu(s_h) * w_h

            scores = jax.lax.fori_loop(0, H, h_step, jnp.zeros((T, max_kv), jnp.float32))
        scores = jnp.where(mask, scores, _NEG_INF)
        page_scores = scores.reshape(T, pages_per_seq, page_size).max(axis=-1)
        vals, pidx = jax.lax.top_k(page_scores, k_pages)
        pidx = jnp.where(vals > _NEG_INF, pidx, -1)
        return jnp.where(in_seq_q[:, None], pidx, out)

    return jax.lax.fori_loop(0, num_seqs, body, out)


@functools.partial(
    jax.jit,
    static_argnames=("k_pages", "pages_per_seq", "q_group", "head_chunk", "unroll_heads"),
)
def streamindex_page_topk_ref_grouped(
    q: jax.Array,
    weights: jax.Array,
    cache_kv: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    *,
    k_pages: int,
    pages_per_seq: int,
    q_group: int,
    head_chunk: int = 8,
    unroll_heads: bool = True,
) -> jax.Array:
    """Page-topk for batches of S requests x ``q_group`` query tokens each
    (spec verify: every valid request contributes exactly G draft tokens, so
    ``cu_q_lens[s+1] - cu_q_lens[s] == G``), vectorized over requests.

    Same selection semantics as ``streamindex_page_topk_ref`` (general causal
    path) but without the per-sequence ``fori_loop``: one flat gather of every
    request's ``pages_per_seq`` pages, one batched ``[S, G, H, max_kv]`` score
    accumulated over head chunks, one batched page max-pool and one top_k.
    Draft token i of a request sits at absolute position ``kv_len - G + i`` and
    only sees keys at positions ``<= kv_len - G + i`` (intra-request causal).

    Rows of padded requests (``seq_lens == 0``) and rows past ``cu_q_lens[-1]``
    are -1. Requests whose q_len != ``q_group`` are treated as padding.

    Returns:
      i32[T, k_pages]  seq-local page ids per query token; -1 for padding.
    """
    T, H, D = q.shape
    page_size = cache_kv.shape[1]
    max_kv = pages_per_seq * page_size
    S = seq_lens.shape[0]
    G = q_group

    q_len = cu_q_lens[1 : S + 1] - cu_q_lens[:S]
    valid = (seq_lens > 0) & (q_len == G)  # [S]
    rows = jnp.minimum(cu_q_lens[:S, None] + jnp.arange(G)[None, :], T - 1)  # [S, G]

    starts = cu_kv_lens[:S] // page_size
    gidx = starts[:, None] + jnp.arange(pages_per_seq)[None, :]
    gidx = jnp.minimum(gidx, page_indices.shape[0] - 1)
    pages = page_indices[gidx]  # [S, pps]
    keys = cache_kv[pages.reshape(-1)].reshape(S, max_kv, D)

    qg = q[rows]  # [S, G, H, D]
    wg = weights[rows].astype(jnp.float32)  # [S, G, H]
    nchunks = -(-H // head_chunk)
    h_pad = nchunks * head_chunk - H
    q_pad = jnp.pad(qg, ((0, 0), (0, 0), (0, h_pad), (0, 0)))
    w_pad = jnp.pad(wg, ((0, 0), (0, 0), (0, h_pad)))

    def h_step(c, acc):
        h0 = c * head_chunk
        q_c = jax.lax.dynamic_slice_in_dim(q_pad, h0, head_chunk, axis=2)  # [S, G, hc, D]
        w_c = jax.lax.dynamic_slice_in_dim(w_pad, h0, head_chunk, axis=2)  # [S, G, hc]
        s_c = jnp.einsum("sghd,skd->sghk", q_c, keys, preferred_element_type=jnp.float32)
        return acc + jnp.einsum("sgh,sghk->sgk", w_c, jax.nn.relu(s_c))

    zeros = jnp.zeros((S, G, max_kv), jnp.float32)
    if unroll_heads:
        scores = zeros
        for c in range(nchunks):
            scores = h_step(c, scores)
    else:
        scores = jax.lax.fori_loop(0, nchunks, h_step, zeros)

    kv_pos = jnp.arange(max_kv)[None, None, :]
    abs_q = (seq_lens[:, None] - G + jnp.arange(G)[None, :])[:, :, None]  # [S, G, 1]
    mask = valid[:, None, None] & (kv_pos < seq_lens[:, None, None]) & (kv_pos <= abs_q)
    scores = jnp.where(mask, scores, _NEG_INF)
    page_scores = scores.reshape(S * G, pages_per_seq, page_size).max(axis=-1)
    vals, pidx = jax.lax.top_k(page_scores, k_pages)
    pidx = jnp.where(vals > _NEG_INF, pidx, -1)  # [S*G, k]

    # Padded requests all alias row cu_q[-1] (== T when there are no pad rows);
    # route them to a sentinel row T so the scatter never has two writers.
    row_valid = jnp.broadcast_to(valid[:, None], (S, G)).reshape(-1)
    pidx = jnp.where(row_valid[:, None], pidx, -1)  # invalid rows write a uniform -1
    dest = jnp.where(row_valid, rows.reshape(-1), T)
    out = jnp.full((T + 1, k_pages), -1, dtype=jnp.int32)
    return out.at[dest].set(pidx)[:T]


@functools.partial(jax.jit, static_argnames=("sm_scale", "pages_per_seq", "v_dim"))
def sparse_mla_ref(
    q: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    pages_per_seq: int,
    v_dim: int,
) -> jax.Array:
    """Reference sparse absorbed-MLA over per-query top-k positions.

    Args:
      q:            f[T, H, D_qk]  packed [nope|rope] query in latent space
      cache_kv:     f[P, page_size, D_kv]  packed [c_kv|k_rope] latent cache
      kv_lens:      i32[S]
      topk_indices: i32[T, k]  kv positions per query, -1 = ignore
      page_indices: i32[S * pages_per_seq]
      cu_q_lens:    i32[S+1]
      distribution: i32[3]
      sm_scale:     softmax scale
      pages_per_seq: static
      v_dim:        latent value dim (kv_lora_rank), <= D_kv

    Returns:
      f[T, H, v_dim]
    """
    T, H, _ = q.shape
    _, page_size, Dkv = cache_kv.shape
    num_seqs = kv_lens.shape[0]
    max_kv = pages_per_seq * page_size

    out = jnp.zeros((T, H, v_dim), dtype=q.dtype)

    def body(seq_id, out):
        q_start = cu_q_lens[seq_id]
        q_end = cu_q_lens[seq_id + 1]
        seq_pages = jax.lax.dynamic_slice_in_dim(
            page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
        )
        kv_flat = cache_kv[seq_pages].reshape(max_kv, Dkv)

        q_pos = jnp.arange(T)
        in_seq = (q_pos >= q_start) & (q_pos < q_end)

        kv_len = kv_lens[seq_id]
        idx = jnp.where(topk_indices >= 0, topk_indices, 0)
        valid = (topk_indices >= 0) & (topk_indices < kv_len) & in_seq[:, None]
        kv_sel = kv_flat[idx]  # [T, k, Dkv]

        logits = jnp.einsum("thd,tkd->thk", q.astype(jnp.float32), kv_sel.astype(jnp.float32))
        logits = logits * sm_scale
        logits = jnp.where(valid[:, None, :], logits, _NEG_INF)
        p = jax.nn.softmax(logits, axis=-1)
        p = jnp.where(valid[:, None, :], p, 0.0)

        v_sel = kv_sel[..., :v_dim]
        o = jnp.einsum("thk,tkd->thd", p, v_sel.astype(jnp.float32)).astype(q.dtype)
        return jnp.where(in_seq[:, None, None], o, out)

    return jax.lax.fori_loop(0, num_seqs, body, out)
