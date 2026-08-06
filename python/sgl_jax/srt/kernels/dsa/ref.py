"""JNP reference implementations for DSA sparse attention.

These are correctness oracles for the Pallas kernels in this directory, and
also serve as the Phase-A e2e path (``--dsa-use-pallas=false``). All functions
are jit-compatible with static shapes.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.dsa.topk import select_indexer_topk

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


@functools.partial(
    jax.jit,
    static_argnames=("k", "pages_per_seq", "one_token_per_seq", "topk_impl"),
)
def score_and_select_index_tokens(
    q_idx: jax.Array,
    idx_weights: jax.Array,
    index_key_cache: jax.Array,
    seq_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    k: int,
    pages_per_seq: int,
    one_token_per_seq: bool = False,
    topk_impl: str,
) -> jax.Array:
    """Score cached index keys and select the top-k token positions.

    ``one_token_per_seq=True`` (decode: T == num_seqs, token i belongs to
    seq i) scores only each iteration's own query row — O(S * max_kv) total
    instead of O(S * T * max_kv); see ``streamindex_page_topk_ref``.

    Scores are ``sum_h relu(q_h · k) * w_h`` per DSA semantics. Because ReLU
    sits before the head sum, the naive ``[T, H, max_kv]`` intermediate OOMs
    at chunked-prefill sizes; instead we accumulate over the H heads (H=32)
    with a ``fori_loop`` so peak memory is a single ``[T, max_kv]`` buffer,
    then hand that score buffer to the configured top-k selection backend.

    Args:
      q_idx:         f[T, H, D]  indexer query heads
      idx_weights:   f[T, H]     per-head mixing weights
      index_key_cache: f[P, page_size, D]  paged indexer keys
      seq_lens:     i32[S]      kv length per sequence
      page_indices: i32[N_pages]  packed; seq i's pages start at cu_kv_lens[i]//page_size
      cu_q_lens:    i32[S+1]
      cu_kv_lens:   i32[S+1]    cumsum of page-aligned kv lens (page_indices stride)
      distribution: i32[3]      (decode_end, prefill_end, num_seqs)
      k:            top-k budget
      pages_per_seq: static, maximum pages materialized per sequence
      topk_impl: selection backend: approximate XLA, exact XLA, or exact
        SparseCore radix selection (``approx``, ``exact_lax``, or ``radix``).

    Returns:
      i32[T, k]  top-k kv positions per query token; -1 for padding.
    """
    T, H, D = q_idx.shape
    page_size = index_key_cache.shape[1]
    max_kv = pages_per_seq * page_size
    num_seqs = seq_lens.shape[0]

    idx_weights_f32 = idx_weights.astype(jnp.float32)
    out = jnp.full((T, k), -1, dtype=jnp.int32)
    if one_token_per_seq:

        def body_decode(seq_id, out):
            kv_len = seq_lens[seq_id]
            seq_pages = jax.lax.dynamic_slice_in_dim(
                page_indices, cu_kv_lens[seq_id] // page_size, pages_per_seq
            )
            seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)
            q_idx_i = jax.lax.dynamic_slice_in_dim(q_idx, seq_id, 1, axis=0)  # [1, H, D]
            idx_weights_i = jax.lax.dynamic_slice_in_dim(
                idx_weights_f32, seq_id, 1, axis=0
            )  # [1, H]
            s = jnp.einsum("thd,kd->thk", q_idx_i, seq_k_idx, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", idx_weights_i, jax.nn.relu(s))  # [1, max_kv]
            mask = (jnp.arange(max_kv) < kv_len)[None, :]
            scores = jnp.where(mask, scores, _NEG_INF)
            vals, idx = select_indexer_topk(scores, k=k, implementation=topk_impl)
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
        seq_k_idx = index_key_cache[seq_pages].reshape(max_kv, D)

        kv_pos = jnp.arange(max_kv)

        q_pos = jnp.arange(T)
        in_seq_q = (q_pos >= q_start) & (q_pos < q_end)
        abs_q = kv_len - (q_end - q_start) + (q_pos - q_start)
        mask = in_seq_q[:, None] & (kv_pos[None, :] < kv_len) & (kv_pos[None, :] <= abs_q[:, None])

        if T * H * max_kv <= 1 << 26:
            s = jnp.einsum("thd,kd->thk", q_idx, seq_k_idx, preferred_element_type=jnp.float32)
            scores = jnp.einsum("th,thk->tk", idx_weights_f32, jax.nn.relu(s))
        else:

            def h_step(h, acc):
                q_idx_h = jax.lax.dynamic_index_in_dim(q_idx, h, axis=1, keepdims=False)
                idx_weights_h = jax.lax.dynamic_index_in_dim(
                    idx_weights_f32, h, axis=1, keepdims=True
                )
                s_h = jnp.einsum(
                    "td,kd->tk", q_idx_h, seq_k_idx, preferred_element_type=jnp.float32
                )
                return acc + jax.nn.relu(s_h) * idx_weights_h

            scores = jax.lax.fori_loop(0, H, h_step, jnp.zeros((T, max_kv), jnp.float32))
        scores = jnp.where(mask, scores, _NEG_INF)
        # Score construction is shared by all selection implementations. The
        # adapter owns approximate candidate refinement and exact radix ordering.
        vals, idx = select_indexer_topk(scores, k=k, implementation=topk_impl)
        n_valid = mask.sum(-1, keepdims=True)
        # Approximate selection and radix input padding can return -inf-scored
        # candidates. Guard both by rank and by value so downstream sparse MLA
        # never gathers stale cache slots.
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

    Vs the token-level path (``score_and_select_index_tokens`` + page union, which
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
