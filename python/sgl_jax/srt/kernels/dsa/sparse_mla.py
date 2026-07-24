"""Sparse absorbed-MLA via the existing dense v2 Pallas kernel.

Two entry points:

- ``sparse_mla_page_level`` (preferred, decode T=1): re-indexes
  ``page_indices`` to only the pages touched by ``topk_indices`` (plus the
  new-token page appended last so the kernel's in-place kv write lands at
  the true slot), then runs the unmodified dense kernel. No kv gather, no
  temp buffer — the kernel DMAs ~k/page_size pages straight from the real
  cache instead of all L/page_size. Page-level granularity (attends the
  whole page, a superset of the token-level topk) so correctness is
  preserved.

- ``sparse_mla_via_dense`` (fallback / T>1 reference): gathers the k
  selected kv rows into a temp page buffer and runs the dense kernel over
  it with kv_len=k. Exact token-level sparse but pays a [T,k,D] HBM gather
  per layer, which dominates at T=1 (0.47ms vs the 0.1ms dense@2K it
  wraps).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from sgl_jax.srt.kernels.mla.v2.kernel import (
    get_kv_cache_shape,
    mla_ragged_paged_attention,
)


@functools.partial(jax.jit, static_argnames=("page_size", "pages_per_seq", "k_pages_max"))
def compute_topk_pages(
    topk_indices: jax.Array,
    *,
    page_size: int,
    pages_per_seq: int,
    k_pages_max: int,
) -> jax.Array:
    """Unique seq-local pages touched by ``topk_indices``, padded to k_pages_max.

    Returns i32[T, k_pages_max] with -1 padding. Computed once per full-indexer
    layer and shared across the IndexShare group so ``sparse_mla_page_level``
    skips its per-layer one_hot+top_k (~0.25ms/layer × 75 layers).
    """
    valid = topk_indices >= 0
    page_local = jnp.where(valid, topk_indices // page_size, pages_per_seq)
    page_hits = jax.nn.one_hot(page_local, pages_per_seq, dtype=jnp.int32)
    page_mask = jnp.any(page_hits, axis=1)  # [T, P]
    n_hit = jnp.sum(page_mask, axis=-1)
    k_eff = min(k_pages_max, pages_per_seq)
    _, hit_pages = jax.lax.top_k(page_mask.astype(jnp.int32), k_eff)
    hit_pages = jnp.pad(hit_pages, ((0, 0), (0, k_pages_max - k_eff)))
    hit_valid = jnp.arange(k_pages_max)[None, :] < n_hit[:, None]
    return jnp.where(hit_valid, hit_pages, -1)


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "page_size",
        "pages_per_seq",
        "kv_lora_rank",
        "k_pages_max",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def sparse_mla_page_level(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    topk_pages: jax.Array | None = None,
    *,
    sm_scale: float,
    page_size: int,
    pages_per_seq: int,
    kv_lora_rank: int,
    k_pages_max: int = 64,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Page-level sparse MLA: dense kernel over only the topk-touched pages.

    Decode-only (each seq contributes exactly one query token). Builds a
    per-seq page list of length ``k_pages_max``:
      [unique pages of topk_indices ... pad ... new-token's page]
    and hands that as ``page_indices`` to the unmodified dense kernel with
    ``kv_len`` set so the kernel's in-place new-kv write lands on the true
    slot (last page × page_size + new_off + 1 - 1).

    Returns ``(o_latent, updated_cache_kv)`` — the cache is written in place
    by the dense kernel, so no separate ``_scatter_paged`` is needed.
    """
    T = ql_nope.shape[0]
    S = kv_lens.shape[0]

    # Per-token seq id (decode: token t belongs to seq t after cu_q_lens=[0..T]).
    seq_id = jnp.searchsorted(cu_q_lens[1:], jnp.arange(T), side="right")
    seq_id = jnp.clip(seq_id, 0, S - 1)
    tok_valid = kv_lens[seq_id] > 0  # [T] — padding seqs (bs padding) have kv_len==0
    # page_indices is packed per-seq at cu_kv_lens[i]//page_size (variable stride
    # via cumsum(aligned_lens)), NOT seq_id*pages_per_seq.
    seq_page_start = cu_kv_lens[seq_id] // page_size  # [T]

    valid = topk_indices >= 0
    safe_idx = jnp.where(valid, topk_indices, 0)
    page_local = safe_idx // page_size  # [T, k] seq-local page ids

    # New token's seq-local slot = kv_len-1; its page/offset. Clamp for padding
    # seqs (kv_len==0) so the -1 doesn't index page_indices[-1] and corrupt the
    # previous real seq's tail page via the kernel's in-place kv write.
    new_abs = jnp.maximum(kv_lens[seq_id] - 1, 0)  # [T]
    new_page_local = new_abs // page_size
    new_off = new_abs % page_size

    if topk_pages is not None:
        # Precomputed by the indexer (shared across the 4-layer IndexShare
        # group), so this branch skips the ~0.25ms one_hot+top_k per layer.
        # Compact valid entries to the front via sort — new_page may sit in
        # the middle of topk_pages (compute_topk_pages doesn't know it), and
        # the sp_local layout below assumes hit_pages[:n_hit_c] are all valid.
        raw = topk_pages[..., : k_pages_max - 1]
        hit_is_valid = (raw >= 0) & (raw != new_page_local[:, None])
        n_hit_c = jnp.minimum(jnp.sum(hit_is_valid, -1), k_pages_max - 1)
        hit_pages = jnp.sort(jnp.where(hit_is_valid, raw, jnp.iinfo(jnp.int32).max), axis=-1)
    else:
        # Unique pages via one-hot OR over the k topk positions (page count is
        # bounded by pages_per_seq which is static). Keeps everything jit-static.
        page_hits = jax.nn.one_hot(page_local, pages_per_seq, dtype=jnp.int32)
        page_hits = jnp.where(valid[..., None], page_hits, 0)
        page_mask = jnp.any(page_hits, axis=1)  # [T, P] bool
        page_mask = page_mask & (jnp.arange(pages_per_seq)[None, :] != new_page_local[:, None])
        n_hit = jnp.sum(page_mask, axis=-1)
        _, hit_pages = jax.lax.top_k(page_mask.astype(jnp.int32), k_pages_max - 1)
        n_hit_c = jnp.minimum(n_hit, k_pages_max - 1)
    n_used = n_hit_c + 1  # +1 for the new-token page

    # Layout: [hit_pages[0..n_hit_c-1], new_page, pad...] so the kernel's
    # kv_len mask (k_span < sp_kv_len) drops the pad tail while keeping the
    # new-token page inside. Padding points at the sentinel last cache page.
    col = jnp.arange(k_pages_max)[None, :]  # [1, kp]
    sp_local = jnp.where(
        col < n_hit_c[:, None],
        jnp.pad(hit_pages, ((0, 0), (0, 1))),
        jnp.where(col == n_hit_c[:, None], new_page_local[:, None], 0),
    )
    sp_phys = jnp.where(
        (col < n_used[:, None]) & tok_valid[:, None],
        page_indices[seq_page_start[:, None] + sp_local],
        cache_kv.shape[0] - 1,
    )

    # Metadata for the dense kernel: T "sequences", each kv_len positions.
    sp_kv_len = (n_hit_c * page_size + new_off + 1).astype(jnp.int32)  # [T]
    sp_page_indices = sp_phys.reshape(-1)  # [T * k_pages_max]
    sp_cu_q = jnp.arange(T + 1, dtype=jnp.int32)
    sp_cu_kv = jnp.arange(T + 1, dtype=jnp.int32) * (k_pages_max * page_size)
    sp_dist = jnp.array([T, T, T], dtype=jnp.int32)

    o, cache_out = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        sp_kv_len,
        sp_page_indices,
        sp_cu_q,
        sp_cu_kv,
        sp_dist,
        sm_scale=sm_scale,
        sliding_window=None,
        soft_cap=None,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return o[..., :kv_lora_rank], cache_out


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "page_size",
        "pages_per_seq",
        "kv_lora_rank",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
)
def sparse_mla_via_dense(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    topk_indices: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    cu_kv_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float,
    page_size: int,
    pages_per_seq: int,
    kv_lora_rank: int,
    num_kv_pages_per_block=None,
    num_queries_per_block=None,
    vmem_limit_bytes: int | None = None,
) -> jax.Array:
    """Absorbed-MLA over per-query top-k, delegating to the dense v2 kernel.

    Args mirror ``mla_ragged_paged_attention`` plus:
      topk_indices: i32[T, k]  per-query KV positions (seq-relative, -1 = pad)
      pages_per_seq: static stride into ``page_indices``

    Returns:
      o_latent: f[T, H, kv_lora_rank]
    """
    T, H, _ = ql_nope.shape
    k = topk_indices.shape[1]
    _, ps_over_pack, pack, kv_dim = cache_kv.shape
    assert ps_over_pack * pack == page_size
    S = kv_lens.shape[0]

    # ── 1. resolve topk seq-relative positions → flat physical slot ──────
    seq_id = jnp.searchsorted(cu_q_lens[1:], jnp.arange(T), side="right")
    seq_id = jnp.clip(seq_id, 0, S - 1)
    seq_page_start = cu_kv_lens[seq_id] // page_size
    valid = topk_indices >= 0
    safe_idx = jnp.where(valid, topk_indices, 0)
    page_local = safe_idx // page_size
    off = safe_idx % page_size
    phys_page = page_indices[seq_page_start[:, None] + page_local]
    slot = phys_page * page_size + off  # [T, k]

    # ── 2. gather kv rows via 1-D index (2-3× faster than 3-D fancy index
    # on TPU) → contiguous [T, k, kv_dim] → temp pages ────────────────────
    cache_flat = cache_kv.reshape(-1, kv_dim)
    kv_gathered = jnp.take(cache_flat, slot.reshape(-1), axis=0).reshape(T, k, kv_dim)
    kv_gathered = jnp.where(valid[..., None], kv_gathered, 0)

    n_tmp_pages = (k + page_size - 1) // page_size
    pad = n_tmp_pages * page_size - k
    kv_padded = jnp.pad(kv_gathered, ((0, 0), (0, pad), (0, 0)))
    tmp_shape = get_kv_cache_shape(T * n_tmp_pages + 1, page_size, kv_dim, cache_kv.dtype)
    tmp_cache = kv_padded.reshape(T * n_tmp_pages, ps_over_pack, pack, kv_dim)
    tmp_cache = jnp.concatenate(
        [tmp_cache, jnp.zeros((tmp_shape[0] - T * n_tmp_pages, *tmp_shape[1:]), cache_kv.dtype)],
        axis=0,
    )

    # ── 3. build metadata: each token is its own "sequence" of kv_len=k ──
    n_valid = jnp.sum(valid, axis=-1).astype(jnp.int32)
    tmp_kv_lens = n_valid  # i32[T]
    tmp_page_indices = jnp.arange(T * n_tmp_pages, dtype=jnp.int32)
    tmp_cu_q = jnp.arange(T + 1, dtype=jnp.int32)
    tmp_cu_kv = jnp.arange(T + 1, dtype=jnp.int32) * (n_tmp_pages * page_size)
    tmp_dist = jnp.array([T, T, T], dtype=jnp.int32)

    # ── 4. run dense kernel over the gathered buffer (O(k) not O(L)) ─────
    o_latent, _ = mla_ragged_paged_attention(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        tmp_cache,
        tmp_kv_lens,
        tmp_page_indices,
        tmp_cu_q,
        tmp_cu_kv,
        tmp_dist,
        sm_scale=sm_scale,
        sliding_window=None,
        soft_cap=None,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    return o_latent[..., :kv_lora_rank]
