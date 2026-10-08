"""CSA paged indexer dispatch and shared completed-group visibility rules.

The native JAX CPU/reference implementation lives in ``dsv4.ref.indexer``.
TPU scoring uses the shared DSA streamindex kernel with completed-group masking.
"""

from __future__ import annotations

import os

import jax
import jax.numpy as jnp

__all__ = [
    "INDEXER_BACKEND_ENV",
    "INVALID_ENTRY",
    "csa_indexer_topk_kernel",
    "kernel_read_layout",
    "resolve_indexer_backend",
    "visible_entries_for_query",
]

# Which implementation scores and selects compressed entries during prefill:
#   "reference": the native-JAX path in ref.indexer (materializes [T, E] scores, lax.top_k);
#   "kernel":    kernels/dsa/streamindex_topk (Pallas scoring straight from the paged
#                indexer cache, completed-groups mask, exact SparseCore selection);
#   "auto":      "kernel" on TPU, "reference" elsewhere.
INDEXER_BACKEND_ENV = "DSV4_INDEXER_BACKEND"
# Kernel block sizes. The compressed page holds page_size // 4 = 32 entries, and the
# kernel needs whole 128-entry KV blocks, so kv pages per block must be a multiple of 4.
_KERNEL_KV_PAGES_PER_BLOCK = int(os.environ.get("DSV4_INDEXER_KV_PAGES_PER_BLOCK", "64"))
# ``DSV4_INDEXER_QUERIES_PER_BLOCK``: query rows per grid step of the prefill /
# mixed indexer kernel (decode stays at 1). 64 is the decode-era default; the
# 8K prefill scores block is [64, N] per step.
_KERNEL_QUERIES_PER_BLOCK = (
    1,
    int(os.environ.get("DSV4_INDEXER_QUERIES_PER_BLOCK", "64")),
    int(os.environ.get("DSV4_INDEXER_QUERIES_PER_BLOCK", "64")),
)

# Packed at the end of each top-k row. -1 rather than an out-of-range positive
# value because that is what kernels/dsa already emits and what the downstream
# sparse gather in M2.4 will check.
INVALID_ENTRY = -1


def visible_entries_for_query(query_positions, ratio: int):
    """Compressed entries a query at each position may select: ``(position+1)//ratio``.

    An entry is selectable only once its group is **complete**, i.e. the token at
    ``(entry + 1) * ratio - 1`` has been consumed. Returns the count, so entry
    ids ``0 .. count-1`` are legal.

    This is what makes a selection stable: it depends only on tokens at or before
    the query, so later tokens in the same chunk cannot change it.

    Same rule as `dsv4.metadata.visible_groups_for_positions` and as the HCA path
    in `kernels/hca` -- kept as a jnp implementation here because this runs inside
    a jitted selection, while the metadata one is host-side NumPy. The equality is
    pinned by a test.
    """
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    return (jnp.asarray(query_positions) + 1) // ratio


def resolve_indexer_backend(backend: str = "auto") -> str:
    """Resolve ``auto`` (env ``DSV4_INDEXER_BACKEND`` first, then the JAX backend)."""
    if backend not in ("auto", "kernel", "reference"):
        raise ValueError(f"unknown CSA indexer backend {backend!r}")
    if backend == "auto":
        backend = os.environ.get(INDEXER_BACKEND_ENV, "auto")
        if backend not in ("auto", "kernel", "reference"):
            raise ValueError(f"{INDEXER_BACKEND_ENV}={backend!r} must be auto, kernel or reference")
    if backend == "auto":
        return "kernel" if jax.default_backend() == "tpu" else "reference"
    return backend


def kernel_read_layout(compressed_rows, seq_lens, q_lens, *, ratio: int, compressed_page_size: int):
    """Per-request page table and gathered-row offsets from the flat read tables.

    ``read_tables`` lists every request's completed entries in order, request after
    request, at flat row ``page * compressed_page_size + entry % compressed_page_size``.
    So request ``r`` owns rows ``[offset_r, offset_r + count_r)`` of the gathered array,
    and the page of its ``p``-th page-aligned entry is that row's page.

    Returns ``(page_indices [B, pages_per_seq], offsets [B])``; entries past a request's
    count are clamped to real rows so every page index is a valid page, and the kernel
    never reads them because ``seq_lens`` bounds each request.
    """
    compressed_rows = jnp.asarray(compressed_rows, jnp.int32)
    seq_lens = jnp.asarray(seq_lens, jnp.int32)
    counts = jnp.where(jnp.asarray(q_lens) > 0, seq_lens // ratio, 0).astype(jnp.int32)
    offsets = jnp.cumsum(counts) - counts
    capacity = compressed_rows.shape[0]
    pages_per_seq = max(1, -(-capacity // compressed_page_size))
    first = (
        offsets[:, None]
        + jnp.arange(pages_per_seq, dtype=jnp.int32)[None, :] * compressed_page_size
    )
    first = jnp.clip(first, 0, capacity - 1)
    return compressed_rows[first] // compressed_page_size, offsets


def csa_indexer_topk_kernel(
    q,
    weights,
    indexer_buffer,
    *,
    compressed_rows,
    seq_lens,
    q_lens,
    cu_q_lens,
    query_request_ids,
    valid_token_mask,
    k: int,
    ratio: int,
    compressed_page_size: int,
    topk_backend: str = "auto",
    return_scores: bool = False,
):
    """`csa_indexer_topk_ref` semantics through ``kernels/dsa/streamindex_topk``.

    Args:
      q: ``[T, H, D]`` indexer queries, request-major in ``cu_q_lens`` order.
      weights: ``[T, H]`` per-head weights.
      indexer_buffer: ``[pages * compressed_page_size, D]`` flat paged indexer cache
        (the ``kv_buffers["indexer"]`` view), already holding this step's records.
      compressed_rows: flat read table (see `kernel_read_layout`).
      seq_lens, q_lens, cu_q_lens: ``[B]``, ``[B]``, ``[B + 1]`` from the metadata.
      query_request_ids, valid_token_mask: ``[T]``.
      k, ratio, compressed_page_size: ``index_topk``, 4, ``page_size // ratio``.

    Returns:
      ``[T, k]`` int32 row indices into the gathered compressed key array (the same
      coordinates `csa_indexer_topk_ref` returns), ``INVALID_ENTRY`` packed at the tail.
      With ``return_scores``: ``(scores [T, E_padded] f32, offsets [B] int32)`` -- the
      raw indexer scores (-inf where a query may not see the entry; column = entry
      index within the query's own request) and each request's first gathered row,
      for `membership_from_scores`.
    """
    from sgl_jax.srt.kernels.dsa.streamindex_topk import streamindex_topk

    if compressed_page_size % 2:
        raise ValueError("compressed_page_size must be even for the paged indexer layout")
    if _KERNEL_KV_PAGES_PER_BLOCK * compressed_page_size % 128:
        raise ValueError("kv pages per block times compressed page size must be a multiple of 128")
    q = jnp.asarray(q)
    num_requests = jnp.asarray(seq_lens).shape[0]
    pages, offsets = kernel_read_layout(
        compressed_rows, seq_lens, q_lens, ratio=ratio, compressed_page_size=compressed_page_size
    )
    cache_kv = jnp.asarray(indexer_buffer).reshape(-1, compressed_page_size // 2, 2, q.shape[-1])
    active = jnp.asarray(q_lens) > 0
    if return_scores:
        scores = streamindex_topk(
            q=q.astype(jnp.bfloat16),
            indexer_weights=jnp.asarray(weights, jnp.float32),
            cache_kv=cache_kv,
            seq_lens=jnp.where(active, jnp.asarray(seq_lens), 0).astype(jnp.int32),
            page_indices=pages.reshape(-1).astype(jnp.int32),
            cu_q_lens=jnp.asarray(cu_q_lens, jnp.int32),
            distribution=jnp.asarray((0, 0, num_requests), jnp.int32),
            k=k,
            compression_ratio=ratio,
            num_kv_pages_per_block=_KERNEL_KV_PAGES_PER_BLOCK,
            num_queries_per_block=_KERNEL_QUERIES_PER_BLOCK,
            topk_backend=topk_backend,
            return_scores=True,
        )
        return scores, offsets
    selected = streamindex_topk(
        q=q.astype(jnp.bfloat16),
        indexer_weights=jnp.asarray(weights, jnp.float32),
        cache_kv=cache_kv,
        seq_lens=jnp.where(active, jnp.asarray(seq_lens), 0).astype(jnp.int32),
        page_indices=pages.reshape(-1).astype(jnp.int32),
        cu_q_lens=jnp.asarray(cu_q_lens, jnp.int32),
        distribution=jnp.asarray((0, 0, num_requests), jnp.int32),
        k=k,
        compression_ratio=ratio,
        num_kv_pages_per_block=_KERNEL_KV_PAGES_PER_BLOCK,
        num_queries_per_block=_KERNEL_QUERIES_PER_BLOCK,
        topk_backend=topk_backend,
    )
    request = jnp.clip(jnp.asarray(query_request_ids), 0, num_requests - 1)
    keep = jnp.asarray(valid_token_mask, bool)[:, None] & (selected >= 0)
    return jnp.where(keep, selected + offsets[request][:, None], INVALID_ENTRY).astype(jnp.int32)


def membership_from_scores(
    scores,
    offsets,
    *,
    q_lens,
    query_request_ids,
    valid_token_mask,
    k: int,
    num_entries: int,
    topk_backend: str = "auto",
):
    """``[T, num_entries]`` bool top-k membership from the indexer scores.

    Two ways to the same mask, chosen at run time: when exactly one request is
    active its gathered rows start at 0, so a row's score column *is* its gathered
    row and the mask is ``score >= k-th largest`` (`kernels/dsv4/topk_threshold`,
    linear in E, no sort; ties admit every tied entry). Otherwise the index path is
    kept: `select_topk_indices` + per-request offsets + `packed_membership`.
    """
    from sgl_jax.srt.kernels.dsa.streamindex_topk import select_topk_indices
    from sgl_jax.srt.kernels.dsv4.topk_threshold import topk_membership_mask
    from sgl_jax.srt.layers.attention.dsv4.attention import packed_membership

    scores = jnp.asarray(scores, jnp.float32)
    offsets = jnp.asarray(offsets, jnp.int32)
    num_requests = offsets.shape[0]
    rows_valid = jnp.asarray(valid_token_mask, bool)[:, None]
    request = jnp.clip(jnp.asarray(query_request_ids), 0, num_requests - 1)

    def _fit(mask):
        width = mask.shape[1]
        if width >= num_entries:
            return mask[:, :num_entries]
        return jnp.pad(mask, ((0, 0), (0, num_entries - width)))

    def by_threshold(s):
        return _fit(topk_membership_mask(s, k)) & rows_valid

    def by_indices(s):
        selected = select_topk_indices(s, k, backend=topk_backend)[:, :k]
        keep = rows_valid & (selected >= 0)
        selected = jnp.where(keep, selected + offsets[request][:, None], INVALID_ENTRY)
        return packed_membership(selected.astype(jnp.int32), num_entries)

    single = jnp.sum((jnp.asarray(q_lens) > 0).astype(jnp.int32)) == 1
    return jax.lax.cond(single, by_threshold, by_indices, scores)
