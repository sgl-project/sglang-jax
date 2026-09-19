"""Paged writes into QSA's compressed indexer-key cache.

The compressed cache holds one entry per ``compress_ratio`` tokens and is given
a page size of ``page_size // compress_ratio``, which makes the token cache's
page table address it unchanged: token ``t``'s entry is ``t // ratio``, on
logical page ``(t // ratio) // (page_size // ratio) == t // page_size``. So this
takes the same ``page_indices`` / ``cu_kv_lens`` the attention metadata already
carries, and differs from ``dsa_sparse_backend._scatter_paged`` only in
addressing by group rather than by token.

Pure page arithmetic, deliberately outside the backend module: the tests that
pin it should not have to import an attention backend to reach it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

SENTINEL_PAGE = 0


def scatter_compressed(
    cache3d: jax.Array,  # [num_pages, compressed_page_size, dim]
    compressed: jax.Array,  # [N, dim]
    groups: jax.Array,  # i32[N]  group index within the request, -1 == skip
    seq_ids: jax.Array,  # i32[N]  which request each row belongs to
    page_indices: jax.Array,  # i32[...]  packed page table
    cu_kv_lens: jax.Array,  # i32[S + 1] page-aligned token cumsum
    *,
    compress_ratio: int,
) -> jax.Array:
    """Write each completed group's compressed key into the paged cache.

    Rows with ``groups[i] < 0`` closed no group and are steered to the reserved
    sentinel page, the same way ``_scatter_paged`` parks its padding: the
    allocator hands out local pages 1.. and reads through page 0 are masked.
    """
    compressed_page_size = cache3d.shape[1]
    page_size = compressed_page_size * compress_ratio

    valid = groups >= 0
    safe_groups = jnp.where(valid, groups, 0)
    page_local = safe_groups // compressed_page_size
    offset = safe_groups % compressed_page_size

    base = cu_kv_lens[seq_ids] // page_size
    page = page_indices[jnp.clip(base + page_local, 0, page_indices.shape[0] - 1)]

    page = jnp.where(valid, page, SENTINEL_PAGE)
    offset = jnp.where(valid, offset, 0)
    return cache3d.at[page, offset].set(compressed.astype(cache3d.dtype))


def compressed_slot(token: jax.Array | int, *, compressed_page_size: int, compress_ratio: int):
    """(logical page, in-page offset) of the entry covering ``token``.

    The identity the shared page table rests on, in one place so tests and the
    backend cannot drift apart on it.
    """
    entry = token // compress_ratio
    return entry // compressed_page_size, entry % compressed_page_size


def as_3d(cache4d: jax.Array) -> jax.Array:
    """[pages, P//packing, packing, dim] -> [pages, P, dim]."""
    pages, per_packing, packing, dim = cache4d.shape
    return cache4d.reshape(pages, per_packing * packing, dim)


def as_4d(cache3d: jax.Array, packing: int) -> jax.Array:
    """Inverse of :func:`as_3d`."""
    pages, page_size, dim = cache3d.shape
    return cache3d.reshape(pages, page_size // packing, packing, dim)
