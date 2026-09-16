"""Padding rows in _scatter_paged must never clobber an allocatable page.

The paged allocator hands out local pages 1..pages_per_rank and reserves
page 0 (allocator.py); the last local page is therefore live. Routing
invalid/padding rows to a live page corrupts offset 0 of that page's
indexer keys whenever the pool is full enough for it to be allocated.
"""

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsa_sparse_backend import _scatter_paged

jax.config.update("jax_platform_name", "cpu")

PAGE_SIZE = 4
NUM_PAGES = 8  # local pages 0..7; allocator hands out 1..7, page 0 reserved
D = 16


def _run_padded_batch(cache):
    """One seq: kv_len=4, 3 new tail tokens (offsets 1..3), padded to T=8.

    Offset 0 of the seq's page holds a pre-existing key that this batch does
    not write — the slot the sentinel routing of padding rows lands on.
    """
    T = 8
    new_tokens = jnp.full((T, D), 777.0, dtype=cache.dtype)
    seq_lens = jnp.array([4], dtype=jnp.int32)
    # seq uses the LAST allocatable page (7): the near-full-pool case.
    page_indices = jnp.array([7, 0, 0, 0], dtype=jnp.int32)
    cu_q_lens = jnp.array([0, 3], dtype=jnp.int32)
    cu_kv_lens = jnp.array([0, PAGE_SIZE], dtype=jnp.int32)
    return _scatter_paged(cache, new_tokens, seq_lens, page_indices, cu_q_lens, cu_kv_lens, 4)


def test_padding_rows_do_not_clobber_live_pages():
    cache = jnp.arange(NUM_PAGES * PAGE_SIZE * D, dtype=jnp.float32).reshape(
        NUM_PAGES, PAGE_SIZE, D
    )
    out = np.asarray(_run_padded_batch(cache))
    ref = np.asarray(cache)

    # The seq's own 3 tail slots (offsets 1..3) on page 7 are written.
    assert (out[7, 1:4] == 777.0).all()
    # The pre-existing key at page 7 offset 0 must survive: padding rows
    # must not be routed onto a live page.
    assert (out[7, 0] == ref[7, 0]).all(), "padding rows clobbered a live page"
    assert (out[1:7] == ref[1:7]).all()


def test_padding_rows_land_on_reserved_page_zero_only():
    cache = jnp.zeros((NUM_PAGES, PAGE_SIZE, D), dtype=jnp.float32)
    out = np.asarray(_run_padded_batch(cache))
    # Padding rows may only ever dirty the reserved page 0 (never handed out
    # by the allocator; reads through it are masked past kv_len).
    dirty = (out != 0).any(axis=2)  # [pages, offsets]
    # page 7: only the seq's own offsets 1..3 may be dirty
    assert not dirty[7, 0], "padding garbage on live page 7 offset 0"
    # pages 1..6 fully clean; page 0 (reserved) is the only allowed dump target
    assert not dirty[1:7].any()
