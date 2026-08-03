from unittest.mock import patch

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.multimodal.in_model import embedding_pool as embedding_pool_module
from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPool


def _read(pool, entry):
    """Reconstruct an entry's stored rows from the paged buffer."""
    rows = []
    for k in range(entry.length):
        page = int(entry.page_ids[k // pool.page_size])
        off = k % pool.page_size
        rows.append(np.asarray(pool.pages[page, off]))
    return np.stack(rows)


def _write(pool, item_hash, emb, ds=None):
    """Cache one item through the packed writer (the pool's only write path).

    ``ds`` is ``[L, D, H]``; it is concatenated onto ``emb``'s feature axis to
    form the ``[L, (1+D)*H]`` layout the pool now stores in one buffer.
    """
    emb = jnp.asarray(emb)
    if ds is not None:
        ds = jnp.asarray(ds)
        emb = jnp.concatenate([emb, ds.reshape(ds.shape[0], -1)], axis=-1)
    (entry,) = pool.write_packed((item_hash,), emb[None], ((0, 0, emb.shape[0]),))
    return entry


def test_write_then_lookup_roundtrips_and_pages_align():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    emb = jnp.asarray([[10.0], [11.0], [12.0]])  # length 3 -> 2 pages
    entry = _write(pool, 1, emb)
    assert entry.length == 3
    assert len(entry.page_ids) == 2  # ceil(3 / 2)
    np.testing.assert_array_equal(_read(pool, entry)[:, 0], [10, 11, 12])
    assert pool.lookup(1) is entry
    assert pool.lookup(999) is None


def test_lru_eviction_frees_pages():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    _write(pool, 0xA, jnp.asarray([[1.0], [2.0], [3.0]]))  # pages 0,1
    _write(pool, 0xB, jnp.asarray([[4.0], [5.0], [6.0]]))  # pages 2,3 -> full
    # No free pages left; writing C must evict the LRU entry (A).
    entry_c = _write(pool, 0xC, jnp.asarray([[7.0]]))
    assert entry_c is not None
    assert pool.lookup(0xA) is None  # evicted
    assert pool.lookup(0xB) is not None  # survived
    np.testing.assert_array_equal(_read(pool, pool.lookup(0xC))[:, 0], [7])


def test_lookup_touch_protects_from_eviction():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    _write(pool, 0xA, jnp.asarray([[1.0], [2.0], [3.0]]))
    _write(pool, 0xB, jnp.asarray([[4.0], [5.0], [6.0]]))
    pool.lookup(0xA)  # A becomes MRU, so B is now the eviction victim
    _write(pool, 0xC, jnp.asarray([[7.0]]))
    assert pool.lookup(0xB) is None
    assert pool.lookup(0xA) is not None


def test_item_larger_than_pool_is_skipped():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    assert _write(pool, 1, jnp.zeros((9, 1), dtype=jnp.float32)) is None  # 5 pages > 4


def test_rewrite_reuses_freed_pages():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    _write(pool, 1, jnp.asarray([[1.0], [2.0], [3.0]]))
    second = _write(pool, 1, jnp.asarray([[9.0]]))
    assert second.length == 1
    # Overwriting the same hash releases the old pages before re-allocating.
    assert len(pool._free_pages) == 3
    np.testing.assert_array_equal(_read(pool, pool.lookup(1))[:, 0], [9])


def test_deepstack_roundtrips():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32, deepstack_dim=2)
    emb = jnp.asarray([[1.0], [2.0]])  # [L=2, H=1]
    ds = jnp.asarray([[[3.0], [4.0]], [[5.0], [6.0]]])  # [L=2, D=2, H=1]
    entry = _write(pool, 1, emb, ds)
    page, off = int(entry.page_ids[0]), 0
    # Deepstack planes are stored after the primary H columns in the same buffer.
    np.testing.assert_array_equal(np.asarray(pool.pages[page, off, pool.hidden :]), [3, 4])


def test_write_packed_roundtrips_lane_offset_and_drops_padding():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    pool._pages = pool.pages.at[-1, -1, 0].set(99)
    packed = jnp.asarray(
        [
            [[100.0], [101.0], [102.0], [103.0]],
            [[200.0], [201.0], [202.0], [203.0]],
        ]
    )

    (entry,) = pool.write_packed((1,), packed, ((0, 1, 3),))

    assert entry is not None
    assert entry.length == 3
    assert len(entry.page_ids) == 2
    np.testing.assert_array_equal(_read(pool, entry)[:, 0], [101, 102, 103])
    # Every non-placement row maps to the positive OOB sentinel and must not
    # wrap around to the final physical pool row.
    assert np.asarray(pool.pages[-1, -1, 0]) == 99


def test_write_packed_keeps_writer_shape_fixed_across_true_lengths():
    pool = EmbeddingPool(num_pages=8, page_size=2, hidden=1, dtype=jnp.float32)
    packed = jnp.arange(8, dtype=jnp.float32).reshape(2, 4, 1)

    with patch.object(
        embedding_pool_module,
        "_scatter_rows",
        wraps=embedding_pool_module._scatter_rows,
    ) as scatter:
        pool.write_packed((1, 2), packed, ((0, 0, 1), (1, 1, 3)))
        pool.write_packed((3,), packed, ((0, 1, 2),))

    assert [call.args[1].shape for call in scatter.call_args_list] == [(2, 4), (2, 4)]
    assert [call.args[2].shape for call in scatter.call_args_list] == [
        (2, 4, 1),
        (2, 4, 1),
    ]


def test_write_packed_deepstack_uses_same_placement():
    pool = EmbeddingPool(
        num_pages=4,
        page_size=2,
        hidden=1,
        dtype=jnp.float32,
        deepstack_dim=2,
    )
    # [1, cap=4, (1+D)*H=3]: primary col 0, deepstack planes cols 1..2.
    packed = jnp.asarray(
        [
            [
                [1.0, 10.0, 11.0],
                [2.0, 20.0, 21.0],
                [3.0, 30.0, 31.0],
                [4.0, 40.0, 41.0],
            ]
        ]
    )

    (entry,) = pool.write_packed((1,), packed, ((0, 1, 2),))

    assert entry is not None
    np.testing.assert_array_equal(_read(pool, entry)[:, 0], [2, 3])
    page0 = int(entry.page_ids[0])
    np.testing.assert_array_equal(
        np.asarray(pool.pages[page0, :2, pool.hidden :]),
        [[20, 21], [30, 31]],
    )


def test_write_packed_scatter_once_and_skips_entries_evicted_during_planning():
    pool = EmbeddingPool(num_pages=2, page_size=1, hidden=1, dtype=jnp.float32)
    packed = jnp.asarray([[[10.0], [20.0], [30.0]]])

    with patch.object(
        embedding_pool_module,
        "_scatter_rows",
        wraps=embedding_pool_module._scatter_rows,
    ) as scatter:
        entries = pool.write_packed(
            (1, 2, 3),
            packed,
            ((0, 0, 1), (0, 1, 1), (0, 2, 1)),
        )

    scatter.assert_called_once()
    assert entries[0] is None
    assert entries[1] is not None
    assert entries[2] is not None
    assert pool.lookup(1) is None
    np.testing.assert_array_equal(_read(pool, pool.lookup(2))[:, 0], [20])
    np.testing.assert_array_equal(_read(pool, pool.lookup(3))[:, 0], [30])


def test_write_packed_duplicate_hash_keeps_only_last_placement():
    pool = EmbeddingPool(num_pages=2, page_size=1, hidden=1, dtype=jnp.float32)
    packed = jnp.asarray([[[10.0], [20.0]]])

    first, last = pool.write_packed(
        (1, 1),
        packed,
        ((0, 0, 1), (0, 1, 1)),
    )

    assert first is None
    assert last is pool.lookup(1)
    np.testing.assert_array_equal(_read(pool, last)[:, 0], [20])
    assert len(pool._free_pages) == 1


def test_write_packed_oversized_item_does_not_disturb_other_entries():
    pool = EmbeddingPool(num_pages=2, page_size=1, hidden=1, dtype=jnp.float32)
    existing = _write(pool, 1, jnp.asarray([[5.0]]))
    packed = jnp.asarray([[[10.0], [11.0], [12.0], [20.0]]])

    oversized, normal = pool.write_packed(
        (1, 2),
        packed,
        ((0, 0, 3), (0, 3, 1)),
    )

    assert oversized is None
    assert pool.lookup(1) is existing
    assert normal is pool.lookup(2)
    np.testing.assert_array_equal(_read(pool, existing)[:, 0], [5])
    np.testing.assert_array_equal(_read(pool, normal)[:, 0], [20])
    assert len(pool._free_pages) == 0


def test_clear_resets_free_list():
    pool = EmbeddingPool(num_pages=4, page_size=2, hidden=1, dtype=jnp.float32)
    _write(pool, 1, jnp.asarray([[1.0], [2.0], [3.0]]))
    pool.clear()
    assert pool.lookup(1) is None
    assert len(pool._free_pages) == 4
