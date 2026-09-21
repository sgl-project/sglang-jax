"""Cache data integrity at capacity and padding boundaries."""

import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.multimodal.in_model.embedding_pool import EmbeddingPool


def _pool(pages=3, page_size=2):
    return EmbeddingPool(pages, page_size, hidden=2, dtype=jnp.float32)


def _write(pool, key, length, value=1):
    (entry,) = pool.write_packed([key], np.full((length, 2), value), [length])
    return entry


def _read(pool, entry):
    return np.asarray(pool.pages)[entry.page_ids].reshape(-1, 2)[: entry.length]


def test_packed_write_handles_page_tail_skipped_item_and_padding():
    pool = _pool(pages=4)
    sentinel = _write(pool, 9, 2, 99)
    packed = np.arange(20, dtype=np.float32).reshape(10, 2)
    entries = pool.write_packed([1, 2, 3], packed, [3, 2, 1], write_mask=[True, False, True])
    np.testing.assert_array_equal(_read(pool, entries[0]), packed[:3])
    np.testing.assert_array_equal(_read(pool, entries[2]), packed[5:6])
    np.testing.assert_array_equal(_read(pool, sentinel), 99)
    assert entries[1] is None and pool.lookup(2) is None


def test_lru_reuses_fragmented_pages_and_clear_restores_full_capacity():
    pool = _pool()
    for key in (1, 2, 3):
        _write(pool, key, 2, key)
    assert pool.contains(2) and not pool.contains(99)  # Presence checks must not touch LRU.
    pool.lookup(1)  # Protect the oldest item by touching it.
    assert _write(pool, 4, 2, 4) is not None
    assert pool.lookup(2) is None
    pool.lookup(3)  # Next victim is page 0, leaving free pages in non-sorted order.
    large = _write(pool, 5, 3, 5)
    assert pool.lookup(1) is None and pool.lookup(4) is None
    np.testing.assert_array_equal(_read(pool, pool.lookup(3)), 3)
    np.testing.assert_array_equal(_read(pool, large), 5)
    pool.clear()
    assert all(pool.lookup(key) is None for key in (3, 5))
    assert _write(pool, 6, 6) is not None


@pytest.mark.parametrize("length", [0, 4, 5], ids=["empty", "exact-capacity", "oversized"])
def test_capacity_boundary_and_oversized_replacement(length):
    pool = _pool(pages=2)
    old = _write(pool, 1, 1, 7)
    result = _write(pool, 1, length, 9)
    if length > 4:
        assert result is None and pool.lookup(1) is old
        np.testing.assert_array_equal(_read(pool, old), 7)
    else:
        assert result.length == length
        np.testing.assert_array_equal(_read(pool, result), np.full((length, 2), 9))


@pytest.mark.parametrize("keys", [[1, 2, 3], [1, 1, 2]], ids=["eviction", "duplicate-hash"])
def test_packed_write_does_not_write_superseded_placements(keys):
    pool = _pool(pages=2, page_size=1)
    packed = np.arange(6, dtype=np.float32).reshape(3, 2)
    first, second, third = pool.write_packed(keys, packed, [1, 1, 1])
    assert first is None
    np.testing.assert_array_equal(_read(pool, second), packed[1:2])
    np.testing.assert_array_equal(_read(pool, third), packed[2:3])
    assert pool.lookup(keys[1]) is second and pool.lookup(keys[2]) is third


@pytest.mark.parametrize(
    "keys, shape, lengths, mask",
    [
        ([1, 2], (2, 2), [1], None),
        ([1], (2, 2), [1], []),
        ([1], (2, 2), [-1], None),
        ([1], (2, 2), [3], None),
        ([1], (2, 3), [1], None),
    ],
    ids=["hash-count", "mask-count", "negative-length", "short-input", "wrong-width"],
)
def test_invalid_write_does_not_damage_resident_data(keys, shape, lengths, mask):
    pool = _pool()
    original = _write(pool, 9, 2, 7)
    with pytest.raises(ValueError):
        pool.write_packed(keys, np.zeros(shape), lengths, write_mask=mask)
    assert pool.lookup(9) is original
    np.testing.assert_array_equal(_read(pool, original), 7)
