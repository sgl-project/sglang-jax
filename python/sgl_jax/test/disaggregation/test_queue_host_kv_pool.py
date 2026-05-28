"""Unit tests for :class:`QueueHostKVPool`.

These run on CPU jax. ``pinned_host`` memory_kind isn't available on
CPU jaxlib so the pool transparently falls back to the default
sharding (see ``host_kv_pool._make_host_sharding``); the queue
semantics under test are independent of the memory kind.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from sgl_jax.srt.mem_cache.host_kv_pool import (
    HostBufferHandle,
    QueueHostKVPool,
    StagedData,
)


def _single_device_mesh() -> Mesh:
    devices = jax.local_devices()
    return Mesh(np.asarray(devices[:1]).reshape(1), axis_names=("x",))


def _make_pool(
    pool_size: int = 4,
    max_tokens_per_buffer: int = 16,
    layer_num: int = 2,
    kv_head_per_rank: int = 2,
    head_dim: int = 8,
) -> QueueHostKVPool:
    mesh = _single_device_mesh()
    spec = PartitionSpec()  # fully replicated; CPU-only fallback
    return QueueHostKVPool(
        pool_size=pool_size,
        max_tokens_per_buffer=max_tokens_per_buffer,
        layer_num=layer_num,
        kv_head_per_rank=kv_head_per_rank,
        head_dim=head_dim,
        dtype=jnp.float32,
        mesh=mesh,
        partition_spec=spec,
    )


def test_init_validates_sizes():
    with pytest.raises(ValueError, match="pool_size"):
        _make_pool(pool_size=0)
    with pytest.raises(ValueError, match="max_tokens_per_buffer"):
        _make_pool(max_tokens_per_buffer=0)


def test_alloc_returns_handle_and_decrements_available():
    pool = _make_pool(pool_size=4, max_tokens_per_buffer=16)
    assert pool.total_size() == 4
    assert pool.available_size() == 4

    h = pool.alloc(num_tokens=8)
    assert isinstance(h, HostBufferHandle)
    assert h.num_tokens == 8
    assert h.buffer.shape == (16, 2, 2, 8)
    assert pool.available_size() == 3


def test_alloc_returns_none_when_exhausted():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=16)
    h1 = pool.alloc(8)
    h2 = pool.alloc(8)
    h3 = pool.alloc(8)
    assert h1 is not None
    assert h2 is not None
    assert h3 is None
    assert pool.available_size() == 0


def test_alloc_returns_none_when_num_tokens_exceeds_per_buffer():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=8)
    assert pool.alloc(num_tokens=9) is None
    assert pool.available_size() == 2


def test_alloc_rejects_non_positive_tokens():
    pool = _make_pool()
    with pytest.raises(ValueError, match="positive"):
        pool.alloc(0)
    with pytest.raises(ValueError, match="positive"):
        pool.alloc(-1)


def test_free_returns_buffer_to_pool_and_allows_realloc():
    pool = _make_pool(pool_size=2)
    h1 = pool.alloc(4)
    h2 = pool.alloc(4)
    assert pool.alloc(4) is None
    pool.free(h1)
    assert pool.available_size() == 1
    h3 = pool.alloc(4)
    assert h3 is not None
    pool.free(h2)
    pool.free(h3)
    assert pool.available_size() == 2


def test_double_free_raises():
    pool = _make_pool(pool_size=2)
    h = pool.alloc(4)
    pool.free(h)
    with pytest.raises(RuntimeError, match="double free"):
        pool.free(h)


def test_get_put_buffer_low_level():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=8)
    bid1, h1 = pool.get_buffer()
    bid2, h2 = pool.get_buffer()
    assert bid1 != bid2
    assert h1.num_tokens == 8 and h2.num_tokens == 8
    with pytest.raises(RuntimeError, match="empty"):
        pool.get_buffer()
    pool.put_buffer(bid1)
    bid3, _ = pool.get_buffer()
    assert bid3 == bid1


def test_copy_from_device_byte_equal_for_prefix():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=8)
    rng = np.random.default_rng(0)
    src_np = rng.integers(0, 200, size=(4, 2, 2, 8), dtype=np.int32).astype(np.float32)
    device_kv = jnp.asarray(src_np)
    staged: StagedData = pool.copy_from_device(device_kv)

    assert isinstance(staged, StagedData)
    assert staged.array.shape == (8, 2, 2, 8)
    # First 4 token rows match the source, remaining 4 are the initial
    # zeros from preallocation.
    got = np.asarray(jax.device_get(staged.array))
    np.testing.assert_array_equal(got[:4], src_np)
    np.testing.assert_array_equal(got[4:], np.zeros((4, 2, 2, 8), dtype=np.float32))

    # Pool's internal slot is updated; getting that buffer back should
    # yield the same content. Drain the rest of the queue first so the
    # FIFO returns the slot we just put back.
    pool.put_buffer(staged.buffer_id)
    drained = []
    while pool.available_size() > 1:
        bid_drain, _ = pool.get_buffer()
        drained.append(bid_drain)
    bid, handle = pool.get_buffer()
    assert bid == staged.buffer_id
    got2 = np.asarray(jax.device_get(handle.buffer))
    np.testing.assert_array_equal(got2[:4], src_np)


def test_copy_from_device_rejects_oversize():
    pool = _make_pool(pool_size=2, max_tokens_per_buffer=4)
    big = jnp.zeros((5, 2, 2, 8), dtype=jnp.float32)
    with pytest.raises(ValueError, match="max_tokens_per_buffer"):
        pool.copy_from_device(big)


def test_copy_from_device_raises_when_pool_exhausted():
    pool = _make_pool(pool_size=1, max_tokens_per_buffer=8)
    # Drain the pool.
    pool.alloc(8)
    src = jnp.zeros((4, 2, 2, 8), dtype=jnp.float32)
    with pytest.raises(RuntimeError, match="exhausted"):
        pool.copy_from_device(src)
