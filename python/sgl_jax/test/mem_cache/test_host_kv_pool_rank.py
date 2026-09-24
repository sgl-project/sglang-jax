"""CPU checks for rank-owned JAX HiCache host pages and transfer pins."""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import PartitionSpec

from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool, make_unit_mesh


def _pool(pool_size=5, dp_size=2):
    # The transfer guard needs a contiguous global device page axis. No TPU
    # kernel runs in these tests: invalid pairs must fail before a transfer.
    device_pool = SimpleNamespace(kv_buffer=[jnp.zeros((16, 1))], dp_size=dp_size, size=16)
    return LRUHostKVPool(
        device_pool=device_pool,
        pool_size=pool_size,
        page_size=1,
        layer_num=1,
        per_layer_shape=(1,),
        dtype=jnp.float32,
        mesh=make_unit_mesh(),
        partition_spec=PartitionSpec(None),
        dp_size=dp_size,
    )


def test_rank_budgets_are_fixed_and_exhaustion_is_isolated():
    pool = _pool()
    assert (pool.total_size(0), pool.total_size(1), pool.total_size()) == (3, 2, 5)
    assert (pool.available_size(0), pool.available_size(1)) == (3, 2)
    rank1 = pool.alloc(2, dp_rank=1)
    rank0 = pool.alloc(3, dp_rank=0)
    np.testing.assert_array_equal(rank1, [3, 4])
    np.testing.assert_array_equal(rank0, [0, 1, 2])
    assert pool.alloc(1, dp_rank=1) is None
    assert pool.reserve() is None  # legacy reserve is rank 0 only
    assert pool.available_size() == 0
    pool.free([0])
    assert pool.available_size(0) == 1
    assert pool.available_size(1) == 0
    assert pool.alloc(1, dp_rank=1) is None


def test_invalid_ranks_fail_without_allocating():
    pool = _pool()
    for rank in (-1, 2, None):
        with pytest.raises(ValueError):
            pool.alloc(1, dp_rank=rank)
    for rank in (-1, 2):
        with pytest.raises(ValueError):
            pool.available_size(rank)
        with pytest.raises(ValueError):
            pool.total_size(rank)
    assert pool.available_size() == 5
    with pytest.raises(ValueError):
        _pool(dp_size=0)


def test_pin_and_unpin_validate_entire_batch_before_changing_locks():
    pool = _pool()
    first = int(pool.alloc(1, dp_rank=0)[0])
    second = int(pool.alloc(1, dp_rank=1)[0])
    with pytest.raises(RuntimeError):
        pool.pin([first, 2])  # page 2 has not been allocated
    pool.free([first])  # failed pin left first freeable
    first = int(pool.alloc(1, dp_rank=0)[0])
    with pytest.raises(ValueError):
        pool.pin([first, first])
    pool.pin([first, second])
    with pytest.raises(RuntimeError):
        pool.free([first, second])
    pool.unpin([first])
    with pytest.raises(RuntimeError):
        pool.unpin([first, second])  # second must remain pinned
    with pytest.raises(RuntimeError):
        pool.free([second])
    pool.unpin([second])
    pool.free([first, second])


def test_transfer_rejects_host_page_owned_by_other_device_rank():
    pool = _pool()
    rank0 = int(pool.alloc(1, dp_rank=0)[0])
    rank1 = int(pool.alloc(1, dp_rank=1)[0])
    # Global device pages 0..7 belong to rank 0, 8..15 to rank 1.
    with pytest.raises(ValueError, match="rank"):
        pool.stage_backup([8], [rank0])
    with pytest.raises(ValueError, match="rank"):
        pool.stage_backup([0, 8], [rank0, rank0])
    with pytest.raises(ValueError, match="rank"):
        pool.flush_load([rank1], [0])
    with pytest.raises(ValueError, match="outside pool range"):
        pool.flush_load([99], [0])


def test_precompile_warms_only_a_transfer_that_fits_one_rank(monkeypatch):
    pool = _pool()
    warmed = []
    monkeypatch.setattr(pool, "stage_backup", lambda pages, handles: warmed.append(len(pages)))
    monkeypatch.setattr(pool, "flush_backup", lambda handles: None)
    monkeypatch.setattr(pool, "stage_load", lambda handles: None)
    monkeypatch.setattr(pool, "flush_load", lambda handles, pages: None)
    pool.precompile_transfers()
    assert warmed == [1, 2, 3]
    assert pool.available_size() == pool.total_size() == 5
