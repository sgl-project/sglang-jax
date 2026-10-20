"""dp>1 round-trip for the MLA page path on emulated CPU devices.

Run in its own process (the flag must be set before the JAX backend initializes):

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \\
        pytest sgl_jax/test/mem_cache/test_host_kv_pool_mla_dp.py

Covers what the single-device test cannot: the host slot sharded over the
``data`` axis (B1 fix: no per-device replication), the gather landing each
device's slice, the ``write_pages_layer`` reshard from slot layout to
page-sharded layout, and the global->shard-local page id mapping for dp>1.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.sharding import PartitionSpec as P  # noqa: E402

from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool  # noqa: E402
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool  # noqa: E402
from sgl_jax.srt.utils.mesh_utils import create_device_mesh  # noqa: E402

DP = 8
_MESH = create_device_mesh(ici_parallelism=[DP, 1], dcn_parallelism=[1, 1])


def _skip_unless_8_devices(tc):
    if jax.device_count() < DP:
        tc.skipTest(
            f"needs {DP} devices; run with XLA_FLAGS=--xla_force_host_platform_device_count={DP}"
        )


def _make_pools(*, dtype=jnp.bfloat16, layer_num=2, pages_per_rank=4, with_indexer=False):
    jax.sharding.set_mesh(_MESH)
    page_size = 128
    # size is the token capacity across all DP ranks; the pool adds one slack
    # page per rank, so pages_per_rank + 1 pages land on each rank.
    pool = MLATokenToKVPool(
        size=pages_per_rank * page_size * DP,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=layer_num,
        mesh=_MESH,
        dp_size=DP,
        indexer_key_dim=128 if with_indexer else 0,
        num_indexer_layers=1 if with_indexer else 0,
    )
    host = LRUHostKVPool(
        device_pool=pool,
        pool_size=16,
        page_size=page_size,
        layer_num=layer_num,
        per_layer_shape=tuple(int(d) for d in pool.kv_buffer[0].shape[1:]),
        dtype=pool.dtype,
        mesh=_MESH,
        partition_spec=pool.kv_sharding.spec,
    )
    return pool, host


def _page(buf, page):
    """Read one page of a (possibly data-sharded) buffer as numpy."""
    return np.asarray(jax.device_get(buf))[page]


def _fill(pool, page, seed):
    outs = []
    for layer in range(pool.layer_num):
        buf = pool.kv_buffer[layer]
        vals = jax.random.normal(
            jax.random.PRNGKey(seed * 100 + layer), tuple(buf.shape[1:]), jnp.float32
        ).astype(buf.dtype)
        pool.kv_buffer[layer] = buf.at[page].set(vals, out_sharding=buf.sharding)
        outs.append(_page(pool.kv_buffer[layer], page))
    idx = []
    for s in range(len(pool.indexer_key_buffer)):
        buf = pool.indexer_key_buffer[s]
        vals = jax.random.normal(
            jax.random.PRNGKey(seed * 1000 + s), tuple(buf.shape[1:]), jnp.float32
        ).astype(buf.dtype)
        pool.indexer_key_buffer[s] = buf.at[page].set(vals, out_sharding=buf.sharding)
        idx.append(_page(pool.indexer_key_buffer[s], page))
    return outs, idx


class TestSlotLayout(unittest.TestCase):
    def test_slot_is_sharded_over_data_not_replicated(self):
        _skip_unless_8_devices(self)
        pool, host = _make_pools()
        g = host._groups[0]
        # MLA bf16 page is (64, 2, 640): the 64 token-group dim is divisible by dp=8.
        self.assertEqual(tuple(g.slot_spec), (None, "data", None, None))
        # 8 local devices / 8-way shard => one full-slot equivalent per host.
        self.assertAlmostEqual(g.replicas_per_host(_MESH), 1.0)
        self.assertEqual(host.host_bytes_per_slot_local(), g.slot_bytes())

    def test_fp8_slot_shardable(self):
        _skip_unless_8_devices(self)
        pool, host = _make_pools(dtype=jnp.float8_e4m3fn)
        self.assertEqual(tuple(host._groups[0].slot_spec), (None, "data", None, None))


class TestDPRoundTrip(unittest.TestCase):
    def test_pages_on_different_ranks_roundtrip(self):
        _skip_unless_8_devices(self)
        pool, host = _make_pools(pages_per_rank=4)
        pages_per_shard = int(pool.kv_buffer[0].shape[0]) // DP
        # One source page on rank 0, one on rank 5; restore onto rank 2 and rank 7.
        src = [1, 5 * pages_per_shard + 2]
        dst = [2 * pages_per_shard + 3, 7 * pages_per_shard + 1]
        origs = [_fill(pool, s, seed=i + 1)[0] for i, s in enumerate(src)]
        untouched = 3 * pages_per_shard + 2
        before = [_page(b, untouched) for b in pool.kv_buffer]

        slots = [host.reserve() for _ in src]
        host.copy_into(src, slots)
        host.copy_to_device(slots, dst)

        for d, orig in zip(dst, origs):
            for layer in range(pool.layer_num):
                np.testing.assert_array_equal(
                    _page(pool.kv_buffer[layer], d), orig[layer], err_msg=f"dst {d} L{layer}"
                )
        for layer, b in enumerate(pool.kv_buffer):
            np.testing.assert_array_equal(_page(b, untouched), before[layer])

    def test_with_indexer_group(self):
        _skip_unless_8_devices(self)
        pool, host = _make_pools(pages_per_rank=3, with_indexer=True)
        self.assertEqual(len(host._groups), 2)
        pages_per_shard = int(pool.kv_buffer[0].shape[0]) // DP
        src, dst = 4 * pages_per_shard + 1, 6 * pages_per_shard + 2
        orig, idx_orig = _fill(pool, src, seed=9)
        slot = host.reserve()
        host.copy_into([src], [slot])
        host.copy_to_device([slot], [dst])
        for layer in range(pool.layer_num):
            np.testing.assert_array_equal(_page(pool.kv_buffer[layer], dst), orig[layer])
        np.testing.assert_array_equal(_page(pool.indexer_key_buffer[0], dst), idx_orig[0])


if __name__ == "__main__":
    unittest.main()
