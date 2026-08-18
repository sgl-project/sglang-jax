"""TPU round-trip tests for :class:`LRUHostKVPool` (HiCache Stage 0), tp=4/dp=1.

The CPU suite (``test_host_kv_pool.py``) monkeypatches the fused-KV Pallas
kernel out (``conftest.py``, gated on ``backend != "tpu"``) and falls back to
default sharding when ``pinned_host`` is unavailable. That leaves the two parts
most likely to break a donating in-place KV write unexercised:

  1. the real ``update_fused_kv_cache_vectorized`` Pallas kernel, and
  2. true ``pinned_host`` placement for the host slots.

This module runs the bit-exact D2H->H2D round-trip on a 4-chip TPU at
``tp=4/dp=1``, covering both. The dp>1 page-axis gather/scatter layout is covered
by the companion ``test_host_kv_pool_tpu_dp.py``: a single process cannot switch
device mesh topology (``[1, 4]`` vs ``[2, 2]``) without the second mesh producing
garbage, so the two layouts live in separate files (each TestFile in
``run_suite.py`` is its own subprocess). Both are registered in
``unit-test-tpu-v6e-4``; skipped off TPU / with <4 devices.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool
from sgl_jax.srt.mem_cache.memory_pool import MHATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _is_tpu_4chip() -> bool:
    try:
        return jax.default_backend() == "tpu" and len(jax.devices()) >= 4
    except RuntimeError:
        return False


def _make_device_pool(
    mesh, *, dp_size, size=32, page_size=1, head_num=4, head_dim=128, layer_num=3
):
    # head_dim must be 128-aligned: copy_to_device writes via the in-place Pallas
    # kernel (update_fused_kv_cache_vectorized), which requires it.
    return MHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=jnp.bfloat16,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        mesh=mesh,
        dp_size=dp_size,
    )


def _make_pool(device_pool, mesh, *, pool_size=8, page_size=1):
    return LRUHostKVPool(
        device_pool=device_pool,
        pool_size=pool_size,
        page_size=page_size,
        layer_num=device_pool.layer_num,
        per_layer_shape=tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:]),
        dtype=device_pool.dtype,
        mesh=mesh,
        partition_spec=device_pool.kv_sharding.spec,
    )


def _read_page(device_pool, layer, idx):
    # Pull the whole (sharded) layer to host and index with numpy: a device-side
    # ``buf[idx]`` gather can't resolve an output sharding under explicit
    # sharding on a multi-chip mesh (test-only read-back, not the path under test).
    return np.asarray(jax.device_get(device_pool.kv_buffer[layer]))[idx]


def _fill(device_pool, idx, seed):
    orig = []
    for layer in range(device_pool.layer_num):
        buf = device_pool.kv_buffer[layer]
        page_shape = tuple(buf.shape[1:])  # avoid a buf[idx] gather just for shape
        vals = jax.random.normal(
            jax.random.PRNGKey(seed * 100 + layer), page_shape, jnp.float32
        ).astype(buf.dtype)
        device_pool.kv_buffer[layer] = buf.at[idx].set(vals, out_sharding=buf.sharding)
        orig.append(_read_page(device_pool, layer, idx))
    return orig


@unittest.skipUnless(_is_tpu_4chip(), "requires a TPU with >=4 chips")
class TestLRUHostKVPoolTPURoundTrip(unittest.TestCase):
    """Bit-exact D2H->H2D round-trip on the real kernel + pinned host, tp=4/dp=1."""

    def _roundtrip(self, page_size=1):
        mesh = create_device_mesh(ici_parallelism=[1, 4], dcn_parallelism=[1, 1])
        device_pool = _make_device_pool(mesh, dp_size=1, page_size=page_size)
        pool = _make_pool(device_pool, mesh, pool_size=8, page_size=page_size)

        # Distinct src/dst device pages so a no-op (forgot to write) can't pass.
        pairs = [(2, 5), (3, 6), (4, 7)]  # 3 pages -> exercises bucket padding too
        origs = {s: _fill(device_pool, s, seed=s + 1) for s, _ in pairs}
        host_pages = [int(p) for p in pool.alloc(len(pairs))]
        srcs = [s for s, _ in pairs]
        dsts = [d for _, d in pairs]

        pool.copy_into(srcs, host_pages)
        pool.copy_to_device(host_pages, dsts)

        for s, d in pairs:
            for layer in range(device_pool.layer_num):
                got = _read_page(device_pool, layer, d)
                np.testing.assert_array_equal(got, origs[s][layer])

    def test_roundtrip_tp4_dp1(self):
        self._roundtrip()

    def test_roundtrip_tp4_dp1_page_size4(self):
        self._roundtrip(page_size=4)

    def test_padding_does_not_clobber_other_pages(self):
        # Device page 0 is the gather/scatter padding target; a 3-page transfer
        # excluding it must leave it untouched (loc=-1 skip works on the real
        # kernel, not just the CPU shim).
        mesh = create_device_mesh(ici_parallelism=[1, 4], dcn_parallelism=[1, 1])
        device_pool = _make_device_pool(mesh, dp_size=1)
        pool = _make_pool(device_pool, mesh, pool_size=8)
        guard = _fill(device_pool, 0, seed=42)
        srcs, dsts = [2, 3, 4], [5, 6, 7]
        for s in srcs:
            _fill(device_pool, s, seed=s + 1)
        host_pages = [int(p) for p in pool.alloc(3)]
        pool.copy_into(srcs, host_pages)
        pool.copy_to_device(host_pages, dsts)
        for layer in range(device_pool.layer_num):
            got = _read_page(device_pool, layer, 0)
            np.testing.assert_array_equal(got, guard[layer])


if __name__ == "__main__":
    unittest.main()
