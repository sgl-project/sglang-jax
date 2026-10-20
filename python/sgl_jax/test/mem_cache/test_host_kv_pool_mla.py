"""HiCache host round-trip against :class:`MLATokenToKVPool` (GLM-5 / DeepSeek layout).

CPU-runnable. The MHA round-trip lives in ``test_host_kv_pool.py``; this file is
the MLA counterpart and the regression gate for the MLA H2D scatter path in
``LRUHostKVPool.flush_load``:

* MLA pages are ``[page_size // packing, packing, kv_dim]`` (tokens folded into a
  ``(32, 4)`` grid for fp8 / ``(64, 2)`` for bf16), with no head axis to carry
  TP sharding. The MHA scatter (per-token ``loc`` + 5-D fused kernel) cannot be
  applied to that layout.
* With ``--attention-backend dsa_sparse`` the pool carries a second buffer list,
  ``indexer_key_buffer``, which must round-trip too or a restored prefix has
  zero indexer keys.
"""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.host_kv_pool import LRUHostKVPool
from sgl_jax.srt.mem_cache.memory_pool import MLATokenToKVPool
from sgl_jax.srt.utils.mesh_utils import create_device_mesh

_MESH = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

# GLM-5 / DeepSeek-V3 latent dims.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
INDEX_HEAD_DIM = 128


def _make_mla_pool(
    *,
    size=16 * 128,
    page_size=128,
    layer_num=3,
    dtype=jnp.bfloat16,
    indexer_key_dim=0,
    num_indexer_layers=0,
):
    jax.sharding.set_mesh(_MESH)
    return MLATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=dtype,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        layer_num=layer_num,
        mesh=_MESH,
        dp_size=1,
        indexer_key_dim=indexer_key_dim,
        num_indexer_layers=num_indexer_layers,
    )


def _make_host_pool(device_pool, *, pool_size=8):
    return LRUHostKVPool(
        device_pool=device_pool,
        pool_size=pool_size,
        page_size=device_pool.page_size,
        layer_num=device_pool.layer_num,
        per_layer_shape=tuple(int(d) for d in device_pool.kv_buffer[0].shape[1:]),
        dtype=device_pool.dtype,
        mesh=_MESH,
        partition_spec=device_pool.kv_sharding.spec,
    )


def _fill_page(device_pool, page, seed):
    """Write random data into device page ``page`` of every layer; return copies."""
    orig = []
    for layer in range(device_pool.layer_num):
        buf = device_pool.kv_buffer[layer]
        vals = jax.random.normal(
            jax.random.PRNGKey(seed * 100 + layer), buf[page].shape, jnp.float32
        ).astype(buf.dtype)
        device_pool.kv_buffer[layer] = buf.at[page].set(vals, out_sharding=buf.sharding)
        orig.append(np.asarray(device_pool.kv_buffer[layer][page]))
    idx_orig = []
    for slot in range(len(device_pool.indexer_key_buffer)):
        buf = device_pool.indexer_key_buffer[slot]
        vals = jax.random.normal(
            jax.random.PRNGKey(seed * 1000 + slot), buf[page].shape, jnp.float32
        ).astype(buf.dtype)
        device_pool.indexer_key_buffer[slot] = buf.at[page].set(vals, out_sharding=buf.sharding)
        idx_orig.append(np.asarray(device_pool.indexer_key_buffer[slot][page]))
    return orig, idx_orig


class TestMLALayoutAssumptions(unittest.TestCase):
    """Pin down the layout facts the scatter must respect."""

    def test_bf16_page_is_token_grid_not_token_axis(self):
        pool = _make_mla_pool(dtype=jnp.bfloat16)
        shape = tuple(pool.kv_buffer[0].shape)
        # (pages, 128//2, 2, 640): dim 1 is NOT page_size.
        self.assertEqual(shape[1:], (64, 2, 640))
        self.assertNotEqual(shape[1], pool.page_size)

    def test_pool_has_no_mha_sharding_attrs(self):
        pool = _make_mla_pool()
        # flush_load reads these on the MHA pool; MLA must not be driven through them.
        self.assertFalse(hasattr(pool, "attention_data_partition_axis"))
        self.assertEqual(pool.kv_partition_axis, "data")


class TestMLAHostRoundTrip(unittest.TestCase):
    DTYPE = jnp.bfloat16

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.device_pool = _make_mla_pool(dtype=self.DTYPE, layer_num=3)
        self.host_pool = _make_host_pool(self.device_pool)

    def _assert_page_equal(self, page, orig):
        for layer in range(self.device_pool.layer_num):
            got = np.asarray(self.device_pool.kv_buffer[layer][page])
            np.testing.assert_array_equal(got, orig[layer], err_msg=f"layer {layer}")

    def test_single_page_roundtrip(self):
        src, dst = 3, 9
        orig, _ = _fill_page(self.device_pool, src, seed=1)
        slot = self.host_pool.reserve()
        self.host_pool.copy_into([src], [slot])  # D2H (works today)
        self.host_pool.copy_to_device([slot], [dst])  # H2D (MLA-unaware today)
        self._assert_page_equal(dst, orig)

    def test_batched_roundtrip_preserves_other_pages(self):
        pairs = [(1, 10), (2, 11), (4, 12)]
        origs = {src: _fill_page(self.device_pool, src, seed=src)[0] for src, _ in pairs}
        untouched = 13
        before = [np.asarray(b[untouched]) for b in self.device_pool.kv_buffer]
        slots = [self.host_pool.reserve() for _ in pairs]
        self.host_pool.copy_into([s for s, _ in pairs], slots)
        self.host_pool.copy_to_device(slots, [d for _, d in pairs])
        for (src, dst), _ in zip(pairs, slots):
            self._assert_page_equal(dst, origs[src])
        for layer, b in enumerate(self.device_pool.kv_buffer):
            np.testing.assert_array_equal(np.asarray(b[untouched]), before[layer])


class TestMLAHostRoundTripFP8(TestMLAHostRoundTrip):
    """fp8 packs 4 tokens per word -> page is (32, 4, 640)."""

    DTYPE = jnp.float8_e4m3fn


class TestMLAWithIndexerRoundTrip(unittest.TestCase):
    """dsa_sparse: the indexer-key buffers must survive the host round-trip too."""

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")
        self.device_pool = _make_mla_pool(
            dtype=jnp.bfloat16,
            layer_num=3,
            indexer_key_dim=INDEX_HEAD_DIM,
            num_indexer_layers=2,
        )
        self.assertEqual(len(self.device_pool.indexer_key_buffer), 2)
        self.host_pool = _make_host_pool(self.device_pool)

    def test_indexer_keys_roundtrip(self):
        src, dst = 2, 7
        orig, idx_orig = _fill_page(self.device_pool, src, seed=5)
        slot = self.host_pool.reserve()
        self.host_pool.copy_into([src], [slot])
        self.host_pool.copy_to_device([slot], [dst])
        for layer in range(self.device_pool.layer_num):
            np.testing.assert_array_equal(
                np.asarray(self.device_pool.kv_buffer[layer][dst]), orig[layer]
            )
        for s in range(len(self.device_pool.indexer_key_buffer)):
            np.testing.assert_array_equal(
                np.asarray(self.device_pool.indexer_key_buffer[s][dst]),
                idx_orig[s],
                err_msg=f"indexer slot {s} not restored",
            )


if __name__ == "__main__":
    unittest.main()
