"""CPU regression for the NextN (MTP) draft pool write-back on the DSA path (#1639 review).

The NextN block writes two page buffers per call, latent KV and indexer keys. The
draft worker hands the model's return value to ``MemoryPools.replace_all``, and the MLA
pool only updates ``indexer_key_buffer`` when it receives the ``(kv, idx)`` tuple. A
bare KV list therefore loses every indexer-key write of the draft model, so the next
step-0 call scores pages against keys that were never stored. These tests pin the
payload format and the property that matters: indexer keys written by one call are
visible to the next.
"""

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.mem_cache.memory_pool import MemoryPools, MLATokenToKVPool
from sgl_jax.srt.models.glm5_moe import nextn_pool_updates
from sgl_jax.test.test_utils import CustomTestCase

PAGE_SIZE = 64
IDX_DIM = 128


def _mesh():
    return jax.sharding.Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))


def _pool():
    return MLATokenToKVPool(
        size=PAGE_SIZE * 4,
        page_size=PAGE_SIZE,
        dtype=jnp.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=1,
        mesh=_mesh(),
        indexer_key_dim=IDX_DIM,
        num_indexer_layers=1,
    )


def _draft_call(pool, slot, value):
    """One NextN forward on a DSA layer: both buffers come back updated (functional
    scatter of ``value`` into row ``slot``); the model returns the pool payload."""
    kv = pool.kv_buffer[0]
    idx = pool.get_indexer_key_buffer(0)
    kv = kv.at[slot].set(jnp.full(kv.shape[1:], value, kv.dtype))
    idx = idx.at[slot].set(jnp.full(idx.shape[1:], value, idx.dtype))
    return nextn_pool_updates(SimpleNamespace(kv=kv, idx=idx, topk_pages=None), is_dsa=True)


class TestNextNIndexerPoolUpdate(CustomTestCase):
    def test_payload_matches_the_target_format(self):
        pool = _pool()
        payload = _draft_call(pool, slot=1, value=1)
        self.assertEqual(set(payload), {"token_to_kv_pool"})
        kv_list, idx_list = payload["token_to_kv_pool"]
        self.assertEqual((len(kv_list), len(idx_list)), (1, 1))
        self.assertEqual(idx_list[0].shape, pool.get_indexer_key_buffer(0).shape)

    def test_non_dsa_payload_is_the_plain_list(self):
        kv = jnp.zeros((4, 8), jnp.bfloat16)
        self.assertEqual(nextn_pool_updates(kv, is_dsa=False), [kv])

    def test_shared_layer_without_indexer_keeps_kv_only(self):
        kv = jnp.zeros((4, 8), jnp.bfloat16)
        payload = nextn_pool_updates(SimpleNamespace(kv=kv, idx=None), is_dsa=True)
        self.assertEqual(payload, {"token_to_kv_pool": [kv]})

    def test_indexer_keys_survive_across_calls(self):
        """Keys written at step 0 of round r must still be in the pool at step 0 of
        round r+1, which is exactly what the bare-list form broke."""
        pool = _pool()
        pools = MemoryPools(token_to_kv_pool=pool)
        pools.replace_all(_draft_call(pool, slot=1, value=1))
        pools.replace_all(_draft_call(pool, slot=2, value=2))
        idx = np.asarray(pool.get_indexer_key_buffer(0).astype(jnp.float32))
        kv = np.asarray(pool.kv_buffer[0].astype(jnp.float32))
        self.assertTrue((idx[1] == 1).all() and (idx[2] == 2).all(), "indexer keys lost")
        self.assertTrue((kv[1] == 1).all() and (kv[2] == 2).all())
        self.assertTrue((idx[0] == 0).all() and (idx[3] == 0).all())

    def test_bare_kv_list_drops_the_indexer_write(self):
        """Documents the defect the format fixes: the list form updates KV only."""
        pool = _pool()
        pools = MemoryPools(token_to_kv_pool=pool)
        payload = _draft_call(pool, slot=1, value=1)
        kv_list, _ = payload["token_to_kv_pool"]
        pools.replace_all(kv_list)
        idx = np.asarray(pool.get_indexer_key_buffer(0).astype(jnp.float32))
        kv = np.asarray(pool.kv_buffer[0].astype(jnp.float32))
        self.assertTrue((kv[1] == 1).all())
        self.assertTrue((idx[1] == 0).all())


if __name__ == "__main__":
    unittest.main()
