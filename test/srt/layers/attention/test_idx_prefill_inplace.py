"""The prefill indexer-key cache write through the 4D in-place run-table scatter
(``compute_pages=False`` branch of ``_maybe_index_prefill_pages``, CPU, 1-device mesh) must
leave live pages bitwise identical to the 3D-view XLA scatter reference (``_scatter_paged``)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsb
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata

PAGE, PACK, D = 8, 2, 128


def _backend(mesh):
    return dsb.DSASparseAttentionBackend(
        index_topk=16,
        index_head_dim=D,
        index_n_heads=4,
        skip_offset=0,
        full_slot={0: 0},
        num_attn_heads=4,
        kv_lora_rank=64,
        qk_nope_head_dim=32,
        qk_rope_head_dim=16,
        v_head_dim=32,
        page_size=PAGE,
        mesh=mesh,
        attention_data_partition_axis="data",
        vmem_limit_bytes=16 << 20,
        num_kv_pages_per_block=4,
        num_queries_per_block=8,
        decode_batch_size=8,
    )


class IdxPrefillInplaceTest(unittest.TestCase):
    def test_live_pages_bitwise_equal(self):
        rng = np.random.default_rng(0)
        mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "tensor"))
        backend = _backend(mesh)
        S, PPS = 3, 4
        q_lens, prefix = [20, 9, 0], [5, 0, 0]
        T = 32
        seq_lens = np.array(prefix) + np.array(q_lens)
        cu_q = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        cu_kv = np.concatenate([[0], np.cumsum([PPS * PAGE] * S)]).astype(np.int32)
        page_indices = rng.permutation(np.arange(1, S * PPS + 1))[: S * PPS].astype(np.int32)
        md = MLAAttentionMetadata()
        md.cu_q_lens = jnp.asarray(cu_q)
        md.cu_kv_lens = jnp.asarray(cu_kv)
        md.page_indices = jnp.asarray(page_indices)
        md.seq_lens = jnp.asarray(seq_lens.astype(np.int32))
        md.distribution = jnp.asarray([0, 0, 2], jnp.int32)
        cache = jnp.asarray(
            rng.standard_normal((S * PPS + 2, PAGE // PACK, PACK, D)).astype(np.float32)
        ).astype(jnp.bfloat16)
        k_idx = jnp.asarray(rng.standard_normal((T, D)).astype(np.float32)).astype(jnp.bfloat16)
        q_idx = jnp.zeros((T, 4, D), jnp.float32)
        w = jnp.zeros((T, 4), jnp.float32)
        with jax.set_mesh(mesh):
            out, pages = backend._maybe_index_prefill_pages(
                True, q_idx, k_idx, w, cache, "data", md, compute_pages=False
            )
        self.assertIsNone(pages)
        got = np.asarray(out)
        ref = np.asarray(
            dsb._scatter_paged(
                cache.reshape(cache.shape[0], PAGE, D),
                k_idx,
                md.seq_lens,
                md.page_indices,
                md.cu_q_lens,
                md.cu_kv_lens,
                PPS,
            ).reshape(cache.shape)
        )
        live = np.asarray(page_indices)
        np.testing.assert_array_equal(got[live], ref[live])
        # something was actually written
        self.assertFalse(np.array_equal(got[live], np.asarray(cache)[live]))


if __name__ == "__main__":
    unittest.main()
