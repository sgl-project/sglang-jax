"""Spec-verify page-topk under tensor sharding (#1811 review, CPU 4-device mesh).

``_maybe_index_prefill_pages`` shards the query rows over the tensor axis when ``T % tp == 0``.
With the batched grouped page-topk (``DSA_IDX_VERIFY_BATCHED=1``, ``q_group`` = draft tokens
per request) a rank whose window holds only part of a verify group sees ``q_len != q_group``
and returns -1 for every page (tp16, 8 requests x 4 tokens: 2 rows per rank). The rows may
only be sharded when every rank holds whole groups; otherwise the selection stays replicated
and must equal the single-device result.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    ).strip()

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsb
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata

PAGE, PACK, D, H = 8, 2, 128, 4
G = 4  # draft tokens per request
PPS = 4


def _backend(mesh):
    return dsb.DSASparseAttentionBackend(
        index_topk=16,
        index_head_dim=D,
        index_n_heads=H,
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


def _verify_batch(rng, prefix):
    S = len(prefix)
    T = S * G
    seq_lens = (np.asarray(prefix) + G).astype(np.int32)
    cu_q = (np.arange(S + 1) * G).astype(np.int32)
    cu_kv = (np.arange(S + 1) * PPS * PAGE).astype(np.int32)
    page_indices = rng.permutation(np.arange(1, S * PPS + 1)).astype(np.int32)
    md = MLAAttentionMetadata()
    md.cu_q_lens = jnp.asarray(cu_q)
    md.cu_kv_lens = jnp.asarray(cu_kv)
    md.page_indices = jnp.asarray(page_indices)
    md.seq_lens = jnp.asarray(seq_lens)
    md.distribution = jnp.asarray([0, 0, S], jnp.int32)
    cache = jnp.asarray(
        rng.standard_normal((S * PPS + 2, PAGE // PACK, PACK, D)).astype(np.float32)
    ).astype(jnp.bfloat16)
    k_idx = jnp.asarray(rng.standard_normal((T, D)).astype(np.float32)).astype(jnp.bfloat16)
    q_idx = jnp.asarray(rng.standard_normal((T, H, D)).astype(np.float32))
    w = jnp.asarray(rng.random((T, H)).astype(np.float32))
    return md, cache, k_idx, q_idx, w


def _sets(a):
    return [sorted(int(x) for x in row if x >= 0) for row in np.asarray(a)]


class IdxPrefillPagesGroupShardTest(unittest.TestCase):
    def _pages(self, tp, md, cache, k_idx, q_idx, w, dp=1):
        devs = np.array(jax.devices()[: dp * tp]).reshape(dp, tp)
        mesh = Mesh(devs, ("data", "tensor"))
        backend = _backend(mesh)
        with (
            mock.patch.multiple(
                dsb,
                _IDX_VERIFY_BATCHED=True,
                _IDX_VERIFY_BATCHED_MIN_BS=1,
                _INDEXER_KERNEL_PREFILL=False,
                _PAGE_TOPK_IDENTITY_MAX_PAGES=-1,  # force real scoring
            ),
            jax.set_mesh(mesh),
        ):
            out, pages = backend._maybe_index_prefill_pages(
                True, q_idx, k_idx, w, cache, "data", md, q_group=G
            )
        return np.asarray(out), np.asarray(pages)

    def test_partial_group_per_rank_stays_replicated(self):
        # 2 requests x 4 tokens on tp=4: 2 rows per rank would split every group
        rng = np.random.default_rng(5)
        md, cache, k_idx, q_idx, w = _verify_batch(rng, prefix=[9, 20])
        ref_cache, ref_pages = self._pages(1, md, cache, k_idx, q_idx, w)
        got_cache, got_pages = self._pages(4, md, cache, k_idx, q_idx, w)
        self.assertEqual(got_pages.shape, ref_pages.shape)
        self.assertEqual(_sets(got_pages), _sets(ref_pages))
        # every valid query row selected at least one page (the pre-fix result was all -1)
        self.assertTrue(all(len(s) > 0 for s in _sets(got_pages)))
        np.testing.assert_array_equal(got_cache, ref_cache)

    def test_whole_groups_per_rank_still_shard(self):
        # 4 requests x 4 tokens on tp=4: one whole group per rank, sharded path kept
        rng = np.random.default_rng(6)
        md, cache, k_idx, q_idx, w = _verify_batch(rng, prefix=[9, 20, 3, 14])
        ref_cache, ref_pages = self._pages(1, md, cache, k_idx, q_idx, w)
        got_cache, got_pages = self._pages(4, md, cache, k_idx, q_idx, w)
        self.assertEqual(_sets(got_pages), _sets(ref_pages))
        self.assertTrue(all(len(s) > 0 for s in _sets(got_pages)))
        np.testing.assert_array_equal(got_cache, ref_cache)

    def test_dp_local_rows_decide_sharding(self):
        # review: DP=2, TP=2, one 4-token request per DP rank. The global row count (8)
        # divided by tp is a whole group, but each DP rank only holds 4 rows, which the
        # tensor axis would split into half-groups (every page -1). Must stay replicated
        # inside each DP rank and match that rank's single-device page sets.
        rng = np.random.default_rng(7)
        ranks = [_verify_batch(rng, prefix=[9]), _verify_batch(rng, prefix=[20])]
        refs = [self._pages(1, *b)[1] for b in ranks]
        # DP layout = per-rank arrays concatenated (each P("data") shard is one rank's
        # self-consistent metadata, page ids local to that rank's cache slice).
        md = MLAAttentionMetadata()
        for f in ("cu_q_lens", "cu_kv_lens", "page_indices", "seq_lens", "distribution"):
            setattr(md, f, jnp.concatenate([getattr(b[0], f) for b in ranks]))
        cache = jnp.concatenate([b[1] for b in ranks])
        k_idx = jnp.concatenate([b[2] for b in ranks])
        q_idx = jnp.concatenate([b[3] for b in ranks])
        w = jnp.concatenate([b[4] for b in ranks])
        _, got_pages = self._pages(2, md, cache, k_idx, q_idx, w, dp=2)
        ref_pages = np.concatenate(refs)
        self.assertEqual(got_pages.shape, ref_pages.shape)
        self.assertEqual(_sets(got_pages), _sets(ref_pages))
        self.assertTrue(all(len(s) > 0 for s in _sets(got_pages)))


if __name__ == "__main__":
    unittest.main()
