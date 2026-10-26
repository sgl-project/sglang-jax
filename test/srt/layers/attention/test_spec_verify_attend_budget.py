"""Spec verify attend page budget (10-26 dp8 p64 regression).

A decode-form TARGET_VERIFY batch hands the attend kernel T = S x G one-query
pseudo-sequences over the S requests' packed page table, so ``len(page_table) // T``
is G times smaller than a request's page span. Sizing the attend page list from it
cut every verify attend to budget / G pages, and when the batched indexer took the
identity shortcut (ascending page ids, allowed when the request span equals the
budget) the kept window was the OLDEST pages: dp8 + MTP GSM p64 0.175 with answers
copied from the few-shot examples. Two checks:

* ``_attend_k_pages_max`` takes the per-REQUEST span (budget clamp + 1).
* the identity shortcut is only taken when the attend budget keeps every page;
  otherwise the grouped path must return score-ordered pages so that truncating
  the list keeps the same top pages as the per-request reference.
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=2"
)

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsb
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata

PAGE, PACK, D, H, G = 8, 2, 128, 4, 4
PPS = 32  # request page span == DSA_PAGE_TOPK_IDENTITY_MAX_PAGES default == index_topk pages
S = 8  # >= _IDX_VERIFY_BATCHED_MIN_BS


def _backend(mesh):
    return dsb.DSASparseAttentionBackend(
        index_topk=PPS * PAGE,
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
    T = S * G
    seq_lens = (np.asarray(prefix) + G).astype(np.int32)
    cu_q = (np.arange(S + 1) * G).astype(np.int32)
    aligned = ((seq_lens + PAGE - 1) // PAGE) * PAGE
    cu_kv = np.concatenate([[0], np.cumsum(aligned)]).astype(np.int32)  # packed layout
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


def _sets(a, keep):
    return [sorted(int(x) for x in row[:keep] if x >= 0) for row in np.asarray(a)]


class SpecVerifyAttendBudgetTest(unittest.TestCase):
    def test_attend_k_pages_max_uses_request_span(self):
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 32):
            self.assertEqual(dsb._attend_k_pages_max(32), 33)  # verify: request span 32
            self.assertEqual(dsb._attend_k_pages_max(128), 33)  # long context: budget clamp
            self.assertEqual(dsb._attend_k_pages_max(8), 9)  # short request
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 0):
            self.assertEqual(dsb._attend_k_pages_max(32), 512)

    def test_identity_only_when_budget_covers_every_page(self):
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 32):
            self.assertTrue(dsb._identity_covers_budget(32))
            self.assertFalse(dsb._identity_covers_budget(33))
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 8):
            self.assertFalse(dsb._identity_covers_budget(32))
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 0):
            self.assertTrue(dsb._identity_covers_budget(4096))

    def _pages(self, tp, md, cache, k_idx, q_idx, w, **patch):
        devs = np.array(jax.devices()[:tp]).reshape(1, tp)
        mesh = Mesh(devs, ("data", "tensor"))
        with mock.patch.multiple(dsb, _INDEXER_KERNEL_PREFILL=False, **patch), jax.set_mesh(mesh):
            _, pages = _backend(mesh)._maybe_index_prefill_pages(
                True, q_idx, k_idx, w, cache, "data", md, q_group=G
            )
        return np.asarray(pages)

    def test_grouped_window_matches_reference_when_budget_truncates(self):
        # budget 8 < request span 32: the identity shortcut would list pages 0..31 and the
        # attend wrapper keeps the first 8 (= oldest). The grouped path must fall back to
        # scoring so the kept window is the reference's top-8.
        rng = np.random.default_rng(1)
        prefix = list(rng.integers(low=int(PPS * PAGE * 0.62), high=int(PPS * PAGE * 0.9), size=S))
        md, cache, k_idx, q_idx, w = _verify_batch(rng, prefix)
        ref = self._pages(1, md, cache, k_idx, q_idx, w, _IDX_VERIFY_BATCHED=False)
        grp = self._pages(
            2,
            md,
            cache,
            k_idx,
            q_idx,
            w,
            _IDX_VERIFY_BATCHED=True,
            _IDX_VERIFY_BATCHED_MIN_BS=1,
            _PAGE_TOPK_IDENTITY_MAX_PAGES=32,
            _PAGE_TOPK_BUDGET=8,
        )
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 8):
            keep = dsb._attend_k_pages_max(PPS) - 1
        self.assertEqual(keep, 8)
        self.assertEqual(_sets(grp, keep), _sets(ref, keep))
        oldest = [list(range(keep))] * (S * G)
        self.assertNotEqual([list(int(x) for x in r[:keep]) for r in grp], oldest)

    def test_grouped_identity_kept_when_budget_covers(self):
        # budget 32 == request span: identity is fine (every page kept) and must still
        # select the same page SET as the reference.
        rng = np.random.default_rng(2)
        prefix = list(rng.integers(low=int(PPS * PAGE * 0.62), high=int(PPS * PAGE * 0.9), size=S))
        md, cache, k_idx, q_idx, w = _verify_batch(rng, prefix)
        ref = self._pages(1, md, cache, k_idx, q_idx, w, _IDX_VERIFY_BATCHED=False)
        grp = self._pages(
            2,
            md,
            cache,
            k_idx,
            q_idx,
            w,
            _IDX_VERIFY_BATCHED=True,
            _IDX_VERIFY_BATCHED_MIN_BS=1,
            _PAGE_TOPK_IDENTITY_MAX_PAGES=32,
            _PAGE_TOPK_BUDGET=32,
        )
        with mock.patch.object(dsb, "_PAGE_TOPK_BUDGET", 32):
            keep = dsb._attend_k_pages_max(PPS) - 1
        self.assertEqual(keep, 32)
        self.assertEqual(_sets(grp, keep), _sets(ref, keep))


if __name__ == "__main__":
    unittest.main()
