"""CPU: opt-in batched (grouped) page-topk dispatch for the spec verify indexer.

``_prefill_page_topk`` is the page-topk selector inside the prefill-form indexer
shard_map; with ``DSA_IDX_VERIFY_BATCHED=1`` and a verify batch (``q_group`` =
draft tokens per request) it must route to ``streamindex_page_topk_ref_grouped``
and return the same per-token page sets (padded to the ``k_pages`` contract).
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsb
from sgl_jax.srt.layers.attention.dsa_sparse_backend import (
    _prefill_page_topk,
    _spec_draft_group,
)

PAGE = 4
D = 8
H = 2
G = 4


def _meta(base_lens, pps):
    S = len(base_lens)
    valid = np.asarray(base_lens) > 0
    ext = np.where(valid, G, 0).astype(np.int32)
    seq_lens = np.where(valid, np.asarray(base_lens) + G, 0).astype(np.int32)
    cu_q = np.concatenate([[0], np.cumsum(ext)]).astype(np.int32)
    cu_kv = (np.arange(S + 1) * pps * PAGE).astype(np.int32)
    pi = (1 + np.arange(S * pps)).astype(np.int32)
    dist = np.array([0, 0, int(valid.sum())], np.int32)
    return [jnp.asarray(x) for x in (seq_lens, pi, cu_q, cu_kv, dist)]


def _sets(a):
    return [sorted(int(x) for x in row if x >= 0) for row in np.asarray(a)]


class VerifyIdxBatchedDispatchTest(unittest.TestCase):
    def _case(self, base_lens, pps, k_pages, q_group, min_bs=1, max_kv=1 << 20, on=True):
        S = len(base_lens)
        T = S * G
        seq_lens, pi, cu_q, cu_kv, dist = _meta(base_lens, pps)
        rng = np.random.default_rng(1)
        q = jnp.asarray(rng.standard_normal((T, H, D)).astype(np.float32))
        w = jnp.asarray(rng.random((T, H)).astype(np.float32))
        cache = jnp.asarray(rng.standard_normal((1 + S * pps, PAGE, D)).astype(np.float32))
        args = (q, w, cache, seq_lens, pi, cu_q, cu_kv, dist)
        kw = dict(k_pages=k_pages, pages_per_seq=pps, num_queries_per_block=None)
        calls = []
        real = dsb.streamindex_page_topk_ref_grouped

        def spy(*a, **k):
            calls.append(k.get("q_group"))
            return real(*a, **k)

        with mock.patch.multiple(
            dsb,
            _IDX_VERIFY_BATCHED=on,
            _IDX_VERIFY_BATCHED_MIN_BS=min_bs,
            _IDX_VERIFY_BATCHED_MAX_KV=max_kv,
            _INDEXER_KERNEL_PREFILL=False,
            streamindex_page_topk_ref_grouped=spy,
        ):
            got = _prefill_page_topk(*args, q_group=q_group, **kw)
            want = _prefill_page_topk(*args, q_group=None, **kw)
        self.assertEqual(got.shape, (T, k_pages))
        self.assertEqual(want.shape, (T, k_pages))
        self.assertEqual(_sets(got), _sets(want))
        return calls

    def test_grouped_path_used_and_matches_ref(self):
        calls = self._case([3, 9, 0, 6], pps=4, k_pages=2, q_group=G)
        self.assertEqual(calls, [G])

    def test_k_pages_above_table_width_is_clamped_and_padded(self):
        calls = self._case([5, 2], pps=3, k_pages=8, q_group=G)
        self.assertEqual(calls, [G])

    def test_not_used_without_q_group(self):
        calls = self._case([3, 9], pps=4, k_pages=2, q_group=None)
        self.assertEqual(calls, [])

    def test_not_used_below_min_bs_or_above_max_kv_or_when_off(self):
        self.assertEqual(self._case([3, 9], pps=4, k_pages=2, q_group=G, min_bs=8), [])
        self.assertEqual(self._case([3, 9], pps=4, k_pages=2, q_group=G, max_kv=PAGE * 3), [])
        self.assertEqual(self._case([3, 9], pps=4, k_pages=2, q_group=G, on=False), [])


class SpecDraftGroupTest(unittest.TestCase):
    def _fb(self, verify, dtn):
        mode = SimpleNamespace(is_target_verify=lambda: verify)
        info = None if dtn is None else SimpleNamespace(draft_token_num=dtn)
        return SimpleNamespace(forward_mode=mode, spec_info=info)

    def test_returns_draft_token_num_for_divisible_verify_batch(self):
        self.assertEqual(_spec_draft_group(self._fb(True, 4), 64), 4)

    def test_none_when_not_verify_or_not_divisible_or_single_token(self):
        self.assertIsNone(_spec_draft_group(self._fb(False, 4), 64))
        self.assertIsNone(_spec_draft_group(self._fb(True, 4), 66))
        self.assertIsNone(_spec_draft_group(self._fb(True, 1), 64))
        self.assertIsNone(_spec_draft_group(self._fb(True, None), 64))


if __name__ == "__main__":
    unittest.main()


class VerifyQgroupGateTest(unittest.TestCase):
    """Grouped verify attention needs the opt-in, a decode-form verify batch and
    at least DSA_SPEC_VERIFY_QGROUP_MIN_BS requests (bs1 gains nothing)."""

    def test_gate(self):
        import sgl_jax.srt.layers.attention.dsa_sparse_backend as be

        with (
            mock.patch.object(be, "_SPEC_VERIFY_QGROUP", True),
            mock.patch.object(be, "_SPEC_VERIFY_QGROUP_MIN_BS", 8),
        ):
            self.assertTrue(be._use_verify_qgroup(True, 4, 64 * 4))
            self.assertTrue(be._use_verify_qgroup(True, 4, 8 * 4))
            self.assertFalse(be._use_verify_qgroup(True, 4, 7 * 4))
            self.assertFalse(be._use_verify_qgroup(True, 4, 4))  # bs1
            self.assertFalse(be._use_verify_qgroup(False, 4, 64 * 4))
            self.assertFalse(be._use_verify_qgroup(True, None, 64 * 4))
        with mock.patch.object(be, "_SPEC_VERIFY_QGROUP", False):
            self.assertFalse(be._use_verify_qgroup(True, 4, 64 * 4))
