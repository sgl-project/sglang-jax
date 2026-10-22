"""Prefill micro-batching (DSA_PREFILL_MICROBATCH): the per-window attention
metadata must place every window row at the same absolute position / cache slot
as the single full-chunk forward does, for windows that cut through requests
(chunked-prefill continuation form) and windows that leave requests with zero
query rows on either side."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention import dsa_sparse_backend as dsb
from sgl_jax.srt.layers.attention.mla_backend import MLAAttentionMetadata

PAGE = 8


def _batch(rng, q_lens, prefix_lens, t_pad, pages_per_seq):
    """Ragged EXTEND metadata the way MLAAttentionBackend.get_forward_metadata lays it out."""
    S = len(q_lens)
    seq_lens = np.array(prefix_lens) + np.array(q_lens)
    cu_q = np.zeros(S + 1, np.int32)
    cu_q[1:] = np.cumsum(q_lens)
    aligned = (seq_lens + PAGE - 1) // PAGE * PAGE
    cu_kv = np.zeros(S + 1, np.int32)
    cu_kv[1:] = np.cumsum(aligned)
    page_indices = rng.permutation(S * pages_per_seq + 1)[: S * pages_per_seq] + 1
    md = MLAAttentionMetadata()
    md.cu_q_lens = jnp.asarray(cu_q)
    md.cu_kv_lens = jnp.asarray(cu_kv)
    md.page_indices = jnp.asarray(page_indices, jnp.int32)
    md.seq_lens = jnp.asarray(seq_lens, jnp.int32)
    md.distribution = jnp.asarray([0, 0, int((seq_lens > 0).sum())], jnp.int32)
    assert cu_q[-1] <= t_pad
    return md


def _landing(md, T):
    page, off, valid = dsb._paged_landing(
        T, PAGE, md.seq_lens, md.page_indices, md.cu_q_lens, md.cu_kv_lens
    )
    return np.asarray(page), np.asarray(off), np.asarray(valid)


class WindowForwardMetadataTest(unittest.TestCase):
    def _check(self, md, t_pad, split):
        full = _landing(md, t_pad)
        md_a = dsb.window_forward_metadata(md, 0, split)
        md_b = dsb.window_forward_metadata(md, split, t_pad - split)
        a = _landing(md_a, split)
        b = _landing(md_b, t_pad - split)
        # valid mask must match row for row; page / offset only matter for
        # valid rows (the writers route invalid rows to the sentinel page)
        np.testing.assert_array_equal(a[2], full[2][:split])
        np.testing.assert_array_equal(b[2], full[2][split:])
        va, vb = full[2][:split], full[2][split:]
        for k in range(2):
            np.testing.assert_array_equal(a[k][va], full[k][:split][va])
            np.testing.assert_array_equal(b[k][vb], full[k][split:][vb])
        # untouched fields: page table layout and distribution
        for w in (md_a, md_b):
            np.testing.assert_array_equal(np.asarray(w.cu_kv_lens), np.asarray(md.cu_kv_lens))
            np.testing.assert_array_equal(np.asarray(w.page_indices), np.asarray(md.page_indices))
            np.testing.assert_array_equal(np.asarray(w.distribution), np.asarray(md.distribution))
            cuq = np.asarray(w.cu_q_lens)
            self.assertTrue((np.diff(cuq) >= 0).all())
            self.assertEqual(cuq[0], 0)
        self.assertLessEqual(int(np.asarray(md_a.cu_q_lens)[-1]), split)
        self.assertLessEqual(int(np.asarray(md_b.cu_q_lens)[-1]), t_pad - split)

    def test_single_request_cut_in_half(self):
        rng = np.random.default_rng(0)
        md = _batch(rng, [250], [0], 256, pages_per_seq=64)
        self._check(md, 256, 128)

    def test_continuation_chunk_cut_in_half(self):
        rng = np.random.default_rng(1)
        md = _batch(rng, [256], [1000], 256, pages_per_seq=256)
        self._check(md, 256, 128)

    def test_multi_request_cuts(self):
        rng = np.random.default_rng(2)
        q_lens = [100, 60, 300, 52]  # 512 total, no padding
        md = _batch(rng, q_lens, [0, 40, 0, 7], 512, pages_per_seq=64)
        for split in (64, 128, 160, 256, 384, 448):
            with self.subTest(split=split):
                self._check(md, 512, split)

    def test_padded_tail_and_zero_seqs(self):
        rng = np.random.default_rng(3)
        q_lens = [100, 60, 0, 0]  # padded batch slots with no tokens
        md = _batch(rng, q_lens, [0, 40, 0, 0], 256, pages_per_seq=64)
        for split in (64, 128, 192):
            with self.subTest(split=split):
                self._check(md, 256, split)

    def test_window_q_pos_matches_absolute_position(self):
        rng = np.random.default_rng(4)
        q_lens = [100, 60, 300, 52]
        prefix = [0, 40, 0, 7]
        md = _batch(rng, q_lens, prefix, 512, pages_per_seq=64)
        split = 160
        md_b = dsb.window_forward_metadata(md, split, 512 - split)
        cuq = np.asarray(md_b.cu_q_lens)
        sl = np.asarray(md_b.seq_lens)
        # request 2 (300 q rows, starts at 160) is fully inside window b: first row at abs pos 0
        self.assertEqual(sl[2] - (cuq[3] - cuq[2]), 0)
        # request 1 (60 rows at 100..160) has no rows in b -> zero q len, kv len kept
        self.assertEqual(cuq[2] - cuq[1], 0)
        self.assertEqual(sl[1], 100)
        md_a = dsb.window_forward_metadata(md, 0, split)
        cuq = np.asarray(md_a.cu_q_lens)
        sl = np.asarray(md_a.seq_lens)
        # request 2 has no rows in a: kv len shrinks to its prefix (0)
        self.assertEqual(cuq[3] - cuq[2], 0)
        self.assertEqual(sl[2], 0)
        # request 3 has no rows in a: kv len shrinks to its prefix (7)
        self.assertEqual(sl[3], 7)


if __name__ == "__main__":
    unittest.main()
