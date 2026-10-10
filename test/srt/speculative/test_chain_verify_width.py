"""_build_chain_verify_arrays must accept a token chain narrower than num_verify_tokens - 1
(width-1 bootstrap seed on MTP + return_logprob requests crashed gp79/gp80 with
'cannot reshape array of shape (1, 2) (size 2) into shape (4,)')."""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.speculative.draft_extend_fused import _build_chain_verify_arrays


class ChainVerifyWidthTest(unittest.TestCase):
    def _run(self, width, bs=2, n=4):
        verified = jnp.arange(bs, dtype=jnp.int32) + 100
        chain = jnp.arange(bs * width, dtype=jnp.int32).reshape(bs, width) + 1
        seq_lens = jnp.array([10, 20][:bs], dtype=jnp.int32)
        return _build_chain_verify_arrays(
            verified_id=verified,
            token_list=chain,
            seq_lens=seq_lens,
            num_verify_tokens=n,
            batch_size=bs,
        )

    def test_full_width_unchanged(self):
        toks, pos, ri, rnt, rns = self._run(width=3)
        np.testing.assert_array_equal(np.asarray(toks), [100, 1, 2, 3, 101, 4, 5, 6])
        np.testing.assert_array_equal(np.asarray(pos), [10, 11, 12, 13, 20, 21, 22, 23])
        np.testing.assert_array_equal(np.asarray(rnt).reshape(2, 4)[0], [1, 2, 3, -1])

    def test_width_one_bootstrap_is_padded_not_crashed(self):
        toks, pos, ri, rnt, rns = self._run(width=1)
        np.testing.assert_array_equal(np.asarray(toks), [100, 1, 1, 1, 101, 2, 2, 2])
        self.assertEqual(np.asarray(pos).shape, (8,))

    def test_wider_than_needed_is_sliced(self):
        toks, *_ = self._run(width=5)
        np.testing.assert_array_equal(np.asarray(toks), [100, 1, 2, 3, 101, 6, 7, 8])

    def test_bs1_width1(self):
        toks, *_ = self._run(width=1, bs=1)
        np.testing.assert_array_equal(np.asarray(toks), [100, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
