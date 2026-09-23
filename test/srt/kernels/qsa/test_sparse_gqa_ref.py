"""CPU tests for the sparse GQA attention reference.

The reference is what the Pallas kernel is checked against, so it gets its own
tests: an oracle nobody has verified is worth nothing.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.qsa.ref import selected_tokens_ref, sparse_gqa_attention_ref
from sgl_jax.test.test_utils import CustomTestCase

RATIO = 4
PAGE_SIZE = 8
HEAD_DIM = 16
SEED = 7


def _softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class TestSelectedTokens(CustomTestCase):
    def test_expansion_and_the_open_group(self):
        """Blocks expand to ``ratio`` tokens each; what follows them is the open
        group, the tokens whose group has not closed yet. How much tail there is
        depends on ``position % ratio``, so all four cases are spelled out.
        """
        for blocks, pos, want in (
            # pos % ratio == 3: the group closes, so there is no tail at all and
            # the last block IS the query's own group.
            ([0, 2], 11, [0, 1, 2, 3, 8, 9, 10, 11]),
            ([], 15, []),
            # 2, 1, 0: one, two and three tokens of the open group come along.
            ([0, 1], 10, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            ([1], 9, [4, 5, 6, 7, 8, 9]),
            ([0, 1], 8, [0, 1, 2, 3, 4, 5, 6, 7, 8]),
            # -1 is padding: it expands to nothing, not to the tokens near zero.
            ([0, -1, -1], 7, [0, 1, 2, 3]),
        ):
            with self.subTest(blocks=blocks, pos=pos):
                self.assertEqual(selected_tokens_ref(blocks, pos, compress_ratio=RATIO), want)


class TestSparseGQAAttentionRef(CustomTestCase):
    def _case(self, n_q_heads, n_kv_heads, t_count=3, n_pages=4):
        rng = np.random.default_rng(SEED)
        k = rng.standard_normal((n_pages, PAGE_SIZE, n_kv_heads, HEAD_DIM)).astype(np.float32)
        v = rng.standard_normal((n_pages, PAGE_SIZE, n_kv_heads, HEAD_DIM)).astype(np.float32)
        q = rng.standard_normal((t_count, n_q_heads, HEAD_DIM)).astype(np.float32)
        return q, k, v

    def test_matches_dense_attention_over_the_same_tokens(self):
        """Given the token set it selects, the reference is plain softmax attention
        -- it adds no masking or scaling of its own."""
        n_q, n_kv, t_count = 4, 1, 3
        q, k, v = self._case(n_q, n_kv, t_count)
        page_table = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
        token_to_req = jnp.zeros((t_count,), jnp.int32)
        positions = jnp.asarray([10, 11, 17], jnp.int32)
        block_ids = jnp.asarray([[0, 1], [0, 2], [1, 3]], jnp.int32)
        sm_scale = HEAD_DIM**-0.5

        got = np.asarray(
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                block_ids,
                positions,
                jnp.asarray(k),
                jnp.asarray(v),
                page_table,
                token_to_req,
                compress_ratio=RATIO,
                sm_scale=sm_scale,
            )
        )

        flat_k = k.reshape(-1, n_kv, HEAD_DIM)
        flat_v = v.reshape(-1, n_kv, HEAD_DIM)
        for t in range(t_count):
            tokens = selected_tokens_ref(
                np.asarray(block_ids)[t], int(positions[t]), compress_ratio=RATIO
            )
            kk = np.repeat(flat_k[tokens], n_q // n_kv, axis=1)  # [N, H, D]
            vv = np.repeat(flat_v[tokens], n_q // n_kv, axis=1)
            scores = np.einsum("hd,nhd->hn", q[t], kk) * sm_scale
            want = np.einsum("hn,nhd->hd", _softmax(scores), vv)
            np.testing.assert_allclose(got[t], want, atol=1e-5, rtol=1e-5)

    def test_query_heads_read_their_own_kv_head(self):
        """With 2 KV heads and 4 query heads, heads 0-1 must read KV head 0 and
        heads 2-3 KV head 1. Zeroing one KV head may only move its own queries."""
        n_q, n_kv, t_count = 4, 2, 2
        q, k, v = self._case(n_q, n_kv, t_count)
        page_table = jnp.asarray([[0, 1, 2, 3]], jnp.int32)
        token_to_req = jnp.zeros((t_count,), jnp.int32)
        positions = jnp.asarray([11, 15], jnp.int32)
        block_ids = jnp.asarray([[0, 1], [1, 2]], jnp.int32)
        common = dict(compress_ratio=RATIO, sm_scale=HEAD_DIM**-0.5)

        base = np.asarray(
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                block_ids,
                positions,
                jnp.asarray(k),
                jnp.asarray(v),
                page_table,
                token_to_req,
                **common,
            )
        )
        v_zeroed = v.copy()
        v_zeroed[:, :, 1, :] = 0.0  # kill KV head 1
        perturbed = np.asarray(
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                block_ids,
                positions,
                jnp.asarray(k),
                jnp.asarray(v_zeroed),
                page_table,
                token_to_req,
                **common,
            )
        )

        np.testing.assert_allclose(base[:, :2], perturbed[:, :2], atol=0, rtol=0)
        self.assertFalse(np.allclose(base[:, 2:], perturbed[:, 2:]))

    def test_rejects_a_head_count_that_is_not_a_multiple(self):
        """Query heads must divide evenly among KV heads, or the GQA mapping is
        undefined."""
        q, k, v = self._case(3, 2, 1)
        with self.assertRaisesRegex(ValueError, "multiple"):
            sparse_gqa_attention_ref(
                jnp.asarray(q),
                jnp.asarray([[0]], jnp.int32),
                jnp.asarray([7], jnp.int32),
                jnp.asarray(k),
                jnp.asarray(v),
                jnp.asarray([[0, 1, 2, 3]], jnp.int32),
                jnp.zeros((1,), jnp.int32),
                compress_ratio=RATIO,
                sm_scale=HEAD_DIM**-0.5,
            )


if __name__ == "__main__":
    unittest.main()
