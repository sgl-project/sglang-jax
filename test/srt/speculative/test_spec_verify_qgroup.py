"""Grouped-query decode-form verify (sparse_mla_page_level group_queries).

With group_queries the G draft tokens of a request run as one ragged sequence
with G query rows. Captured kernel metadata must give every query exactly the
same visible (page, offset) set as the per-token pseudo-sequence layout when
all pages are selected -- including a group that crosses a page boundary,
where the previous page is placed right before the new page so the tail G
positions are the G consecutive tokens.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from functools import partial
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa import sparse_mla as sparse_mla_mod

PS, KP, PPS, RANK, ROPE, HEADS, G = 128, 8, 16, 512, 64, 4, 4
# request A: positions 200..203 (inside one page); request B: 254..257 (crosses 255|256)
LAST_POS = [203, 257]


def _inputs():
    T = len(LAST_POS) * G
    cache = jnp.zeros((2 * PPS + 1, PS // 2, 2, RANK + 128), jnp.bfloat16)
    ql = jnp.zeros((T, HEADS, RANK), jnp.bfloat16)
    qpe = jnp.zeros((T, HEADS, ROPE), jnp.bfloat16)
    kv = jnp.zeros((T, RANK), jnp.bfloat16)
    kpe = jnp.zeros((T, ROPE), jnp.bfloat16)
    # per-token pseudo-sequences: token i of request r has kv_len = pos + 1
    kv_lens = np.array([p - (G - 1) + i + 1 for p in LAST_POS for i in range(G)], np.int32)
    # every page below the token's page is selected (all-pages case): token
    # indices 0, 128, 256 -> pages 0, 1, 2 (entries >= kv_len are still "selected"
    # pages < new page only matter; extra pages are harmless for the set test)
    topk = np.tile(np.array([0, 128, 256, -1], np.int32), (T, 1))
    for t, kl in enumerate(kv_lens):
        topk[t, topk[t] >= kl] = -1  # only positions the token can see
    # packed page table: request A pages 10.., request B pages 20..
    page_indices = np.concatenate([10 + np.arange(PPS), 20 + np.arange(PPS)]).astype(np.int32)
    cu_q = np.arange(T + 1, dtype=np.int32)
    cu_kv = np.array([0] * G + [PPS * PS] * G + [PPS * PS], np.int32)
    dist = np.array([T, T, T], np.int32)
    return tuple(
        jnp.asarray(a)
        for a in (ql, qpe, kv, kpe, cache, kv_lens, topk, page_indices, cu_q, cu_kv, dist)
    )


def _visible_sets(meta, n_tokens):
    """Per query token: set of (physical page, offset) the kernel may attend."""
    kv_lens, page_indices, cu_q, _cu_kv, _dist = meta
    kv_lens = np.asarray(kv_lens)
    pages = np.asarray(page_indices).reshape(len(kv_lens), KP)
    cu_q = np.asarray(cu_q)
    out = [None] * n_tokens
    for s in range(len(kv_lens)):
        q_len = int(cu_q[s + 1] - cu_q[s])
        for i in range(q_len):
            bound = int(kv_lens[s]) - q_len + i + 1
            vis = {(int(pages[s, k // PS]), k % PS) for k in range(bound)}
            out[int(cu_q[s]) + i] = vis
    return out


class GroupedVerifyLayoutTest(unittest.TestCase):
    def _run(self, page_share_group, group_queries):
        captured = {}

        def fake_kernel(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
            kv_lens,
            page_indices,
            cu_q,
            cu_kv,
            dist,
            **kw,
        ):
            captured["meta"] = (kv_lens, page_indices, cu_q, cu_kv, dist)
            return jnp.zeros((ql_nope.shape[0], HEADS, RANK), jnp.bfloat16), cache_kv

        fn = partial(
            sparse_mla_mod.sparse_mla_page_level,
            sm_scale=1.0,
            page_size=PS,
            pages_per_seq=PPS,
            kv_lora_rank=RANK,
            k_pages_max=KP,
            page_share_group=page_share_group,
            group_queries=group_queries,
        )
        with (
            jax.disable_jit(),
            mock.patch.object(sparse_mla_mod, "mla_ragged_paged_attention", fake_kernel),
        ):
            o, _ = fn(*_inputs())
        self.assertEqual(o.shape[0], len(LAST_POS) * G)
        return captured["meta"]

    def test_grouped_metadata_shape_and_distribution(self):
        kv_lens, page_indices, cu_q, cu_kv, dist = self._run(G, True)
        self.assertEqual(kv_lens.shape, (2,))
        self.assertEqual(list(np.asarray(cu_q)), [0, 4, 8])
        self.assertEqual(list(np.asarray(dist)), [0, 0, 2])
        # request A: pages 0 (hit) + page 1 (new, offsets 0..75) -> 128 + 76
        self.assertEqual(int(kv_lens[0]), 128 + 75 + 1)
        # request B crosses 255|256: pages 0 (hit) + page 1 (previous, full) + page 2 (new, offs 0..1)
        self.assertEqual(int(kv_lens[1]), 128 + 128 + 1 + 1)
        pages = np.asarray(page_indices).reshape(2, KP)
        self.assertEqual(list(pages[1, :3]), [20, 21, 22])
        self.assertEqual(list(pages[0, :2]), [10, 11])

    def test_every_query_sees_the_same_positions_as_per_token(self):
        n = len(LAST_POS) * G
        per_token = _visible_sets(self._run(1, False), n)
        grouped = _visible_sets(self._run(G, True), n)
        for t in range(n):
            self.assertEqual(grouped[t], per_token[t], f"token {t}")
        # sanity: own position is the last visible one of each token
        for t, kl in enumerate([p - (G - 1) + i + 1 for p in LAST_POS for i in range(G)]):
            self.assertEqual(len(per_token[t]), kl)

    def test_group_queries_needs_a_group(self):
        # group_queries without page_share_group > 1 falls back to per-token metadata
        kv_lens, _pi, cu_q, _cu_kv, dist = self._run(1, True)
        self.assertEqual(kv_lens.shape, (len(LAST_POS) * G,))
        self.assertEqual(list(np.asarray(dist)), [8, 8, 8])


if __name__ == "__main__":
    unittest.main()
