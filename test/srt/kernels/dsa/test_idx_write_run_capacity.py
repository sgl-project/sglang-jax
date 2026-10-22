"""Run-table capacity of the prefill indexer-key cache write (#1811 review).

``paged_write_back`` has no overflow fallback on the run-table path: runs past ``r_cap`` are
silently dropped. The KV write sizes the table as ``T // page_size + 2 * S + 2`` (at most
``ceil(q / page_size) + 1`` runs per sequence); the indexer-key write must use the same bound,
the previous ``T // page_size + S + 34`` overflows once more than ~34 sequences cross a page
boundary.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.dsa.sparse_mla_prefill_qblock import _build_write_runs

jax.config.update("jax_platform_name", "cpu")

PAGE = 128
PK = 2


def _kv_write_cap(T, S):
    return T // PAGE + 2 * S + 2


def _old_idx_cap(T, S):
    return T // PAGE + S + 34


def _crossing_batch(S, G=4):
    """S requests x G draft tokens, every request straddling two non-adjacent pages."""
    loc = []
    for s in range(S):
        pa, pb = 1 + s, 2 + S + (S - 1 - s)  # never adjacent pages
        half = G // 2
        loc += [pa * PAGE + PAGE - half + i for i in range(half)]
        loc += [pb * PAGE + i for i in range(G - half)]
    return jnp.asarray(np.array(loc, np.int32))


class IdxWriteRunCapacityTest(unittest.TestCase):
    def test_old_cap_overflows_kv_write_cap_holds(self):
        S, G = 64, 4
        loc = _crossing_batch(S, G)
        T = int(loc.shape[0])
        _, n_raw = _build_write_runs(loc, kv_packing=PK, r_cap=4096)
        n_raw = int(n_raw)
        self.assertEqual(n_raw, 2 * S)
        self.assertGreater(n_raw, _old_idx_cap(T, S))  # 128 runs > 2 + 64 + 34
        self.assertLessEqual(n_raw, _kv_write_cap(T, S))

    def test_kv_write_cap_bounds_random_ragged_batches(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            S = int(rng.integers(1, 48))
            q_lens = rng.integers(1, 3 * PAGE, size=S)
            starts = rng.integers(0, PAGE, size=S)
            pages = rng.permutation(np.arange(1, 4 * S + 8))
            loc, used = [], 0
            for q, st in zip(q_lens, starts):
                n_pages = (st + q + PAGE - 1) // PAGE
                pg = pages[used : used + n_pages]
                used += n_pages
                for i in range(q):
                    pos = st + i
                    loc.append(int(pg[pos // PAGE]) * PAGE + pos % PAGE)
            T = len(loc)
            T_pad = -(-T // PK) * PK
            loc = np.array(loc + [-1] * (T_pad - T), np.int32)
            _, n_raw = _build_write_runs(jnp.asarray(loc), kv_packing=PK, r_cap=8192)
            self.assertLessEqual(int(n_raw), _kv_write_cap(T_pad, S), (S, q_lens, starts))


if __name__ == "__main__":
    unittest.main()
