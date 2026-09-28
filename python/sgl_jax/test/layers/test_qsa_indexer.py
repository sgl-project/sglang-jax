"""QSA indexer CPU tests.

Runs in the ``unit-test-cpu`` suite. Nothing here builds a KV pool or a
backend: the indexer's projections, group compression and block selection are
pure functions, and those are what this pins.

The oracles loop over groups and heads explicitly rather than broadcasting, so
a reshape or grouping bug in the vectorized implementation does not reproduce
in the reference as well.

Run:
    python -m unittest python.sgl_jax.test.layers.test_qsa_indexer
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.layers.attention.qsa_indexer import QSAIndexer, select_blocks
from sgl_jax.srt.layers.embeddings import RotaryEmbedding

HIDDEN_SIZE = 32
N_HEADS = 4
KV_HEADS = 1
HEAD_DIM = 16
ROTARY_DIM = 8
RATIO = 4
BUDGET = 8  # -> block_topk = 2, expansion width = BUDGET + RATIO - 1 = 11
EPSILON = 1e-6
SEED = 42


def _make_mesh():
    devices = np.array(jax.devices())
    return Mesh(
        devices[:1].reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _make_rotary():
    return RotaryEmbedding(
        head_size=ROTARY_DIM,
        rotary_dim=ROTARY_DIM,
        max_position_embeddings=1024,
        base=10000,
        is_neox_style=True,
        dtype=jnp.float32,
    )


def _make_indexer(mesh):
    with jax.set_mesh(mesh):
        return QSAIndexer(
            hidden_size=HIDDEN_SIZE,
            indexer_n_heads=N_HEADS,
            indexer_kv_heads=KV_HEADS,
            indexer_head_dim=HEAD_DIM,
            indexer_budget=BUDGET,
            indexer_compress_ratio=RATIO,
            rotary_dim=ROTARY_DIM,
            mesh=mesh,
            rms_norm_eps=EPSILON,
            params_dtype=jnp.float32,
        )


def _ref_compress_batch(raw_keys, positions, cu_q_lens, slots, rings):
    """fp64 oracle for a packed batch, by explicit loops.

    Rebuilds each request's logical key run -- the live rows of its ring
    followed by this step's keys -- and indexes that run by absolute position.
    It therefore shares no index arithmetic with the implementation, which walks
    backwards from each token instead.

    Returns ``({token index: (group id, pooled key)}, {slot: expected ring})``.
    """
    x = np.asarray(raw_keys, np.float64)
    pos = np.asarray(positions)
    cu = np.asarray(cu_q_lens)
    pooled, rings_out = {}, {}
    for b in range(len(cu) - 1):
        lo, hi = int(cu[b]), int(cu[b + 1])
        slot = int(slots[b])
        ring = np.asarray(rings[slot], np.float64)
        if lo == hi:
            rings_out[slot] = ring
            continue

        carried_in = int(pos[lo]) % RATIO
        live = ring[RATIO - carried_in :] if carried_in else ring[:0]
        run = np.concatenate([live, x[lo:hi]])
        run_start = int(pos[lo]) - carried_in

        for i in range(lo, hi):
            p = int(pos[i])
            if (p + 1) % RATIO:
                continue  # this token does not close a group
            acc = np.zeros(x.shape[1], np.float64)
            for j in range(RATIO):
                acc += run[(p - j) - run_start]
            pooled[i] = (p // RATIO, acc / RATIO)

        last_pos = int(pos[hi - 1])
        open_from = ((last_pos + 1) // RATIO) * RATIO
        tail = run[open_from - run_start : last_pos + 1 - run_start]
        out = np.zeros_like(ring)
        if len(tail):
            out[RATIO - len(tail) :] = tail
        rings_out[slot] = out
    return pooled, rings_out


def _batch_of_one(layer, raw_keys, start_position, ring, rotary):
    """A one-request batch: the minimal shape for checking numerics and the
    ring, where a second request would only add noise."""
    t = raw_keys.shape[0]
    positions = start_position + jnp.arange(t, dtype=jnp.int32)
    rings = jnp.zeros((1, RATIO, HEAD_DIM), jnp.float32).at[0].set(ring)
    compressed, groups, _, rings_out = layer.compress_batch(
        raw_keys,
        positions,
        jnp.asarray([0, t], jnp.int32),
        jnp.asarray([0], jnp.int32),
        rings,
        rotary,
    )
    return compressed, groups, rings_out[0]


def _closing_rows(groups):
    return [i for i in range(len(groups)) if int(groups[i]) >= 0]


class TestQSAIndexer(unittest.TestCase):
    def test_rejects_bad_config(self):
        """Constructor gates: one indexer KV head, a ratio of at least 2, a
        budget divisible by it, and a rotary width the head can hold."""
        mesh = _make_mesh()
        for overrides, match in [
            ({"indexer_kv_heads": 2}, "indexer_kv_heads=1"),
            ({"indexer_compress_ratio": 1}, "at least 2"),
            ({"indexer_budget": BUDGET + 1}, "divisible"),
            ({"rotary_dim": HEAD_DIM + 8}, "exceeds"),
        ]:
            kwargs = dict(
                hidden_size=HIDDEN_SIZE,
                indexer_n_heads=N_HEADS,
                indexer_kv_heads=KV_HEADS,
                indexer_head_dim=HEAD_DIM,
                indexer_budget=BUDGET,
                indexer_compress_ratio=RATIO,
                rotary_dim=ROTARY_DIM,
                mesh=mesh,
            )
            kwargs.update(overrides)
            with (
                self.subTest(**overrides),
                self.assertRaisesRegex(ValueError, match),
                jax.set_mesh(mesh),
            ):
                QSAIndexer(**kwargs)

    def test_project_splits_query_and_leaves_the_key_raw(self):
        """The key must come back un-normed and un-rotated: both happen later,
        on the pooled group, at the group's first position."""
        mesh = _make_mesh()
        layer = _make_indexer(mesh)
        rotary = _make_rotary()
        rng = np.random.default_rng(SEED)
        hidden = jnp.array(rng.standard_normal((6, HIDDEN_SIZE)).astype(np.float32))
        positions = jnp.arange(6, dtype=jnp.int32)

        with jax.set_mesh(mesh):
            query, raw_key = layer.project(hidden, positions, rotary)
            fused, _ = layer.index_qk_proj(hidden)

        self.assertEqual(query.shape, (6, N_HEADS, HEAD_DIM))
        self.assertEqual(raw_key.shape, (6, HEAD_DIM))
        # The key half of the fused projection is passed through untouched.
        np.testing.assert_allclose(
            np.asarray(raw_key), np.asarray(fused[:, N_HEADS * HEAD_DIM :]), atol=0, rtol=0
        )
        # The query half is not: it was normed and rotated.
        self.assertFalse(
            np.allclose(
                np.asarray(query).reshape(6, -1),
                np.asarray(fused[:, : N_HEADS * HEAD_DIM]),
            )
        )
        # Only the leading rotary_dim dims rotate; the tail passes through.
        normed = np.asarray(
            layer.q_layernorm(fused[:, : N_HEADS * HEAD_DIM].reshape(6, N_HEADS, HEAD_DIM))
        )
        np.testing.assert_allclose(
            np.asarray(query)[..., ROTARY_DIM:], normed[..., ROTARY_DIM:], atol=1e-6, rtol=1e-6
        )

    def _selection_inputs(self, layer):
        """One prefill sequence of 24 tokens = 6 compressed entries.

        Pages hold 4 entries each and the sequence owns 2 pages, so 8 cache
        rows back 6 real entries; the last 2 are padding the mask must drop.
        Entry e scores strictly higher than e-1, which makes the expected top-k
        closed form instead of tie-dependent.
        """
        n_tokens, page_size, pages_per_seq = 24, 4, 2
        n_entries = n_tokens // RATIO
        max_kv = page_size * pages_per_seq

        direction = np.zeros(HEAD_DIM, np.float32)
        direction[0] = 1.0
        keys = np.zeros((max_kv, HEAD_DIM), np.float32)
        for e in range(n_entries):
            keys[e] = direction * (e + 1)
        # The pool's layout; fp32 packs one entry per row.
        cache = jnp.array(keys.reshape(pages_per_seq, page_size, 1, HEAD_DIM))
        query = jnp.array(np.tile(direction, (n_tokens, N_HEADS, 1)))

        meta = dict(
            seq_lens=jnp.array([n_tokens], jnp.int32),  # uncompressed tokens
            page_table=jnp.arange(pages_per_seq, dtype=jnp.int32).reshape(1, pages_per_seq),
            cu_q_lens=jnp.array([0, n_tokens], jnp.int32),
            distribution=jnp.array([0, 1, 1], jnp.int32),
        )
        return query, cache, meta, n_entries

    def test_select_blocks_matches_the_qsa_formula(self):
        """QSA scores ``sum_h relu(q_h . k)``; streamindex_topk scores that same
        thing times per-head weights, so all-ones weights must reproduce it.
        Scores increase with the entry index here, so the expected answer is the
        two highest *visible* entries -- the visibility rule is under test too.
        """
        mesh = _make_mesh()
        layer = _make_indexer(mesh)
        query, cache, meta, n_entries = self._selection_inputs(layer)

        with jax.set_mesh(mesh):
            block_ids = np.asarray(
                select_blocks(
                    query,
                    cache,
                    block_topk=layer.block_topk,
                    compress_ratio=layer.compress_ratio,
                    use_kernel=False,
                    **meta,
                )
            )

        self.assertEqual(block_ids.shape, (24, BUDGET // RATIO))
        for p in range(24):
            n_visible = min((p + 1) // RATIO, n_entries)
            got = sorted(int(b) for b in block_ids[p] if b >= 0)
            want = sorted(range(n_visible))[-min(n_visible, BUDGET // RATIO) :]
            self.assertEqual(got, want, f"query position {p}")


class TestCompressBatch(unittest.TestCase):
    """The ragged entry point against the single-request one it generalises."""

    @staticmethod
    def _inputs(rng, layer, mesh, q_lens, start_positions, max_reqs=6):
        t_count = sum(q_lens)
        cu = np.concatenate([[0], np.cumsum(q_lens)]).astype(np.int32)
        positions = np.concatenate(
            [np.arange(sp, sp + n, dtype=np.int32) for sp, n in zip(start_positions, q_lens)]
        )
        raw = rng.standard_normal((t_count, HEAD_DIM)).astype(np.float32)
        slots = rng.permutation(max_reqs)[: len(q_lens)].astype(np.int32)
        rings = rng.standard_normal((max_reqs, RATIO, HEAD_DIM)).astype(np.float32)
        # Only the rows a right-aligned ring would really hold are meaningful;
        # zero the rest so a stale row cannot silently contribute.
        for b, sp in enumerate(start_positions):
            carried = sp % RATIO
            rings[slots[b], : RATIO - carried] = 0.0
        return (
            jnp.asarray(raw),
            jnp.asarray(positions),
            jnp.asarray(cu),
            jnp.asarray(slots),
            jnp.asarray(rings),
        )

    def test_carries_the_open_group_across_steps(self):
        """Six tokens in two steps must compress identically to one step of six.

        This is the bug the ring exists to prevent: without it the trailing
        group of a step is either dropped or pooled from a short mean.
        """
        mesh = _make_mesh()
        layer = _make_indexer(mesh)
        rotary = _make_rotary()
        rng = np.random.default_rng(SEED)
        raw = jnp.array(rng.standard_normal((6, HEAD_DIM)).astype(np.float32))
        empty = jnp.zeros((RATIO, HEAD_DIM), jnp.float32)

        with jax.set_mesh(mesh):
            one_shot, groups_one, ring_one = _batch_of_one(layer, raw, 0, empty, rotary)
            _, groups_a, ring_a = _batch_of_one(layer, raw[:3], 0, empty, rotary)
            second, groups_b, ring_b = _batch_of_one(layer, raw[3:], 3, ring_a, rotary)

        # Step one completes nothing; step two closes group 0 from the ring.
        self.assertEqual(_closing_rows(groups_a), [])
        self.assertEqual(_closing_rows(groups_b), [0])
        self.assertEqual(_closing_rows(groups_one), [3])
        self.assertEqual(int(groups_b[0]), 0)
        self.assertEqual(int(groups_one[3]), 0)
        np.testing.assert_allclose(
            np.asarray(second)[0], np.asarray(one_shot)[3], atol=1e-6, rtol=1e-6
        )
        # Both leave the same two-token tail (tokens 4 and 5) in the ring.
        # The ring is right aligned: the open group's keys sit at its tail.
        np.testing.assert_allclose(
            np.asarray(ring_b)[-2:], np.asarray(ring_one)[-2:], atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(np.asarray(ring_a)[-3:], np.asarray(raw)[:3], atol=0)

    def test_matches_an_fp64_oracle(self):
        """Ragged batch under jit against explicit fp64 loops: which tokens
        close a group, what each group pools, where it is rotated, and each
        request's new ring.

        The oracle rebuilds every request's logical key run and indexes it by
        absolute position, so it reaches the answer by a different route than
        the implementation's backwards walk. It pools, then norms, then rotates
        at the group's first position; each of the three is silent if wrong.
        """
        mesh = _make_mesh()
        layer = _make_indexer(mesh)
        rotary = _make_rotary()
        rng = np.random.default_rng(0)

        q_lens = [7, 1, 4, 5, 3, 1, 0]
        # A mix of group-aligned and mid-group starts, so the ring matters, and
        # between them the four possible carry offsets all occur. The two
        # one-token requests are decode steps: the one at 11 closes a group
        # almost entirely out of the ring, the one at 13 closes none. The empty
        # request has no tokens this step and must keep its ring.
        start_positions = [0, 13, 8, 6, 2, 11, 5]
        raw, positions, cu, slots, rings = self._inputs(
            rng, layer, mesh, q_lens, start_positions, max_reqs=8
        )
        want_pooled, want_rings = _ref_compress_batch(raw, positions, cu, slots, rings)

        @jax.jit
        def compress(raw, positions, cu, slots, rings):
            return layer.compress_batch(raw, positions, cu, slots, rings, rotary)

        with jax.set_mesh(mesh):
            compressed, groups, seq_ids, rings_out = compress(raw, positions, cu, slots, rings)
            rows = _closing_rows(groups)
            normed = layer.k_layernorm(
                jnp.asarray(np.stack([want_pooled[i][1] for i in rows]).astype(np.float32))
            )
            first = jnp.asarray([want_pooled[i][0] * RATIO for i in rows], jnp.int32)
            rot, _ = rotary(first, normed[:, None, :ROTARY_DIM], normed[:, None, :ROTARY_DIM])
            expected = jnp.concatenate([rot[:, 0, :], normed[:, ROTARY_DIM:]], axis=-1)

        self.assertEqual(rows, sorted(want_pooled))
        cu_np = np.asarray(cu)
        for i in rows:
            self.assertEqual(int(groups[i]), want_pooled[i][0])
            self.assertEqual(int(seq_ids[i]), int(np.searchsorted(cu_np[1:], i, side="right")))
        np.testing.assert_allclose(
            np.asarray(compressed)[rows], np.asarray(expected), rtol=1e-5, atol=1e-5
        )
        for slot, want in want_rings.items():
            np.testing.assert_allclose(
                np.asarray(rings_out[slot]),
                want,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"ring for slot {slot}",
            )

    def test_a_padded_request_leaves_the_last_slot_alone(self):
        """Batch padding carries slot -1. Its ring must not land on the last
        slot, where a live request's new ring would be overwritten."""
        mesh = _make_mesh()
        layer = _make_indexer(mesh)
        rotary = _make_rotary()
        rng = np.random.default_rng(4)
        raw, positions, cu, _, rings = self._inputs(rng, layer, mesh, [5], [2])
        last = rings.shape[0] - 1

        with jax.set_mesh(mesh):
            _, _, _, alone = layer.compress_batch(
                raw, positions, cu, jnp.asarray([last], jnp.int32), rings, rotary
            )
            _, _, _, padded = layer.compress_batch(
                raw,
                positions,
                jnp.concatenate([cu, cu[-1:]]),
                jnp.asarray([last, -1], jnp.int32),
                rings,
                rotary,
            )
        self.assertFalse(np.array_equal(np.asarray(alone[last]), np.asarray(rings[last])))
        np.testing.assert_array_equal(np.asarray(padded), np.asarray(alone))


if __name__ == "__main__":
    unittest.main()
