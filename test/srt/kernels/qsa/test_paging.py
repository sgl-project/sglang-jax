"""CPU tests for the compressed-cache page walk.

A wrong page walk does not raise -- a query silently reads another request's
keys. It is also invisible to any kernel-vs-kernel check, because both sides
would walk the same wrong page. So the page identity is asserted head-on, and
the scatter is checked by reading back through the *token* page table rather
than through the arithmetic that wrote it.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.qsa.paging import (
    SENTINEL_PAGE,
    as_3d,
    as_4d,
    compressed_slot,
    scatter_compressed,
)
from sgl_jax.test.test_utils import CustomTestCase

RATIO = 4
PAGE_SIZE = 16
PC = PAGE_SIZE // RATIO
DIM = 8
NUM_PAGES = 12


def _fixture(seq_token_lens, seed=0):
    """A page table whose physical pages are shuffled, so a walk that ignores it
    cannot pass by accident."""
    rng = np.random.default_rng(seed)
    pages_per_seq = max((n + PAGE_SIZE - 1) // PAGE_SIZE for n in seq_token_lens)
    n_seqs = len(seq_token_lens)
    # Page 0 is the reserved sentinel; hand out the rest.
    table = rng.permutation(np.arange(1, NUM_PAGES))[: n_seqs * pages_per_seq]
    page_indices = jnp.asarray(table.astype(np.int32))
    aligned = [((n + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE for n in seq_token_lens]
    cu_kv_lens = jnp.asarray(np.concatenate([[0], np.cumsum(aligned)]).astype(np.int32))
    return page_indices, cu_kv_lens, pages_per_seq


class TestCompressedSlot(CustomTestCase):
    def test_the_page_identity(self):
        """Entry t//ratio must land on the same logical page as token t."""
        for token in range(PAGE_SIZE * 6):
            page, offset = compressed_slot(token, compressed_page_size=PC, compress_ratio=RATIO)
            self.assertEqual(page, token // PAGE_SIZE)
            self.assertEqual(offset, (token % PAGE_SIZE) // RATIO)

    def test_every_token_of_a_group_agrees(self):
        """All ratio tokens of a group must resolve to one slot, or the group would
        need more than one page-table lookup to place."""
        for group in range(PAGE_SIZE * 6 // RATIO):
            slots = {
                compressed_slot(group * RATIO + j, compressed_page_size=PC, compress_ratio=RATIO)
                for j in range(RATIO)
            }
            self.assertEqual(len(slots), 1)


class TestScatterCompressed(CustomTestCase):
    def _run(self, seq_token_lens, groups, seq_ids, seed=0):
        page_indices, cu_kv_lens, _ = _fixture(seq_token_lens, seed)
        cache = jnp.zeros((NUM_PAGES, PC, DIM), jnp.float32)
        n = len(groups)
        payload = jnp.asarray((np.arange(n)[:, None] + 1.0) * np.ones((n, DIM)), jnp.float32)
        out = scatter_compressed(
            cache,
            payload,
            jnp.asarray(np.asarray(groups, np.int32)),
            jnp.asarray(np.asarray(seq_ids, np.int32)),
            page_indices,
            cu_kv_lens,
            compress_ratio=RATIO,
        )
        return out, payload, page_indices, cu_kv_lens

    def test_reads_back_through_the_token_page_table(self):
        """Write by group, read by token: the two must meet."""
        seq_token_lens = [PAGE_SIZE * 2, PAGE_SIZE * 3]
        # tokens 3, 7, 35 close groups 0, 1, 8 of their request
        tokens = [3, 7, 35, 3, 19]
        seq_ids = [0, 0, 1, 1, 1]
        groups = [t // RATIO for t in tokens]
        out, payload, page_indices, cu_kv_lens = self._run(seq_token_lens, groups, seq_ids)

        for i, (token, seq) in enumerate(zip(tokens, seq_ids)):
            phys = page_indices[cu_kv_lens[seq] // PAGE_SIZE + token // PAGE_SIZE]
            offset = (token % PAGE_SIZE) // RATIO
            np.testing.assert_array_equal(np.asarray(out[phys, offset]), np.asarray(payload[i]))

    def test_two_requests_do_not_collide(self):
        """Same group index, different requests, different physical pages."""
        out, payload, page_indices, cu_kv_lens = self._run(
            [PAGE_SIZE * 2, PAGE_SIZE * 2], groups=[0, 0], seq_ids=[0, 1]
        )
        p0 = page_indices[cu_kv_lens[0] // PAGE_SIZE]
        p1 = page_indices[cu_kv_lens[1] // PAGE_SIZE]
        self.assertNotEqual(int(p0), int(p1))
        np.testing.assert_array_equal(np.asarray(out[p0, 0]), np.asarray(payload[0]))
        np.testing.assert_array_equal(np.asarray(out[p1, 0]), np.asarray(payload[1]))

    def test_crossing_a_page_boundary(self):
        """Group PC-1 is the last entry of page 0; group PC is the first of page 1."""
        out, payload, page_indices, cu_kv_lens = self._run(
            [PAGE_SIZE * 3], groups=[PC - 1, PC], seq_ids=[0, 0]
        )
        base = cu_kv_lens[0] // PAGE_SIZE
        np.testing.assert_array_equal(
            np.asarray(out[page_indices[base], PC - 1]), np.asarray(payload[0])
        )
        np.testing.assert_array_equal(
            np.asarray(out[page_indices[base + 1], 0]), np.asarray(payload[1])
        )

    def test_padding_goes_to_the_sentinel_and_leaves_live_pages_alone(self):
        """Rows that closed no group are steered to the reserved page 0 rather than
        clobbering a live page's first entry."""
        out, payload, page_indices, cu_kv_lens = self._run(
            [PAGE_SIZE * 2], groups=[-1, 1, -1], seq_ids=[0, 0, 0]
        )
        live = page_indices[cu_kv_lens[0] // PAGE_SIZE]
        self.assertNotEqual(int(live), SENTINEL_PAGE)
        np.testing.assert_array_equal(np.asarray(out[live, 1]), np.asarray(payload[1]))
        # Slot 0 of the live page was never written.
        np.testing.assert_array_equal(np.asarray(out[live, 0]), np.zeros(DIM))


class TestLayoutViews(CustomTestCase):
    def test_3d_4d_round_trip(self):
        """The packed 4-D buffer and the 3-D view the page walk uses are the same
        bytes in either direction."""
        c4 = jnp.asarray(np.arange(NUM_PAGES * PC * DIM, dtype=np.float32)).reshape(
            NUM_PAGES, PC // 2, 2, DIM
        )
        np.testing.assert_array_equal(np.asarray(as_4d(as_3d(c4), 2)), np.asarray(c4))

    def test_3d_view_page_size_matches_the_ratio(self):
        """The 3-D view's page axis is page_size // ratio, the number the walk
        divides by."""
        c4 = jnp.zeros((NUM_PAGES, PC // 2, 2, DIM))
        self.assertEqual(as_3d(c4).shape[1], PAGE_SIZE // RATIO)


if __name__ == "__main__":
    unittest.main()
