"""CPU tests for the compressed-cache page walk.

A wrong page walk does not raise -- a query silently reads another request's
keys. It is also invisible to any kernel-vs-kernel check, because both sides
would walk the same wrong page. So every destination is spelled out rather than
derived from the arithmetic that wrote it.
"""

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.kernels.qsa.paging import SENTINEL_PAGE, scatter_compressed
from sgl_jax.test.test_utils import CustomTestCase

RATIO = 4
PAGE_SIZE = 16
PC = PAGE_SIZE // RATIO
DIM = 8
NUM_PAGES = 12


class TestScatterCompressed(CustomTestCase):
    def test_every_group_lands_on_its_own_requests_page(self):
        """Two requests of different lengths on shuffled physical pages.

        Request 0 owns pages (7, 3) and request 1 owns (10, 1, 5), in the packed
        list the metadata carries. Both requests close group 0, so a walk that
        ignores the request collides; request 0 closes groups PC-1 and PC on
        either side of its page boundary; rows that closed no group go to the
        sentinel page and nowhere else.
        """
        page_indices = jnp.asarray([7, 3, 10, 1, 5], jnp.int32)
        cu_kv_lens = jnp.asarray([0, 2 * PAGE_SIZE, 5 * PAGE_SIZE], jnp.int32)
        # (request, group) per row, and where the row must land.
        rows = [
            ((0, 0), (7, 0)),
            ((0, PC - 1), (7, PC - 1)),
            ((0, PC), (3, 0)),
            ((1, 0), (10, 0)),
            ((1, 2 * PC + 1), (5, 1)),
            ((1, -1), None),
            ((0, -1), None),
        ]
        payload = np.arange(1, len(rows) + 1, dtype=np.float32)[:, None] * np.ones(DIM)

        out = scatter_compressed(
            jnp.zeros((NUM_PAGES, PC, DIM), jnp.float32),
            jnp.asarray(payload, jnp.float32),
            jnp.asarray([group for (_, group), _ in rows], jnp.int32),
            jnp.asarray([seq for (seq, _), _ in rows], jnp.int32),
            page_indices,
            cu_kv_lens,
            compress_ratio=RATIO,
        )

        want = np.zeros((NUM_PAGES, PC, DIM), np.float32)
        for i, (_, destination) in enumerate(rows):
            if destination is not None:
                want[destination] = payload[i]
        live = np.arange(NUM_PAGES) != SENTINEL_PAGE
        np.testing.assert_array_equal(np.asarray(out)[live], want[live])


if __name__ == "__main__":
    unittest.main()
