"""CPU unit tests for V-2 vision patch bucketing (host NumPy logic).

The merge-unit padding is the bug-prone part of V-2 (a wrong reshape/slice silently corrupts which
visual token lands where). These tests pin the canonical (t, llm_h, llm_w, m*m) layout, the
zero-fill placement, the already-aligned pass-through, and multi-item concatenation -- all on CPU,
no JAX, so they guard the logic the pod-only end-to-end check can't isolate.
"""

from __future__ import annotations

import unittest

import numpy as np

from sgl_jax.srt.model_executor.vision_bucketing import bucket_pad_images


def _make_pixels(t: int, llm_h: int, llm_w: int, merge: int, dim: int) -> np.ndarray:
    """Patches in (t, llm_h, llm_w, merge*merge) row-major; patch i = [i]*dim so it's traceable."""
    n = t * llm_h * llm_w * merge * merge
    return np.repeat(np.arange(n, dtype=np.float32)[:, None], dim, axis=1)


class TestBucketPadImages(unittest.TestCase):
    def test_pads_to_bucket_multiple_keeps_real_units_in_place(self):
        merge, dim, bucket = 2, 3, 4
        t, llm_h, llm_w = 1, 3, 2  # h=6, w=4
        px = _make_pixels(t, llm_h, llm_w, merge, dim)

        padded, grids, real = bucket_pad_images(
            px, [(t, llm_h * merge, llm_w * merge)], merge, bucket
        )

        # padded grid rounds each LLM dim up to a multiple of bucket -> (1, 4*2, 4*2) patches
        self.assertEqual(grids, ((t, 8, 8),))
        np.testing.assert_array_equal(real, np.array([[llm_h, llm_w]], dtype=np.int32))
        self.assertEqual(padded.shape, (t * 4 * 4 * merge * merge, dim))

        canon = padded.reshape(t, 4, 4, merge * merge, dim)
        # real units preserved exactly in the top-left sub-block, in canonical order
        np.testing.assert_array_equal(
            canon[:, :llm_h, :llm_w, :, :], px.reshape(t, llm_h, llm_w, merge * merge, dim)
        )
        # everything outside the real sub-block is zero padding
        self.assertTrue(np.all(canon[:, llm_h:, :, :, :] == 0))
        self.assertTrue(np.all(canon[:, :, llm_w:, :, :] == 0))

    def test_already_aligned_is_passthrough(self):
        merge, dim, bucket = 2, 4, 4
        t, llm_h, llm_w = 1, 4, 4  # already a bucket multiple
        px = _make_pixels(t, llm_h, llm_w, merge, dim)

        padded, grids, real = bucket_pad_images(
            px, [(t, llm_h * merge, llm_w * merge)], merge, bucket
        )

        np.testing.assert_array_equal(padded, px)  # untouched, no padding
        self.assertEqual(grids, ((t, llm_h * merge, llm_w * merge),))
        np.testing.assert_array_equal(real, np.array([[llm_h, llm_w]], dtype=np.int32))

    def test_multi_item_concatenation_and_dims(self):
        merge, dim, bucket = 2, 2, 4
        items = [(1, 2, 2), (1, 5, 1)]  # LLM (2,2)->bucket(4,4); (5,1)->bucket(8,4)
        grids_in = [(t, h * merge, w * merge) for (t, h, w) in items]
        pxs = [_make_pixels(t, h, w, merge, dim) for (t, h, w) in items]
        px = np.concatenate(pxs, axis=0)

        padded, grids, real = bucket_pad_images(px, grids_in, merge, bucket)

        np.testing.assert_array_equal(real, np.array([[2, 2], [5, 1]], dtype=np.int32))
        self.assertEqual(grids, ((1, 8, 8), (1, 16, 8)))  # (4,4)*2 and (8,4)*2 patch dims
        expected_len = (1 * 4 * 4 + 1 * 8 * 4) * merge * merge
        self.assertEqual(padded.shape, (expected_len, dim))

        # each item's real units recoverable from its own padded block
        off = 0
        for (t, h, w), src in zip(items, pxs):
            ph = ((h + bucket - 1) // bucket) * bucket
            pw = ((w + bucket - 1) // bucket) * bucket
            n = t * ph * pw * merge * merge
            block = padded[off : off + n].reshape(t, ph, pw, merge * merge, dim)
            np.testing.assert_array_equal(
                block[:, :h, :w, :, :], src.reshape(t, h, w, merge * merge, dim)
            )
            off += n

    def test_real_units_compactable_in_canonical_order(self):
        # The consumer compacts valid units to the front; verify the real units, read in canonical
        # (t, llm_h, llm_w) row-major from the padded block, equal the original unit sequence.
        merge, dim, bucket = 2, 1, 4
        t, llm_h, llm_w = 1, 3, 3
        px = _make_pixels(t, llm_h, llm_w, merge, dim)
        padded, grids, _ = bucket_pad_images(px, [(t, llm_h * merge, llm_w * merge)], merge, bucket)
        ph = pw = 4
        canon = padded.reshape(t, ph, pw, merge * merge, dim)
        real_units = canon[:, :llm_h, :llm_w, :, :].reshape(-1, dim)
        np.testing.assert_array_equal(real_units, px)


if __name__ == "__main__":
    unittest.main()
