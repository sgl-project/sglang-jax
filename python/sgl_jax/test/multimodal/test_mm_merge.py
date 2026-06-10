"""Tests for the canonical multimodal merge (sgl_jax.srt.mm_core.merge).

Requires jax (the merge uses jnp scatter). Run on the project's env (Python >=3.12):
    python -m pytest python/sgl_jax/test/multimodal/test_mm_merge.py
"""

import unittest

import jax.numpy as jnp

from sgl_jax.srt.mm_core.merge import FusedEmbed, merge

# pad_values use the MM_PAD_SHIFT regime (>> any real token id), see design doc §3.6.3.
PAD_A = 1_000_007
PAD_B = 1_000_009


class TestMerge(unittest.TestCase):
    def test_pure_text_passthrough(self):
        t = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
        out = merge(t, [], [], jnp.array([1, 2, 3, 4]))
        self.assertIsInstance(out, FusedEmbed)
        self.assertTrue(bool(jnp.array_equal(out.embed, t)))

    def test_single_item_scatter(self):
        # rows 1,2 are placeholders (PAD_A); rows 0,3 are text.
        ids = jnp.array([5, PAD_A, PAD_A, 6])
        t = jnp.zeros((4, 3), dtype=jnp.float32)
        feats = jnp.array([[1, 1, 1], [2, 2, 2]], dtype=jnp.float32)
        out = merge(t, [feats], [PAD_A], ids)
        self.assertTrue(bool(jnp.array_equal(out.embed[1], feats[0])))
        self.assertTrue(bool(jnp.array_equal(out.embed[2], feats[1])))
        self.assertTrue(bool(jnp.array_equal(out.embed[0], t[0])))  # text untouched
        self.assertTrue(bool(jnp.array_equal(out.embed[3], t[3])))

    def test_multiple_items_ordered(self):
        # item A (PAD_A) at row 1; item B (PAD_B) at rows 3,4. Concat order = A then B.
        ids = jnp.array([0, PAD_A, 0, PAD_B, PAD_B])
        t = jnp.zeros((5, 2), dtype=jnp.float32)
        fa = jnp.array([[1, 1]], dtype=jnp.float32)
        fb = jnp.array([[2, 2], [3, 3]], dtype=jnp.float32)
        out = merge(t, [fa, fb], [PAD_A, PAD_B], ids)
        self.assertTrue(bool(jnp.array_equal(out.embed[1], fa[0])))
        self.assertTrue(bool(jnp.array_equal(out.embed[3], fb[0])))
        self.assertTrue(bool(jnp.array_equal(out.embed[4], fb[1])))
        self.assertTrue(bool(jnp.array_equal(out.embed[0], t[0])))
        self.assertTrue(bool(jnp.array_equal(out.embed[2], t[2])))

    def test_pad_values_never_touch_text_rows(self):
        # Only row 0 holds a pad_value; small real token ids must never be placeholders.
        ids = jnp.array([PAD_A, 5, 7, 9])
        t = jnp.ones((4, 2), dtype=jnp.float32)
        feats = jnp.array([[9, 9]], dtype=jnp.float32)
        out = merge(t, [feats], [PAD_A], ids)
        self.assertTrue(bool(jnp.array_equal(out.embed[0], feats[0])))
        for r in (1, 2, 3):
            self.assertTrue(bool(jnp.array_equal(out.embed[r], t[r])))

    def test_count_mismatch_is_safe(self):
        # 3 placeholders but only 2 features: no crash; surplus placeholder left as text,
        # and a text row stays intact (mode="drop" + OOB fill, contract rule 1 safety).
        ids = jnp.array([PAD_A, PAD_A, PAD_A, 5])
        t = jnp.zeros((4, 2), dtype=jnp.float32)
        feats = jnp.array([[1, 1], [2, 2]], dtype=jnp.float32)
        out = merge(t, [feats], [PAD_A], ids)
        self.assertEqual(out.embed.shape, (4, 2))
        self.assertTrue(bool(jnp.array_equal(out.embed[3], t[3])))  # text row intact

    def test_more_features_than_placeholders_drops_surplus(self):
        # 1 placeholder but 2 features: extra feature maps OOB and is dropped (no crash).
        ids = jnp.array([0, PAD_A, 0])
        t = jnp.zeros((3, 2), dtype=jnp.float32)
        feats = jnp.array([[1, 1], [2, 2]], dtype=jnp.float32)
        out = merge(t, [feats], [PAD_A], ids)
        self.assertEqual(out.embed.shape, (3, 2))
        self.assertTrue(bool(jnp.array_equal(out.embed[1], feats[0])))

    def test_deepstack_densify_into_visual_rows(self):
        # seq 4: [text, visual(PAD_A), visual(PAD_A), text]; 2 visual tokens, 2 levels.
        t = jnp.zeros((4, 3), dtype=jnp.float32)
        ids = jnp.array([1, PAD_A, PAD_A, 2])
        vfeat = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
        ds = jnp.stack([jnp.full((2, 3), 10.0), jnp.full((2, 3), 20.0)])  # [2 levels, 2, 3]
        out = merge(t, [vfeat], [PAD_A], ids, deepstack=(ds, [PAD_A]))
        self.assertEqual(out.deepstack_embed.shape, (2, 4, 3))
        # visual rows (1,2) hold per-level features; text rows (0,3) stay 0
        self.assertTrue(bool(jnp.all(out.deepstack_embed[0, 1] == 10.0)))
        self.assertTrue(bool(jnp.all(out.deepstack_embed[1, 2] == 20.0)))
        self.assertTrue(bool(jnp.all(out.deepstack_embed[:, 0] == 0.0)))
        self.assertTrue(bool(jnp.all(out.deepstack_embed[:, 3] == 0.0)))

    def test_deepstack_excludes_audio_rows(self):
        # [text, visual(PAD_A), audio(PAD_B), text]; deepstack keyed on the visual item only.
        t = jnp.zeros((4, 2), dtype=jnp.float32)
        ids = jnp.array([1, PAD_A, PAD_B, 2])
        vfeat = jnp.ones((1, 2), dtype=jnp.float32)
        afeat = jnp.full((1, 2), 5.0, dtype=jnp.float32)
        ds = jnp.full((1, 1, 2), 9.0)  # [1 level, 1 visual token, 2]
        out = merge(t, [vfeat, afeat], [PAD_A, PAD_B], ids, deepstack=(ds, [PAD_A]))
        self.assertTrue(bool(jnp.all(out.deepstack_embed[0, 1] == 9.0)))  # visual row
        self.assertTrue(bool(jnp.all(out.deepstack_embed[0, 2] == 0.0)))  # audio row untouched

    def test_no_deepstack_is_none(self):
        t = jnp.zeros((2, 2), dtype=jnp.float32)
        out = merge(t, [jnp.ones((1, 2))], [PAD_A], jnp.array([PAD_A, 1]))
        self.assertIsNone(out.deepstack_embed)


if __name__ == "__main__":
    unittest.main()
