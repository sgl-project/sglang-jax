"""Tests for the Qwen2.5-VL in-model path (refactor M3 step 1+2).

Requires jax. Run on the project env (Python >=3.12):
    python -m pytest python/sgl_jax/test/multimodal/test_inmodel_qwen25vl.py
"""

import unittest

import jax.numpy as jnp

PAD_A = 1_000_007
PAD_B = 1_000_009


class TestForwardBatchMMFields(unittest.TestCase):
    """M3 step1: ForwardBatch carries mm inputs through its pytree for in-forward merge."""

    def _fb(self, **kw):
        from sgl_jax.srt.model_executor.forward_batch_info import (
            ForwardBatch,
            ForwardMode,
        )

        base = dict(
            bid=0,
            forward_mode=next(iter(ForwardMode)),
            batch_size=1,
            input_ids=jnp.array([1, 2, 3, 4]),
            req_pool_indices=jnp.array([0]),
            seq_lens=jnp.array([4]),
            out_cache_loc=jnp.array([0, 1, 2, 3]),
        )
        base.update(kw)
        return ForwardBatch(**base)

    def test_contains_mm_inputs(self):
        self.assertFalse(self._fb().contains_mm_inputs())
        self.assertTrue(
            self._fb(
                mm_pixel_values=jnp.ones((4, 8)), mm_grid_thw=((1, 2, 2),)
            ).contains_mm_inputs()
        )

    def test_pytree_roundtrip_preserves_mm_fields(self):
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

        fb = self._fb(
            mm_pixel_values=jnp.ones((4, 8)),
            mm_grid_thw=((1, 2, 2),),
            mm_pad_values=(PAD_A,),
        )
        children, aux = fb.tree_flatten()
        fb2 = ForwardBatch.tree_unflatten(aux, children)
        self.assertTrue(bool(jnp.array_equal(fb2.mm_pixel_values, fb.mm_pixel_values)))
        self.assertEqual(fb2.mm_grid_thw, ((1, 2, 2),))  # static aux preserved
        self.assertEqual(fb2.mm_pad_values, (PAD_A,))
        self.assertTrue(fb2.contains_mm_inputs())

    def test_text_only_batch_unaffected(self):
        from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch

        fb = self._fb()
        fb2 = ForwardBatch.tree_unflatten(*reversed(fb.tree_flatten()))
        self.assertIsNone(fb2.mm_pixel_values)
        self.assertFalse(fb2.contains_mm_inputs())


class TestInModelRegistration(unittest.TestCase):
    """M3 step2: the in-model VLM resolves via the standard srt ModelRegistry."""

    def test_resolves_via_standard_registry(self):
        from sgl_jax.srt.models.registry import ModelRegistry

        cls, arch = ModelRegistry.resolve_model_cls("Qwen2_5_VLForConditionalGeneration")
        self.assertEqual(cls.__name__, "Qwen2_5_VLForConditionalGeneration")
        self.assertEqual(arch, "Qwen2_5_VLForConditionalGeneration")
        # capability declarations present (mm_core.capability / U3)
        from sgl_jax.srt.mm_core import capability as cap

        self.assertTrue(cap.is_multimodal_arch(cls))
        self.assertEqual(cap.supported_modalities(cls), {"image", "video"})

    def test_has_load_weights(self):
        from sgl_jax.srt.models.registry import ModelRegistry

        cls, _ = ModelRegistry.resolve_model_cls("Qwen2_5_VLForConditionalGeneration")
        self.assertTrue(hasattr(cls, "load_weights"))


if __name__ == "__main__":
    unittest.main()
