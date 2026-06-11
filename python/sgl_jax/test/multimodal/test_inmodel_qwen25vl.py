"""Tests for the Qwen2.5-VL in-model path (refactor M3 step 1+2).

Requires jax. Run on the project env (Python >=3.12):
    python -m pytest python/sgl_jax/test/multimodal/test_inmodel_qwen25vl.py
"""

import unittest


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
