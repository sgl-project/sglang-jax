"""M3 regression: Qwen2_5_VLProcessor arch-keyed registration into the mm_core registry.

Registry resolution only (no HF AutoProcessor instantiation -> no torch/model-cache needed).
The full process() image->input_ids+mm_items transform is validated on the TPU dev pod
(tmp/refactor/test_qwen_vl_processor.py): registry resolve + process() PASS, placeholders==N.
"""

from __future__ import annotations

import unittest

from sgl_jax.srt.mm_core.processor import (
    get_processor_cls,
    import_processor_classes,
    supported_processor_archs,
)


class TestQwenVLProcessorRegistry(unittest.TestCase):
    def test_registers_and_resolves_by_arch(self):
        import_processor_classes("sgl_jax.srt.multimodal.processors")
        cls = get_processor_cls(["Qwen2_5_VLForConditionalGeneration"])
        self.assertIsNotNone(cls)
        self.assertEqual(cls.__name__, "Qwen2_5_VLProcessor")
        self.assertIn("Qwen2_5_VLForConditionalGeneration", supported_processor_archs())

    def test_unknown_arch_resolves_none(self):
        import_processor_classes("sgl_jax.srt.multimodal.processors")
        self.assertIsNone(get_processor_cls(["NonexistentArchForTest"]))


if __name__ == "__main__":
    unittest.main()
