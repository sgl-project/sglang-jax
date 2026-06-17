"""Capability-first ``is_multimodal_model`` (review code-review.md §9-§11).

``is_multimodal_model`` resolves the served ``architectures[0]`` to its registered model class and
reads the per-class capability, instead of trusting the hf_config ``vision_config``/``audio_config``
key heuristic. These tests pin the two failure modes the rewrite fixes:

  - false-NEGATIVE: a real VLM whose checkpoint config carries no vision_config (Fuyu /
    Phi-3.5-vision / Phi-4-multimodal) used to be classified text-only -> media silently dropped.
  - false-POSITIVE: a text model whose config auto-injects a vision_config (grok-2) used to be
    forced onto the multimodal path.

and that the config-key heuristic survives ONLY as a fallback for an arch with no registered class.
Needs the real ModelRegistry, so it runs where jax is importable (pod CPU), not on the bare
interpreter.
"""

import types
import unittest

from sgl_jax.srt.configs.model_config import is_multimodal_model
from sgl_jax.srt.mm_core import capability as cap
from sgl_jax.srt.models.registry import ModelRegistry


def _cfg(arch, **kw):
    """Minimal hf_config stand-in: an ``architectures`` list + whatever config keys the test sets."""
    return types.SimpleNamespace(architectures=[arch], **kw)


def _a_registered_multimodal_arch():
    for arch, c in ModelRegistry.models.items():
        if isinstance(c, type) and cap.is_multimodal_arch(c):
            return arch
    raise unittest.SkipTest("no multimodal model registered")


def _a_registered_text_arch():
    for arch, c in ModelRegistry.models.items():
        if isinstance(c, type) and not cap.is_multimodal_arch(c):
            return arch
    raise unittest.SkipTest("no text-only model registered")


class TestIsMultimodalModelCapabilityFirst(unittest.TestCase):
    def test_vlm_with_flat_config_is_multimodal(self):
        # The core fix: a registered VLM whose config has NO vision_config/audio_config must still
        # be detected as multimodal (capability wins over the absent config key).
        arch = _a_registered_multimodal_arch()
        self.assertTrue(is_multimodal_model(_cfg(arch)))  # no vision_config at all

    def test_text_model_with_spurious_vision_config_is_not_multimodal(self):
        # The grok-2 false-positive: a registered text model whose config injects a vision_config
        # must stay text-only (capability wins over the spurious config key).
        arch = _a_registered_text_arch()
        cfg = _cfg(arch, vision_config=types.SimpleNamespace())
        self.assertFalse(is_multimodal_model(cfg))

    def test_unresolvable_arch_falls_back_to_config_key(self):
        # An arch with no registered class isn't natively servable; is_multimodal then degrades to
        # the best-effort config-key proxy (top-level and thinker_config nesting).
        self.assertFalse(is_multimodal_model(_cfg("NotARegisteredArch__xyz")))
        self.assertTrue(
            is_multimodal_model(
                _cfg("NotARegisteredArch__xyz", vision_config=types.SimpleNamespace())
            )
        )
        self.assertTrue(
            is_multimodal_model(
                _cfg(
                    "NotARegisteredArch__xyz",
                    thinker_config=types.SimpleNamespace(audio_config=types.SimpleNamespace()),
                )
            )
        )


class TestSupportedModalitiesDeclarations(unittest.TestCase):
    """The in-model VLM declares explicit supported_modalities (review §11.6 hardening).

    Only MiMo-V2.5 runs in-model on this branch; Qwen2.5-VL / Qwen3-Omni are staged (multi-stage)
    and so are not registered in the in-model ModelRegistry here."""

    EXPECTED = {
        "MiMoV2_5ForConditionalGeneration": {"image", "video", "audio"},
    }

    def test_explicit_sets(self):
        for arch, expected in self.EXPECTED.items():
            c = ModelRegistry.models.get(arch)
            if not isinstance(c, type):
                self.skipTest(f"{arch} not registered")
            self.assertEqual(cap.supported_modalities(c), expected, arch)


if __name__ == "__main__":
    unittest.main()
