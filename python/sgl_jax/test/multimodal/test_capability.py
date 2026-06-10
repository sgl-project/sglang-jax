"""Tests for mm_core.capability (pure python; runnable on any interpreter)."""

import unittest

from sgl_jax.srt.mm_core import capability as cap


class _TextOnly:
    """A plain text LLM: no encoders, no markers."""


class _ImageVLM:
    audio_kind = None
    has_deepstack = False

    def encode_image(self, items):  # noqa: D401 - test stub
        return None


class _OmniVLM:
    audio_kind = "features"
    has_deepstack = True

    def encode_image(self, items):
        return None

    def encode_video(self, items):
        return None

    def encode_audio(self, items):
        return None


class _CodesAudioVLM:
    audio_kind = "codes"

    def encode_audio(self, items):
        return None


class _ExplicitMarker:
    is_multimodal = True  # explicit marker, no encoders


class _ExplicitModalities:
    supported_modalities = ("image",)  # explicit override; no encode_image method


class TestCapability(unittest.TestCase):
    def test_text_only_is_not_multimodal(self):
        self.assertFalse(cap.is_multimodal_arch(_TextOnly))
        self.assertEqual(cap.supported_modalities(_TextOnly), set())

    def test_image_vlm(self):
        self.assertTrue(cap.is_multimodal_arch(_ImageVLM))
        self.assertEqual(cap.supported_modalities(_ImageVLM), {"image"})
        self.assertIsNone(cap.audio_kind(_ImageVLM))
        self.assertFalse(cap.has_deepstack(_ImageVLM))

    def test_omni_vlm(self):
        self.assertTrue(cap.is_multimodal_arch(_OmniVLM))
        self.assertEqual(cap.supported_modalities(_OmniVLM), {"image", "video", "audio"})
        self.assertEqual(cap.audio_kind(_OmniVLM), "features")
        self.assertTrue(cap.has_deepstack(_OmniVLM))

    def test_codes_audio(self):
        self.assertEqual(cap.audio_kind(_CodesAudioVLM), "codes")
        self.assertEqual(cap.supported_modalities(_CodesAudioVLM), {"audio"})

    def test_explicit_marker(self):
        self.assertTrue(cap.is_multimodal_arch(_ExplicitMarker))

    def test_explicit_modalities_override(self):
        self.assertEqual(cap.supported_modalities(_ExplicitModalities), {"image"})
        self.assertTrue(cap.is_multimodal_arch(_ExplicitModalities))


if __name__ == "__main__":
    unittest.main()
