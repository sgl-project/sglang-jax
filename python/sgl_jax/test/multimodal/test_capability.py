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

    def embed_mm(self, input_ids, **kwargs):  # satisfies the EMBED_MM contract (review H-1)
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

    def embed_mm(self, input_ids, **kwargs):
        return None


class _CodesAudioVLM:
    audio_kind = "codes"

    def encode_audio(self, items):
        return None


class _ExplicitMarker:
    is_multimodal = True  # explicit marker, no encoders


class _ExplicitModalities:
    supported_modalities = ("image",)  # explicit override; no encode_image method


class _ExplicitContractVLM:
    """Spells out the whole EMBED_MM contract explicitly (no **kwargs) -> must pass (review H-1)."""

    def encode_image(self, items):
        return None

    def embed_mm(
        self,
        input_ids,
        mm_pixel_values=None,
        mm_grid_thw=None,
        mm_pixel_values_videos=None,
        mm_video_grid_thw=None,
        mm_audio_features=None,
        mm_audio_feature_lengths=None,
        mm_audio_codes=None,
        mm_real_llm_dims=None,
        mm_real_video_llm_dims=None,
    ):
        return None


class _DriftedEmbedMM:
    """embed_mm predating M5/V-2: missing mm_audio_codes/mm_real_*, no **kwargs -> the H-1 drift."""

    def encode_image(self, items):
        return None

    def embed_mm(
        self,
        input_ids,
        mm_pixel_values=None,
        mm_grid_thw=None,
        mm_pixel_values_videos=None,
        mm_video_grid_thw=None,
        mm_audio_features=None,
        mm_audio_feature_lengths=None,
    ):
        return None


class _NoEmbedMM:
    """Declares a modality encoder (mm-capable) but never defines embed_mm at all."""

    def encode_image(self, items):
        return None


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


class TestReconcile(unittest.TestCase):
    """★5 startup reconciliation (design §3.5.5), via the pure find_capability_inconsistencies."""

    def test_consistent_passes(self):
        registered = {"Img": _ImageVLM, "Txt": _TextOnly}
        # mm-capable _ImageVLM has a processor; served proxy True matches its capability True.
        errs = cap.find_capability_inconsistencies(registered, {"Img"}, ["Img"], True)
        self.assertEqual(errs, [])

    def test_served_proxy_mismatch_flagged(self):
        # config-proxy said NOT multimodal, but the served class declares capability -> drift.
        errs = cap.find_capability_inconsistencies({"Img": _ImageVLM}, {"Img"}, ["Img"], False)
        self.assertTrue(any("mismatch" in e for e in errs), errs)

    def test_missing_processor_flagged(self):
        # mm-capable model with no registered processor -> media would be silently dropped.
        errs = cap.find_capability_inconsistencies({"Img": _ImageVLM}, set(), ["Img"], True)
        self.assertTrue(any("no registered processor" in e for e in errs), errs)

    def test_text_only_needs_no_processor(self):
        errs = cap.find_capability_inconsistencies({"Txt": _TextOnly}, set(), ["Txt"], False)
        self.assertEqual(errs, [])

    def test_multiple_missing_processors_listed(self):
        registered = {"Img": _ImageVLM, "Omni": _OmniVLM, "Txt": _TextOnly}
        errs = cap.find_capability_inconsistencies(registered, set(), ["Txt"], False)
        # both mm-capable archs reported missing; text-only excluded
        joined = " ".join(errs)
        self.assertIn("Img", joined)
        self.assertIn("Omni", joined)
        self.assertNotIn("Txt", joined)

    # ----- (3) uniform embed_mm contract drift (review H-1) -----

    def test_kwargs_embed_mm_satisfies_contract(self):
        # _ImageVLM/_OmniVLM use **kwargs -> accept the whole contract, no contract error.
        registered = {"Img": _ImageVLM, "Omni": _OmniVLM}
        errs = cap.find_capability_inconsistencies(registered, {"Img", "Omni"}, ["Img"], True)
        self.assertFalse(any("embed_mm" in e for e in errs), errs)

    def test_explicit_full_contract_passes(self):
        # A model that spells out all contract params (no **kwargs) must not be flagged.
        errs = cap.find_capability_inconsistencies(
            {"Full": _ExplicitContractVLM}, {"Full"}, [], False
        )
        self.assertFalse(any("embed_mm" in e for e in errs), errs)

    def test_drifted_embed_mm_flagged(self):
        # The exact H-1 failure: embed_mm missing the M5/V-2 params and no **kwargs.
        errs = cap.find_capability_inconsistencies({"Drift": _DriftedEmbedMM}, {"Drift"}, [], False)
        contract_errs = [e for e in errs if "Drift" in e and "embed_mm" in e]
        self.assertTrue(contract_errs, errs)
        msg = contract_errs[0]
        self.assertIn("mm_audio_codes", msg)
        self.assertIn("mm_real_llm_dims", msg)
        self.assertIn("mm_real_video_llm_dims", msg)

    def test_missing_embed_mm_flagged(self):
        # mm-capable but no embed_mm method at all.
        errs = cap.find_capability_inconsistencies({"NoEmbed": _NoEmbedMM}, {"NoEmbed"}, [], False)
        self.assertTrue(any("NoEmbed" in e and "no embed_mm" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
