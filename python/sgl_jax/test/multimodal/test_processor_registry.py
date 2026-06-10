"""Tests for mm_core.processor registry + caps (pure stdlib; runnable anywhere)."""

import unittest

from sgl_jax.srt.mm_core import processor as P


class _ArchA:
    pass


class _ArchB:
    pass


class _ProcAB(P.BaseMultimodalProcessor):
    models = [_ArchA, _ArchB]

    def process(self, *, images=None, videos=None, audios=None, text=None):
        return None


class TestProcessorRegistry(unittest.TestCase):
    def setUp(self):
        P._PROCESSOR_REGISTRY.clear()

    def test_caps_defaults(self):
        c = P.MediaInputCaps()
        self.assertEqual(c.fps, 2.0)
        self.assertEqual(c.fps_max_frames, 768)
        self.assertEqual(c.video_max_pixels, 768 * 28 * 28)
        self.assertEqual(c.max_aspect_ratio, 200)

    def test_register_and_lookup_by_arch_name(self):
        P.register_processor(_ProcAB)
        self.assertIs(P.get_processor_cls("_ArchA"), _ProcAB)
        self.assertIs(P.get_processor_cls(["nope", "_ArchB"]), _ProcAB)
        self.assertEqual(P.supported_processor_archs(), {"_ArchA", "_ArchB"})

    def test_unknown_arch_returns_none(self):
        P.register_processor(_ProcAB)
        self.assertIsNone(P.get_processor_cls("Unknown"))
        self.assertIsNone(P.get_processor_cls([]))

    def test_duplicate_registration_raises(self):
        P.register_processor(_ProcAB)

        class _Other(P.BaseMultimodalProcessor):
            models = [_ArchA]

            def process(self, *, images=None, videos=None, audios=None, text=None):
                return None

        with self.assertRaises(AssertionError):
            P.register_processor(_Other)

    def test_base_is_abstract(self):
        with self.assertRaises(TypeError):
            P.BaseMultimodalProcessor()


if __name__ == "__main__":
    unittest.main()
