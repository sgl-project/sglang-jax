from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np


def _install_import_stubs():
    jax_stub = types.ModuleType("jax")
    jax_stub.Array = np.ndarray
    jax_stub.nn = types.SimpleNamespace(gelu=lambda value: value)
    jax_stub.sharding = types.SimpleNamespace(Mesh=object)
    jax_numpy_stub = types.ModuleType("jax.numpy")
    jax_numpy_stub.swapaxes = np.swapaxes
    jax_numpy_stub.concatenate = np.concatenate
    jax_numpy_stub.repeat = np.repeat
    jax_numpy_stub.int32 = np.int32
    jax_numpy_stub.bfloat16 = np.float32
    jax_stub.numpy = jax_numpy_stub
    sys.modules.setdefault("jax", jax_stub)
    sys.modules.setdefault("jax.numpy", jax_numpy_stub)

    flax_stub = types.ModuleType("flax")
    nnx_stub = types.ModuleType("flax.nnx")
    nnx_stub.Module = object
    nnx_stub.Rngs = lambda *args, **kwargs: None
    nnx_stub.List = list
    flax_stub.nnx = nnx_stub
    sys.modules.setdefault("flax", flax_stub)
    sys.modules.setdefault("flax.nnx", nnx_stub)

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.PretrainedConfig = type("PretrainedConfig", (), {})
    sys.modules.setdefault("transformers", transformers_stub)

    model_config_stub = types.ModuleType("sgl_jax.srt.configs.model_config")
    model_config_stub.ModelConfig = type("ModelConfig", (), {})
    sys.modules.setdefault("sgl_jax.srt.configs.model_config", model_config_stub)

    embeddings_stub = types.ModuleType("sgl_jax.srt.layers.embeddings")
    embeddings_stub.Embed = type("Embed", (), {})
    sys.modules.setdefault("sgl_jax.srt.layers.embeddings", embeddings_stub)

    linear_stub = types.ModuleType("sgl_jax.srt.layers.linear")
    linear_stub.LinearBase = type("LinearBase", (), {})
    sys.modules.setdefault("sgl_jax.srt.layers.linear", linear_stub)

    backbone_stub = types.ModuleType("sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_backbone")
    backbone_stub.MiMoAudioTransformer = type("MiMoAudioTransformer", (), {})
    sys.modules.setdefault(
        "sgl_jax.srt.multimodal.models.mimo_audio.mimo_audio_backbone",
        backbone_stub,
    )

    weight_utils_stub = types.ModuleType("sgl_jax.srt.utils.weight_utils")
    weight_utils_stub.WeightLoader = type("WeightLoader", (), {})
    weight_utils_stub.WeightMapping = type("WeightMapping", (), {})
    sys.modules.setdefault("sgl_jax.srt.utils.weight_utils", weight_utils_stub)


_install_import_stubs()

from sgl_jax.srt.multimodal.models.mimo_v2_5.embedding import (  # noqa: E402
    MiMoV25AudioUnderstandingEncoder,
)


class TestMiMoV25AudioEncoderAxes(unittest.TestCase):
    def _new_encoder(self):
        encoder = MiMoV25AudioUnderstandingEncoder.__new__(MiMoV25AudioUnderstandingEncoder)
        encoder.audio_channels = 20
        encoder.group_size = 4
        return encoder

    def test_time_major_codes_are_channel_last_even_when_time_equals_channels(self):
        encoder = self._new_encoder()
        codes = np.arange(20 * 20, dtype=np.int32).reshape(20, 20)

        normalized = encoder._ensure_channel_first_audio_codes(codes)

        self.assertEqual(normalized.shape, (1, 20, 20))
        np.testing.assert_array_equal(normalized, np.swapaxes(codes[None, ...], 1, 2))

    def test_channel_major_codes_are_preserved(self):
        encoder = self._new_encoder()
        codes = np.arange(20 * 5, dtype=np.int32).reshape(20, 5)

        normalized = encoder._ensure_channel_first_audio_codes(codes)

        self.assertEqual(normalized.shape, (1, 20, 5))
        np.testing.assert_array_equal(normalized, codes[None, ...])

    def test_batched_time_major_codes_are_channel_last(self):
        encoder = self._new_encoder()
        codes = np.arange(2 * 5 * 20, dtype=np.int32).reshape(2, 5, 20)

        normalized = encoder._ensure_channel_first_audio_codes(codes)

        self.assertEqual(normalized.shape, (2, 20, 5))
        np.testing.assert_array_equal(normalized, np.swapaxes(codes, 1, 2))

    def test_batched_channel_major_codes_are_preserved(self):
        encoder = self._new_encoder()
        codes = np.arange(2 * 20 * 5, dtype=np.int32).reshape(2, 20, 5)

        normalized = encoder._ensure_channel_first_audio_codes(codes)

        self.assertEqual(normalized.shape, (2, 20, 5))
        np.testing.assert_array_equal(normalized, codes)

    def test_rejects_missing_channel_axis(self):
        encoder = self._new_encoder()

        with self.assertRaisesRegex(ValueError, "audio_codes must be shaped"):
            encoder._ensure_channel_first_audio_codes(np.zeros((5, 8), dtype=np.int32))

    def test_rejects_empty_time_major_codes(self):
        encoder = self._new_encoder()

        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            encoder._ensure_channel_first_audio_codes(np.zeros((0, 20), dtype=np.int32))

    def test_rejects_empty_channel_major_codes(self):
        encoder = self._new_encoder()

        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            encoder._ensure_channel_first_audio_codes(np.zeros((20, 0), dtype=np.int32))

    def test_rejects_empty_batched_codes(self):
        encoder = self._new_encoder()

        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            encoder._ensure_channel_first_audio_codes(np.zeros((0, 4, 20), dtype=np.int32))
        with self.assertRaisesRegex(ValueError, "audio_codes cannot be empty"):
            encoder._ensure_channel_first_audio_codes(np.zeros((2, 0, 20), dtype=np.int32))

    def test_group_audio_codes_pads_tail_by_repeating_last_frame(self):
        encoder = self._new_encoder()
        codes = np.arange(1 * 20 * 5, dtype=np.int32).reshape(1, 20, 5)

        grouped = encoder._group_audio_codes(codes)

        self.assertEqual(grouped.shape, (1, 2, 4, 20))
        np.testing.assert_array_equal(grouped[:, 0], np.swapaxes(codes[:, :, :4], 1, 2))
        np.testing.assert_array_equal(grouped[:, 1, 0], codes[:, :, 4])
        np.testing.assert_array_equal(grouped[:, 1, 1], codes[:, :, 4])
        np.testing.assert_array_equal(grouped[:, 1, 2], codes[:, :, 4])
        np.testing.assert_array_equal(grouped[:, 1, 3], codes[:, :, 4])

    def test_init_rejects_partial_rotary_factor(self):
        # Guard lives in MiMoV25AudioUnderstandingEncoder.__init__ (mimo-v2.5 only),
        # not in the shared MiMoAudioAttention, so mimo-audio is unaffected. It must
        # refuse a partial-rotary checkpoint instead of silently using full rotary.
        cfg = SimpleNamespace(
            partial_rotary_factor=0.5,
            audio_channels=20,
            speech_vocab_size=1280,
            group_size=4,
            input_local_dim=1024,
        )
        with self.assertRaisesRegex(ValueError, "full rotary"):
            MiMoV25AudioUnderstandingEncoder(cfg, mesh=None, rngs=None)

    def test_init_rejects_non_upstream_audio_tower_contract(self):
        base = dict(
            partial_rotary_factor=1.0,
            audio_channels=20,
            speech_vocab_size=1280,
            group_size=4,
            input_local_dim=1024,
        )
        bad_configs = [
            ("input_full_attention", False, "input_full_attention"),
            ("add_post_norm", False, "add_post_norm"),
            ("projection_layers", 1, "projection_layers"),
            ("input_local_hidden_dropout", 0.1, "input_local_hidden_dropout"),
        ]
        for key, value, message in bad_configs:
            with self.subTest(key=key):
                cfg = SimpleNamespace(**base, **{key: value})
                with self.assertRaisesRegex(ValueError, message):
                    MiMoV25AudioUnderstandingEncoder(cfg, mesh=None, rngs=None)


if __name__ == "__main__":
    unittest.main()
