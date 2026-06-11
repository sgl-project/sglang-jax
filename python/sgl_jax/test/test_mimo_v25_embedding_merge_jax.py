"""Real-jnp (CPU backend) coverage for MiMoV2_5Embedding scatter + encoder axes.

The other mimo_v25 unit tests stub jnp with numpy, which hides jnp-specific
behavior of ``.at[].set`` and dtype handling (review D6-3 / D6-4). This module
exercises the actual jax ops and is skipped when jax is unavailable.
"""

import unittest

try:
    import jax  # noqa: F401
    import jax.numpy as jnp

    _HAS_JAX = True
except Exception:  # pragma: no cover - jax optional in some envs
    _HAS_JAX = False


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestMiMoV25ScatterJax(unittest.TestCase):
    def _new_embedding(self):
        from sgl_jax.srt.multimodal.models.mimo_v2_5.embedding import MiMoV2_5Embedding

        return MiMoV2_5Embedding.__new__(MiMoV2_5Embedding)

    def test_scatter_modality_fills_placeholder_positions(self):
        obj = self._new_embedding()
        input_ids = jnp.array([1, 7, 7, 2], dtype=jnp.int32)
        input_embeds = jnp.zeros((4, 3), dtype=jnp.float32)
        modality_embeds = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=jnp.float32)

        merged = obj._scatter_modality(input_ids, input_embeds, modality_embeds, token_id=7)

        # placeholder rows get features, text rows stay zero (no token-0 clobber)
        self.assertEqual(merged[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(merged[1].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(merged[2].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(merged[3].tolist(), [0.0, 0.0, 0.0])

    def test_scatter_modality_none_is_passthrough(self):
        obj = self._new_embedding()
        input_ids = jnp.array([1, 2, 3], dtype=jnp.int32)
        input_embeds = jnp.ones((3, 2), dtype=jnp.float32)
        self.assertTrue(
            bool((obj._scatter_modality(input_ids, input_embeds, None, 7) == input_embeds).all())
        )
        # also a no-op when token_id is unresolved
        feats = jnp.zeros((1, 2), dtype=jnp.float32)
        self.assertTrue(
            bool(
                (obj._scatter_modality(input_ids, input_embeds, feats, None) == input_embeds).all()
            )
        )

    def test_encoder_ensure_channel_first_square_is_time_major(self):
        from sgl_jax.srt.multimodal.models.mimo_v2_5.audio_encoder import (
            MiMoV25AudioUnderstandingEncoder,
        )

        encoder = MiMoV25AudioUnderstandingEncoder.__new__(MiMoV25AudioUnderstandingEncoder)
        encoder.audio_channels = 20
        codes = jnp.arange(20 * 20, dtype=jnp.int32).reshape(20, 20)
        out = encoder._ensure_channel_first_audio_codes(codes)
        # [T=20, C=20] time-major -> [B=1, C=20, T=20]
        self.assertEqual(tuple(out.shape), (1, 20, 20))


class _FakeAudioTower:
    """Returns features only when audio_codes are actually provided."""

    def __init__(self, out=None):
        self.out = out

    def __call__(self, *, input_features=None, audio_feature_lengths=None, audio_codes=None):
        return self.out if audio_codes is not None else None


@unittest.skipUnless(_HAS_JAX, "jax not installed")
class TestMiMoV25AnyModalitySubset(unittest.TestCase):
    """MiMo-V2.5 must accept any subset of {text, audio, vision}, not all-present."""

    def _new_embedding(self, *, audio_out=None):
        from sgl_jax.srt.multimodal.models.mimo_v2_5.embedding import MiMoV2_5Embedding

        emb = MiMoV2_5Embedding.__new__(MiMoV2_5Embedding)
        emb.dtype = jnp.float32
        emb.audio_token_id = 7
        emb.image_token_id = 8
        emb.video_token_id = 9
        emb.text_embed_tokens = lambda ids: jnp.zeros((ids.shape[0], 3), dtype=jnp.float32)
        emb.audio_encoder = _FakeAudioTower(audio_out)
        emb.visual = None  # vision not wired
        return emb

    def test_text_only_request(self):
        emb = self._new_embedding()
        ids = jnp.array([1, 2, 3], dtype=jnp.int32)
        out = emb(ids)  # no audio/image/video inputs at all
        self.assertEqual(tuple(out.input_embeds.shape), (3, 3))
        self.assertTrue(bool((out.input_embeds == 0).all()))  # pure text, no scatter

    def test_audio_only_request(self):
        audio_out = jnp.array([[5.0, 5.0, 5.0]], dtype=jnp.float32)
        emb = self._new_embedding(audio_out=audio_out)
        ids = jnp.array([1, 7, 2], dtype=jnp.int32)  # one audio placeholder, no vision
        out = emb(ids, audio_codes=jnp.zeros((4, 20), dtype=jnp.int32))
        self.assertEqual(out.input_embeds[1].tolist(), [5.0, 5.0, 5.0])
        self.assertEqual(out.input_embeds[0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(out.input_embeds[2].tolist(), [0.0, 0.0, 0.0])

    def test_audio_token_id_present_but_no_audio_input_is_noop(self):
        # token id is configured, but the request carries no audio -> no scatter
        emb = self._new_embedding(audio_out=jnp.array([[5.0, 5.0, 5.0]], dtype=jnp.float32))
        ids = jnp.array([1, 7, 2], dtype=jnp.int32)
        out = emb(ids)  # audio_codes=None -> fake tower returns None
        self.assertTrue(bool((out.input_embeds == 0).all()))

    def test_image_input_raises_until_vision_wired(self):
        emb = self._new_embedding()
        ids = jnp.array([1, 8, 2], dtype=jnp.int32)
        with self.assertRaises(NotImplementedError):
            emb(ids, pixel_values=jnp.zeros((1, 4), dtype=jnp.float32))

    def test_video_input_raises_until_vision_wired(self):
        emb = self._new_embedding()
        ids = jnp.array([1, 9, 2], dtype=jnp.int32)
        with self.assertRaises(NotImplementedError):
            emb(ids, pixel_values_videos=jnp.zeros((1, 4), dtype=jnp.float32))


if __name__ == "__main__":
    unittest.main()
