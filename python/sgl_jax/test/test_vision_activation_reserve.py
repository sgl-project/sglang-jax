"""Unit tests for the G1 vision/audio activation-reserve sizing (design §5.7).

Regression for review P1-a: MiMo-V2.5 ships nested vision_config / audio_config as plain dicts, so
the old getattr-based sizing silently degraded every field to Whisper defaults (16 / 1280 / 1500).
These tests assert dict-form configs are read correctly and the audio reserve is sized by tower
FORM (codes tower vs whisper tower), exercising the pure helpers without a ModelRunner.
"""

import unittest

from sgl_jax.srt.model_executor.model_runner_kv_cache_mixin import (
    _audio_activation_reserve_bytes,
    _cfg_get,
    _mm_activation_reserve_bytes,
    _tower_attn_reserve_bytes,
    _vision_probe_geometry,
)


class _Obj:
    """Stand-in for an HF config object (attribute access)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# MiMo-V2.5 codes audio tower, exactly as AutoConfig leaves it (a plain dict).
MIMO_AUDIO = {
    "input_local_dim": 1024,
    "input_local_attn_heads": 16,
    "input_local_head_dim": 64,
    "group_size": 4,
    "audio_channels": 20,
}
MIMO_VISION = {"num_heads": 16, "hidden_size": 1280, "depth": 32}
# Full MiMo-V2.5 vision geometry, exactly as the checkpoint nests it (a plain dict; note `in_chans`,
# not `in_channels`). The AOT probe must read these dict-safe -> patch_dim 1536, not the 1176 the old
# getattr-default build produced.
MIMO_VISION_GEOM = {
    "patch_size": 16,
    "temporal_patch_size": 2,
    "in_chans": 3,
    "spatial_merge_size": 2,
    "num_heads": 32,
    "hidden_size": 1280,
}
# Whisper-form audio tower (Qwen3-Omni style), as an object.
WHISPER_AUDIO = _Obj(encoder_attention_heads=20, d_model=1280, max_source_positions=1500)

WHISPER_DEFAULT = _tower_attn_reserve_bytes(16, 1280, 1500)  # the old silent fallback value


class TestCfgGet(unittest.TestCase):
    def test_reads_dict_and_object(self):
        self.assertEqual(_cfg_get({"a": 5}, "a", default=1), 5)
        self.assertEqual(_cfg_get(_Obj(a=5), "a", default=1), 5)

    def test_first_present_of_several(self):
        self.assertEqual(_cfg_get({"b": 7}, "a", "b", default=1), 7)

    def test_missing_returns_default_for_dict(self):
        # The bug: getattr(dict, "missing") raised AttributeError -> caught by getattr's default ->
        # silently returned the default for EVERY field. _cfg_get must do the same *only* when the
        # key is truly absent, while actually finding present keys.
        self.assertEqual(_cfg_get({"x": 1}, "missing", default=99), 99)
        self.assertIsNone(_cfg_get({}, "missing"))


class TestAudioReserve(unittest.TestCase):
    def test_codes_tower_is_not_whisper_default(self):
        codes = _audio_activation_reserve_bytes(MIMO_AUDIO, max_patches=4096)
        self.assertNotEqual(codes, WHISPER_DEFAULT)
        self.assertGreater(codes, 0)

    def test_codes_tower_scales_with_real_fields(self):
        base = _audio_activation_reserve_bytes(MIMO_AUDIO, max_patches=4096)
        wider = _audio_activation_reserve_bytes(
            {**MIMO_AUDIO, "input_local_dim": 2048}, max_patches=4096
        )
        more_tokens = _audio_activation_reserve_bytes(MIMO_AUDIO, max_patches=8192)
        self.assertGreater(wider, base)  # reads input_local_dim (was silently ignored before)
        self.assertGreater(more_tokens, base)  # scales with the token budget

    def test_whisper_tower_uses_source_positions(self):
        a = _audio_activation_reserve_bytes(WHISPER_AUDIO, max_patches=4096)
        self.assertEqual(a, _tower_attn_reserve_bytes(20, 1280, 1500))

    def test_none_config_is_zero(self):
        self.assertEqual(_audio_activation_reserve_bytes(None, max_patches=4096), 0)


class TestMMReserve(unittest.TestCase):
    def test_zero_when_no_max_patches(self):
        self.assertEqual(_mm_activation_reserve_bytes(MIMO_VISION, MIMO_AUDIO, 0, 0), 0)

    def test_dict_vision_fields_are_read(self):
        base = _mm_activation_reserve_bytes(MIMO_VISION, None, 4096, 0)
        wider = _mm_activation_reserve_bytes({**MIMO_VISION, "hidden_size": 2560}, None, 4096, 0)
        self.assertGreater(wider, base)  # hidden_size actually consumed (dict-safe)

    def test_aot_vision_takes_priority(self):
        aot = 12_345_678
        self.assertEqual(_mm_activation_reserve_bytes(MIMO_VISION, None, 4096, aot), aot)

    def test_vision_plus_audio_is_additive(self):
        v_only = _mm_activation_reserve_bytes(MIMO_VISION, None, 4096, 0)
        v_plus_a = _mm_activation_reserve_bytes(MIMO_VISION, MIMO_AUDIO, 4096, 0)
        self.assertEqual(v_plus_a, v_only + _audio_activation_reserve_bytes(MIMO_AUDIO, 4096))


class TestVisionProbeGeometry(unittest.TestCase):
    """Regression for the G1 AOT probe shape bug: the probe built a (N, 1176) dummy because it read
    the nested-dict vision_config with bare getattr (-> default patch_size 14, 3*2*14*14 = 1176), and
    the ViT patch_embed rejected it ('Expected flattened patch dim 1536, got input shape (N, 1176)').
    The patch_embed contract is patch_dim == in_chans * temporal_patch * patch_size**2 == 1536."""

    def test_patch_dim_is_1536_from_dict_config(self):
        _, patch_dim, _, _ = _vision_probe_geometry(MIMO_VISION_GEOM, max_patches=8192)
        self.assertEqual(patch_dim, 1536)  # 3 * 2 * 16 * 16; was 1176 with the getattr-default bug

    def test_old_getattr_default_would_be_1176(self):
        # Document the bug: a dict whose real keys are *missing* (the failure mode of plain getattr on
        # a dict) degrades to patch_size 14 -> the broken 1176.
        _, patch_dim, _, _ = _vision_probe_geometry({}, max_patches=8192)
        self.assertEqual(patch_dim, 3 * 2 * 14 * 14)  # 1176, the degraded value

    def test_patches_cover_max_patches(self):
        # side is rounded UP so patches >= max_patches (never under-measure the worst case).
        patches, _, grid, _ = _vision_probe_geometry(MIMO_VISION_GEOM, max_patches=8000)
        self.assertGreaterEqual(patches, 8000)
        (t, h, w) = grid[0]
        self.assertEqual(t * h * w, patches)  # grid_thw is consistent with the pixel_values rows
        self.assertEqual(h % 2, 0)  # multiple of spatial_merge_size

    def test_seq_after_merge(self):
        patches, _, _, seq = _vision_probe_geometry(MIMO_VISION_GEOM, max_patches=8192)
        # input_ids length = LLM tokens after the 2x2 spatial merge (+ small slack)
        self.assertEqual(seq, patches // 4 + 16)

    def test_bucket_size_rounds_grid_up(self):
        _, _, grid, _ = _vision_probe_geometry(MIMO_VISION_GEOM, max_patches=8000, bucket_size=64)
        (_, h, _) = grid[0]
        self.assertEqual((h // 2) % 64, 0)  # LLM-grid side is a bucket multiple


if __name__ == "__main__":
    unittest.main()
