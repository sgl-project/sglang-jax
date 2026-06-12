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


if __name__ == "__main__":
    unittest.main()
