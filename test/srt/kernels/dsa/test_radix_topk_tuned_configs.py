"""Tests for score-size/top-k radix tuning lookup."""

import pytest

from sgl_jax.srt.kernels.radix_topk import tuned_configs


def test_make_supported_exact_config_and_alignment():
    config = tuned_configs.make_radix_topk_config(
        num_seq_windows=1,
        digit_width=4,
        num_digits=8,
        use_tc_tiling_on_sc=False,
    )

    assert config.num_digits == 8
    assert config.input_alignment == 256


def test_lookup_is_keyed_by_score_size_and_topk(monkeypatch):
    config = tuned_configs.RadixTopKConfig(num_seq_windows=1)
    monkeypatch.setattr(tuned_configs, "_device_name", lambda: "test-tpu")
    monkeypatch.setitem(
        tuned_configs.TUNED_RADIX_TOPK_CONFIGS,
        "test-tpu",
        {(135168, 2048): config},
    )

    assert tuned_configs.get_tuned_radix_topk_config(135168, 2048) == config
    assert tuned_configs.get_tuned_radix_topk_config(135168, 1024) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_seq_windows": 0},
        {"num_seq_windows": 2},
        {"digit_width": 7, "num_digits": 5},
        {"digit_width": 8, "num_digits": 8},
        {"digit_width": 4, "num_digits": 4},
        {"digit_width": 4, "num_digits": 8, "use_tc_tiling_on_sc": True},
    ],
)
def test_rejects_non_exact_or_unsupported_configs(kwargs):
    with pytest.raises(ValueError):
        tuned_configs.RadixTopKConfig(**kwargs)
