from types import SimpleNamespace

import pytest

from sgl_jax.srt.configs.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4LayerType,
    classify_layers,
    hash_moe_layer_flags,
    mhc_param_shapes,
    mix_hc_width,
    trunk_compress_ratios,
)


def test_flash_backbone_uses_only_trunk_ratios():
    config = DeepseekV4Config()
    types = classify_layers(config)

    assert len(config.compress_ratios) == 46
    assert len(types) == config.num_hidden_layers == 43
    assert types[:2] == (DeepseekV4LayerType.SWA_ONLY,) * 2
    assert types[2:] == (DeepseekV4LayerType.C4A, DeepseekV4LayerType.C128A) * 20 + (
        DeepseekV4LayerType.C4A,
    )
    assert config.attention_layer_types == types
    assert hash_moe_layer_flags(config) == (True,) * 3 + (False,) * 40
    assert config.hash_moe_layers == hash_moe_layer_flags(config)


def test_moe_and_mhc_contract_uses_epic_flash_defaults():
    config = DeepseekV4Config()

    assert config.hidden_size == 4096
    assert config.n_routed_experts == 256
    assert config.num_experts_per_tok == 6
    assert config.n_shared_experts == 1
    assert config.num_hash_layers == 3
    assert config.hc_mult == 4
    assert config.hc_sinkhorn_iters == 20
    assert mix_hc_width(config.hc_mult) == 24
    assert mhc_param_shapes(config) == {
        "fn": (24, 16384),
        "base": (24,),
        "scale": (3,),
        "head_fn": (4, 16384),
        "head_base": (4,),
        "head_scale": (1,),
    }


def test_mutable_defaults_are_isolated_and_mtp_tail_is_not_backbone():
    first = DeepseekV4Config()
    second = DeepseekV4Config()
    first.compress_ratios[0] = 4
    first.dspark_target_layer_ids.append(10)

    assert second.compress_ratios[0] == 0
    assert second.dspark_target_layer_ids == [40, 41, 42]
    assert classify_layers(second)[-1] == DeepseekV4LayerType.C4A
    assert DeepseekV4Config.from_dict(second.to_dict()).compress_ratios == second.compress_ratios


@pytest.mark.parametrize("ratio", [2, 16])
def test_unsupported_backbone_ratio_is_rejected(ratio):
    config = DeepseekV4Config(compress_ratios=[ratio] + [0] * 42)
    with pytest.raises(ValueError, match="compress_ratio"):
        classify_layers(config)


@pytest.mark.parametrize("ratio", [-1, 0, 1])
def test_epic_swa_ratio_classification_is_preserved(ratio):
    config = DeepseekV4Config(compress_ratios=[ratio] + [0] * 42)
    assert classify_layers(config)[0] == DeepseekV4LayerType.SWA_ONLY


def test_epic_config_boundaries_are_preserved():
    with pytest.raises(ValueError, match="compress_ratios has"):
        classify_layers(DeepseekV4Config(compress_ratios=[0] * 42))
    assert classify_layers(DeepseekV4Config(compress_ratios=[4.0] + [0] * 42))[0] == (
        DeepseekV4LayerType.C4A
    )
    with pytest.raises(ValueError, match="num_hash_layers"):
        hash_moe_layer_flags(DeepseekV4Config(num_hash_layers=44))


def test_missing_ratio_list_keeps_epic_swa_only_fallback():
    config = SimpleNamespace(num_hidden_layers=2, compress_ratios=None)
    assert trunk_compress_ratios(config) == (1, 1)
    assert classify_layers(config) == (DeepseekV4LayerType.SWA_ONLY,) * 2
