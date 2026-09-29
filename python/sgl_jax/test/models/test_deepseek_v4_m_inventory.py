import pytest

from sgl_jax.srt.configs.deepseek_v4 import DeepseekV4Config
from sgl_jax.srt.models.deepseek_v4 import (
    Disposition,
    build_weight_mappings,
    classify_checkpoint,
    expected_trunk_keys,
)


def test_flash_inventory_partitions_exactly_between_m_and_e():
    config = DeepseekV4Config()
    required = expected_trunk_keys(config)
    facts = classify_checkpoint(config, required | {"mtp.2.dspark.weight"})
    m_keys = {key for key, fact in facts.items() if fact.disposition is Disposition.M_OWNED}
    e_keys = {key for key, fact in facts.items() if fact.disposition is Disposition.E_OWNED}

    assert m_keys == set(build_weight_mappings(config))
    assert not m_keys & e_keys
    assert m_keys | e_keys == required
    assert facts["mtp.2.dspark.weight"].disposition is Disposition.DROPPED
    assert "layers.0.ffn.gate.tid2eid" in e_keys
    assert "layers.3.ffn.gate.bias" in e_keys


@pytest.mark.parametrize(
    "key",
    [
        "layers.0.attn.compressor.wkv.weight",
        "layers.2.ffn.gate.bias",
        "layers.3.ffn.gate.tid2eid",
        "layers.3.attn.unknown.weight",
        "layers.43.attn_norm.weight",
    ],
)
def test_invalid_checkpoint_key_is_rejected(key):
    with pytest.raises(ValueError):
        classify_checkpoint(DeepseekV4Config(), [key])
