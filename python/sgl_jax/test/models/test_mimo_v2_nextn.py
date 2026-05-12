"""Weight-mapping coverage test for MiMoV2MTPForCausalLM.

Verifies, without touching TPU, that the mapping dict covers exactly the
tensors present in V2.5-Pro's ``model_mtp.safetensors`` for each layer
index, and that ``mtp_layer_idx`` correctly selects per-layer keys.
"""

from types import SimpleNamespace

import pytest

# Ground truth: keys from XiaomiMiMo/MiMo-V2.5-Pro model_mtp.safetensors header
# (16 tensors per layer × 3 layers = 48). Derived from the live HF header.
_PER_LAYER_KEYS = [
    "eh_proj.weight",
    "enorm.weight",
    "final_layernorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.down_proj.weight_scale_inv",
    "mlp.gate_proj.weight",
    "mlp.gate_proj.weight_scale_inv",
    "mlp.up_proj.weight",
    "mlp.up_proj.weight_scale_inv",
    "pre_mlp_layernorm.weight",
    "self_attn.attention_sink_bias",
    "self_attn.o_proj.weight",
    "self_attn.qkv_proj.weight",
    "self_attn.qkv_proj.weight_scale_inv",
]


def _hf_keys(layer_idx: int) -> set[str]:
    return {f"model.mtp.layers.{layer_idx}.{k}" for k in _PER_LAYER_KEYS}


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_weight_mapping_covers_safetensors(layer_idx: int):
    from sgl_jax.srt.models.mimo_v2_nextn import MiMoV2MTPForCausalLM

    model = SimpleNamespace(
        mtp_layer_idx=layer_idx,
        loader=SimpleNamespace(is_static_quant=True, is_quant_ignored=lambda k: False),
    )
    mappings = MiMoV2MTPForCausalLM._create_weight_mappings(model)

    expected = _hf_keys(layer_idx)
    got = set(mappings.keys())
    assert (
        got == expected
    ), f"layer {layer_idx}: missing={sorted(expected - got)} extra={sorted(got - expected)}"

    # Per-layer selection: mappings for layer N must not reference any other layer.
    for other in {0, 1, 2} - {layer_idx}:
        assert not any(f"mtp.layers.{other}." in k for k in got)

    # Target paths are unique (no two HF keys write the same JAX param).
    targets = []
    for m in mappings.values():
        tp = m.target_path
        targets.extend(tp if isinstance(tp, list) else [tp])
    assert len(targets) == len(set(targets)), "duplicate target_path"
