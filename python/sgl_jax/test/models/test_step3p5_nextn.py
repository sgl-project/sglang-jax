"""CPU weight-mapping coverage test for Step3p5MTPForCausalLM.

Verifies, without touching TPU, that the mapping dict covers exactly the 17
MTP tensors present per layer in the real Step-3.5-Flash checkpoint index
(``model.layers.{45,46,47}.*``), that ``mtp_layer_idx`` selects the right
per-layer keys, and that the draft architecture resolves via the registry.

Ground truth: the 51 keys delivered from the live
``model.safetensors.index.json`` (17 per layer × 3 layers).
"""

from types import SimpleNamespace

import pytest

# 17 per-layer suffixes from the real checkpoint index (BF16, separate q/k/v,
# shared_head nested under `.transformer.`).
_PER_LAYER_SUFFIXES = [
    "eh_proj.weight",
    "enorm.weight",
    "hnorm.weight",
    "input_layernorm.weight",
    "mlp.down_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "post_attention_layernorm.weight",
    "self_attn.g_proj.weight",
    "self_attn.k_norm.weight",
    "self_attn.k_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.v_proj.weight",
    "transformer.shared_head.norm.weight",
    "transformer.shared_head.output.weight",
]

_MTP_START = 45  # num_hidden_layers (45) → MTP layers are 45/46/47.


def _hf_keys(abs_layer: int) -> set[str]:
    return {f"model.layers.{abs_layer}.{s}" for s in _PER_LAYER_SUFFIXES}


def _mock_model(layer_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        mtp_layer_idx=layer_idx,
        model=SimpleNamespace(mtp_abs_layer=_MTP_START + layer_idx),
    )


@pytest.mark.parametrize("layer_idx", [0, 1, 2])
def test_weight_mapping_covers_index(layer_idx: int):
    from sgl_jax.srt.models.step3p5_nextn import Step3p5MTPForCausalLM

    mappings = Step3p5MTPForCausalLM._create_weight_mappings(_mock_model(layer_idx))

    expected = _hf_keys(_MTP_START + layer_idx)
    got = set(mappings.keys())
    assert (
        got == expected
    ), f"layer {layer_idx}: missing={sorted(expected - got)} extra={sorted(got - expected)}"

    # Per-layer selection: layer N maps only its own keys.
    for other in {0, 1, 2} - {layer_idx}:
        assert not any(f"model.layers.{_MTP_START + other}." in k for k in got)

    # No two source keys write the same JAX param.
    targets = []
    for m in mappings.values():
        tp = m.target_path
        targets.extend(tp if isinstance(tp, list) else [tp])
    assert len(targets) == len(set(targets)), "duplicate target_path"


def test_key_targets_are_correct():
    from sgl_jax.srt.models.step3p5_nextn import Step3p5MTPForCausalLM

    m = Step3p5MTPForCausalLM._create_weight_mappings(_mock_model(0))

    # Own LM head loads from the checkpoint's shared_head.output (not the target).
    lm = m["model.layers.45.transformer.shared_head.output.weight"]
    assert lm.target_path == "lm_head.embedding"
    assert lm.transpose is False
    # Final norm before the LM head.
    assert m["model.layers.45.transformer.shared_head.norm.weight"].target_path == (
        "model.shared_head_norm.weight"
    )
    # GemmaRMSNorm params are `.weight` (not `.scale`), enorm/hnorm live on the model.
    assert m["model.layers.45.enorm.weight"].target_path == "model.enorm.weight"
    assert m["model.layers.45.hnorm.weight"].target_path == "model.hnorm.weight"
    # eh_proj is a plain Linear → transposed into [in, out].
    assert m["model.layers.45.eh_proj.weight"].transpose is True
    # Attention/MLP land on the reused decoder block.
    assert m["model.layers.45.self_attn.q_proj.weight"].target_path == (
        "model.mtp_block.self_attn.q_proj.weight"
    )
    assert m["model.layers.45.mlp.gate_proj.weight"].target_path == (
        "model.mtp_block.mlp.gate_proj.weight"
    )


def test_registry_resolves_draft_arch():
    from sgl_jax.srt.models.registry import ModelRegistry

    model_cls, _ = ModelRegistry.resolve_model_cls(["Step3p5MTPForCausalLM"])
    assert model_cls.__name__ == "Step3p5MTPForCausalLM"
    # MTP owns its LM head; only the token embedding is injected from the target.
    assert model_cls.load_lm_head_from_target is False
