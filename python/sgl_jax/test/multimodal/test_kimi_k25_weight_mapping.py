import types
from types import SimpleNamespace

import pytest


def _make_stub(num_hidden_layers, first_k_dense_replace, n_routed_experts):
    from flax import nnx

    from sgl_jax.srt.layers.embeddings import ParallelLMHead
    from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vl_generation import (
        KimiK25ForConditionalGeneration,
    )

    model = SimpleNamespace(
        lm_head=nnx.eval_shape(lambda: ParallelLMHead(16, 8)),
        config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            first_k_dense_replace=first_k_dense_replace,
            n_routed_experts=n_routed_experts,
            moe_layer_freq=1,
            moe_backend="epmoe",
        ),
        hf_weight_prefix="language_model.",
        loader=SimpleNamespace(
            is_static_quant=False,
            is_quant_ignored=lambda k: False,
        ),
    )
    model._create_layer_mappings = types.MethodType(
        KimiK25ForConditionalGeneration._create_layer_mappings, model
    )

    config = SimpleNamespace(
        num_hidden_layers=num_hidden_layers,
        first_k_dense_replace=first_k_dense_replace,
        n_routed_experts=n_routed_experts,
        quantization_config=None,
    )
    return KimiK25ForConditionalGeneration._create_weight_mappings(model, config)


def test_embed_tokens_has_language_model_prefix():
    mappings = _make_stub(num_hidden_layers=1, first_k_dense_replace=0, n_routed_experts=None)
    assert "language_model.model.embed_tokens.weight" in mappings
    assert (
        mappings["language_model.model.embed_tokens.weight"].target_path
        == "model.embed_tokens.embedding"
    )


def test_lm_head_has_language_model_prefix():
    mappings = _make_stub(num_hidden_layers=1, first_k_dense_replace=0, n_routed_experts=None)
    assert "language_model.lm_head.weight" in mappings
    assert mappings["language_model.lm_head.weight"].target_path == "lm_head.embedding"


def test_layer_keys_have_language_model_prefix():
    mappings = _make_stub(num_hidden_layers=2, first_k_dense_replace=0, n_routed_experts=None)
    layer_keys = [k for k in mappings if "layers." in k]
    assert all(k.startswith("language_model.") for k in layer_keys)
    assert all(
        not m.target_path.startswith("language_model.")
        for k in layer_keys
        for m in [mappings[k]]
        if isinstance(m.target_path, str)
    )


def test_no_target_has_language_model_prefix():
    mappings = _make_stub(num_hidden_layers=2, first_k_dense_replace=1, n_routed_experts=256)
    for m in mappings.values():
        targets = m.target_path[0] if isinstance(m.target_path, list) else [m.target_path]
        for t in targets:
            assert not t.startswith("language_model."), f"target_path should not have prefix: {t}"


@pytest.mark.parametrize("enable_dp_lm_head", [None, False, True])
@pytest.mark.parametrize("dp_size", [1, 2, 8])
def test_nested_config_preserves_lm_head_policy(monkeypatch, enable_dp_lm_head, dp_size):
    import jax
    import numpy as np
    from flax import nnx
    from jax.sharding import AxisType, Mesh

    from sgl_jax.srt.models import deepseek_v3
    from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vl_generation import (
        KimiK25ForConditionalGeneration,
    )

    if len(jax.devices()) < 8:
        pytest.skip("Requires 8 devices; set JAX_NUM_CPU_DEVICES=8")

    # Keep the real wrapper, LM head, logits processor, and weight mapping;
    # the transformer backbone is irrelevant to this construction contract.
    monkeypatch.setattr(deepseek_v3, "DeepseekV3Model", lambda *args, **kwargs: nnx.Module())
    text_config = SimpleNamespace(
        num_attention_heads=8,
        vocab_size=10,
        hidden_size=8,
        n_routed_experts=None,
        num_hidden_layers=0,
        enable_dp_lm_head=not bool(enable_dp_lm_head),
    )
    config = SimpleNamespace(text_config=text_config)
    if enable_dp_lm_head is not None:
        config.enable_dp_lm_head = enable_dp_lm_head
    mesh = Mesh(
        np.array(jax.devices()[:8]).reshape(dp_size, 8 // dp_size),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )
    model = nnx.eval_shape(lambda: KimiK25ForConditionalGeneration(config=config, mesh=mesh))

    enabled = bool(enable_dp_lm_head)
    axes = ("tensor", None) if enabled else (("data", "tensor"), None)
    partitions = 8 // dp_size if enabled else 8
    padding = -text_config.vocab_size % partitions
    assert model.lm_head.enable_dp_lm_head is enabled
    assert model.logits_processor.enable_dp_lm_head is enabled
    assert model.lm_head.kernel_axes == axes
    assert model.lm_head.embedding.value.shape == (10 + padding, 8)
    mapping = model._create_weight_mappings(SimpleNamespace(quantization_config=None))[
        "language_model.lm_head.weight"
    ]
    assert mapping.sharding == axes
    assert mapping.pad_width == (((0, padding), (0, 0)) if padding else None)
