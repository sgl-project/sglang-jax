import types
from types import SimpleNamespace


def _make_stub(num_hidden_layers, first_k_dense_replace, n_routed_experts):
    from sgl_jax.srt.multimodal.models.kimi_k25.kimi_k25_vl_generation import (
        KimiK25ForConditionalGeneration,
    )

    model = SimpleNamespace(
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
