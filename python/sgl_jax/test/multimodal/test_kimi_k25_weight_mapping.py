import types
from types import SimpleNamespace

import pytest


def _make_stub(
    num_hidden_layers,
    first_k_dense_replace,
    n_routed_experts,
    quant_config=None,
    moe_backend="epmoe",
):
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
            n_shared_experts=1,
            moe_layer_freq=1,
            moe_backend=moe_backend,
        ),
        hf_weight_prefix="language_model.",
        loader=SimpleNamespace(
            is_static_quant=quant_config.is_static_checkpoint if quant_config else False,
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
        quantization_config=quant_config,
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


def test_int4_moe_weight_mappings():
    import jax.numpy as jnp

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig

    quant_config = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[],
        moe_weight_dtype=getattr(jnp, "int4", None) or getattr(jnp, "uint4", None),
    )
    mappings = _make_stub(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=16,
        quant_config=quant_config,
    )

    # Layer 0 (dense) linear weights should be unquantized (.weight)
    assert "language_model.model.layers.0.mlp.gate_proj.weight" in mappings

    # Layer 1 (MoE) expert weights should use weight_packed
    expert_group_wi_0 = mappings["__MOE_EXPERTS__model.layers.1.mlp.wi_0"]
    assert any(".weight_packed" in k for k in expert_group_wi_0.target_path[1:])

    # Layer 1 (MoE) scales should use .weight_scale and tensor sharding
    scale_group_wi_0 = mappings["__MOE_EXPERTS__model.layers.1.mlp.wi_0_scale"]
    scale_group_wo = mappings["__MOE_EXPERTS__model.layers.1.mlp.wo_scale"]
    assert any(".weight_scale" in k for k in scale_group_wi_0.target_path[1:])
    assert scale_group_wi_0.sharding == ("expert", "tensor", None)
    assert scale_group_wo.sharding == ("expert", None, "tensor")

    # Shared experts should map to layer.shared_experts (non-fused path)
    shared_gate = mappings["language_model.model.layers.1.mlp.shared_experts.gate_proj.weight"]
    assert shared_gate.target_path == "model.layers.1.shared_experts.gate_proj.weight"


def test_dynamic_int4_and_static_fp8_moe_weight_mappings():
    import jax.numpy as jnp

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig

    # 1. Dynamic INT4 (is_static_checkpoint=False) must load ordinary .weight tensors
    dyn_int4_config = QuantizationConfig(
        is_static_checkpoint=False,
        linear_rules=[],
        moe_weight_dtype=getattr(jnp, "int4", None) or getattr(jnp, "uint4", None),
    )
    dyn_mappings = _make_stub(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=16,
        quant_config=dyn_int4_config,
    )
    expert_group_dyn = dyn_mappings["__MOE_EXPERTS__model.layers.1.mlp.wi_0"]
    assert all(k.endswith(".weight") for k in expert_group_dyn.target_path[1:])
    assert "__MOE_EXPERTS__model.layers.1.mlp.wi_0_scale" not in dyn_mappings

    # 2. Static FP8 must keep replicated scale_sharding ("expert", None, None) and .weight_scale_inv
    fp8_config = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[],
        moe_weight_dtype=jnp.float8_e4m3fn,
    )
    fp8_mappings = _make_stub(
        num_hidden_layers=2,
        first_k_dense_replace=1,
        n_routed_experts=16,
        quant_config=fp8_config,
    )
    scale_group_fp8 = fp8_mappings["__MOE_EXPERTS__model.layers.1.mlp.wi_0_scale"]
    assert all(k.endswith(".weight_scale_inv") for k in scale_group_fp8.target_path[1:])
    assert scale_group_fp8.sharding == ("expert", None, None)


def test_static_int4_fused_mapping_rejected():
    import jax.numpy as jnp

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig

    config = QuantizationConfig(is_static_checkpoint=True, moe_weight_dtype=jnp.int4)
    with pytest.raises(ValueError, match="require moe_backend='epmoe'"):
        _make_stub(2, 1, 16, config, moe_backend="fused")


@pytest.mark.parametrize(
    "ignored,rule,quantized",
    [
        (["model.layers.0.mlp"], ".*", True),
        (["model.layers.0.mlp.gate_proj"], ".*", False),
        (["gate_proj"], ".*", False),
        ([], r"model/layers\[0\]/mlp/gate_proj", True),
        ([], r"model/layers\[0\]/self_attn/.*", False),
    ],
)
def test_linear_mapping_matches_quantized_model(ignored, rule, quantized):
    import jax
    import numpy as np
    from flax import nnx
    from jax.sharding import AxisType, Mesh

    from sgl_jax.srt.configs.quantization_config import QuantizationConfig
    from sgl_jax.srt.layers.linear import LinearBase, QuantizedLinear
    from sgl_jax.srt.utils.quantization.quantization_utils import (
        apply_linear_quantization,
    )

    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )

    class MLP(nnx.Module):
        def __init__(self):
            self.gate_proj = LinearBase(64, 64, mesh, use_bias=False, kernel_axes=(None, "tensor"))

    class Layer(nnx.Module):
        def __init__(self):
            self.mlp = MLP()

    class Body(nnx.Module):
        def __init__(self):
            self.layers = nnx.data([Layer()])

    class Model(nnx.Module):
        def __init__(self):
            self.model = Body()

    config = QuantizationConfig(
        is_static_checkpoint=True,
        linear_rules=[{"module_path": rule, "weight_dtype": "float8_e4m3fn"}],
        ignored_layers=ignored,
    )
    with jax.set_mesh(mesh):
        model = nnx.eval_shape(Model)
        apply_linear_quantization(SimpleNamespace(quantization_config=config), model, True)
    layer = model.model.layers[0].mlp.gate_proj
    assert isinstance(layer, QuantizedLinear) is quantized
    mapping = _make_stub(1, 1, None, config)["language_model.model.layers.0.mlp.gate_proj.weight"]
    suffix = mapping.target_path.rsplit(".", 1)[1]
    assert suffix == ("weight_q" if quantized else "weight")
    assert hasattr(layer, suffix)
