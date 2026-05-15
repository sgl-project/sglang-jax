from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from transformers import PretrainedConfig

from sgl_jax.srt.layers.radix_attention import AttentionType
from sgl_jax.srt.models.gemma3 import (
    Gemma3ForCausalLM,
    _get_forward_batch_input_embeds,
    _get_layer_type,
    _get_rope_theta,
    get_attention_sliding_window_size,
)
from sgl_jax.srt.models.registry import ModelRegistry


class _CaptureAttentionBackend:
    def __init__(self):
        self.q = None
        self.k = None
        self.v = None
        self.layer = None

    def __call__(self, q, k, v, layer, forward_batch, token_to_kv_pool, **kwargs):
        del forward_batch, token_to_kv_pool, kwargs
        self.q = q
        self.k = k
        self.v = v
        self.layer = layer
        output = jnp.zeros(
            (q.shape[0], layer.q_head_num * layer.head_dim),
            dtype=q.dtype,
        )
        return output, jnp.zeros((1,), dtype=q.dtype)


def _single_device_mesh():
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1, 1)
    return jax.sharding.Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _tiny_gemma3_config(**overrides):
    values = {
        "architectures": ["Gemma3ForCausalLM"],
        "vocab_size": 128,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "query_pre_attn_scalar": 8,
        "sliding_window": 8,
        "layer_types": ["sliding_attention", "full_attention"],
        "attention_bias": False,
        "hidden_activation": "gelu_pytorch_tanh",
        "rope_parameters": {
            "sliding_attention": {"rope_theta": 10000.0},
            "full_attention": {"rope_theta": 1000000.0},
        },
        "tie_word_embeddings": True,
    }
    values.update(overrides)
    return PretrainedConfig(**values)


def _tiny_outer_conditional_config(text_config):
    return PretrainedConfig(
        architectures=["Gemma3ForConditionalGeneration"],
        text_config=text_config,
    )


def _build_model(config, dtype=jnp.bfloat16):
    mesh = _single_device_mesh()
    with jax.set_mesh(mesh):
        return Gemma3ForCausalLM(config, mesh=mesh, dtype=dtype)


def test_registry_exposes_gemma3_causal_lm():
    assert ModelRegistry.resolve_model_cls(["Gemma3ForCausalLM"]) == (
        Gemma3ForCausalLM,
        "Gemma3ForCausalLM",
    )


def test_rope_and_sliding_window_helpers_use_gemma3_conventions():
    config = _tiny_gemma3_config()

    assert get_attention_sliding_window_size(config) == 8
    assert _get_rope_theta(config, "sliding_attention") == 10000.0
    assert _get_rope_theta(config, "full_attention") == 1000000.0


def test_layer_type_falls_back_to_sliding_window_pattern():
    config = _tiny_gemma3_config(layer_types=None, sliding_window_pattern=3)

    assert [_get_layer_type(config, i) for i in range(6)] == [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]


def test_tiny_text_model_initializes_gemma3_specific_attention():
    model = _build_model(_tiny_gemma3_config())
    layer0 = model.model.layers[0].self_attn
    layer1 = model.model.layers[1].self_attn

    assert layer0.head_dim == 8
    assert layer0.scaling == 8**-0.5
    assert layer0.q_norm.weight[...].shape == (8,)
    assert layer0.k_norm.weight[...].shape == (8,)
    assert layer0.attn.sliding_window_size == 8
    assert layer0.attn.logit_cap is None
    assert layer0.rotary_emb.base == 10000.0
    assert layer1.attn.sliding_window_size is None
    assert layer1.rotary_emb.base == 1000000.0
    assert model.logits_processor.soft_cap is None


def test_text_decoder_uses_causal_decoder_attention_type():
    model = _build_model(_tiny_gemma3_config())

    assert [layer.self_attn.attn.attn_type for layer in model.model.layers] == [
        AttentionType.DECODER,
        AttentionType.DECODER,
    ]


def test_attention_applies_qk_norm_and_partial_rope_before_backend():
    model = _build_model(
        _tiny_gemma3_config(num_hidden_layers=1, partial_rotary_factor=0.5),
        dtype=jnp.float32,
    )
    attn = model.model.layers[0].self_attn
    backend = _CaptureAttentionBackend()
    positions = jnp.asarray([1, 2, 3], dtype=jnp.int32)
    forward_batch = SimpleNamespace(positions=positions, attn_backend=backend)
    hidden_states = (
        jnp.arange(3 * model.config.hidden_size, dtype=jnp.float32).reshape(
            3,
            model.config.hidden_size,
        )
        / 10.0
    )

    attn.q_norm.weight[...] = jnp.linspace(0.1, 0.8, attn.head_dim)
    attn.k_norm.weight[...] = jnp.linspace(-0.2, 0.5, attn.head_dim)

    raw_q, _ = attn.q_proj(hidden_states)
    raw_k, _ = attn.k_proj(hidden_states)
    expected_q = attn.q_norm(raw_q.reshape(-1, attn.num_heads, attn.head_dim))
    expected_k = attn.k_norm(raw_k.reshape(-1, attn.num_kv_heads, attn.head_dim))
    normed_q = expected_q
    normed_k = expected_k
    expected_q, expected_k = attn.rotary_emb(positions, expected_q, expected_k)

    with jax.set_mesh(model.mesh):
        attn(hidden_states, forward_batch, token_to_kv_pool=None)

    np.testing.assert_allclose(
        np.asarray(backend.q),
        np.asarray(expected_q),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(backend.k),
        np.asarray(expected_k),
        rtol=1e-5,
    )
    assert np.asarray(backend.v).shape == (3, 1, attn.head_dim)
    rotary_dim = attn.rotary_emb.rotary_dim
    assert rotary_dim == attn.head_dim // 2
    np.testing.assert_allclose(
        np.asarray(backend.q[..., rotary_dim:]),
        np.asarray(normed_q[..., rotary_dim:]),
        rtol=1e-5,
    )
    assert not np.allclose(
        np.asarray(backend.q[..., :rotary_dim]),
        np.asarray(normed_q[..., :rotary_dim]),
    )
    assert not np.allclose(
        np.asarray(backend.k[..., :rotary_dim]),
        np.asarray(normed_k[..., :rotary_dim]),
    )


def test_decoder_layer_defers_mlp_residual_until_model_final_add():
    model = _build_model(_tiny_gemma3_config(num_hidden_layers=1), dtype=jnp.float32)
    layer = model.model.layers[0]
    backend = _CaptureAttentionBackend()
    forward_batch = SimpleNamespace(
        positions=jnp.asarray([0, 1], dtype=jnp.int32),
        attn_backend=backend,
    )
    hidden_states = (
        jnp.arange(2 * model.config.hidden_size, dtype=jnp.float32).reshape(
            2,
            model.config.hidden_size,
        )
        / 10.0
        + 0.5
    )

    layer.mlp.gate_proj.weight[...] = jnp.zeros_like(layer.mlp.gate_proj.weight[...])
    layer.mlp.up_proj.weight[...] = jnp.zeros_like(layer.mlp.up_proj.weight[...])
    layer.mlp.down_proj.weight[...] = jnp.zeros_like(layer.mlp.down_proj.weight[...])

    with jax.set_mesh(model.mesh):
        output, residual, _ = layer(
            hidden_states,
            forward_batch,
            token_to_kv_pool=None,
            residual=None,
        )

    np.testing.assert_allclose(
        np.asarray(output),
        np.zeros_like(output),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(residual),
        np.asarray(hidden_states),
        rtol=1e-6,
    )


def test_text_weight_mappings_include_qk_norms():
    model = _build_model(_tiny_gemma3_config())
    mappings = model._create_weight_mappings()

    q_norm = mappings["model.layers.0.self_attn.q_norm.weight"]
    k_norm = mappings["model.layers.0.self_attn.k_norm.weight"]

    assert q_norm.target_path == "model.layers.0.self_attn.q_norm.weight"
    assert q_norm.sharding == (None,)
    assert q_norm.transpose is False
    assert k_norm.target_path == "model.layers.0.self_attn.k_norm.weight"
    assert k_norm.sharding == (None,)
    assert k_norm.transpose is False


def test_conditional_language_model_weight_mappings_use_prefixed_hf_keys():
    text_config = _tiny_gemma3_config()
    model = _build_model(_tiny_outer_conditional_config(text_config))
    mappings = model._create_weight_mappings()

    assert (
        mappings["language_model.model.embed_tokens.weight"].target_path
        == "model.embed_tokens.embedding"
    )
    assert (
        mappings["language_model.model.layers.0.self_attn.q_norm.weight"].target_path
        == "model.layers.0.self_attn.q_norm.weight"
    )
    assert (
        mappings["language_model.model.layers.0.self_attn.k_norm.weight"].sharding
        == (None,)
    )
    assert "model.layers.0.self_attn.q_norm.weight" not in mappings


def test_forward_batch_input_embedding_only_used_for_extend_modes():
    class Mode:
        def __init__(self, result):
            self.result = result

        def is_extend_or_draft_extend_or_mixed(self):
            return self.result

    input_embedding = jnp.ones((2, 16), dtype=jnp.bfloat16)

    assert (
        _get_forward_batch_input_embeds(
            SimpleNamespace(forward_mode=Mode(True), input_embedding=input_embedding)
        )
        is input_embedding
    )
    assert (
        _get_forward_batch_input_embeds(
            SimpleNamespace(forward_mode=Mode(False), input_embedding=input_embedding)
        )
        is None
    )
