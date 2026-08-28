import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.bailing_moe_v3 import BailingMoeV3Config
from sgl_jax.srt.hf_transformers_utils import get_config
from sgl_jax.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
    attn_backend_wrapper,
)
from sgl_jax.srt.layers.attention.linear.kda_backend import KDAAttnBackend
from sgl_jax.srt.layers.moe import EPMoE
from sgl_jax.srt.model_loader.arch import get_model_architecture
from sgl_jax.srt.models.bailing_moe_v3 import (
    BailingKDAAttention,
    BailingMLA,
    BailingMoeV3DecoderLayer,
    BailingMoeV3ForCausalLM,
)
from sgl_jax.srt.utils.mesh_utils import create_device_mesh


def _tiny_config(**overrides):
    defaults = dict(
        architectures=["BailingMoeV3ForCausalLM"],
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        q_lora_rank=4,
        kv_lora_rank=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        layer_group_size=2,
        short_conv_kernel_size=4,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=8,
        moe_shared_expert_intermediate_size=8,
        first_k_dense_replace=1,
        max_position_embeddings=128,
    )
    defaults.update(overrides)
    return BailingMoeV3Config(**defaults)


def test_ling3_config_exposes_recurrent_radix_state_layout():
    cfg = _tiny_config(num_hidden_layers=8, layer_group_size=4)

    assert cfg.linear_layer_ids == [0, 1, 2, 4, 5, 6]
    assert cfg.full_attention_layer_ids == [3, 7]
    state = cfg.linear_state_params
    assert state.layers == cfg.linear_layer_ids
    assert state.num_heads == 2
    assert state.head_dim == 8
    assert state.conv_kernel_size == 4


def test_ling3_config_rejects_invalid_layer_group_size():
    with pytest.raises(ValueError, match="layer_group_size must be positive"):
        _tiny_config(layer_group_size=0)


def test_bailing_hybrid_disk_config_routes_to_ling3(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "bailing_hybrid",
                "architectures": ["BailingMoeV3ForCausalLM"],
                "num_hidden_layers": 2,
                "layer_group_size": 2,
            }
        )
    )

    cfg = get_config(str(tmp_path), trust_remote_code=False)

    assert isinstance(cfg, BailingMoeV3Config)
    assert cfg.architectures == ["BailingMoeV3ForCausalLM"]


def test_ling3_architecture_and_hybrid_backend_are_registered():
    cfg = _tiny_config()
    model_cls, arch = get_model_architecture(
        SimpleNamespace(
            hf_config=cfg,
            model_impl="auto",
            model_path="",
            revision=None,
        )
    )
    assert model_cls is BailingMoeV3ForCausalLM
    assert arch == "BailingMoeV3ForCausalLM"

    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    full_backend = object()
    backend = attn_backend_wrapper(
        SimpleNamespace(
            linear_recurrent_config=cfg,
            kimi_linear_config=None,
            bailing_moe_v3_config=cfg,
            qwen3_5_hybrid_config=None,
            lightning_config=None,
            mesh=mesh,
        ),
        full_backend,
    )
    assert isinstance(backend, HybridLinearAttnBackend)
    assert backend.full_attn_backend is full_backend
    assert isinstance(backend.linear_attn_backend, KDAAttnBackend)
    assert backend.full_attn_layers == frozenset({1})


def test_ling3_model_builds_kda_mla_and_epmoe_on_current_main():
    cfg = _tiny_config(use_absorbed_mla=False)
    cfg.ep_size = 1
    cfg.moe_backend = "epmoe"
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])

    with jax.set_mesh(mesh):
        model = BailingMoeV3ForCausalLM(cfg, mesh, dtype=jnp.float32)

    assert isinstance(model.model.layers[0].self_attn, BailingKDAAttention)
    assert isinstance(model.model.layers[1].self_attn, BailingMLA)
    assert model.model.layers[1].self_attn.use_absorbed is False
    assert isinstance(model.model.layers[1].experts, EPMoE)
    assert model.model.layers[1].topk.mesh is mesh


def test_ling3_kda_lower_bound_is_used_by_decode_gate():
    layer = SimpleNamespace(
        A_log=nnx.Param(jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)),
        dt_bias=nnx.Param(jnp.array([0.0, 1.0, -1.0, 0.5], dtype=jnp.float32)),
        kda_lower_bound=-5.0,
    )
    gate = jnp.zeros((1, 2, 2), dtype=jnp.float32)

    actual = KDAAttnBackend()._fused_kda_gate(layer, gate)
    expected = -5.0 * jax.nn.sigmoid(layer.dt_bias.value.reshape(2, 2))

    assert jnp.allclose(actual, expected[None, ...])


def test_ling3_kda_lower_bound_is_threaded_to_prefill(monkeypatch):
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    backend = KDAAttnBackend(mesh=mesh)
    captured = {}

    def fake_chunk_kda(q, k, v, g, beta, *, initial_state, lower_bound, **kwargs):
        captured["lower_bound"] = lower_bound
        return v, initial_state

    monkeypatch.setattr(
        "sgl_jax.srt.layers.attention.linear.kda_backend.chunk_kda",
        fake_chunk_kda,
    )
    layer = SimpleNamespace(
        A_log=nnx.Param(jnp.zeros((1, 1, 2, 1), dtype=jnp.float32)),
        dt_bias=nnx.Param(jnp.zeros((4,), dtype=jnp.float32)),
        kda_lower_bound=-5.0,
        scale=0.5,
    )
    qkv = jnp.zeros((1, 2, 2), dtype=jnp.float32)

    with jax.set_mesh(mesh):
        output, final_state = backend._forward_extend(
            qkv,
            qkv,
            qkv,
            qkv,
            jnp.zeros((1, 2), dtype=jnp.float32),
            jnp.zeros((1, 2, 2, 2), dtype=jnp.float32),
            jnp.array([0, 1], dtype=jnp.int32),
            layer,
        )

    assert output.shape == qkv.shape
    assert final_state.shape == (1, 2, 2, 2)
    assert captured["lower_bound"] == -5.0


def test_ling3_epmoe_preserves_data_parallel_output_layout():
    mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
    captured = {}

    class Identity:
        def __call__(self, value):
            return value

    class Attention:
        def __call__(self, positions, hidden_states, forward_batch, pool):
            return hidden_states, None

    class Gate:
        bias = None

        def __call__(self, hidden_states):
            return jnp.zeros((hidden_states.shape[0], 4), dtype=jnp.float32)

    class TopKCapture:
        def __call__(self, router_logits, correction_bias, **kwargs):
            captured["routing_sharding"] = kwargs["routing_sharding"]
            tokens = router_logits.shape[0]
            return jnp.ones((tokens, 1)), jnp.zeros((tokens, 1), dtype=jnp.int32)

    class ExpertsCapture:
        def __call__(self, hidden_states, topk_weights, topk_ids, *, out_sharding):
            captured["out_sharding"] = out_sharding
            return hidden_states

    layer = object.__new__(BailingMoeV3DecoderLayer)
    object.__setattr__(layer, "mesh", mesh)
    object.__setattr__(layer, "is_kda", False)
    object.__setattr__(layer, "is_moe_layer", True)
    object.__setattr__(layer, "use_fused", False)
    object.__setattr__(layer, "input_layernorm", Identity())
    object.__setattr__(layer, "post_attention_layernorm", Identity())
    object.__setattr__(layer, "self_attn", Attention())
    object.__setattr__(layer, "shared_experts", None)
    object.__setattr__(layer, "moe_gate", Gate())
    object.__setattr__(layer, "topk", TopKCapture())
    object.__setattr__(layer, "experts", ExpertsCapture())

    hidden_states = jnp.ones((1, 4), dtype=jnp.float32)
    layer(
        jnp.zeros((1,), dtype=jnp.int32),
        hidden_states,
        SimpleNamespace(),
        SimpleNamespace(token_to_kv_pool=None, recurrent_state_pool=None),
    )

    expected = NamedSharding(mesh, P("data", None))
    assert captured["routing_sharding"] == expected
    assert captured["out_sharding"] == expected
