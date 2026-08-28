import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
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


def _pure_dp_mesh():
    return create_device_mesh(
        ici_parallelism=[jax.device_count(), 1],
        dcn_parallelism=[1, 1],
    )


def _single_device_mesh():
    return create_device_mesh(
        ici_parallelism=[1, 1],
        dcn_parallelism=[1, 1],
        devices=jax.devices()[:1],
    )


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

    mesh = _pure_dp_mesh()
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


def test_ling3_model_builds_kda_mla_and_replicated_moe_on_current_main():
    cfg = _tiny_config(use_absorbed_mla=False)
    cfg.ep_size = 8
    cfg.moe_backend = "epmoe"
    mesh = _pure_dp_mesh()

    with jax.set_mesh(mesh):
        model = BailingMoeV3ForCausalLM(cfg, mesh, dtype=jnp.float32)

    assert isinstance(model.model.layers[0].self_attn, BailingKDAAttention)
    assert isinstance(model.model.layers[1].self_attn, BailingMLA)
    assert model.model.layers[1].self_attn.use_absorbed is False
    assert isinstance(model.model.layers[1].experts, EPMoE)
    assert model.model.layers[1].experts.replicate_experts is True
    assert model.model.layers[1].experts.ep_size == 1
    assert model.model.layers[1].experts.tp_size == 1
    assert model.model.layers[1].experts.wi_0.value.sharding.spec == P(None, None, None)
    assert model.model.layers[1].topk.mesh is mesh

    mappings = model._create_weight_mappings()
    expert_mapping = mappings["__MOE_EXPERTS__model.layers.1.experts.wi_0"]
    assert expert_mapping.sharding == (None, None, None)
    assert expert_mapping.physical_to_logical_map is None


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


def test_replicated_moe_matches_local_dense_reference():
    mesh = _pure_dp_mesh()
    with jax.set_mesh(mesh):
        moe = EPMoE(
            hidden_size=4,
            num_experts=4,
            num_experts_per_tok=2,
            ep_size=4,
            mesh=mesh,
            intermediate_dim=8,
            weight_dtype=jnp.float32,
            dtype=jnp.float32,
            replicate_experts=True,
        )

        hidden_states = jax.device_put(
            jnp.arange(32, dtype=jnp.float32).reshape(8, 4) / 10,
            NamedSharding(mesh, P("data", None)),
        )
        topk_ids = jax.device_put(
            jnp.array(
                [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2], [1, 3], [2, 0], [3, 1]],
                dtype=jnp.int32,
            ),
            NamedSharding(mesh, P("data", None)),
        )
        topk_weights = jax.device_put(
            jnp.array(
                [
                    [0.7, 0.3],
                    [0.6, 0.4],
                    [0.8, 0.2],
                    [0.5, 0.5],
                    [0.9, 0.1],
                    [0.4, 0.6],
                    [0.3, 0.7],
                    [0.2, 0.8],
                ],
                dtype=jnp.float32,
            ),
            NamedSharding(mesh, P("data", None)),
        )
        output = moe(
            hidden_states,
            topk_weights,
            topk_ids,
            out_sharding=NamedSharding(mesh, P("data", None)),
        )

        hidden_np = np.asarray(jax.device_get(hidden_states))
        ids_np = np.asarray(jax.device_get(topk_ids))
        weights_np = np.asarray(jax.device_get(topk_weights))
        wi_0_np = np.asarray(jax.device_get(moe.wi_0.value))
        wi_1_np = np.asarray(jax.device_get(moe.wi_1.value))
        wo_np = np.asarray(jax.device_get(moe.wo.value))
        expected = np.zeros_like(hidden_np)
        for token_idx, token in enumerate(hidden_np):
            for weight, expert_id in zip(weights_np[token_idx], ids_np[token_idx]):
                gate = token @ wi_0_np[expert_id]
                up = token @ wi_1_np[expert_id]
                silu_gate = gate / (1.0 + np.exp(-gate))
                expected[token_idx] += weight * ((silu_gate * up) @ wo_np[expert_id])

    assert output.sharding == NamedSharding(mesh, P("data", None))
    np.testing.assert_allclose(np.asarray(jax.device_get(output)), expected, rtol=2e-4, atol=2e-4)


def test_ling3_kda_lower_bound_is_threaded_to_prefill(monkeypatch):
    # This unit only checks lower_bound plumbing into the prefill kernel. Keep
    # it independent of the process-wide virtual-device count used by DP tests.
    mesh = _single_device_mesh()
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


def test_ling3_replicated_moe_preserves_data_parallel_output_layout():
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
