from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from sgl_jax.srt.models.minimax_m3 import (
    MiniMaxM3Attention,
    MiniMaxM3DecoderLayer,
    MiniMaxM3MLP,
    swigluoai,
)


@pytest.fixture
def mesh():
    m = jax.make_mesh((1, 1), ("data", "tensor"))
    with jax.sharding.set_mesh(m):
        yield m


def _ref_swigluoai(gate, up, alpha, limit):
    gate = np.clip(gate, a_max=limit, a_min=None)
    up = np.clip(up, -limit, limit)
    return (up + 1.0) * gate * (1.0 / (1.0 + np.exp(-gate * alpha)))


@pytest.mark.unit
def test_swigluoai_matches_ref():
    rng = np.random.default_rng(0)
    gate = rng.standard_normal((7, 16)).astype(np.float32) * 10  # exercise clamp
    up = rng.standard_normal((7, 16)).astype(np.float32) * 10
    out = np.asarray(swigluoai(jnp.asarray(gate), jnp.asarray(up), alpha=1.702, limit=7.0))
    ref = _ref_swigluoai(gate, up, alpha=1.702, limit=7.0)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


@pytest.mark.unit
def test_mlp_forward_shape(mesh):
    mlp = MiniMaxM3MLP(
        hidden_size=64,
        intermediate_size=128,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        mesh=mesh,
        dtype=jnp.float32,
    )
    out = mlp(jnp.ones((3, 64), dtype=jnp.float32))
    assert out.shape == (3, 64)


def _tiny_text_config(**overrides):
    cfg = SimpleNamespace(
        hidden_size=128,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=32,
        rms_norm_eps=1e-6,
        rotary_dim=16,
        rope_theta=5_000_000,
        max_position_embeddings=2048,
        moe_layer_freq=[0, 0, 1, 1],
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        intermediate_size=64,
        dense_intermediate_size=96,
        shared_intermediate_size=64,
        routed_scaling_factor=2.0,
        scoring_func="sigmoid",
        use_routing_bias=True,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        ep_size=1,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


@pytest.mark.unit
def test_attention_eval_shape(mesh):
    cfg = _tiny_text_config()
    attn = nnx.eval_shape(
        lambda: MiniMaxM3Attention(cfg, mesh=mesh, layer_id=0, dtype=jnp.bfloat16)
    )
    assert attn.q_norm.weight.shape == (cfg.head_dim,)
    assert attn.k_norm.weight.shape == (cfg.head_dim,)
    assert attn.q_proj.weight.shape[-1] == cfg.num_attention_heads * cfg.head_dim


@pytest.mark.unit
def test_decoder_layer_dense_dispatch(mesh):
    cfg = _tiny_text_config()
    layer = nnx.eval_shape(
        lambda: MiniMaxM3DecoderLayer(cfg, mesh=mesh, layer_id=0, dtype=jnp.bfloat16)
    )
    assert layer.is_moe is False
    assert hasattr(layer, "mlp") and not hasattr(layer, "moe_gate")
    assert layer.mlp.gate_proj.weight.shape[-1] == cfg.dense_intermediate_size


@pytest.mark.unit
def test_moe_layer_freq_dispatch():
    cfg = _tiny_text_config()
    assert [bool(cfg.moe_layer_freq[i]) for i in range(4)] == [False, False, True, True]


@pytest.mark.unit
def test_epmoe_swigluoai_activation(mesh):
    from sgl_jax.srt.layers.moe import EPMoE

    moe = EPMoE(
        hidden_size=32,
        num_experts=2,
        num_experts_per_tok=1,
        intermediate_dim=64,
        mesh=mesh,
        ep_size=1,
        weight_dtype=jnp.float32,
        dtype=jnp.float32,
        activation="swigluoai",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        layer_id=0,
    )
    assert moe.activation == "swigluoai"
    assert moe.swiglu_alpha == 1.702
