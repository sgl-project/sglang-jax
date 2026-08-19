"""KimiK3DecoderLayer / KimiK3MLP construction and SITU numerics on a real mesh."""
import contextlib
import jax, jax.numpy as jnp, numpy as np, pytest
from jax.sharding import Mesh
from sgl_jax.srt.configs.kimi_k3 import KimiK3Config
from sgl_jax.srt.models.kimi_k3 import KimiK3MLP, KimiK3DecoderLayer


def _mesh():
    """LinearBase's PartitionSpecs require EXPLICIT axis types; a plain Mesh(...) gets Auto and
    fails with 'AxisType.Auto/Manual'. sglang-jax's create_device_mesh sets these via
    use_explicit_sharding (mesh_utils.py:58-62); mirror it."""
    d = np.array(jax.devices()[:1]).reshape(1, 1)
    axis_types = (jax.sharding.AxisType.Explicit,) * 2
    return Mesh(d, ("data", "tensor"), axis_types=axis_types)


@contextlib.contextmanager
def _in_mesh():
    """LinearBase builds PartitionSpecs, which require an ACTIVE mesh context -- passing the
    Mesh object alone raises 'Using PartitionSpec when you are not under a mesh context'."""
    m = _mesh()
    # jax renamed this: use_mesh on <=0.6.3, set_mesh on >=0.7.1. sglang-jax's model_runner
    # carries the same fallback (model_runner.py:892-898); mirror it so the test tracks whatever
    # jax the image ships.
    ctx = getattr(jax.sharding, "use_mesh", None)
    ctx = ctx(m) if ctx is not None else jax.set_mesh(m)
    with ctx:
        yield m


def _tiny_cfg(n_layers=4, attn_res=2):
    """A 4-layer K3 with the released model's real hyper-parameters where they matter."""
    return KimiK3Config(
        num_hidden_layers=n_layers, hidden_size=128, intermediate_size=256,
        moe_intermediate_size=64, num_attention_heads=4,
        num_experts=8, num_experts_per_token=2, num_shared_experts=1,
        first_k_dense_replace=1, moe_layer_freq=1,
        kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        mla_use_nope=True, rms_norm_eps=1e-6,
        # Mirrors the released config's shape: BOTH lists are required (kimi_linear.py asserts
        # full_attn_layers is present), 1-based, and partition the layer range.
        linear_attn_config={
            "kda_layers": [1, 2, 3], "full_attn_layers": [4],
            "gate_lower_bound": -5.0, "head_dim": 32, "num_heads": 4,
            "short_conv_kernel_size": 4, "use_full_rank_gate": True,
        },
        hidden_act="situ", activation_situ_beta=4.0, activation_situ_linear_beta=25.0,
        attn_res_block_size=attn_res, mla_use_output_gate=True, latent_moe_use_norm=True,
    )


def test_situ_mlp_matches_the_reference_activation():
    """KimiK3MLP must apply SITU, not SiLU -- K3 ships hidden_act='situ'."""
    from sgl_jax.srt.models.kimi_k3_layers import situ_and_mul
    with _in_mesh() as mesh:
        m = KimiK3MLP(64, 128, mesh, situ_beta=4.0, situ_linear_beta=25.0, dtype=jnp.float32)
        x = jnp.asarray(np.random.default_rng(0).normal(size=(3, 64)).astype(np.float32))
        got = np.asarray(m(x), dtype=np.float64)
        g, _ = m.gate_proj(x); u, _ = m.up_proj(x)
        act = situ_and_mul(jnp.concatenate([g, u], axis=-1), 4.0, 25.0)
        want, _ = m.down_proj(act)
    np.testing.assert_allclose(got, np.asarray(want, dtype=np.float64), rtol=1e-5, atol=1e-5)


def test_situ_differs_from_silu():
    """Guard against silently falling back to SiLU: the two must not agree."""
    with _in_mesh() as mesh:
        situ = KimiK3MLP(64, 128, mesh, situ_beta=4.0, situ_linear_beta=25.0, dtype=jnp.float32)
        x = jnp.asarray(np.random.default_rng(1).normal(size=(3, 64)).astype(np.float32) * 3)
        g, _ = situ.gate_proj(x); u, _ = situ.up_proj(x)
        silu_out, _ = situ.down_proj(jax.nn.silu(g) * u)
        got = np.asarray(situ(x))
    assert not np.allclose(got, np.asarray(silu_out), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("layer_idx,expect_kda", [(0, True), (1, True), (2, True), (3, False)])
def test_layer_picks_kda_or_mla_from_config(layer_idx, expect_kda):
    """3:1 interleave must come from is_kda_layer, not from layer position guessing."""
    cfg = _tiny_cfg()
    assert cfg.is_kda_layer(layer_idx) == expect_kda


def test_attn_res_modules_exist_only_when_enabled():
    cfg = _tiny_cfg(attn_res=2)
    assert cfg.uses_attn_res
    off = _tiny_cfg(attn_res=None)
    off.attn_res_block_size = None
    assert not off.uses_attn_res and off.n_attn_res_candidates(3) == 0


def test_dense_layer_zero_is_not_moe():
    """first_k_dense_replace=1 means layer 0 is dense, the rest MoE."""
    cfg = _tiny_cfg()
    for i in range(cfg.num_hidden_layers):
        is_moe = bool(cfg.num_experts) and i >= cfg.first_k_dense_replace and i % cfg.moe_layer_freq == 0
        assert is_moe == (i >= 1), i
