import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from sgl_jax.srt.layers.moe import EPMoE


def _naive_moe_ref(x, wi_0, wi_1, wo, topk_w, topk_ids, swiglu_limit):
    """Per-expert loop reference (fp32). wi_0/wi_1=[E,h,i], wo=[E,i,h]."""
    x = x.astype(jnp.float32)
    out = jnp.zeros((x.shape[0], wo.shape[-1]), jnp.float32)
    for t in range(x.shape[0]):
        acc = jnp.zeros((wo.shape[-1],), jnp.float32)
        for slot in range(topk_ids.shape[1]):
            e = int(topk_ids[t, slot])
            w = topk_w[t, slot]
            gate = jax.nn.silu(x[t] @ wi_0[e].astype(jnp.float32))
            up = x[t] @ wi_1[e].astype(jnp.float32)
            if swiglu_limit is not None:
                gate = jnp.clip(gate, max=swiglu_limit)
                up = jnp.clip(up, -swiglu_limit, swiglu_limit)
            acc = acc + w * ((gate * up) @ wo[e].astype(jnp.float32))
        out = out.at[t].set(acc)
    return out


def _build_epmoe(mesh, num_experts=4, hidden=16, inter_dim=32, swiglu_limit=None):
    return EPMoE(
        hidden_size=hidden,
        num_experts=num_experts,
        num_experts_per_tok=2,
        ep_size=1,
        mesh=mesh,
        intermediate_dim=inter_dim,
        weight_dtype=jnp.float32,
        dtype=jnp.float32,
        swiglu_limit=swiglu_limit,
    )


@pytest.fixture
def mesh():
    return Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1),
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def test_clamp_matches_naive_loop(mesh):
    E, H, inter, T, K, L = 4, 16, 32, 6, 2, 0.5
    rng = np.random.default_rng(0)
    m = _build_epmoe(mesh, E, H, inter, swiglu_limit=L)
    m.wi_0.value = jnp.asarray(rng.normal(0, 3.0, (E, H, inter)), jnp.float32)
    m.wi_1.value = jnp.asarray(rng.normal(0, 3.0, (E, H, inter)), jnp.float32)
    m.wo.value = jnp.asarray(rng.normal(0, 1.0, (E, inter, H)), jnp.float32)
    x = jnp.asarray(rng.normal(0, 3.0, (T, H)), jnp.float32)
    topk_ids = jnp.asarray(rng.integers(0, E, (T, K)), jnp.int32)
    topk_w = jnp.asarray(rng.uniform(0.1, 1.0, (T, K)), jnp.float32)
    got = m(x, topk_w, topk_ids)
    ref = _naive_moe_ref(x, m.wi_0.value, m.wi_1.value, m.wo.value, topk_w, topk_ids, L)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-3, atol=2e-3)


def test_default_matches_unclamped_loop(mesh):
    E, H, inter, T, K = 4, 16, 32, 6, 2
    rng = np.random.default_rng(1)
    m = _build_epmoe(mesh, E, H, inter, swiglu_limit=None)
    m.wi_0.value = jnp.asarray(rng.normal(0, 1.0, (E, H, inter)), jnp.float32)
    m.wi_1.value = jnp.asarray(rng.normal(0, 1.0, (E, H, inter)), jnp.float32)
    m.wo.value = jnp.asarray(rng.normal(0, 1.0, (E, inter, H)), jnp.float32)
    x = jnp.asarray(rng.normal(0, 1.0, (T, H)), jnp.float32)
    topk_ids = jnp.asarray(rng.integers(0, E, (T, K)), jnp.int32)
    topk_w = jnp.asarray(rng.uniform(0.1, 1.0, (T, K)), jnp.float32)
    got = m(x, topk_w, topk_ids)
    ref = _naive_moe_ref(x, m.wi_0.value, m.wi_1.value, m.wo.value, topk_w, topk_ids, None)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), rtol=2e-3, atol=2e-3)
