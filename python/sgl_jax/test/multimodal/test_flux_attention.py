import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.multimodal.layers.rotary_embedding import NDRotaryEmbedding
from sgl_jax.srt.multimodal.models.dits.flux import FluxAttention, _sdpa_attention


def _mesh(dp, tp):
    if jax.device_count() < dp * tp:
        pytest.skip(f"requires {dp * tp} devices")
    return Mesh(
        np.asarray(jax.devices()[: dp * tp]).reshape(dp, tp),
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


@pytest.mark.parametrize("dp,tp,batch", [(1, 1, 1), (1, 4, 1), (1, 4, 2), (2, 2, 2)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("causal,scale", [(False, None), (True, 0.25)])
def test_sdpa_preserves_values_and_nontrivial_sharding(dp, tp, batch, dtype, causal, scale):
    mesh = _mesh(dp, tp)
    rng = np.random.default_rng(42)
    sharding = NamedSharding(mesh, P("data", None, "tensor", None))
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        q, k, v = [
            jax.device_put(jnp.asarray(rng.normal(size=(batch, 8, 4, 8)), dtype), sharding)
            for _ in range(3)
        ]
        output = jax.jit(lambda q, k, v: _sdpa_attention(q, k, v, causal, scale))(q, k, v)

    # Full matmul precision makes this dense reference comparable on CPU and TPU.
    # Preserve the softmax scale, causal mask and dtype.
    q_np, k_np, v_np = [np.asarray(x, np.float32) for x in (q, k, v)]
    scores = np.einsum("bthd,bshd->bhts", q_np, k_np) * (8**-0.5 if scale is None else scale)
    if causal:
        scores = np.where(np.tril(np.ones((8, 8), dtype=bool)), scores, -np.inf)
    probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhts,bshd->bthd", probs, v_np)
    expected = np.asarray(jnp.asarray(expected, dtype), np.float32)
    np.testing.assert_allclose(np.asarray(output, np.float32), expected, atol=2e-6, rtol=2e-5)
    assert output.dtype == dtype
    assert output.sharding.spec == P(None if batch == 1 else "data", None, "tensor", None)
    for operand in (q, k, v):
        assert operand.sharding == sharding


def test_flux_single_device_fallback_with_rotary_matches_batched_execution():
    mesh = _mesh(1, 1)
    with jax.set_mesh(mesh), jax.default_matmul_precision("highest"):
        attention = FluxAttention(
            query_dim=32, heads=4, dim_head=8, attention_impl="usp", mesh=mesh, pre_only=True
        )
        rope = NDRotaryEmbedding([2, 2, 4], 10000, use_real=True, repeat_interleave_real=True)
        positions = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
        rotary = rope(positions)
        hidden = jax.device_put(
            np.random.default_rng(0).normal(size=(1, 8, 32)).astype(np.float32),
            NamedSharding(mesh, P()),
        )
        output = jax.jit(lambda x: attention(x, image_rotary_emb=rotary))(hidden)
        # Duplicating the sample avoids a singleton batch while keeping the
        # same weights and rotary values. The first sample must be unchanged.
        expected = jax.jit(lambda x: attention(x, image_rotary_emb=rotary))(
            jnp.repeat(hidden, 2, axis=0)
        )[:1]
    np.testing.assert_allclose(output, expected, atol=1e-5, rtol=1e-5)
    assert output.shape == (1, 8, 32)
