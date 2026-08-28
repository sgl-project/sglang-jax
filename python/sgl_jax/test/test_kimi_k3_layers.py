"""Parity tests for the K3-specific JAX layers vs a numpy oracle.

The oracle is transcribed LINE BY LINE from the K3 PyTorch reference implementation. Where torch
is unavailable this is a faithful transcription rather than an execution of the torch module; the
executable torch-vs-jax cross-check lives in ``test_kimi_k3_torch_parity.py``, which runs the
reference functions directly when torch is installed. The math here is small enough that
transcription is a defensible oracle.
"""
import jax, jax.numpy as jnp, numpy as np, pytest
from sgl_jax.srt.models.kimi_k3_layers import situ_and_mul, attention_residual_apply


def _oracle_situ(x, beta, linear_beta):
    """torch: gate,up = x.chunk(2,-1); gate = beta*tanh(gate/beta)*sigmoid(gate);
              up = linear_beta*tanh(up/linear_beta) if linear_beta else up; return gate*up"""
    gate, up = np.split(x.astype(np.float64), 2, axis=-1)
    g = beta * np.tanh(gate / beta) * (1.0 / (1.0 + np.exp(-gate)))
    u = linear_beta * np.tanh(up / linear_beta) if linear_beta is not None else up
    return g * u


def _oracle_attnres(prefix_sum, block_residuals, norm_scale, proj_kernel, eps):
    """torch: values=cat((blocks, prefix[...,None,:]),-2); scores=proj(norm(values));
              p=softmax(scores.float(),-2); return (p*values.float()).sum(-2)"""
    v = np.concatenate(
        (block_residuals.astype(np.float64), prefix_sum.astype(np.float64)[..., None, :]), axis=-2
    )
    var = np.mean(v**2, axis=-1, keepdims=True)
    normed = v / np.sqrt(var + eps) * norm_scale.astype(np.float64)
    scores = normed @ proj_kernel.astype(np.float64)
    e = np.exp(scores - scores.max(axis=-2, keepdims=True))
    p = e / e.sum(axis=-2, keepdims=True)
    return (p * v).sum(axis=-2)


@pytest.mark.parametrize("beta,linear_beta", [(1.0, None), (2.0, 3.0), (0.5, 1.5)])
def test_situ_matches_pytorch_reference(beta, linear_beta):
    x = np.random.default_rng(0).normal(size=(7, 256)).astype(np.float32)
    got = np.asarray(situ_and_mul(jnp.asarray(x), beta, linear_beta), dtype=np.float64)
    want = _oracle_situ(x, beta, linear_beta)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_situ_soft_clips_to_beta():
    """SITU's defining property vs SiLU: the gate branch is bounded by +/-beta."""
    beta = 2.0
    x = np.concatenate([np.full((4, 8), 1e4), np.ones((4, 8))], axis=-1).astype(np.float32)
    out = np.asarray(situ_and_mul(jnp.asarray(x), beta, None))
    assert np.all(out <= beta + 1e-3), out.max()


@pytest.mark.parametrize("n_blocks", [1, 3, 8])
def test_attention_residual_matches_pytorch_reference(n_blocks):
    rng = np.random.default_rng(1)
    hidden, eps = 128, 1e-6
    prefix = rng.normal(size=(5, hidden)).astype(np.float32)
    blocks = rng.normal(size=(5, n_blocks, hidden)).astype(np.float32)
    nscale = rng.normal(size=(hidden,)).astype(np.float32)
    pk = rng.normal(size=(hidden, 1)).astype(np.float32) * 0.05
    got = np.asarray(
        attention_residual_apply(jnp.asarray(prefix), jnp.asarray(blocks),
                                 jnp.asarray(nscale), jnp.asarray(pk), eps), dtype=np.float64)
    want = _oracle_attnres(prefix, blocks, nscale, pk, eps)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_attention_residual_is_a_convex_combination():
    """Weights are a softmax over candidates, so the output must lie in their convex hull."""
    rng = np.random.default_rng(2)
    hidden = 64
    prefix = rng.normal(size=(3, hidden)).astype(np.float32)
    blocks = rng.normal(size=(3, 4, hidden)).astype(np.float32)
    out = np.asarray(attention_residual_apply(
        jnp.asarray(prefix), jnp.asarray(blocks),
        jnp.ones((hidden,), jnp.float32), jnp.zeros((hidden, 1), jnp.float32), 1e-6))
    allv = np.concatenate([blocks, prefix[:, None, :]], axis=-2)
    assert np.all(out <= allv.max(axis=-2) + 1e-4) and np.all(out >= allv.min(axis=-2) - 1e-4)
    # zero projection => uniform softmax => plain mean
    np.testing.assert_allclose(out, allv.mean(axis=-2), rtol=1e-4, atol=1e-4)


def test_attnres_scoring_needs_highest_precision_on_tpu():
    """Regression: TPU's default einsum precision is bf16, and these scores feed a softmax.

    Measured on v7x, default precision gives ~37% max relative error against the fp32 oracle
    while HIGHEST gives ~2e-7. This test pins the tolerance that only HIGHEST can meet, so a
    future edit that drops the precision= argument fails here instead of silently degrading the
    mixing weights. On CPU both paths are exact, so this only bites on TPU.
    """
    rng = np.random.default_rng(7)
    hidden, eps = 128, 1e-6
    prefix = rng.normal(size=(4, hidden)).astype(np.float32)
    blocks = rng.normal(size=(4, 3, hidden)).astype(np.float32)
    ns = rng.normal(size=(hidden,)).astype(np.float32)
    pk = (rng.normal(size=(hidden, 1)) * 0.05).astype(np.float32)
    got = np.asarray(attention_residual_apply(
        jnp.asarray(prefix), jnp.asarray(blocks), jnp.asarray(ns), jnp.asarray(pk), eps),
        dtype=np.float64)
    want = _oracle_attnres(prefix, blocks, ns, pk, eps)
    rel = np.abs(got - want) / (np.abs(want) + 1e-9)
    assert rel.max() < 1e-4, f"max rel err {rel.max():.3e} -- scoring einsum lost precision"
