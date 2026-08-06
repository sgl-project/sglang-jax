"""TPU correctness and production-shape smoke tests for exact DSA."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.kernels.dsa.exact_attention import sparse_core_tensor_core_dsa


def _require_sparse_core_tpu() -> None:
    if not any("TPU" in str(device).upper() for device in jax.devices()):
        pytest.skip("requires a TPU")
    from jax.experimental.pallas import tpu as pltpu

    if pltpu.get_tpu_info().sparse_core is None:
        pytest.skip("requires a TPU target with SparseCore support")


def _reference(q_latent, q_rope, cache, slots, counts, scale):
    gathered = cache[slots]
    q = jnp.concatenate([q_latent, q_rope], axis=-1).astype(jnp.float32)
    k = gathered[..., : q.shape[-1]].astype(jnp.float32)
    scores = jnp.einsum("qhd,qkd->qhk", q, k) * scale
    valid = jnp.arange(slots.shape[1])[None, :] < counts[:, None]
    scores = jnp.where(valid[:, None, :], scores, -1.0e30)
    probabilities = jax.nn.softmax(scores, axis=-1) * valid[:, None, :]
    normalizer = jnp.maximum(probabilities.sum(axis=-1, keepdims=True), 1.0e-30)
    probabilities /= normalizer
    return jnp.einsum("qhk,qkv->qhv", probabilities, gathered[..., : q_latent.shape[-1]])


def test_exact_dsa_matches_reference() -> None:
    _require_sparse_core_tpu()
    q_size, heads, latent, rope, topk, cache_size = 128, 8, 512, 64, 128, 512
    keys = jax.random.split(jax.random.PRNGKey(17), 4)
    q_latent = (jax.random.normal(keys[0], (q_size, heads, latent)) * 0.1).astype(jnp.bfloat16)
    q_rope = (jax.random.normal(keys[1], (q_size, heads, rope)) * 0.1).astype(jnp.bfloat16)
    cache = (jax.random.normal(keys[2], (cache_size, 640)) * 0.1).astype(jnp.bfloat16)
    slots = jax.random.randint(keys[3], (q_size, topk), 0, cache_size, dtype=jnp.int32)
    counts = (jnp.arange(q_size, dtype=jnp.int32) * 13) % (topk + 1)
    slots = jnp.where(jnp.arange(topk)[None, :] < counts[:, None], slots, 0)
    scale = jnp.asarray(576**-0.5, jnp.float32)

    actual = sparse_core_tensor_core_dsa(
        q_latent,
        q_rope,
        cache,
        slots,
        counts,
        scale,
        bq_sparse=128,
        bq=32,
        b_topk=128,
    )
    expected = _reference(q_latent, q_rope, cache, slots, counts, scale)
    actual, expected = jax.block_until_ready((actual, expected))

    np.testing.assert_allclose(
        np.asarray(actual, np.float32), np.asarray(expected, np.float32), rtol=5e-3, atol=5e-3
    )


@pytest.mark.parametrize(
    ("q_size", "bq_sparse", "bq"),
    [(2, 2, 1), (128, 128, 32)],
    ids=["decode-c2-h64-k2048", "extend-h64-k2048"],
)
def test_exact_dsa_glm52_production_shape_smoke(q_size, bq_sparse, bq) -> None:
    _require_sparse_core_tpu()
    q_latent = jnp.zeros((q_size, 64, 512), jnp.bfloat16)
    q_rope = jnp.zeros((q_size, 64, 64), jnp.bfloat16)
    cache = jnp.zeros((4096, 640), jnp.bfloat16)
    slots = jnp.zeros((q_size, 2048), jnp.int32)
    counts = jnp.zeros((q_size,), jnp.int32)

    output = sparse_core_tensor_core_dsa(
        q_latent,
        q_rope,
        cache,
        slots,
        counts,
        jnp.asarray(576**-0.5, jnp.float32),
        bq_sparse=bq_sparse,
        bq=bq,
        b_topk=128,
    )
    output.block_until_ready()
    assert output.shape == q_latent.shape
    assert jnp.all(output == 0)
