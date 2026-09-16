"""Accuracy of ordinary JAX Gemma RMSNorm through TT compiler fusion."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sgl_jax.srt.layers.layernorm import GemmaRMSNorm


@pytest.fixture(autouse=True)
def isolated_mesh():
    with jax.set_mesh(jax.sharding.Mesh(np.empty((), dtype=object), ())):
        yield


@pytest.mark.skipif(jax.default_backend() != "tt", reason="requires a TT device")
@pytest.mark.parametrize("shape", [(1, 4096), (1, 16, 256)])
@pytest.mark.parametrize("trace", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_device_gemma_norm(shape, trace, offset):
    rng = np.random.default_rng(35)
    x = rng.normal(size=shape).astype(jnp.bfloat16)
    weight = rng.uniform(-0.5, 0.5, shape[-1]).astype(np.float32)
    x32 = x.astype(np.float32)
    expected = (
        (x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + 1e-6) * (int(offset) + weight))
        .astype(jnp.bfloat16)
        .astype(np.float32)
    )

    def normalize(x, w):
        layer = GemmaRMSNorm(shape[-1], add_unit_offset=offset)
        layer.weight[...] = w
        return layer(x)

    f = jax.jit(
        normalize, compiler_options={"optimization_level": "1", "enable_trace": str(trace).lower()}
    )
    x, weight = jax.device_put(x), jax.device_put(weight)
    for _ in range(3):
        actual = np.asarray(f(x, weight), np.float32)
        # Native normalization uses FP32 accumulation; its arithmetic and
        # final BF16 rounding differ from the unfused JAX expression.
        assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.005
        np.testing.assert_allclose(actual, expected, rtol=0.015, atol=0.002)
