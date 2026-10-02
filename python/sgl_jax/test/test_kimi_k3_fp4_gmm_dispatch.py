"""Kimi-K3 fp4 MoE matmul dispatch: tokamax's gmm_v2 when usable, the dequant reference otherwise.

The kernel path must receive the packed operands untouched (native float4_e2m1fn weights plus fp32
block scales); the reference path must widen them to bf16 before the standard grouped matmul. Both
routing decisions are checked here without a TPU by stubbing the kernel and the generation probe.
"""

import inspect

import jax.numpy as jnp
import numpy as np
import pytest

import sgl_jax.srt.layers.moe as moe_mod
import sgl_jax.srt.models.kimi_k3 as k3
from sgl_jax.srt.models.kimi_k3 import KimiK3EPMoE

NUM_EXPERTS, SIZE_K, SIZE_N, BLOCK = 2, 64, 8, 32


def _operands():
    rng = np.random.default_rng(0)
    w = jnp.asarray(rng.choice([-2.0, -1.0, 0.0, 0.5, 1.0, 3.0], (NUM_EXPERTS, SIZE_K, SIZE_N)))
    rhs = w.astype(jnp.float4_e2m1fn)
    scale = jnp.asarray(
        rng.choice([0.5, 1.0, 2.0], (NUM_EXPERTS, SIZE_K // BLOCK, 1, SIZE_N)), dtype=jnp.float32
    )
    return rhs, scale


def _kwargs(rhs, scale):
    return dict(
        lhs=jnp.ones((4, SIZE_K), jnp.bfloat16),
        rhs=rhs,
        rhs_scale=scale,
        rhs_bias=None,
        zero_initialize=False,
        activation_quantized_dtype=None,
        group_sizes=jnp.array([2, 2], jnp.int32),
        preferred_element_type=jnp.bfloat16,
        group_offset=jnp.array([0], jnp.int32),
        maybe_quantize_lhs=False,
        acc_dtype=jnp.float32,
    )


def _moe():
    moe = object.__new__(KimiK3EPMoE)
    object.__setattr__(moe, "fp4", True)
    return moe


def test_kernel_path_receives_packed_operands(monkeypatch):
    seen = {}
    monkeypatch.setattr(k3, "_tpu_generation", lambda: 7)
    monkeypatch.setattr(k3, "_tokamax_fp4_gmm", lambda: lambda **kw: seen.update(kw) or "kernel")
    rhs, scale = _operands()
    assert _moe()._call_gmm(**_kwargs(rhs, scale)) == "kernel"
    assert seen["rhs"] is rhs and seen["rhs"].dtype == jnp.float4_e2m1fn
    assert seen["rhs_scale"] is scale
    assert "activation_quantized_dtype" not in seen


@pytest.mark.parametrize(
    "generation, kernel", [(6, lambda **kw: pytest.fail("kernel on v6")), (7, None), (None, None)]
)
def test_reference_path_widens_with_block_scales(monkeypatch, generation, kernel):
    seen = {}
    monkeypatch.setattr(k3, "_tpu_generation", lambda: generation)
    monkeypatch.setattr(k3, "_tokamax_fp4_gmm", lambda: kernel)
    monkeypatch.setattr(moe_mod, "gmm", lambda **kw: seen.update(kw) or "reference")
    rhs, scale = _operands()
    assert _moe()._call_gmm(**_kwargs(rhs, scale)) == "reference"
    assert seen["rhs"].dtype == jnp.bfloat16
    expected = rhs.astype(jnp.float32) * jnp.repeat(scale[:, :, 0, :], BLOCK, axis=1)
    np.testing.assert_array_equal(np.asarray(seen["rhs"], np.float32), np.asarray(expected))
    assert "rhs_scale" not in seen and "rhs_bias" not in seen


def test_call_kwargs_bind_to_the_real_tokamax_signature():
    tokamax_gmm = pytest.importorskip("tokamax._src.ops.experimental.gmm_v2.gmm_v2").gmm_v2
    kwargs = _kwargs(*_operands())
    kwargs.pop("activation_quantized_dtype")
    inspect.signature(tokamax_gmm).bind(**kwargs)
