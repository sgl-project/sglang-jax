"""CPU check: block-quant QuantizedLinear -> per-channel fp8 rewrite keeps the output close and NaN-free."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.linear import QuantizedLinear
from sgl_jax.srt.utils.quantization.per_channel_requant import (
    requantize_linears_per_channel,
)

BLOCK = 128
F8 = jnp.float8_e4m3fn


def _block_quant(w):
    n_out, n_in = w.shape
    wb = w.reshape(n_out // BLOCK, BLOCK, n_in // BLOCK, BLOCK)
    amax = jnp.max(jnp.abs(wb), axis=(1, 3), keepdims=True)
    scale = amax / 448.0
    wq = (wb / scale).astype(F8).reshape(n_out, n_in)
    return wq, scale[:, 0, :, 0]  # [out_blocks, in_blocks]


class _Holder(nnx.Module):
    def __init__(self, lin):
        self.lin = lin


def _make(mesh, kernel_axes, n_out, n_in, key):
    w = jax.random.normal(key, (n_out, n_in), jnp.float32) * 0.02
    wq, s2 = _block_quant(w)
    wq = jax.device_put(wq, NamedSharding(mesh, P(kernel_axes[1], kernel_axes[0])))
    lin = QuantizedLinear(
        weight_q=wq,
        weight_scale=s2,
        bias=None,
        activation_dtype=None,
        mesh=mesh,
        kernel_axes=kernel_axes,
        compute_dtype=jnp.float32,
        weight_block_size=(BLOCK, BLOCK),
    )
    w_deq = wq.astype(jnp.float32) * jnp.repeat(jnp.repeat(s2, BLOCK, 0), BLOCK, 1)
    return lin, w_deq


def test_requant_per_channel_close_and_nan_free():
    mesh = jax.make_mesh(
        (1, 1), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    key = jax.random.PRNGKey(0)
    with jax.set_mesh(mesh):
        for kernel_axes in ((None, "tensor"), ("tensor", None)):
            key, k1, k2 = jax.random.split(key, 3)
            lin, w_deq = _make(mesh, kernel_axes, 512, 384, k1)
            x = jax.random.normal(k2, (64, 384), jnp.float32)
            x = x.at[0].multiply(30.0).at[-1].set(0.0)
            x = jax.device_put(
                x.astype(jnp.bfloat16), NamedSharding(mesh, P("data", kernel_axes[0]))
            )
            ref = np.asarray(x.astype(jnp.float32)) @ np.asarray(w_deq).T
            for mode in ("a16", "a8"):
                h = _Holder(_make(mesh, kernel_axes, 512, 384, k1)[0])
                n = requantize_linears_per_channel(h, mode)
                assert n == 1
                assert h.lin.weight_scale.value.ndim == 1 and h.lin.weight_scale.value.shape == (
                    512,
                )
                assert h.lin.weight_block_size is None
                assert (h.lin.activation_dtype == F8) == (mode == "a8")
                out = h.lin(x)
                out = out[0] if isinstance(out, tuple) else out
                o32 = np.asarray(out.astype(jnp.float32))
                assert not np.isnan(o32).any()
                rel = np.max(np.abs(o32 - ref)) / np.max(np.abs(ref))
                assert rel < (0.06 if mode == "a8" else 0.03), (kernel_axes, mode, rel)


def test_skip_regex_leaves_layer_untouched():
    mesh = jax.make_mesh(
        (1, 1), ("data", "tensor"), axis_types=(jax.sharding.AxisType.Explicit,) * 2
    )
    with jax.set_mesh(mesh):
        h = _Holder(_make(mesh, (None, "tensor"), 256, 256, jax.random.PRNGKey(3))[0])
        assert requantize_linears_per_channel(h, "a8", skip=r"^lin$") == 0
        assert h.lin.weight_scale.value.ndim == 3
