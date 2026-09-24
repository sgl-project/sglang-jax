"""One-time output-row padding of replicated-N block-quant linears.

The block-wise quantized matmul wrapper pads ``w_q`` / ``w_scale`` to the tuned
out block on every call when ``n_out`` is not aligned; ``QuantizedLinear.
pad_out_rows`` does it once at load time and the call path slices the result
back to the logical width. These CPU tests pin that contract with a reference
block-wise matmul in place of the Pallas kernel.
"""

import os

if "--xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=4"
    )
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.kernels.quantized_matmul import kernel as qmm_kernel
from sgl_jax.srt.layers import linear as linear_mod
from sgl_jax.srt.layers.linear import (
    QuantizedLinear,
    prepad_replicated_quantized_linears,
)

N_IN, N_OUT, BLOCK, MULT = 256, 40, 128, 64


def _mesh():
    devices = np.array(jax.devices()[:4]).reshape(1, 4)
    return jax.sharding.Mesh(
        devices,
        axis_names=("data", "tensor"),
        axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
    )


def _reference_blockwise(x, w_q, w_scale, block_size, x_q_dtype, tuned_value):
    """Dequantised matmul with the wrapper's calling convention."""
    del x_q_dtype, tuned_value
    in_blocks = w_q.shape[1] // block_size
    scale = jnp.repeat(w_scale[:, 0, :].T, block_size, axis=1)  # [n_out, n_in]
    w = w_q.astype(jnp.float32) * scale[:, : w_q.shape[1]]
    del in_blocks
    return jnp.dot(x.astype(jnp.float32), w.T)


def _weights(n_out=N_OUT, n_in=N_IN, seed=0):
    rng = np.random.default_rng(seed)
    w_q = jnp.asarray(rng.integers(-3, 4, size=(n_out, n_in)).astype(np.int8)).astype(
        jnp.float8_e4m3fn
    )
    w_scale = jnp.asarray(rng.uniform(0.5, 1.5, size=(n_in // BLOCK, 1, n_out)), jnp.float32)
    return w_q, w_scale


class LocalMatmulPrepadTest(unittest.TestCase):
    def test_prepadded_weight_matches_reference_and_keeps_logical_lookup(self):
        w_q, w_scale = _weights()
        x = jnp.asarray(np.random.default_rng(1).standard_normal((3, N_IN)), jnp.bfloat16)
        seen = []

        def fake_lookup(**kw):
            seen.append(kw["n_out"])
            return None

        pad = MULT - N_OUT
        w_q_p = jnp.pad(w_q, ((0, pad), (0, 0)))
        w_scale_p = jnp.pad(w_scale, ((0, 0), (0, 0), (0, pad)))
        kwargs = dict(
            quantize_activation=False,
            reduce_axis=None,
            compute_dtype=jnp.float32,
            weight_block_size=(BLOCK, BLOCK),
            allow_narrow_n_blockwise=True,
        )
        with (
            mock.patch.object(
                qmm_kernel, "get_blockwise_kernel", return_value=_reference_blockwise
            ),
            mock.patch.object(qmm_kernel, "get_safe_blockwise_tuned_value", fake_lookup),
        ):
            ref = qmm_kernel.xla_quantized_matmul_local(x, w_q, w_scale, **kwargs)
            out = qmm_kernel.xla_quantized_matmul_local(
                x, w_q_p, w_scale_p, n_out_valid=N_OUT, **kwargs
            )
        self.assertEqual(out.shape, (3, N_OUT))
        np.testing.assert_array_equal(np.asarray(out), np.asarray(ref))
        # tuned-block lookup sees the logical width, not the padded row count
        self.assertEqual(seen, [N_OUT, N_OUT])

    def test_per_channel_path_slices_to_logical_width(self):
        rng = np.random.default_rng(2)
        w_q = jnp.asarray(rng.integers(-3, 4, size=(96, N_IN)).astype(np.int8)).astype(jnp.bfloat16)
        w_scale = jnp.ones((96,), jnp.float32)
        x = jnp.asarray(rng.standard_normal((2, N_IN)), jnp.bfloat16)
        out = qmm_kernel.xla_quantized_matmul_local(
            x, w_q, w_scale, quantize_activation=False, reduce_axis=None, n_out_valid=N_OUT
        )
        self.assertEqual(out.shape, (2, N_OUT))


class QuantizedLinearPrepadTest(unittest.TestCase):
    def _layer(self, mesh, kernel_axes, scale_3d=True):
        w_q, w_scale = _weights()
        if not scale_3d:
            w_scale = jnp.ones((N_OUT,), jnp.float32)
        spec_w = P(kernel_axes[1], kernel_axes[0])
        w_q = jax.device_put(w_q, NamedSharding(mesh, spec_w))
        return QuantizedLinear(
            weight_q=w_q,
            weight_scale=w_scale,
            bias=None,
            activation_dtype=None,
            mesh=mesh,
            kernel_axes=kernel_axes,
            weight_block_size=(BLOCK, BLOCK),
            allow_narrow_n_blockwise=True,
        )

    def test_pad_out_rows_replicated_only_and_idempotent(self):
        mesh = _mesh()
        with jax.set_mesh(mesh):
            layer = self._layer(mesh, (None, None))
            self.assertTrue(layer.pad_out_rows(MULT))
            self.assertEqual(layer.weight_q.value.shape, (MULT, N_IN))
            self.assertEqual(layer.weight_scale.value.shape, (N_IN // BLOCK, 1, MULT))
            self.assertEqual(layer.n_out_valid, N_OUT)
            self.assertEqual(
                float(jnp.abs(layer.weight_q.value[N_OUT:].astype(jnp.float32)).sum()), 0.0
            )
            self.assertFalse(layer.pad_out_rows(MULT))  # idempotent

            sharded = self._layer(mesh, (None, "tensor"))
            self.assertFalse(sharded.pad_out_rows(MULT))
            self.assertIsNone(sharded.n_out_valid)

            per_channel = self._layer(mesh, (None, None), scale_3d=False)
            self.assertFalse(per_channel.pad_out_rows(MULT))

            aligned = self._layer(mesh, (None, None))
            self.assertFalse(aligned.pad_out_rows(8))  # 40 % 8 == 0 -> nothing to do

    def test_forward_unchanged_after_prepad(self):
        mesh = _mesh()
        x = jnp.asarray(np.random.default_rng(3).standard_normal((4, N_IN)), jnp.bfloat16)
        with (
            jax.set_mesh(mesh),
            mock.patch.object(
                qmm_kernel, "get_blockwise_kernel", return_value=_reference_blockwise
            ),
            mock.patch.object(qmm_kernel, "get_safe_blockwise_tuned_value", return_value=None),
        ):
            xs = jax.device_put(x, NamedSharding(mesh, P("data", None)))
            layer = self._layer(mesh, (None, None))
            before, _ = layer(xs)
            self.assertTrue(layer.pad_out_rows(MULT))
            after, _ = layer(xs)
        self.assertEqual(after.shape, (4, N_OUT))
        np.testing.assert_array_equal(np.asarray(after), np.asarray(before))

    def test_prepad_helper_walks_module_and_honours_env_switch(self):
        mesh = _mesh()

        class Holder(linear_mod.nnx.Module):
            def __init__(self, a, b):
                self.a = a
                self.b = b

        with jax.set_mesh(mesh):
            holder = Holder(self._layer(mesh, (None, None)), self._layer(mesh, (None, "tensor")))
            with mock.patch.dict(os.environ, {"SGLANG_JAX_QMM_PREPAD_N": "0"}):
                self.assertEqual(prepad_replicated_quantized_linears(holder, MULT), 0)
            self.assertEqual(prepad_replicated_quantized_linears(holder, MULT), 1)
            self.assertEqual(holder.a.n_out_valid, N_OUT)
            self.assertIsNone(holder.b.n_out_valid)


if __name__ == "__main__":
    unittest.main()
