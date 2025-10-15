import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh

from sgl_jax.srt.layers.linear import LinearBase


class TestLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        use_bias: bool = True,
        params_dtype: jnp.dtype = jnp.bfloat16,
        kernel_axes=None,
        rngs: nnx.Rngs | None = None,
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias
        self.skip_bias_add = False
        self.params_dtype = params_dtype
        self.kernel_axes = kernel_axes or (None, None)
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            use_bias=use_bias,
            skip_bias_add=False,
            params_dtype=params_dtype,
            kernel_axes=self.kernel_axes,
            rngs=rngs,
        )


def _single_device_mesh() -> Mesh:
    devices = np.array(jax.devices()[:1])
    return Mesh(devices, ("data",))


class TestLinearLazyInit(unittest.TestCase):
    def test_linear_materialize_and_forward_with_bias(self):
        rngs = nnx.Rngs(0)
        layer = TestLinear(
            input_size=8,
            output_size=4,
            use_bias=True,
            params_dtype=jnp.bfloat16,
            kernel_axes=(None, None),
            rngs=rngs,
        )

        self.assertIsInstance(layer.weight.value, jax.ShapeDtypeStruct)
        if layer.bias is not None:
            self.assertIsInstance(layer.bias.value, jax.ShapeDtypeStruct)

        mesh = _single_device_mesh()
        layer.materialize(mesh, rngs)

        self.assertIsInstance(layer.weight.value, jax.Array)
        if layer.use_bias:
            self.assertIsInstance(layer.bias.value, jax.Array)

        x = jnp.ones((2, layer.input_size), dtype=jnp.float32)
        y, b = layer(x)

        self.assertEqual(y.shape, (2, layer.output_size))
        self.assertIsNone(b)  # skip_bias_add=False

        # Validate numerics against explicit computation
        expected = jnp.dot(x, layer.weight.value)
        if layer.use_bias:
            expected = expected + layer.bias.value
        self.assertTrue(
            np.allclose(
                np.array(y, dtype=np.float32),
                np.array(expected, dtype=np.float32),
                atol=1e-2,
                rtol=1e-2,
            )
        )

    def test_linear_materialize_and_forward_no_bias(self):
        rngs = nnx.Rngs(1)
        layer = TestLinear(
            input_size=16,
            output_size=3,
            use_bias=False,
            params_dtype=jnp.bfloat16,
            kernel_axes=(None, None),
            rngs=rngs,
        )

        mesh = _single_device_mesh()
        layer.materialize(mesh, rngs)

        x = jnp.arange(0, 32, dtype=jnp.float32).reshape(2, 16)
        y, b = layer(x)

        self.assertEqual(y.shape, (2, 3))
        self.assertIsNone(b)

        expected = jnp.dot(x, layer.weight.value)
        self.assertTrue(
            np.allclose(
                np.array(y, dtype=np.float32),
                np.array(expected, dtype=np.float32),
                atol=1e-2,
                rtol=1e-2,
            )
        )


if __name__ == "__main__":
    unittest.main()
