import unittest

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P

from sgl_jax.srt.layers.linear import LinearBase
from sgl_jax.srt.utils.mesh_utils import create_device_mesh
from sgl_jax.test.test_utils import CustomTestCase

mesh = create_device_mesh(ici_parallelism=[1, -1], dcn_parallelism=[1, 1])
jax.sharding.set_mesh(mesh)


class TestLinearTp(CustomTestCase):
    @unittest.skipIf(
        mesh.shape.get("tensor", 1) < 2,
        "Row-parallel regression test requires at least 2 tensor-parallel devices.",
    )
    def test_row_parallel_linear_matches_dense_dot(self):
        batch, in_dim, out_dim = 4, 256, 128
        x_host = (
            jnp.arange(batch * in_dim, dtype=jnp.float32).reshape(batch, in_dim) / 128.0
        ).astype(jnp.bfloat16)
        w_host = (
            jnp.arange(in_dim * out_dim, dtype=jnp.float32).reshape(in_dim, out_dim) / 256.0
        ).astype(jnp.bfloat16)

        x = jax.device_put(x_host, NamedSharding(mesh, P(None, "tensor")))
        w = jax.device_put(w_host, NamedSharding(mesh, P("tensor", None)))

        with jax.set_mesh(mesh):
            linear = LinearBase(
                input_size=in_dim,
                output_size=out_dim,
                use_bias=False,
                mesh=mesh,
                kernel_axes=("tensor", None),
                params_dtype=jnp.bfloat16,
            )
            linear.weight = nnx.Param(w, out_sharding=P("tensor", None))

            out, bias = linear(x)

        ref = jnp.dot(x_host, w_host)

        np.testing.assert_allclose(
            np.asarray(out),
            np.asarray(ref),
            rtol=1e-2,
            atol=1e-2,
        )
        self.assertIsNone(bias)


if __name__ == "__main__":
    unittest.main()
