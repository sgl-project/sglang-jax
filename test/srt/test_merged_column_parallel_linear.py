"""Unit tests for :class:`MergedColumnParallelLinear`.

The primitive is a generalised drop-in for a stack of independent
column-parallel ``LinearBase``s fused into one bigger GEMM (better MXU
utilization on TPU). Weight loading is the caller's responsibility (no
built-in loader yet — see class docstring). The forward identity is
just ``LinearBase``'s matmul, so tests focus on:

* the merged weight has shape ``[input_size, sum(output_sizes)]``;
* default no-bias behaviour matches ``LinearBase``;
* construction rejects component sizes that don't divide TP — the
  divisibility guard the per-rank block-concat layout depends on.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_merged_column_parallel_linear.py -v
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh

from sgl_jax.srt.layers.linear import MergedColumnParallelLinear


def _mesh_1x1():
    devices = mesh_utils.create_device_mesh((8,))[:1].reshape((1, 1))
    return Mesh(
        devices, ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _mesh_1xN(n: int):
    devices = mesh_utils.create_device_mesh((8,))[:n].reshape((1, n))
    return Mesh(
        devices, ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


class MergedColumnParallelInitTest(unittest.TestCase):
    def test_weight_shape_is_sum_of_output_sizes(self):
        """Single merged weight of width ``sum(output_sizes)``."""
        mesh = _mesh_1x1()
        with jax.set_mesh(mesh):
            layer = MergedColumnParallelLinear(
                input_size=32, output_sizes=[64, 64, 128], mesh=mesh,
            )
            self.assertEqual(layer.weight.value.shape, (32, 64 + 64 + 128))
            self.assertEqual(layer.output_sizes, [64, 64, 128])

    def test_no_bias_by_default(self):
        mesh = _mesh_1x1()
        with jax.set_mesh(mesh):
            layer = MergedColumnParallelLinear(
                input_size=8, output_sizes=[4, 4], mesh=mesh,
            )
            self.assertIsNone(layer.bias)

    def test_rejects_non_divisible_component(self):
        """Each component size must independently divide TP, so the
        per-rank block-concat boundary aligns with the TP cut."""
        mesh = _mesh_1xN(2)
        with jax.set_mesh(mesh):
            with self.assertRaises(ValueError) as ctx:
                MergedColumnParallelLinear(
                    input_size=16, output_sizes=[3, 4], mesh=mesh,  # 3 % 2 == 1
                )
            self.assertIn("divisible by TP=2", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
