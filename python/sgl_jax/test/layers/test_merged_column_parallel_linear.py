"""Unit tests for :class:`MergedColumnParallelLinear`.

The primitive is a generalised drop-in for a stack of independent
column-parallel ``LinearBase``s fused into one bigger GEMM (better MXU
utilization on TPU). Weight loading is the caller's responsibility (no
built-in loader yet — see class docstring). The forward identity is
just ``LinearBase``'s matmul, so tests focus on:

* the merged weight has shape ``[input_size, sum(output_sizes)]``;
* default no-bias behaviour matches ``LinearBase``;
* construction rejects component sizes that don't divide TP — the
  divisibility guard the per-rank block-concat layout depends on;
* ``stripe_merged_weight`` + ``split_merged_output`` recover each
  component's output locally, also after quantization replaces the layer.

Run with:
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        python -m pytest test/srt/test_merged_column_parallel_linear.py -v
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.experimental import mesh_utils
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.configs.quantization_config import QuantizationConfig
from sgl_jax.srt.layers.linear import (
    MergedColumnParallelLinear,
    QuantizedLinear,
    split_merged_output,
    stripe_merged_weight,
)
from sgl_jax.srt.utils.quantization.quantization_utils import apply_linear_quantization


def _mesh_1x1():
    devices = mesh_utils.create_device_mesh((8,))[:1].reshape((1, 1))
    return Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


def _mesh_1xN(n: int):
    devices = mesh_utils.create_device_mesh((8,))[:n].reshape((1, n))
    return Mesh(
        devices,
        ("data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit),
    )


class MergedColumnParallelInitTest(unittest.TestCase):
    def test_weight_shape_is_sum_of_output_sizes(self):
        """Single merged weight of width ``sum(output_sizes)``."""
        mesh = _mesh_1x1()
        with jax.set_mesh(mesh):
            layer = MergedColumnParallelLinear(
                input_size=32,
                output_sizes=[64, 64, 128],
                mesh=mesh,
            )
            self.assertEqual(layer.weight.value.shape, (32, 64 + 64 + 128))
            self.assertEqual(layer.output_sizes, [64, 64, 128])

    def test_no_bias_by_default(self):
        mesh = _mesh_1x1()
        with jax.set_mesh(mesh):
            layer = MergedColumnParallelLinear(
                input_size=8,
                output_sizes=[4, 4],
                mesh=mesh,
            )
            self.assertIsNone(layer.bias)

    def test_rejects_non_divisible_component(self):
        """Each component size must independently divide TP, so the
        per-rank block-concat boundary aligns with the TP cut."""
        mesh = _mesh_1xN(2)
        with jax.set_mesh(mesh):
            with self.assertRaises(ValueError) as ctx:
                MergedColumnParallelLinear(
                    input_size=16,
                    output_sizes=[3, 4],
                    mesh=mesh,  # 3 % 2 == 1
                )
            self.assertIn("divisible by TP=2", str(ctx.exception))


class MergedColumnParallelSplitTest(unittest.TestCase):
    def test_stripe_then_split_recovers_components(self):
        """With a striped weight, ``split`` returns each component's output,
        sharded over ``"tensor"``, without moving data between devices."""
        sizes = [16, 16, 32]
        rng = np.random.default_rng(0)
        weight = rng.normal(size=(8, sum(sizes))).astype(np.float32)
        x = rng.normal(size=(4, 8)).astype(np.float32)
        expected = np.split(x @ weight, np.cumsum(sizes)[:-1], axis=-1)
        for tp in (1, 2, 4):
            with self.subTest(tp=tp):
                parts, hlo = self._split_on_mesh(tp, sizes, weight, x)
                for part, want in zip(parts, expected):
                    self.assertEqual(part.sharding.spec, P("data", "tensor"))
                    np.testing.assert_allclose(np.asarray(part), want, rtol=1e-5, atol=1e-5)
                for collective in ("all-gather", "all-to-all", "collective-permute"):
                    self.assertNotIn(collective, hlo)

    @staticmethod
    def _split_on_mesh(tp, sizes, weight, x):
        """Runs the layer with a striped weight on a 1 x tp mesh and splits its
        output; returns the parts and the compiled HLO text."""
        mesh = _mesh_1xN(tp)
        with jax.set_mesh(mesh):
            layer = MergedColumnParallelLinear(
                input_size=weight.shape[0], output_sizes=sizes, mesh=mesh, params_dtype=jnp.float32
            )
            layer.weight.value = jax.device_put(
                stripe_merged_weight(weight, sizes, tp), NamedSharding(mesh, P(None, "tensor"))
            )
            run = jax.jit(lambda x: split_merged_output(layer(x)[0], sizes, mesh))
            x = jax.device_put(x, NamedSharding(mesh, P("data", None)))
            return run(x), run.lower(x).compile().as_text()

    def test_split_after_quantization(self):
        """Quantization replaces the layer with a ``QuantizedLinear``; its
        output keeps the striped layout and splits the same way."""
        sizes = [16, 16, 32]
        rng = np.random.default_rng(0)
        weight = rng.normal(size=(8, sum(sizes))).astype(np.float32)
        x = rng.normal(size=(4, 8)).astype(np.float32)
        expected = np.split(x @ weight, np.cumsum(sizes)[:-1], axis=-1)
        mesh = _mesh_1xN(2)

        class Block(nnx.Module):
            def __init__(self):
                self.proj = MergedColumnParallelLinear(
                    input_size=8, output_sizes=sizes, mesh=mesh, params_dtype=jnp.float32
                )

        config = SimpleNamespace(
            quantization_config=QuantizationConfig(
                linear_rules=[{"module_path": ".*", "weight_dtype": "int8"}]
            )
        )
        with jax.set_mesh(mesh):
            block = Block()
            block.proj.weight.value = jax.device_put(
                stripe_merged_weight(weight, sizes, 2), NamedSharding(mesh, P(None, "tensor"))
            )
            apply_linear_quantization(config, block)
            self.assertIsInstance(block.proj, QuantizedLinear)
            x = jax.device_put(x, NamedSharding(mesh, P("data", None)))
            parts = jax.jit(lambda x: split_merged_output(block.proj(x)[0], sizes, mesh))(x)
        for part, want in zip(parts, expected):
            np.testing.assert_allclose(np.asarray(part, np.float32), want, rtol=0.05, atol=0.1)


if __name__ == "__main__":
    unittest.main()
