import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.utils.jax_utils import effective_axis


class TestEffectiveAxis(unittest.TestCase):
    """Unit tests for the effective_axis sharding helper."""

    def test_concrete_named_sharding_match(self):
        mesh = jax.sharding.Mesh(jax.devices()[:1], ("data",))
        x = jax.device_put(jnp.ones((4, 8)), NamedSharding(mesh, P("data", None)))
        self.assertEqual(effective_axis(x, 0, "data"), "data")
        self.assertIsNone(effective_axis(x, 0, "tensor"))
        self.assertIsNone(effective_axis(x, 1, "data"))

    def test_concrete_named_sharding_mismatch(self):
        mesh = jax.sharding.Mesh(jax.devices()[:1], ("tensor",))
        x = jax.device_put(jnp.ones((4, 8)), NamedSharding(mesh, P(None, "tensor")))
        self.assertIsNone(effective_axis(x, 0, "tensor"))
        self.assertEqual(effective_axis(x, 1, "tensor"), "tensor")

    def test_replicated_returns_none(self):
        x = jnp.ones((4, 8))
        self.assertIsNone(effective_axis(x, 0, "data"))
        self.assertIsNone(effective_axis(x, 1, "tensor"))

    def test_dim_out_of_range(self):
        x = jax.device_put(
            jnp.ones((4,)),
            NamedSharding(jax.sharding.Mesh(jax.devices()[:1], ("data",)), P("data")),
        )
        self.assertIsNone(effective_axis(x, 1, "data"))


if __name__ == "__main__":
    unittest.main()
