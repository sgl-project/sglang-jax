import unittest
from argparse import Namespace

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from sgl_jax.srt.layers.moe import FusedMoE, EPMoE


class TestFusedMoe(unittest.TestCase):

    def setUp(self):
        if not jax.devices():
            self.skipTest("JAX not available")

        self.mesh = jax.sharding.Mesh([jax.devices()], ("data", "tensor"))

        # Initialize random seeds for reproducible results
        self.rng_key = jax.random.PRNGKey(42)
        np.random.seed(42)

    def test_forward(self):
        config = Namespace()
        config.hidden_size = 128
        src_layer = FusedMoE(
            config=config,
            num_experts=8,
            num_experts_per_tok=2,
            mesh=self.mesh,
            rngs=nnx.Rngs(default=0),
        )
        std_layer = EPMoE(
            config=config,
            num_experts=8,
            num_experts_per_tok=2,
            mesh=self.mesh,
            rngs=nnx.Rngs(default=0),
        )
        router_logits = jax.random.normal(self.rng_key, shape=(1, 8), dtype=jnp.float32)
        inputs = jax.random.normal(self.rng_key, shape=(1, 128), dtype=jnp.float32)

        std_output = std_layer(inputs, router_logits)
        src_output = src_layer(inputs, router_logits)

        print(src_output)
        print(std_output)

        assert jnp.allclose(src_output, std_output)


if __name__ == "__main__":
    unittest.main()
