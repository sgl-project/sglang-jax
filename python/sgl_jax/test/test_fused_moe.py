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

        with jax.sharding.use_mesh(self.mesh):
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
                expert_parallel_size=1,
            )
            src_layer.wi_0.value = std_layer.wi_0.value
            src_layer.wi_1.value = std_layer.wi_1.value
            src_layer.wo.value = std_layer.wo.value

            src_layer_state = nnx.state(src_layer)
            src_layer_state_pspecs = nnx.get_partition_spec(src_layer_state)
            src_layer_state = jax.lax.with_sharding_constraint(src_layer_state, src_layer_state_pspecs)
            nnx.update(src_layer, src_layer_state)

            std_layer_state = nnx.state(std_layer)
            std_layer_state_pspecs = nnx.get_partition_spec(std_layer_state)
            std_layer_state = jax.lax.with_sharding_constraint(std_layer_state, std_layer_state_pspecs)
            nnx.update(std_layer, std_layer_state)

            router_logits = jax.random.normal(self.rng_key, shape=(10, 128, 8), dtype=jnp.float32)
            inputs = jax.random.normal(self.rng_key, shape=(10, 128), dtype=jnp.float32)

            std_output = std_layer(inputs, router_logits)
            src_output = src_layer(inputs, router_logits)

        print(src_output)
        print(std_output)

        assert jnp.allclose(src_output, std_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
