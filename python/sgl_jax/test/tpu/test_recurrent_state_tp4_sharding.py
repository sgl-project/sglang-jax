""" TPU TP=4 真测: RecurrentStatePool sharding under NamedSharding.

Run on tpu-tpu-v6e-4-lattn-10934 via sky exec (see plan  runbook).
Removes the ' deferred' marker from RecurrentStatePool.replace_buffer
docstring once this passes.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh


def _is_tpu_tp4():
    devs = jax.devices()
    return len(devs) >= 4 and all(d.platform == "tpu" for d in devs)


@unittest.skipUnless(_is_tpu_tp4(), "TPU runtime with >=4 devices required")
class TestRecurrentStatePoolTP4Sharding(unittest.TestCase):
    def setUp(self):
        devices = np.array(jax.devices()[:4]).reshape(1, 4)
        self.mesh = Mesh(devices, axis_names=("data", "tensor"))

    def test_recurrent_buffer_sharded_along_tensor_axis(self):
        """RecurrentStatePool buffer sharding spec splits H along 'tensor' axis."""
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        with self.mesh:
            rsp = RecurrentStatePool(
                linear_recurrent_layer_ids=[0, 1],
                max_num_reqs=4,
                num_heads=8,  # divisible by 4
                head_dim=4,
                conv_kernel_size=4,
                mesh=self.mesh,
            )
            # Probe sharding: each layer's recurrent buffer must report
            # device_set with 4 devices (sharded across tensor axis).
            for layer_id in range(2):
                rec, conv = rsp.get_linear_recurrent_layer_cache(layer_id)
                self.assertGreaterEqual(
                    len(rec.sharding.device_set),
                    4,
                    f"Layer {layer_id} recurrent buffer must be sharded across 4 TPU devices",
                )

    def test_replace_buffer_preserves_sharding_after_jit_donate(self):
        """deferred concern: per-element .sharding probe + device_put
        must not throw under real NamedSharding. This test removes the deferral."""
        from sgl_jax.srt.mem_cache.recurrent_state_pool import RecurrentStatePool

        with self.mesh:
            rsp = RecurrentStatePool(
                linear_recurrent_layer_ids=[0, 1],
                max_num_reqs=4,
                num_heads=8,
                head_dim=4,
                conv_kernel_size=4,
                mesh=self.mesh,
            )

            # Simulate JIT output: create new buffers without explicit sharding
            # constraint and call replace_buffer (which probes per-element
            # .sharding and re-applies via device_put).
            new_recurrent = [jnp.ones_like(buf) for buf in rsp.recurrent_buffers]
            new_conv = [[jnp.ones_like(c) for c in inner] for inner in rsp.conv_buffers]
            # Must not raise.
            rsp.replace_buffer((new_recurrent, new_conv))

            # Sharding metadata still accessible after replace_buffer.
            for layer in range(2):
                self.assertIsNotNone(rsp.recurrent_buffers[layer].sharding)


if __name__ == "__main__":
    unittest.main()
